# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding
)
from .m2 import MultiheadAttentionC, MultiheadAttentionV

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024
PAD_WORD = '<blank>'

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def build_embedding(vocab, embed_dim):
    num_embeddings = len(vocab)
    padding_idx = vocab.stoi[PAD_WORD]
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    return emb

class TransformerEncoderC(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Controller stream

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, vocab, embed_tokens):
        super().__init__()
        self.vocab = vocab

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) 

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayerC(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.g_layer_norm = LayerNorm(embed_dim)
        else:
            self.g_layer_norm = None

    def forward(self, src_tokens, encoder_mode="gumbel", encoder_temperature=-1, need_weights=False, **unused):
        # embed tokens and positions
        embedding = self.embed_tokens(src_tokens)        
        g = self.embed_scale * embedding
        g += self.embed_positions(src_tokens)
        g = F.dropout(g, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        g = g.transpose(0, 1)
        
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
            
        # encoder layers
        attn_list = []
        attn_data_list = []
        for layer in self.layers:
            g, attn, attn_data = layer(
                g,
                encoder_padding_mask=encoder_padding_mask, 
                encoder_mode=encoder_mode, 
                encoder_temperature=encoder_temperature,
                need_weights=need_weights)
            attn_list.append(attn)
            attn_data_list.append(attn_data)

        if self.g_layer_norm:
            g = self.g_layer_norm(g)

        return {
            'encoder_g': g,  # T x B x C
            'encoder_attn': attn_list,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_attn_data_list' : attn_data_list
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerEncoderV(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Value Stream.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, vocab, embed_tokens):
        super().__init__()
        self.vocab = vocab

        self.dropout = args.dropout

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        )

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayerV(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.h_layer_norm = LayerNorm(embed_dim)
        else:
            self.h_layer_norm = None

    def forward(self, src_tokens, attn_list):
        # embed tokens and positions
        embedding = self.embed_tokens(src_tokens)
        h = self.embed_scale * embedding
        h += self.embed_positions(src_tokens)
        h = F.dropout(h, p=self.dropout, training=self.training)
        # B x T x C -> T x B x C
        h = h.transpose(0, 1)

        for i,layer in enumerate(self.layers):
            h = layer(
                h, 
                attn_list[i]
                )

        if self.h_layer_norm:
            h = self.h_layer_norm(h)

        return {
            'encoder_h': h,  # T x B x C
        }

class TransformerEncoderLayerC(nn.Module):
    """Encoder layer block.
    Controller Stream.
    """

    def __init__(self, args):
        super().__init__()        
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttentionC(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True
        )
        self.g_self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)

        self.normalize_before = args.encoder_normalize_before
        self.g_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.g_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.g_final_layer_norm = LayerNorm(self.embed_dim)        

    def forward(self, g, encoder_padding_mask=None, encoder_mode="gumbel", encoder_temperature=-1, need_weights=False):
        g_residual = g
        g = self.maybe_layer_norm(self.g_self_attn_layer_norm, g, before=True)
        g, attn, attn_data = self.self_attn(
            g=g,
            key_padding_mask=encoder_padding_mask, 
            mode=encoder_mode,
            temperature=encoder_temperature,
            need_weights=need_weights)
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = g_residual + g
        g = self.maybe_layer_norm(self.g_self_attn_layer_norm, g, after=True)

        g_residual = g
        g = self.maybe_layer_norm(self.g_final_layer_norm, g, before=True)
        g = self.activation_fn(self.g_fc1(g))
        g = F.dropout(g, p=self.activation_dropout, training=self.training)
        g = self.g_fc2(g)
        g = F.dropout(g, p=self.dropout, training=self.training)
        g = g_residual + g
        g = self.maybe_layer_norm(self.g_final_layer_norm, g, after=True)
        return g, attn, attn_data

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

class TransformerEncoderLayerV(nn.Module):
    """Encoder layer block.
    Value stream.
    """

    def __init__(self, args):
        super().__init__()
        self.layer_num = i
        
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttentionV(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True
        )
        self.h_self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)

        self.normalize_before = args.encoder_normalize_before
        self.h_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.h_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.h_final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, h, attn):
        h_residual = h
        h = self.maybe_layer_norm(self.h_self_attn_layer_norm, h, before=True)
        h = self.self_attn(
            h=h, 
            attn=attn)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h_residual + h
        h = self.maybe_layer_norm(self.h_self_attn_layer_norm, h, after=True)

        h_residual = h
        h = self.maybe_layer_norm(self.h_final_layer_norm, h, before=True)
        h = self.activation_fn(self.h_fc1(h))
        h = F.dropout(h, p=self.activation_dropout, training=self.training)
        h = self.h_fc2(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h_residual + h
        h = self.maybe_layer_norm(self.h_final_layer_norm, h, after=True)
        return h

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
