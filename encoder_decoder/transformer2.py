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
from .m2 import MultiheadAttention

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

class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, vocab, embed_tokens, embed_tokens2=None):
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

        self.share_embeddings = args.encoder_share_embeddings
        if not self.share_embeddings:
            assert embed_tokens2 is not None
            self.embed_tokens2 = embed_tokens2
            self.embed_positions2 = PositionalEmbedding(
                args.max_source_positions, embed_dim, self.padding_idx,
                learned=args.encoder_learned_pos,
            )
        else:
            self.split_ff = Linear(embed_dim, 2*embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args, i)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.g_layer_norm = LayerNorm(embed_dim)
            self.h_layer_norm = LayerNorm(embed_dim)
        else:
            self.g_layer_norm = None
            self.h_layer_norm = None

    def forward(self, src_tokens, data_holder=None, encoder_mode="gumbel", encoder_temperature=-1, need_weights=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        if self.share_embeddings:
            embedding = self.embed_tokens(src_tokens)
            if data_holder is not None:
                if data_holder.permute_embed is not None:
                    bsz, _, dim = embedding.shape
                    for i in range(bsz):
                        perm = torch.randperm(dim)
                        embedding[i,data_holder.permute_embed] = embedding[i,data_holder.permute_embed,perm]
                if data_holder.keep_grads:
                    data_holder.embedding = embedding
                    data_holder.embedding.retain_grad()
            x = self.embed_scale * embedding
            x += self.embed_positions(src_tokens)
            x = F.dropout(x, p=self.dropout, training=self.training)

            # B x T x C -> T x B x C
            x = x.transpose(0, 1)
            g, h = self.split_ff(x).chunk(2, dim=-1)
        else:
            embedding0 = self.embed_tokens(src_tokens)
            embedding = self.embed_tokens2(src_tokens)
            
            if data_holder is not None:
                if data_holder.permute_embed is not None:
                    bsz, _, dim = embedding.shape
                    for i in range(bsz):
                        perm = torch.randperm(dim)
                        embedding0[i,data_holder.permute_embed] = embedding0[i,data_holder.permute_embed,perm]
                        embedding[i,data_holder.permute_embed] = embedding[i,data_holder.permute_embed,perm]
            
            g = self.embed_scale * embedding0
            g += self.embed_positions(src_tokens)
            g = F.dropout(g, p=self.dropout, training=self.training)
            # B x T x C -> T x B x C
            g = g.transpose(0, 1)
            
            
            if data_holder is not None:
                if data_holder.keep_grads:
                    data_holder.embedding = embedding
                    data_holder.embedding.retain_grad()
            h = self.embed_scale * embedding
            h += self.embed_positions2(src_tokens)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # B x T x C -> T x B x C
            h = h.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
            
#         if encoder_padding_mask is None:
#             print("None")
#         else:
#             print(encoder_padding_mask.shape)
#             print((1-encoder_padding_mask).sum(-1))

        # encoder layers
        attn_data_list = []
        for layer in self.layers:
            g, h, attn_data = layer(
                g,
                h, 
                data_holder=data_holder,
                encoder_padding_mask=encoder_padding_mask, 
                encoder_mode=encoder_mode, 
                encoder_temperature=encoder_temperature,
                need_weights=need_weights)
            attn_data_list.append(attn_data)

        if self.g_layer_norm:
            g = self.g_layer_norm(g)
        if self.h_layer_norm:
            h = self.h_layer_norm(h)

        return {
            'encoder_g': g,  # T x B x C
            'encoder_h': h,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
            'encoder_attn_data_list' : attn_data_list
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_g'] is not None:
            encoder_out['encoder_g'] = \
                encoder_out['encoder_g'].index_select(1, new_order)
        if encoder_out['encoder_h'] is not None:
            encoder_out['encoder_h'] = \
                encoder_out['encoder_h'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, i):
        super().__init__()
        self.layer_num = i
        
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout, self_attention=True,
            layer_num = self.layer_num
        )
        self.g_self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.h_self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu')
        )
        self.activation_dropout = getattr(args, 'activation_dropout', 0)

        self.normalize_before = args.encoder_normalize_before
        self.h_to_g = args.encoder_h_to_g
        self.g_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.g_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.h_fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.h_fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.g_final_layer_norm = LayerNorm(self.embed_dim)
        self.h_final_layer_norm = LayerNorm(self.embed_dim)
        

    def forward(self, g, h, data_holder=None, encoder_padding_mask=None, encoder_mode="gumbel", encoder_temperature=-1, need_weights=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        g_residual = g
        h_residual = h
        g = self.maybe_layer_norm(self.g_self_attn_layer_norm, g, before=True)
        h = self.maybe_layer_norm(self.h_self_attn_layer_norm, h, before=True)
        g, h, attn_data = self.self_attn(
            g=g,
            h=h, 
            data_holder=data_holder,
            key_padding_mask=encoder_padding_mask, 
            mode=encoder_mode,
            temperature=encoder_temperature,
            need_weights=need_weights)
        g = F.dropout(g, p=self.dropout, training=self.training)
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.h_to_g:
            g = g_residual + g + h
        else:
            g = g_residual + g
        h = h_residual + h
        g = self.maybe_layer_norm(self.g_self_attn_layer_norm, g, after=True)
        h = self.maybe_layer_norm(self.h_self_attn_layer_norm, h, after=True)

        g_residual = g
        h_residual = h
        g = self.maybe_layer_norm(self.g_final_layer_norm, g, before=True)
        h = self.maybe_layer_norm(self.h_final_layer_norm, h, before=True)
        g = self.activation_fn(self.g_fc1(g))
        h = self.activation_fn(self.h_fc1(h))
        g = F.dropout(g, p=self.activation_dropout, training=self.training)
        h = F.dropout(h, p=self.activation_dropout, training=self.training)
        g = self.g_fc2(g)
        h = self.h_fc2(h)
        g = F.dropout(g, p=self.dropout, training=self.training)
        h = F.dropout(h, p=self.dropout, training=self.training)
        g = g_residual + g
        h = h_residual + h
        g = self.maybe_layer_norm(self.g_final_layer_norm, g, after=True)
        h = self.maybe_layer_norm(self.h_final_layer_norm, h, after=True)
        return g, h, attn_data

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
