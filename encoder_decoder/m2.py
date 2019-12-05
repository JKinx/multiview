# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import sys

from fairseq import utils

def sample_gumbel(input):
    # sample from a gumbel distribution Gumbel(0,1)
    # u ~ Uniform(0,1)
    # g = -log(-log(u))
    noise = torch.empty_like(input).uniform_()
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    return noise

def gumbel_softmax_sample(log_probs, temperature):
    # sample from gumble softmax to approximate sampling from categorical distribution
    noise = sample_gumbel(log_probs)
    x = (log_probs + noise) / temperature
    x = F.softmax(x, dim=-1)
    return x.view_as(log_probs)

def gumbel_perturb(logits, temperature):
    noise = sample_gumbel(logits)
    return (logits + noise) / temperature

class MultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 self_attention=False, layer_num=-1,
                 encoder_decoder_attention=False):
        super().__init__()
        self.layer_num = layer_num
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        assert self_attention ^ encoder_decoder_attention
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.g_in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.g_in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.g_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.h_in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        if bias:
            self.h_in_proj_bias = Parameter(torch.Tensor(2 * embed_dim))
        self.h_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.g_in_proj_weight)
        nn.init.xavier_uniform_(self.g_out_proj.weight)
        nn.init.xavier_uniform_(self.h_in_proj_weight)
        nn.init.xavier_uniform_(self.h_out_proj.weight)
        if self.g_in_proj_bias is not None:
            nn.init.constant_(self.g_in_proj_bias, 0.)
            nn.init.constant_(self.g_out_proj.bias, 0.)
            nn.init.constant_(self.h_in_proj_bias, 0.)
            nn.init.constant_(self.h_out_proj.bias, 0.)

    def forward(self, g, h=None, g_p=None, h_p=None, data_holder=None, key_padding_mask=None, attn_mask=None, incremental_state=None, need_weights=True, static_kv=False, mode="gumbel", temperature=-1):
        """Input shape: Time x Batch x Channel
        g, h = hidden vectors
        g_p, h_p = encoder hidden vectors while in encoder_decoder mode

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = g.size()
        assert embed_dim == self.embed_dim
        assert list(g.size()) == [tgt_len, bsz, embed_dim]
        assert list(h.size()) == [tgt_len, bsz, embed_dim] if h is not None else True

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'g_prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    g_p = h_p = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            assert h is not None
            q, gk, gv = self.g_in_proj_qkv(g)
            hk, hv = self.h_in_proj_kv(h)
#             hk = None
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.g_in_proj_q(g)
            if g_p is None and h_p is None:
                gk = gv = None
                hk = hv = None
            else:
                assert g_p is not None and h_p is not None
                gk, gv = self.g_in_proj_kv(g_p)
                hk, hv = self.h_in_proj_kv(h_p)
#                 hk = None
        else:
            raise ValueError("Invalid Mode")

        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if gk is not None:
            gk = gk.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if gv is not None:
            gv = gv.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if hk is not None:
            hk = hk.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if hv is not None:
            hv = hv.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
        src_len = gk.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        g_attn_weights = torch.bmm(q, gk.transpose(1, 2))
        assert list(g_attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        h_attn_weights = torch.bmm(q, hk.transpose(1, 2))
        assert list(h_attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            g_attn_weights += attn_mask
            h_attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            g_attn_weights = g_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            g_attn_weights = g_attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            g_attn_weights = g_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

            h_attn_weights = h_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            h_attn_weights = h_attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            h_attn_weights = h_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            
        if data_holder is not None:
            if data_holder.permute_attn is not None:
                idx = data_holder.permute_attn
                for i in range(bsz * self.num_heads):
                    perm = torch.randperm(src_len)
                    g_attn_weights[i] = g_attn_weights[i,:,perm]
#                     for j in range(tgt_len):
#                         perm = torch.randperm(src_len)
#                         g_attn_weights[i,j] = g_attn_weights[i,j,perm]
                
        g_attn_p = utils.softmax(
                g_attn_weights, dim=-1, onnx_trace=False,
            ).type_as(g_attn_weights)
        g_attn_p = F.dropout(g_attn_p, p=self.dropout, training=self.training)

        h_attn_p = utils.softmax(
            h_attn_weights, dim=-1, onnx_trace=False,
        ).type_as(h_attn_weights)
        
        if data_holder is not None:
            if data_holder.detach:
                g_attn_p = g_attn_p.detach()

        g_attn = torch.bmm(g_attn_p, gv)

        if mode == "gumbel":
            assert(temperature != -1)
            sample = gumbel_softmax_sample(F.log_softmax(h_attn_weights, dim=-1), temperature)
        elif mode == "greedy":
            sample_id = h_attn_weights.view(-1, src_len).argmax(dim=-1)
            sample_id = sample_id.view(bsz * self.num_heads, tgt_len, 1)
            sample = h_attn_weights.new_full(h_attn_weights.size(), 0.).to(h_attn_weights)
            sample.scatter_(2, sample_id, 1.)
        elif mode == "soft":
            sample = h_attn_p
        elif mode == "control-g":
            assert(temperature != -1)
            if self.training:
                sample = gumbel_softmax_sample(F.log_softmax(g_attn_weights, dim=-1), temperature)
            else:
                sample_id = g_attn_weights.view(-1, src_len).argmax(dim=-1)
                sample_id = sample_id.view(bsz * self.num_heads, tgt_len, 1)
                sample = g_attn_weights.new_full(g_attn_weights.size(), 0.).to(g_attn_weights)
                sample.scatter_(2, sample_id, 1.)
        elif mode == "control":
            sample = g_attn_p
        else:
            raise ValueError("Invalid Attention Mode!")
        if data_holder is not None:
            if data_holder.detach:
                sample = sample.detach()
            data_holder.attn.append(sample)
        h_attn = torch.bmm(sample, hv)
        
        assert list(g_attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        assert list(h_attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        g_attn = g_attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        g_attn = self.g_out_proj(g_attn)

        h_attn = h_attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        h_attn = self.h_out_proj(h_attn)

        if need_weights:
            attn_data = {
                "g_attn" : g_attn_p.view(bsz, self.num_heads, tgt_len, src_len), 
                "h_attn" : h_attn_p.view(bsz, self.num_heads, tgt_len, src_len), 
                "sample" : sample.view(bsz, self.num_heads, tgt_len, src_len)
            }
        else:
            attn_data = None

        return g_attn, h_attn, attn_data

    def g_in_proj_qkv(self, g):
        return self._g_in_proj(g).chunk(3, dim=-1)

    def g_in_proj_q(self, g):
        return self._g_in_proj(g, end=self.embed_dim)

    def g_in_proj_kv(self, g):
        return self._g_in_proj(g, start=self.embed_dim).chunk(2, dim=-1)

    def h_in_proj_kv(self, h):
        return self._h_in_proj(h).chunk(2, dim=-1)

    def _g_in_proj(self, input, start=0, end=None):
        weight = self.g_in_proj_weight
        bias = self.g_in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def _h_in_proj(self, input, start=0, end=None):
        weight = self.h_in_proj_weight
        bias = self.h_in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(
            self,
            incremental_state,
            'attn_state',
        ) or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(
            self,
            incremental_state,
            'attn_state',
            buffer,
        )
