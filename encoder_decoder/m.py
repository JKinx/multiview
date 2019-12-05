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

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, h, h_p=None, data_holder=None, key_padding_mask=None, attn_mask=None, incremental_state=None, need_weights=True, static_kv=False, mode="soft", temperature=-1):
        """Input shape: Time x Batch x Channel
        h = hidden vectors
        h_p = encoder hidden vectors while in encoder_decoder mode

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = h.size()
        assert embed_dim == self.embed_dim
        assert list(h.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    h_p = None
        else:
            saved_state = None

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(h)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(h)
            if h_p is None:
                k = v = None
            else:
                k = self.in_proj_k(h_p)
                v = self.in_proj_v(h_p)
        else:
            raise ValueError("Invalid Mode")

        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1, self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)

        src_len = k.size(1)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                attn_weights += attn_mask
            elif attn_mask.dim() == 3:
                mask = torch.zeros_like(attn_mask).float()
                mask = mask.masked_fill(attn_mask, float('-inf'))
                mask = mask.unsqueeze(1)
                attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                attn_weights += mask
                attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            else:
                raise NotImplementedError

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
            
        if data_holder is not None:
            if data_holder.permute_attn is not None:
                idx = data_holder.permute_attn
                for i in range(bsz * self.num_heads):
                    perm = torch.randperm(src_len)
                    attn_weights[i] = attn_weights[i,:,perm]
            
        attn_p = utils.softmax(
            attn_weights, dim=-1, onnx_trace=False,
        ).type_as(attn_weights)

        if mode == "soft":
            if data_holder is not None:
                if data_holder.detach:
                    attn_p = attn_p.detach()
                data_holder.attn.append(attn_p)
            attn = torch.bmm(attn_p, v)
            sample = None
        elif mode == "gumbel":
            assert(temperature != -1)
            sample = gumbel_softmax_sample(F.log_softmax(attn_weights, dim = -1), temperature)
            if data_holder is not None:
                if data_holder.detach:
                    sample = sample.detach()
                data_holder.attn.append(sample)
            attn = torch.bmm(sample, v)
        elif mode == "greedy":
            sample_id = attn_weights.view(-1, src_len).argmax(dim=-1)
            sample_id = sample_id.view(bsz * self.num_heads, tgt_len, 1)
            sample = attn_weights.new_full(attn_weights.size(), 0.).to(attn_weights)
            sample.scatter_(2, sample_id, 1.)
            if data_holder is not None:
                if data_holder.detach:
                    sample = sample.detach()
                data_holder.attn.append(sample)
            attn = torch.bmm(sample, v)
        else:
            raise ValueError("Invalid Attention Mode!")
            
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            attn_data = {"attn" : attn_p.view(bsz, self.num_heads, tgt_len, src_len)}
            if sample is not None:
                attn_data.update({"sample" : sample.view(bsz, self.num_heads, tgt_len, src_len)})
        else:
            attn_data = None

        return attn, attn_data

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
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