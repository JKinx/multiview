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

class MultiheadAttentionC(nn.Module):
    """Multi-headed attention.
    Controller stream.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 self_attention=False, encoder_decoder_attention=False):
        super().__init__()
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

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.g_in_proj_weight)
        nn.init.xavier_uniform_(self.g_out_proj.weight)
        if self.g_in_proj_bias is not None:
            nn.init.constant_(self.g_in_proj_bias, 0.)
            nn.init.constant_(self.g_out_proj.bias, 0.)

    def forward(self, g, g_p=None, key_padding_mask=None, attn_mask=None,need_weights=True, mode="gumbel", temperature=-1):
        """Input shape: Time x Batch x Channel
        g = hidden vectors
        g_p = encoder hidden vectors while in encoder_decoder mode

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = g.size()
        assert embed_dim == self.embed_dim
        assert list(g.size()) == [tgt_len, bsz, embed_dim]

        if self.self_attention:
            # self-attention
            q, gk, gv = self.g_in_proj_qkv(g)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.g_in_proj_q(g)
            assert g_p is not None
            gk, gv = self.g_in_proj_kv(g_p)
        else:
            raise ValueError("Invalid Mode")

        q *= self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        gk = gk.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        gv = gv.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
        src_len = gk.size(1)

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        g_attn_weights = torch.bmm(q, gk.transpose(1, 2))
        assert list(g_attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            g_attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            g_attn_weights = g_attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            g_attn_weights = g_attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            g_attn_weights = g_attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
                
        g_attn_p = utils.softmax(
                g_attn_weights, dim=-1, onnx_trace=False,
            ).type_as(g_attn_weights)
        g_attn_p = F.dropout(g_attn_p, p=self.dropout, training=self.training)
        
        g_attn = torch.bmm(g_attn_p, gv)

        if mode == "control-g":
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
        
        assert list(g_attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        
        g_attn = g_attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        g_attn = self.g_out_proj(g_attn)

        if need_weights:
            attn_data = {
                "g_attn" : g_attn_p.view(bsz, self.num_heads, tgt_len, src_len).detach(), 
                "sample" : sample.view(bsz, self.num_heads, tgt_len, src_len).detach()
            }
        else:
            attn_data = None

        return g_attn, sample, attn_data

    def g_in_proj_qkv(self, g):
        return self._g_in_proj(g).chunk(3, dim=-1)

    def g_in_proj_q(self, g):
        return self._g_in_proj(g, end=self.embed_dim)

    def g_in_proj_kv(self, g):
        return self._g_in_proj(g, start=self.embed_dim).chunk(2, dim=-1)

    def _g_in_proj(self, input, start=0, end=None):
        weight = self.g_in_proj_weight
        bias = self.g_in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class MultiheadAttentionV(nn.Module):
    """Multi-headed attention.
    Value stream.
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True,
                 self_attention=False, encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim = embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        assert self_attention ^ encoder_decoder_attention
        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        self.h_in_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        if bias:
            self.h_in_proj_bias = Parameter(torch.Tensor(embed_dim))
        self.h_out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.h_in_proj_weight)
        nn.init.xavier_uniform_(self.h_out_proj.weight)
        if self.h_in_proj_bias is not None:
            nn.init.constant_(self.h_in_proj_bias, 0.)
            nn.init.constant_(self.h_out_proj.bias, 0.)

    def forward(self, h=None, h_p=None, attn=None):
        """Input shape: Time x Batch x Channel
        h = hidden vectors
        h_p = encoder hidden vectors while in encoder_decoder mode

        attn = attention from controller state
            shape : bsz * num_heads X trg_len X src_len
        """

        tgt_len, bsz, embed_dim = g.size()
        assert embed_dim == self.embed_dim
        assert list(h.size()) == [tgt_len, bsz, embed_dim]

        if self.self_attention:
            # self-attention
            assert h is not None
            hv = self.h_in_proj_v(h)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            assert h_p is not None
            hv = self.h_in_proj_v(h_p)
        else:
            raise ValueError("Invalid Mode")

        hv = hv.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        
        h_attn = torch.bmm(attn, hv)
        assert list(h_attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        h_attn = h_attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        h_attn = self.h_out_proj(h_attn)

        return h_attn

    def h_in_proj_v(self, h):
        return self._h_in_proj(h)

    def _h_in_proj(self, input, start=0, end=None):
        weight = self.h_in_proj_weight
        bias = self.h_in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
