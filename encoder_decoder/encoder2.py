from .transformer2 import *
from torch import nn

class Model(nn.Module):
    def __init__(self, encoder, predictor, h_mode="first"):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.h_mode = h_mode
        
    def forward(self, x, data_holder=None, **attn_args):
        if self.h_mode == "first":
            h = self.encoder(x, data_holder=data_holder, **attn_args)["encoder_h"][0]
        else:
            h = self.encoder(x, data_holder=data_holder, **attn_args)["encoder_h"].mean(0)
        
        out = self.predictor(h)
        
        if data_holder is not None:
            data_holder.hidden = h
            if data_holder.keep_grads:
                data_holder.hidden.retain_grad()
            data_holder.predict = out
        return out

def make_model(args, src_vocab, trg_vocab):
    embed_tokens = build_embedding(
            src_vocab, args.encoder_embed_dim
        )
    if not args.encoder_share_embeddings:
        embed_tokens2 = build_embedding(
            src_vocab, args.encoder_embed_dim
        )
    else:
        embed_tokens2 = None
    encoder = TransformerEncoder(args, src_vocab, embed_tokens, embed_tokens2)
    predictor = Linear(args.encoder_embed_dim, 1, bias=True)
    return Model(encoder, predictor, args.h_mode)