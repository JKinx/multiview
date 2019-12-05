from .transformer2 import *
from torch import nn

class Model(nn.Module):
    def __init__(self, encoderC, encoderV, predictor, h_mode="first"):
        super().__init__()
        self.encoderC = encoderC
        self.encoderV = encoderV
        self.predictor = predictor
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.h_mode = h_mode
        
    def forward(self, x, **attn_args):
        attn_list = self.encoderC(x, **attn_args)["encoder_attn"]
        if self.h_mode == "first":
            h = self.encoderV(x, attn_list)["encoder_h"][0]
        else:
            h = self.encoderV(x, attn_list)["encoder_h"].mean(0)
        
        out = self.predictor(h)
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
        embed_tokens2 = embed_tokens
    encoderC = TransformerEncoderC(args, src_vocab, embed_tokens)
    encoderV = TransformerEncoderV(args, src_vocab, embed_tokens2)
    predictor = Linear(args.encoder_embed_dim, 1, bias=True)
    return Model(encoderC, encoderV, predictor, args.h_mode)