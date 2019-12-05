import argparse
import sys
import math

import torch
import torchtext

from data import *
from encoder_decoder.encoder2 import *
from optim import *
from encoder_decoder.utils import AverageMeter
import time 
import numpy as np
from copy import deepcopy as dc
import random

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
CLS = '<cls>'

args = {"activation_fn" : 'relu',
        "dropout" : 0.4, 
        "attention_dropout": 0.,
        "activation_dropout": 0.,
        "encoder_embed_dim": 64,
        "encoder_ffn_embed_dim": 128,
        "encoder_layers": 6,
        "encoder_attention_heads": 8,
        "encoder_normalize_before": True,
        "encoder_learned_pos": False,
        "warmup_updates" : 3000,
        "warmup_init_lr": 1e-07,
        "adam_betas": '(0.9, 0.999)',
        "max_source_positions": 500,
        "adam_eps": 1e-8,
        "weight_decay": 0.00,
        "lr": 0.0001,
        "clip_norm" : -1,
        "batch_size": 2000,
        "epochs": 50,
        "print_every": 100,
        "temperature": -1.,
        "h_to_g" : False,
        "share_embeddings" : False,
        "mode" : "control-g",
        "has_dev" : True,
        "h_mode" : "first"
       }

parser = argparse.ArgumentParser(
    description='train_encoder_fair.py',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, help="Path to the dataset")
parser.add_argument("--max-source-positions", type=int, default=args["max_source_positions"], help="Maximum Src Length")
parser.add_argument("--batch-size", type=int, default=args["batch_size"], help="Number of tokens per minibatch")
parser.add_argument("--epochs", type=int, default=args["epochs"], help="Number of Epochs")
parser.add_argument("--train-from", default="", help="Model Path to train from")
parser.add_argument("--print-every", type=int, default=args["print_every"], help="Interval to print logs")
parser.add_argument("--save-to", type=str, required=True, help="Model Path to save to")
parser.add_argument("--temperature", type=float, help="Gumbel temperature")
parser.add_argument("--mode", type=str, default=args["mode"], help="Attention Mode")

parser.add_argument('--lr', default=args["lr"], type=float, metavar='N',
                    help='learning rate')
parser.add_argument('--warmup-updates', default=args["warmup_updates"], type=int, metavar='N',
                    help='warmup the learning rate linearly for the first N updates')
parser.add_argument('--warmup-init-lr', default=args["warmup_init_lr"], type=float, metavar='LR',
                    help='initial learning rate during warmup phase; default is args.lr')
parser.add_argument('--clip-norm', default=args["clip_norm"], type=float, metavar='LR',
                    help='max gradient norm')

parser.add_argument('--adam-betas', default=args["adam_betas"], metavar='B',
                            help='betas for Adam optimizer')
parser.add_argument('--adam-eps', type=float, default=args["adam_eps"], metavar='D',
                    help='epsilon for Adam optimizer')
parser.add_argument('--weight-decay', '--wd', default=args["weight_decay"], type=float, metavar='WD',
                            help='weight decay')

parser.add_argument('--activation-fn', default=args["activation_fn"],
                            help='activation function to use')
parser.add_argument('--dropout', type=float, default=args["dropout"], metavar='D',
                    help='dropout probability')
parser.add_argument('--attention-dropout', type=float, default=args["attention_dropout"], metavar='D',
                    help='dropout probability for attention weights')
parser.add_argument('--activation-dropout', '--relu-dropout', default=args["activation_dropout"], type=float, metavar='D',
                    help='dropout probability after activation in FFN.')
parser.add_argument('--encoder-embed-dim', type=int, default=args["encoder_embed_dim"], metavar='N',
                    help='encoder embedding dimension')
parser.add_argument('--encoder-ffn-embed-dim', type=int, default=args["encoder_ffn_embed_dim"], metavar='N',
                    help='encoder embedding dimension for FFN')
parser.add_argument('--encoder-layers', type=int, default=args["encoder_layers"], metavar='N',
                    help='num encoder layers')
parser.add_argument('--encoder-attention-heads', type=int, default=args["encoder_attention_heads"], metavar='N',
                    help='num encoder attention heads')
parser.add_argument('--encoder-normalize-before', type=lambda string: True if string == "True" else False, default=args["encoder_normalize_before"],
                    help='apply layernorm before each encoder block')
parser.add_argument('--encoder-learned-pos', type=lambda string: True if string == "True" else False, default=args["encoder_learned_pos"], 
                    help='use learned positional embeddings in the encoder')
parser.add_argument('--encoder-h-to-g', type=lambda string: True if string == "True" else False, default=args["h_to_g"], 
                    help='Information flow from h to g after attention')

parser.add_argument('--encoder-share-embeddings', type=lambda string: True if string == "True" else False, default=args["share_embeddings"], 
                    help='Share embeddings for g and h')
parser.add_argument('--out-activation', type=str, default="softmax")

parser.add_argument('--has-dev', type=lambda string: True if string == "True" else False, default=args["has_dev"], 
                    help='Has dev set')

parser.add_argument("--h-mode", type=str, default=args["h_mode"], help="Encoder h Mode")

opts = parser.parse_args()

def gumbel_to_greedy(attn_args):
    for key in attn_args:
        if attn_args[key] == "gumbel":
            attn_args[key] = "greedy"
    return attn_args

def main(opts):
    SRC = torchtext.data.Field(
                    pad_token=PAD_WORD,
                    unk_token=UNK_WORD,
                    init_token=CLS)

    TRG = torchtext.data.Field(sequential=False, unk_token=None, pad_token=None)
    
    if opts.has_dev:
        train, val, test = BCDataset.splits(SRC, TRG,
                path=opts.dataset, train='train', validation='dev', test='test',
                max_len=opts.max_source_positions)
    else:
        train, test = BCDataset.splits(SRC, TRG,
                path=opts.dataset, train='train', validation=None, test='test',
                max_len=opts.max_source_positions)
        random.seed(606)
        train, val = train.split(0.9, random_state=random.getstate())

    SRC.build_vocab(train.src)
    TRG.build_vocab(train.trg)

    batch_size_fn = batch_size_fn_with_padding

    BATCH_SIZE = opts.batch_size
    
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                        repeat=False, sort_key=lambda x: len(x.src),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True, sort_within_batch=True)

    val_iter = MyIterator(val, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                            repeat=False, sort_key=lambda x: len(x.src),
                            batch_size_fn=batch_size_fn, train=False)

    test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=torch.device('cuda:0'),
                                repeat=False, sort_key=lambda x: len(x.src),
                                batch_size_fn=batch_size_fn, train=False)

    print (' '.join(sys.argv))

    trg_vocab_size = len(TRG.vocab.itos)
    src_vocab_size = len(SRC.vocab.itos)
    print('SRC Vocab Size: %d, TRG Vocab Size: %d'%(src_vocab_size, trg_vocab_size))

    print ('Building Model')

    model = make_model(opts, SRC.vocab, TRG.vocab)
    print(model)
        
    model.cuda()
    
    num_updates = 0
    if opts.train_from != "":
        checkpoint = torch.load(opts.train_from)
        model.load_state_dict(checkpoint['model'])
        num_updates = checkpoint["num_updates"]
        num_updates = 0
    print(num_updates)
        
    optimizer = FairseqAdam(opts, list(model.parameters()))
    lr_scheduler = InverseSquareRootSchedule(opts, optimizer)
        
    lr_scheduler.step_update(num_updates)

    print(opts)
    best_vacc = -float("inf")

    default_attn_args = {"encoder_mode" : opts.mode, 
                         "encoder_temperature" : opts.temperature}
    sys.stdout.flush()
    for epoch in range(opts.epochs):
        print ('')
        print ('Epoch: %d' %epoch)
        print ('Training')
        sys.stdout.flush()
        model.train()

        batch_time_meter = AverageMeter()
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        start = time.time()
        attn_args = dc(default_attn_args)

        for i, batch in enumerate(train_iter):
            src = batch.src.transpose(0, 1) # batch, len
            trg = batch.trg 
            n = trg.shape[0]
            
            prediction_probs = model(src, **attn_args)
            loss = model.criterion(prediction_probs, trg.unsqueeze(-1).float()).mean()

            loss.backward()
            optimizer.clip_grad_norm(opts.clip_norm)
            optimizer.step()
            num_updates += 1
            lr_scheduler.step_update(num_updates)
            optimizer.zero_grad()
                
            accuracy = ((prediction_probs.sigmoid() > 0.5).long().squeeze() == trg).sum().item() / n

            accuracy_meter.update(accuracy, n)
            loss_meter.update(loss.item(), n)
            batch_time_meter.update(time.time() - start)
            start = time.time()
            

            if i % opts.print_every == 0:
                print("batch_idx : %d"%(i), end=" ")
                print("accuracy : %.4f"%(accuracy_meter.avg), end=" ")
                print("loss : %.4f"%(loss_meter.avg), end=" ")
                print("batch_time : %.4f"%(batch_time_meter.avg))
                sys.stdout.flush()

        print ('Validation')
        sys.stdout.flush()
        vbatch_time_meter = AverageMeter()
        vloss_meter = AverageMeter()
        vaccuracy_meter = AverageMeter()
        model.eval()

        start = time.time()
        attn_args = gumbel_to_greedy(dc(default_attn_args))

        with torch.no_grad():
            for i, batch in enumerate(val_iter):
                src = batch.src.transpose(0, 1) # batch, len
                trg = batch.trg 

                n = trg.shape[0]

                prediction_probs = model(src, **attn_args)
                loss = model.criterion(prediction_probs, trg.unsqueeze(-1).float()).mean()

                lr_scheduler.step(epoch, loss.item())  
                accuracy = ((prediction_probs.sigmoid() > 0.5).long().squeeze() == trg).sum().item() / n

                vaccuracy_meter.update(accuracy, n)
                vloss_meter.update(loss.item(), n)
                vbatch_time_meter.update(time.time() - start)
                start = time.time()

            print("accuracy : %.4f"%(vaccuracy_meter.avg), end=" ")
            print("loss : %.4f"%(vloss_meter.avg), end=" ")
            print("batch_time : %.4f"%(vbatch_time_meter.avg))
            sys.stdout.flush()

            if vaccuracy_meter.avg > best_vacc:
                best_vacc = vaccuracy_meter.avg
                torch.save({'model': model.state_dict(), 
                            'opts': opts, 'optimizer': optimizer.state_dict(), 'lr_scheduler' : lr_scheduler, 
                            'num_updates' : num_updates, "val":best_vacc}, '%s.best.pt'%(opts.save_to))

        model.train()

if __name__ == '__main__':
    main(opts)