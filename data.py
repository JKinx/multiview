from torchtext import data
import io
import torch.nn.functional as F
import torch
from tqdm import tqdm
import pickle
import math
from collections import defaultdict
import numpy as np
import os

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(b)

def batch_size_fn_with_padding(new, count, sofar):
    """
    In token batching scheme, the number of sequences is limited
    such that the total number of src/trg tokens (including padding)
    in a batch <= batch_size
    """
    # Maintains the longest src length in the current batch
    global max_src_in_batch
    # Reset current longest length at a new batch (count=1)
    if count == 1:
        max_src_in_batch = 0
    if not hasattr(new, 'src'):
        new_src = 0
    else:
        new_src = len(new.src)
        
    # Src: <bos> w1 ... wN <eos>
    max_src_in_batch = max(max_src_in_batch, new_src + 2)
    src_elements = count * max_src_in_batch
    return src_elements

def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)

class BCDataset(data.Dataset):
    def __init__(self, path, SRC, TRG, max_len=float("inf"), **kwargs):
        fields = [('src', SRC), ('trg', TRG)]
        src_text = []
        trg_text = []
        too_long = 0
        with open(path + ".text") as tf,\
            open(path + ".label") as lf:
            
            for line, label in zip(tf, lf):
                text = line.split()
                if len(text) > max_len:
                    too_long += 1
                    continue
                src_text.append(text)
                trg_text.append(label)
        print("Path : %s, Total : %d, Dropped : %d"%(path, len(src_text), too_long))
                
        examples = [data.Example.fromlist([src, trg], fields) \
                    for (src, trg) in zip(src_text, trg_text)]
                
        super(BCDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, SRC, TRG, root='.',
               train='train', validation='dev', test='test', max_len=float("inf"), **kwargs):
        return super(BCDataset, cls).splits(
            root=root, train=train, validation=validation, test=test,
            SRC=SRC, TRG=TRG, max_len=max_len, **kwargs)
