import numpy as np
from copy import deepcopy as dc
import torch
from tqdm import tqdm_notebook as tqdm

class DataHolder() : 
    def __init__(self, detach=False, keep_grads=False, permute_embed=None, permute_attn=None) :
        self.detach = detach
        self.keep_grads = keep_grads
        self.permute_embed = permute_embed
        self.permute_attn = permute_attn
        self.hidden = None
        self.predict = None
        self.attn = []
        
class AverageMeter:
    def __init__(self):
        self.value = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count
        
def remove_and_run(model, data_iter, attn_args) :
    model.eval()
    outputs = []
    targets = []
    corrects = []
    
    for batch in tqdm(data_iter):
        src = batch.src[0].transpose(0, 1) # batch, len
        trg = batch.trg 
        
        targets.append(trg.cpu().data.numpy())
        
        po = np.zeros((src.shape[0], src.shape[1], 1))
        
        n = src.shape[1]
        
        correct = model(src, **attn_args).sigmoid().cpu().data.numpy()
        corrects.append(correct)
        
        for i in range(n):
            in_src = torch.cat([src[:, :i], src[:, i+1:]], dim=-1)
            
            pred = model(in_src, **attn_args)
            po[:, i] = pred.sigmoid().cpu().data.numpy()

        outputs.append(po)
    targets = [x for y in targets for x in y]
    outputs = [x for y in outputs for x in y]
    corrects = [x for y in corrects for x in y]
    
    return outputs, targets, corrects

def tvd(a, b):
    return np.abs(a - b).sum(1) / 2

def process_loo(ori, loos):
    ori = np.concatenate([ori, 1 - ori], 0)
    loos = loos.reshape(-1)[1:]
    loos = np.stack([loos, 1 - loos]).transpose(1,0)
    tvds = tvd(loos, ori)
    tvds /= tvds.sum()
    return tvds

def gradient_mem(model, data_iter, attn_args):
    model.eval()
    
    grads = []
    sum_attns = []
    src_lens = []
    
    for batch in tqdm(data_iter):
        
        src = batch.src[0].transpose(0, 1) # batch, len
        src_lens.append(batch.src[1].cpu().data.numpy())
        seqlen = src.shape[-1]
        trg = batch.trg 
        
        data_holder = DataHolder(True, True)
        
        pred = model(src, data_holder=data_holder, **attn_args)
        
        pred.sigmoid().sum().backward()
        g = data_holder.embedding.grad
        
        em = data_holder.embedding
        g1 = (g * em).sum(-1)

        grads_xxex = g1.cpu().data.numpy()

        grads.append(grads_xxex)
        
        attn = [el.view(-1,4,seqlen,seqlen) for el in data_holder.attn]
        sum_attn = torch.zeros(attn[0].shape[0], seqlen, seqlen).cuda()
        for at in attn:
            sum_attn += at.sum(1)
        sum_attn = sum_attn.sum(1)
        sum_attns.append(sum_attn.cpu().data.numpy())
        
    grads = [x for y in grads for x in y]
    sum_attns = [x for y in sum_attns for x in y]
    src_lens = [x for y in src_lens for x in y]
        
    return grads, sum_attns, src_lens

def process_grads(grads) :
    new_grads = []
    for grad in grads:
        new_grad = np.abs(grad[1:])
        new_grad /= new_grad.sum()
        new_grads.append(new_grad)
    return new_grads

def permute_embed(model, data_iter, attn_args) :
    model.eval()
    outputs = []
    targets = []
    corrects = []
    
    for batch in tqdm(data_iter):
        src = batch.src[0].transpose(0, 1) # batch, len
        trg = batch.trg 
        
        targets.append(trg.cpu().data.numpy())
        
        po = np.zeros((src.shape[0], src.shape[1], 1))
        
        n = src.shape[1]
        
        correct = model(src, **attn_args).sigmoid().cpu().data.numpy()
        corrects.append(correct)
        
        for i in range(n):
            data_holder = DataHolder(True, True, i)
            
            pred = model(src, data_holder=data_holder, **attn_args)
            po[:, i] = pred.sigmoid().cpu().data.numpy()

        outputs.append(po)
    targets = [x for y in targets for x in y]
    outputs = [x for y in outputs for x in y]
    corrects = [x for y in corrects for x in y]
    
    return outputs, targets, corrects

def dist(nlayers, nheads, attn, batch_idx):
    prev_AD = None
    ADs = []
    seqlen = attn[0].shape[-1]
    for l in range(nlayers):
        for h in range(nheads):
            if h == nheads-1:
                AD = dc(attn[l][batch_idx,h].cpu().data.numpy())
                for i in range(seqlen):
                    for j in range(seqlen):
                        if AD[i][j] == 1:
                            AD[i][j] = 1
                        else:
                            AD[i][j] = float('inf')
                        if i == j:
                            AD[i][j] = 0

                for h2 in range(nheads-1):
                    attention_dependencies2 = attn[l][batch_idx,h2].cpu().data.numpy() 
                    for i in range(seqlen):
                        for j in range(seqlen):
                            if attention_dependencies2[i][j] == 1:
                                AD[i][j] = min(1, AD[i][j])

                AD_static = dc(AD)
                if prev_AD is not None:
                    for i in range(seqlen):
                        for j in range(seqlen):
                            for k in range(seqlen):
                                if AD_static[i][j] < float('inf') and prev_AD[j][k] < float('inf'):
                                    AD[i][k] = min(AD[i][k], AD_static[i][j]+prev_AD[j][k])

                prev_AD = AD
                ADs.append(dc(AD))
    return ADs
        
"""Top-k kendall-tau distance.

This module generalise kendall-tau as defined in [1].
It returns a distance: 0 for identical (in the sense of top-k) lists and 1 if completely different.

Example:
    Simply call kendall_top_k with two same-length arrays of ratings (or also rankings), length of the top elements k (default is the maximum length possible), and p (default is 0, see [1]) as parameters:

        import kendall
        a = np.array([1,2,3,4,5])
        b = np.array([5,4,3,2,1])
        kendall.kendall_top_k(a,b,k=4)

Author: Alessandro Checco
    https://github.com/AlessandroChecco
References
[1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. "Comparing top k lists." SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
"""
import numpy as np
import scipy.stats as stats
import scipy.special as special

def kendall_top_k(a,b,k=None,p=0.5): #zero is equal 1 is max distance, compare with 1-scipy.stats.kendalltau(a,b)/2+1/2
    """
    kendall_top_k(np.array,np.array,k,p)
    This function generalise kendall-tau as defined in [1] Fagin, Ronald, Ravi Kumar, and D. Sivakumar. "Comparing top k lists." SIAM Journal on Discrete Mathematics 17.1 (2003): 134-160.
    It returns a distance: 0 for identical (in the sense of top-k) lists and 1 if completely different.

    Example:
        Simply call it with two same-length arrays of ratings (or also rankings), length of the top elements k (default is the maximum length possible), and p (default is 0, see [1]) as parameters:

            $ a = np.array([1,2,3,4,5])
            $ b = np.array([5,4,3,2,1])
            $ kendall_top_k(a,b,k=4)
    """

    a = np.array(a)
    b = np.array(b)
    if k is None:
        k = a.size
    if a.size != b.size:
        raise NameError('The two arrays need to have same lengths')
    k = min(k,a.size)
    a_top_k = np.argpartition(a,-k)[-k:]
    b_top_k = np.argpartition(b,-k)[-k:]
    common_items = np.intersect1d(a_top_k,b_top_k)
    only_in_a = np.setdiff1d(a_top_k, common_items)
    only_in_b = np.setdiff1d(b_top_k, common_items)
    kendall = (1 - (stats.kendalltau(a[common_items], b[common_items])[0]/2+0.5)) * (common_items.size**2) #case 1
    if np.isnan(kendall): # degenerate case with only one item (not defined by Kendall)
        kendall = 0
    for i in common_items: #case 2
        for j in only_in_a:
            if a[i] < a[j]:
                kendall += 1
        for j in only_in_b:
            if b[i] < b[j]:
                kendall += 1
    kendall += 2*p * special.binom(k-common_items.size,2)     #case 4
    kendall /= ((only_in_a.size + only_in_b.size + common_items.size)**2 ) #normalization
    return np.array([kendall, 1.0])
