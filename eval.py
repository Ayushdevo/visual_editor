"""Leave-one-out Recall@K, full-ranking mAP, and fixed validation triplets."""
import numpy as np
import torch
from torch.utils.data import DataLoader
from retrieval import L2Index

@torch.no_grad()
def embed(model, dataset, device, batch_size=256):
    model.eval()
    zs,ys = [],[]
    for x,y in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        zs.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(zs),np.concatenate(ys)

def evaluate(z,y,chunk=128):
    """Every test image queries the other N-1 images; self is excluded."""
    z, y = np.asarray(z), np.asarray(y)
    index = L2Index(z)
    hits = {k:0 for k in (1,5,10)}
    aps = []
    for start in range(0,len(z),chunk):
        stop = min(start+chunk,len(z))
        rank,_ = index.search(z[start:stop],len(z)-1,np.arange(start,stop))
        rel = y[rank] == y[start:stop,None]
        for k in hits:
            hits[k] += int(rel[:,:k].any(1).sum())
        precision = rel.cumsum(1)/np.arange(1,rel.shape[1]+1)
        counts = (y[:,None]==y[start:stop][None,:]).sum(0)-1
        if (counts <= 0).any():
            raise ValueError('Every query needs at least one relevant gallery image')
        aps.extend(((precision*rel).sum(1)/counts).tolist())
    return {**{f'Recall@{k}':v/len(z) for k,v in hits.items()},'mAP':float(np.mean(aps))}

def triplet_accuracy(z,y,seed=42):
    """One fixed random positive and negative per validation anchor."""
    z, y = np.asarray(z), np.asarray(y)
    rng = np.random.default_rng(seed)
    p,n = [],[]
    for i,c in enumerate(y):
        p.append(rng.choice(np.flatnonzero((y==c)&(np.arange(len(y))!=i))))
        n.append(rng.choice(np.flatnonzero(y!=c)))
    dp = ((z-z[p])**2).sum(1)
    dn = ((z-z[n])**2).sum(1)
    return float((dp<dn).mean())
