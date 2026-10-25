"""Deterministic per-class 80/10/10 split and balanced P-by-K batches."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.datasets import CIFAR100
from torchvision import transforms as T

MEAN, STD = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)

def transform(train=False):
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if train else []
    return T.Compose(ops + [T.ToTensor(), T.Normalize(MEAN, STD)])

class Images(Dataset):
    def __init__(self, images, labels, ids, train=False):
        self.images, self.labels, self.ids = images, labels, np.asarray(ids)
        self.targets = labels[self.ids]
        self.transform = transform(train)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, i):
        from PIL import Image
        j = self.ids[i]
        return self.transform(Image.fromarray(self.images[j])), int(self.labels[j])

def load(root='data', seed=42):
    """Pool official train/test sets, then split each of 100 classes 480/60/60."""
    a = CIFAR100(root, train=True, download=True)
    b = CIFAR100(root, train=False, download=True)
    x = np.concatenate([a.data,b.data])
    y = np.asarray(a.targets+b.targets)
    rng = np.random.default_rng(seed)
    splits = [[],[],[]]
    for c in range(100):
        ids = rng.permutation(np.flatnonzero(y == c))
        for dst, part in zip(splits, [ids[:480],ids[480:540],ids[540:]]):
            dst.extend(part.tolist())
    return [Images(x,y,ids,i==0) for i,ids in enumerate(splits)], a.classes

class PKSampler(Sampler):
    """Each batch has P distinct classes and K distinct samples per class."""
    def __init__(self, targets, p=16, k=4, seed=42):
        self.groups = [np.flatnonzero(np.asarray(targets)==c) for c in np.unique(targets)]
        self.p,self.k,self.seed,self.epoch = p,k,seed,0
        if p < 2 or p > len(self.groups) or k < 2 or any(len(g)<k for g in self.groups):
            raise ValueError('Need >=2 classes and >=2 samples per selected class')
        self.batches = len(targets)//(p*k)
    def __len__(self):
        return self.batches
    def __iter__(self):
        rng = np.random.default_rng(self.seed+self.epoch)
        for _ in range(self.batches):
            chosen = rng.choice(len(self.groups),self.p,replace=False)
            yield np.concatenate([rng.choice(self.groups[c],self.k,replace=False) for c in chosen]).tolist()

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
