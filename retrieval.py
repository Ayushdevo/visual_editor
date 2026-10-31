"""Exact NumPy squared-L2 index, with stable tie breaking."""
import numpy as np

class L2Index:
    def __init__(self, embeddings):
        self.gallery = np.array(embeddings,dtype=np.float32,copy=True)
        self.norms = (self.gallery**2).sum(1)
    def distances(self, queries):
        q = np.atleast_2d(queries).astype(np.float32)
        return np.maximum((q*q).sum(1,keepdims=True)+self.norms[None,:]-2*q@self.gallery.T,0)
    def search(self, queries, k=5, exclude=None):
        d = self.distances(queries)
        if exclude is not None:
            d[np.arange(len(d)),np.asarray(exclude)] = np.inf
        ids = np.argsort(d,axis=1,kind='stable')[:,:k]
        return ids,np.take_along_axis(d,ids,axis=1)
