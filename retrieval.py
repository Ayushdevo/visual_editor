"""Exact NumPy squared-L2 index, with stable tie breaking."""
import numpy as np

class L2Index:
    def __init__(self, embeddings):
        self.gallery = np.array(embeddings,dtype=np.float64,copy=True)
        if self.gallery.ndim != 2 or 0 in self.gallery.shape:
            raise ValueError('Gallery must be a nonempty embedding matrix')
        if not np.isfinite(self.gallery).all():
            raise ValueError('Gallery embeddings must be finite')
    def distances(self, queries):
        q = np.atleast_2d(queries).astype(np.float64)
        if q.ndim != 2 or q.shape[1] != self.gallery.shape[1]:
            raise ValueError('Query matrix dimension must match gallery embeddings')
        if not np.isfinite(q).all():
            raise ValueError('Query embeddings must be finite')
        # Direct differences avoid cancellation for nearby, large-magnitude vectors.
        # Process one query at a time to bound temporary memory by gallery size.
        distances = np.empty((len(q), len(self.gallery)), dtype=np.float64)
        for row, query in enumerate(q):
            difference = self.gallery - query
            distances[row] = np.einsum('ij,ij->i', difference, difference)
        return distances
    def search(self, queries, k=5, exclude=None):
        d = self.distances(queries)
        if exclude is not None:
            d[np.arange(len(d)),np.asarray(exclude)] = np.inf
        ids = np.argsort(d,axis=1,kind='stable')[:,:k]
        return ids,np.take_along_axis(d,ids,axis=1)
