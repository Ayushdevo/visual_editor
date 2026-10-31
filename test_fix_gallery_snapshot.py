import numpy as np
from retrieval import L2Index

def test_index_is_independent_of_caller_mutation():
    gallery = np.array([[0., 0.], [3., 4.]], dtype=np.float32)
    index = L2Index(gallery)
    gallery[:] = 100
    np.testing.assert_allclose(index.distances([[3., 4.]]), [[25., 0.]])
