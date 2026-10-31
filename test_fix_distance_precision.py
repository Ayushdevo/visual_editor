import numpy as np
from retrieval import L2Index

def test_large_offset_neighbors_keep_true_distance_order():
    gallery = np.array([[10000.1, 10000.], [10000., 10000.]])
    index = L2Index(gallery)
    ids, distances = index.search([[10000., 10000.]], 2)
    np.testing.assert_array_equal(ids, [[1, 0]])
    np.testing.assert_allclose(distances, [[0., 0.01]], rtol=1e-6)

def test_tiny_float64_separations_are_preserved():
    d = L2Index([[1.00000001], [1.]]).distances([[1.]])
    assert d[0, 0] > 0
    assert d[0, 1] == 0
