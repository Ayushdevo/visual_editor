import numpy as np
from retrieval import L2Index

def test_excluded_neighbors_never_reappear_at_end_of_ranking():
    ids, distances = L2Index([[0.], [1.]]).search([[0.]], 10, [0])
    np.testing.assert_array_equal(ids, [[1]])
    assert np.isfinite(distances).all()

def test_singleton_self_exclusion_returns_empty_result():
    ids, d = L2Index([[0.]]).search([[0.]], 1, [0])
    assert ids.shape == d.shape == (1, 0)
