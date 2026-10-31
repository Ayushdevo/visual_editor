import pytest
from retrieval import L2Index

@pytest.mark.parametrize('k', [-1, True, 1.5])
def test_neighbor_count_requires_nonnegative_integer(k):
    with pytest.raises(ValueError, match='k must'):
        L2Index([[0.], [1.], [2.]]).search([[0.]], k)

def test_zero_neighbors_has_empty_result():
    ids, d = L2Index([[0.], [1.]]).search([[0.]], 0)
    assert ids.shape == d.shape == (1, 0)
