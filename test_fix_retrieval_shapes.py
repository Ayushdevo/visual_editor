import numpy as np
import pytest
from retrieval import L2Index

@pytest.mark.parametrize('query', [[[1.]], [[[1., 2.]]]])
def test_query_dimension_must_match_gallery(query):
    with pytest.raises(ValueError, match='dimension|matrix'):
        L2Index([[0., 0.], [1., 1.]]).distances(query)

@pytest.mark.parametrize('gallery', [[], np.empty((2, 0)), [[[1., 2.]]]])
def test_gallery_requires_nonempty_matrix(gallery):
    with pytest.raises(ValueError, match='matrix'):
        L2Index(gallery)


def test_complex_embeddings_are_rejected():
    values = np.array([[1 + 2j, 3 + 0j]])
    with pytest.raises(ValueError, match="Gallery embeddings must be real"):
        L2Index(values)
    with pytest.raises(ValueError, match="Query embeddings must be real"):
        L2Index([[1.0, 3.0]]).search(values)
