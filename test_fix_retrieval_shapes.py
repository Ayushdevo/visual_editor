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
