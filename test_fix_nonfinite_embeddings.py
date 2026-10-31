import numpy as np
import pytest
from retrieval import L2Index

@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_gallery_is_rejected(bad):
    with pytest.raises(ValueError, match='finite'):
        L2Index([[0., 1.], [bad, 1.]])

@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_nonfinite_query_is_rejected(bad):
    with pytest.raises(ValueError, match='finite'):
        L2Index([[0., 1.]]).search([[bad, 1.]])
