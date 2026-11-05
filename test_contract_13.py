import numpy as np
import pytest
from retrieval import L2Index
def test_distance_overflow():
    with pytest.raises(ValueError,match='range'): L2Index([[1e308]]).search([[-1e308]])
def test_large_representable_distance():
    ids,d=L2Index([[0.],[1e150]]).search([[0.]],2)
    assert ids.tolist()==[[0,1]]
    np.testing.assert_allclose(d,[[0.,1e300]])
