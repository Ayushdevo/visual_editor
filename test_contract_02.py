import pytest
from data import PKSampler
@pytest.mark.parametrize('labels', [[], [[0,0],[1,1]], 1])
def test_label_shape(labels):
    with pytest.raises(ValueError, match='one-dimensional'):
        PKSampler(labels,2,2)
