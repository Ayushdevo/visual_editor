import pytest
from data import PKSampler
@pytest.mark.parametrize('p,k', [(2.5,2),(2,2.5),(True,2),(2,True)])
def test_batch_dimensions(p,k):
    with pytest.raises(ValueError, match='integers'):
        PKSampler([0,0,1,1],p,k)
