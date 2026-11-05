import pytest
import torch
from loss import triplet_loss
@pytest.mark.parametrize('z,y', [(torch.zeros(4),torch.zeros(4)),(torch.zeros(4,2),torch.zeros(4,1)),(torch.zeros(4,2),torch.zeros(3)),(torch.zeros(4,0),torch.zeros(4))])
def test_triplet_shapes(z,y):
    with pytest.raises(ValueError,match='aligned'): triplet_loss(z,y)
