import pytest
import torch
from loss import triplet_loss
@pytest.mark.parametrize('value', [float('nan'),float('inf'),-float('inf')])
def test_nonfinite_loss_inputs(value):
    z=torch.zeros(4,2); z[0,0]=value
    with pytest.raises(ValueError,match='finite'): triplet_loss(z,torch.tensor([0,0,1,1]))
def test_integer_embeddings():
    with pytest.raises(ValueError,match='floating'): triplet_loss(torch.ones(4,2,dtype=torch.int64),torch.tensor([0,0,1,1]))
