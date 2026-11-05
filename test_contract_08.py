import pytest
import torch
from loss import triplet_loss
def test_nan_labels():
    with pytest.raises(ValueError,match='Labels'): triplet_loss(torch.zeros(4,2),torch.tensor([0.,0.,1.,float('nan')]))
def test_label_device():
    with pytest.raises(ValueError,match='same device'): triplet_loss(torch.zeros(4,2),torch.empty(4,device='meta'))
