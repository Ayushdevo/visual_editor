import pytest
import torch
from optimizers import Optimizer

def test_duplicate_parameter_cannot_be_updated_twice():
    parameter = torch.tensor([1.], requires_grad=True)
    with pytest.raises(ValueError, match='Duplicate'):
        Optimizer([parameter, parameter], kind='sgd')
