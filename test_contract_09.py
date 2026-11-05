import pytest
import torch
from optimizers import Optimizer
@pytest.mark.parametrize('params', [[], [1], [torch.tensor([1])]])
def test_invalid_parameters(params):
    with pytest.raises(ValueError): Optimizer(params)
def test_nonleaf_parameter():
    p=torch.ones(1,requires_grad=True)
    with pytest.raises(ValueError,match='leaf'): Optimizer([p*2])
