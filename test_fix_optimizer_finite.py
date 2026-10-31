import pytest
import torch
from optimizers import Optimizer

@pytest.mark.parametrize('option', ['lr', 'eps', 'weight_decay'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_optimizer_parameters_are_rejected(option, value):
    with pytest.raises(ValueError, match='finite'):
        Optimizer([torch.tensor([1.], requires_grad=True)], **{option:value})
