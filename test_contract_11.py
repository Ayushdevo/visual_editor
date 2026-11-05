import pytest
import torch
from eval import embed
def test_empty_dataset():
    model=torch.nn.Linear(2,2)
    with pytest.raises(ValueError,match='empty dataset'): embed(model,[], 'cpu')
    assert model.training
