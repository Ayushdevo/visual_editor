import pytest
import torch
from loss import triplet_loss

@pytest.mark.parametrize('margin', [-.1, float('nan'), float('inf')])
def test_invalid_margin_cannot_define_training_objective(margin):
    with pytest.raises(ValueError, match='margin'):
        triplet_loss(torch.zeros((4, 2)), torch.tensor([0, 0, 1, 1]), margin=margin)

def test_invalid_mining_is_not_silently_ignored_on_degenerate_batch():
    with pytest.raises(ValueError, match='mining'):
        triplet_loss(torch.zeros((2, 2)), torch.tensor([0, 0]), mining='typo')
