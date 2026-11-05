"""Randomly initialized, fixed ResNet-18 embedding architecture."""
import torch
from torch import nn
from torchvision.models import resnet18

class EmbeddingModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        from numbers import Integral
        if isinstance(dim, bool) or not isinstance(dim, Integral) or dim < 1:
            raise ValueError('dim must be a positive integer')
        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, dim)

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)

