"""Randomly initialized, fixed ResNet-18 embedding architecture."""
import torch
from torch import nn
from torchvision.models import resnet18

class EmbeddingModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = resnet18(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, dim)

    def forward(self, x):
        z = self.net(x)
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
