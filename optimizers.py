"""Tensor-only SGD momentum, Adam, and fixed-beta NAdam (Dozat form)."""
import math
import torch

class Optimizer:
    def __init__(self, params, kind='adam', lr=1e-3, beta1=0.9,
                 beta2=0.999, eps=1e-8, momentum=0.9, weight_decay=0):
        if not all(math.isfinite(value) for value in
                   (lr, beta1, beta2, eps, momentum, weight_decay)):
            raise ValueError('Optimizer parameters must be finite')
        if kind not in {'adam', 'nadam', 'sgd'}:
            raise ValueError(kind)
        if lr <= 0 or eps <= 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not 0 <= momentum < 1 or weight_decay < 0:
            raise ValueError('Invalid optimizer parameters')
        self.params = list(params)
        self.kind, self.lr = kind, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.mu, self.wd = momentum, weight_decay
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad
            if self.wd:
                g = g + self.wd * p
            if i not in self.state:
                self.state[i] = [0, torch.zeros_like(p), torch.zeros_like(p)]
            s = self.state[i]
            s[0] += 1
            t, m, v = s
            if self.kind == 'sgd':
                m.mul_(self.mu).add_(g)
                p.add_(m, alpha=-self.lr)
            else:
                m.mul_(self.b1).add_(g, alpha=1-self.b1)
                v.mul_(self.b2).add_(g.square(), alpha=1-self.b2)
                mh = m / (1-self.b1**t)
                vh = v / (1-self.b2**t)
                if self.kind == 'nadam':
                    mh = self.b1 * mh + (1-self.b1)*g/(1-self.b1**t)
                p.add_(mh / (vh.sqrt()+self.eps), alpha=-self.lr)
