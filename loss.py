"""Own Euclidean triplet objective and in-batch mining; no library losses."""
import math
import torch

def triplet_loss(z, labels, margin=0.3, mining='hard'):
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('margin must be finite and nonnegative')
    if mining not in {'hard', 'random'}:
        raise ValueError('mining must be hard or random')
    if z.ndim != 2 or z.shape[1] == 0 or labels.ndim != 1 or len(labels) != len(z):
        raise ValueError('Expected an embedding matrix and one aligned label per row')
    if not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError('Embeddings must be finite floating-point values')
    if labels.device != z.device:
        raise ValueError('Embeddings and labels must use the same device')
    if labels.is_complex() or not torch.isfinite(labels).all():
        raise ValueError('Labels must be finite real values')
    # Explicit differences keep the diagonal at exactly zero.
    d = ((z[:, None] - z[None, :]).square().sum(-1) + 1e-12).sqrt()
    same = labels[:, None].eq(labels[None, :])
    positive = same & ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    negative = ~same
    valid = positive.any(1) & negative.any(1)
    if not valid.any():
        return z.sum() * 0
    if mining == 'hard':
        dp = d.masked_fill(~positive, -torch.inf).max(1).values
        dn = d.masked_fill(~negative, torch.inf).min(1).values
    elif mining == 'random':
        # Hold hardest-positive selection fixed; vary negative mining only.
        dp = d.masked_fill(~positive, -torch.inf).max(1).values
        n = torch.rand_like(d).masked_fill(~negative, -1).argmax(1)
        rows = torch.arange(len(z), device=z.device)
        dn = d[rows, n]
    else:
        raise ValueError(mining)
    return (dp[valid] - dn[valid] + margin).clamp_min(0).mean()

