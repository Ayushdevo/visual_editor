"""Own Euclidean triplet objective and in-batch mining; no library losses."""
import torch

def triplet_loss(z, labels, margin=0.3, mining='hard'):
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
