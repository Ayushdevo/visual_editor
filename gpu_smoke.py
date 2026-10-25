"""Three real training batches on CUDA; no checkpoints or evaluation metrics."""
import json
import torch
import torchvision
from torch.utils.data import DataLoader
from data import load, PKSampler, seed_all
from model import EmbeddingModel
from loss import triplet_loss
from optimizers import Optimizer


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; run with .venv/Scripts/python.exe')
    seed_all(42)
    a = torch.ones((256, 256), device='cuda')
    assert (a @ a).eq(256).all().item()
    print(json.dumps({'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                      'tensor_matmul': 'passed'}), flush=True)
    (train, _, _), _ = load()
    loader = DataLoader(train, batch_sampler=PKSampler(train.targets, 16, 4, 42), num_workers=0)
    model = EmbeddingModel().to('cuda').train()
    opt = Optimizer(model.parameters(), weight_decay=1e-4)
    before = model.net.fc.weight.detach().clone()
    for step, (x, y) in enumerate(loader):
        opt.zero_grad()
        loss = triplet_loss(model(x.to('cuda')), y.to('cuda'))
        assert torch.isfinite(loss).item()
        loss.backward()
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        opt.step()
        torch.cuda.synchronize()
        print(json.dumps({'step': step + 1, 'batch_size': len(x), 'device': str(next(model.parameters()).device),
                          'loss': loss.item()}), flush=True)
        if step == 2:
            break
    assert not torch.equal(before, model.net.fc.weight)
    print('PASS: three CUDA training steps, finite gradients, weights updated. This does not test embedding collapse.')


if __name__ == '__main__':
    main()
