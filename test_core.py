"""Small correctness tests; run pytest -q after installing requirements."""
import numpy as np
import pytest
from retrieval import L2Index

def test_retrieval_self_exclusion():
    z = np.array([[1.,0.],[0.9,0.1],[-1.,0.]])
    ids,d = L2Index(z).search(z,2,exclude=np.arange(3))
    assert np.array_equal(ids[:,0],[1,0,1])
    assert all(i not in row for i,row in enumerate(ids))
    np.testing.assert_allclose(d[0],[0.02,4],atol=1e-6)

def test_metrics_perfect_clusters():
    pytest.importorskip('torch')
    from eval import evaluate
    z = np.array([[1,0],[1,0],[-1,0],[-1,0]],dtype=np.float32)
    result = evaluate(z,np.array([0,0,1,1]),chunk=2)
    assert result['Recall@1'] == 1 and result['mAP'] == 1

def test_hard_triplet_and_gradients():
    torch = pytest.importorskip('torch')
    from loss import triplet_loss
    z = torch.tensor([[0.],[2.],[3.],[7.]],requires_grad=True)
    y = torch.tensor([0,0,1,1])
    # Hardest positive: 2,2,4,4; negative: 3,1,1,5.
    loss = triplet_loss(z,y,margin=0.3)
    assert loss.item() == pytest.approx((0+1.3+3.3+0)/4)
    loss.backward()
    assert torch.isfinite(z.grad).all()

@pytest.mark.parametrize('kind',['adam','sgd'])
def test_optimizer_against_reference(kind):
    torch = pytest.importorskip('torch')
    from optimizers import Optimizer
    p = torch.tensor([1.,-2.],requires_grad=True)
    q = p.detach().clone().requires_grad_()
    own = Optimizer([p],kind=kind,lr=0.01,weight_decay=0.02)
    reference = (torch.optim.Adam([q],lr=0.01,weight_decay=0.02) if kind=='adam'
                 else torch.optim.SGD([q],lr=0.01,momentum=0.9,weight_decay=0.02))
    for _ in range(5):
        own.zero_grad(); reference.zero_grad()
        p.square().sum().backward(); q.square().sum().backward()
        own.step(); reference.step()
        torch.testing.assert_close(p,q)

def test_nadam_first_update():
    torch = pytest.importorskip('torch')
    from optimizers import Optimizer
    p = torch.tensor([1.],requires_grad=True)
    opt = Optimizer([p],kind='nadam',lr=0.01)
    p.grad = torch.tensor([2.]); opt.step()
    assert p.item() == pytest.approx(0.981,abs=1e-6)
