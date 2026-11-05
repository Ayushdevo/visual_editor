import pytest
import torch
from optimizers import Optimizer
def test_sparse_gradient_does_not_partially_update():
    p=torch.ones(2,requires_grad=True); q=torch.ones(2,requires_grad=True)
    opt=Optimizer([p,q]); p.grad=torch.ones(2)
    q.grad=torch.sparse_coo_tensor([[0]],[1.],(2,))
    with pytest.raises(ValueError,match='Sparse'): opt.step()
    assert torch.equal(p,torch.ones(2)) and opt.state=={}
