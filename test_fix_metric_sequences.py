import pytest
from eval import evaluate, triplet_accuracy

def test_metrics_accept_python_sequences():
    z = [[1., 0.], [1., 0.], [-1., 0.], [-1., 0.]]
    y = [0, 0, 1, 1]
    assert evaluate(z, y)['mAP'] == pytest.approx(1)
    assert triplet_accuracy(z, y) == pytest.approx(1)
