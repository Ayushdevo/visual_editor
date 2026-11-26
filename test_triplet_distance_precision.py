import numpy as np
import pytest

pytest.importorskip('torch')
from eval import triplet_accuracy


@pytest.mark.parametrize('dtype,scale', [(np.uint8, 1), (np.float32, 1e20)])
def test_triplet_accuracy_computes_distances_in_float64(dtype, scale):
    z = (np.array([[0], [255], [1], [2]], dtype=np.float64) * scale).astype(dtype)
    labels = np.array([0, 0, 1, 1])
    for seed in range(10):
        assert triplet_accuracy(z, labels, seed) == triplet_accuracy(z.astype(np.float64), labels, seed)


def test_triplet_accuracy_rejects_distance_overflow():
    z = np.array([[1e308], [-1e308], [0], [1]])
    with pytest.raises(ValueError, match='floating-point range'):
        triplet_accuracy(z, [0, 0, 1, 1])
