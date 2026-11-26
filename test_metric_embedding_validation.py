import numpy as np
import pytest

pytest.importorskip('torch')
from eval import evaluate, triplet_accuracy


@pytest.mark.parametrize('metric', [evaluate, triplet_accuracy])
def test_rejects_complex_embeddings(metric):
    with pytest.raises(ValueError, match='real'):
        metric(np.array([[0j], [1j], [2j], [3j]]), [0, 0, 1, 1])


@pytest.mark.parametrize('metric', [evaluate, triplet_accuracy])
def test_rejects_zero_dimensional_embeddings(metric):
    with pytest.raises(ValueError, match='dimension'):
        metric(np.empty((4, 0)), [0, 0, 1, 1])
