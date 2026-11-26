import numpy as np
import pytest

pytest.importorskip('torch')
from eval import evaluate, triplet_accuracy


@pytest.mark.parametrize('metric', [evaluate, triplet_accuracy])
@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf])
def test_numeric_labels_must_be_finite(metric, invalid):
    with pytest.raises(ValueError, match='labels must be finite'):
        metric(np.arange(4).reshape(-1, 1), [0, 0, 1, invalid])


@pytest.mark.parametrize('metric', [evaluate, triplet_accuracy])
def test_string_labels_are_supported(metric):
    metric(np.array([[0], [1], [10], [11]]), ['a', 'a', 'b', 'b'])
