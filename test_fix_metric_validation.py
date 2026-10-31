import numpy as np
import pytest
from eval import evaluate, triplet_accuracy

@pytest.mark.parametrize('chunk', [-1, 0, True, 1.5])
def test_invalid_chunks_raise_value_error(chunk):
    with pytest.raises(ValueError, match='chunk'):
        evaluate(np.zeros((4, 2)), np.array([0, 0, 1, 1]), chunk)

@pytest.mark.parametrize('fn', [evaluate, triplet_accuracy])
@pytest.mark.parametrize('z,y', [(np.empty((0, 2)), []), (np.zeros((2, 2)), [0, 0, 1]),
                                 (np.zeros((4, 2)), [[0], [0], [1], [1]])])
def test_invalid_label_alignment_is_rejected(fn, z, y):
    with pytest.raises(ValueError, match='labels|samples'):
        fn(z, y)
