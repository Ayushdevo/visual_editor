import numpy as np
import pytest
from eval import triplet_accuracy
@pytest.mark.parametrize('labels', [[0,0],[0,1,1]])
def test_invalid_triplet_population(labels):
    with pytest.raises(ValueError,match='two classes'): triplet_accuracy(np.zeros((len(labels),2)),labels)
