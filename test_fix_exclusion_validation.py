import numpy as np
import pytest
from retrieval import L2Index

@pytest.mark.parametrize('exclude', [[-1, 0], [0], [0., 1.], [True, False], [3, 0]])
def test_exclusion_requires_valid_integer_indices_for_every_query(exclude):
    with pytest.raises(ValueError, match='exclude'):
        L2Index([[0.], [1.], [2.]]).search([[0.], [1.]], 1, exclude)
