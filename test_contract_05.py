import pytest
from model import EmbeddingModel
@pytest.mark.parametrize('dim', [0,-1,True,1.5])
def test_invalid_dimensions(dim):
    with pytest.raises(ValueError,match='positive integer'): EmbeddingModel(dim)
