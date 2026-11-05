import numpy as np
import pytest
from data import Images
def test_label_lists():
    ds=Images(np.zeros((2,32,32,3),dtype=np.uint8),[3,4],[1])
    assert ds.targets.tolist()==[4]
def test_misaligned_images():
    with pytest.raises(ValueError,match='aligned'):
        Images(np.zeros((1,32,32,3)),[0,1],[0])
