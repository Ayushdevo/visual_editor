import numpy as np
import pytest
from data import Images
def test_split_ids_are_owned():
    ids=np.array([0,1]); ds=Images(np.zeros((2,32,32,3),dtype=np.uint8),np.array([0,1]),ids)
    ids[0]=1
    assert ds.ids.tolist()==[0,1]
@pytest.mark.parametrize('ids', [[-1],[2],[0.5],[[0]]])
def test_invalid_split_ids(ids):
    with pytest.raises(ValueError): Images(np.zeros((2,32,32,3)),np.array([0,1]),ids)
