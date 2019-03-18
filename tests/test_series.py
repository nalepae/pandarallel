import pytest

from pandarallel import pandarallel

import pandas as _pd
import numpy as np
import math


@pytest.fixture(scope="session")
def plasma_client():
    pandarallel.initialize()

def test_series(plasma_client):
    # dummy series
    ser = _pd.Series(np.random.randint(0, 10, 100))
    
    # various apply
    def f(i) : return i**2
    ser_apply       = ser.apply(f)
    ser_parallel    = ser.parallel_apply(f)

    # check consistancy
    assert (ser_apply == ser_parallel).all()


