import pytest

from pandarallel import pandarallel

import pandas as _pd
import numpy as np
import math




def test_series() : 
    
    ser = _pd.Series(np.random.randint(0, 1000, 100000))
    pandarallel.initialize()

    def f(i) : 
        return i**2

    ser_apply = ser.apply(f)
    ser_parallel = ser.parallel_apply(f)

    assert (ser.apply == ser_parallel).all()


