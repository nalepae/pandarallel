import itertools
from enum import Enum
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame, Index


def chunk(nb_item: int, nb_chunks: int, start_offset=0) -> List[slice]:
    """
    Return `nb_chunks` slices of approximatively `nb_item / nb_chunks` each.

    Parameters
    ----------
    nb_item : int
        Total number of items

    nb_chunks : int
        Number of chunks to return

    start_offset : int
        Shift start of slice by this amount

    Returns
    -------
    A list of slices

    Examples
    --------
    >>> chunks = chunk(103, 4)
    >>> chunks
    [slice(0, 26, None), slice(26, 52, None), slice(52, 78, None), slice(78, 103, None)]
    """
    if nb_item <= nb_chunks:
        return [slice(max(0, idx - start_offset), idx + 1) for idx in range(nb_item)]

    quotient = nb_item // nb_chunks
    remainder = nb_item % nb_chunks

    quotients = [quotient] * nb_chunks
    remainders = [1] * remainder + [0] * (nb_chunks - remainder)

    nb_elems_per_chunk = [
        quotient + remainder for quotient, remainder in zip(quotients, remainders)
    ]

    accumulated = list(itertools.accumulate(nb_elems_per_chunk))
    shifted_accumulated = accumulated.copy()
    shifted_accumulated.insert(0, 0)
    shifted_accumulated.pop()

    return [
        slice(max(0, begin - start_offset), end)
        for begin, end in zip(shifted_accumulated, accumulated)
    ]


def df_indexed_like(df: DataFrame, axes: List[Index]) -> bool:
    """
    Returns whether a data frame is indexed in the way specified by the
    provided axes.

    Used by DataFrameGroupBy to determine whether a group has been modified.

    Function adapted from pandas.core.groupby.ops._is_indexed_like

    Parameters
    ----------
    df : DataFrame
        The data frame in question

    axes : List[Index]
        The axes to which the data frame is compared

    Returns
    -------
    Whether or not the data frame is indexed in the same wa as the axes.
    """
    if isinstance(df, DataFrame):
        return df.axes[0].equals(axes[0])

    return False


def get_pandas_version() -> Tuple[int, int]:
    major_str, minor_str, *_ = pd.__version__.split(".")
    return int(major_str), int(minor_str)


class WorkerStatus(int, Enum):
    Running = 0
    Success = 1
    Error = 2
