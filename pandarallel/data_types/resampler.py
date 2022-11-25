import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union, cast

import pandas as pd
from pandas.core.resample import Resampler

from ..utils import chunk
from .generic import DataType


class Resampler:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int,  resampler: Resampler, **kwargs
        ) -> Iterator[List[Tuple[Any, pd.DataFrame]]]:
            chunks = chunk(resampler.ngroups, nb_workers)
            iterator = iter(resampler)

            for chunk_ in chunks:
                yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

        @staticmethod
        def work(
            data: List[Tuple[Any, pd.DataFrame]],
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> List[Tuple[int, pd.DataFrame, bool]]:
            def compute_result(
                key: int, df: pd.DataFrame
            ) -> Tuple[int, pd.DataFrame, bool]:
                result = user_defined_function(
                    df, *user_defined_function_args, **user_defined_function_kwargs
                )
                return key, result

            return [compute_result(key, df) for key, df in data]

        @staticmethod
        def reduce(datas: Iterable[Tuple[Any, pd.DataFrame]], extra: Dict[str, Any]) -> pd.DataFrame:
            keys, values = zip(*[item for sublist in datas for item in sublist])
            if isinstance(values[0], pd.Series):
                return pd.DataFrame(values, index=keys)
            elif isinstance(values[0], pd.DataFrame):
                return pd.concat(values, copy=False, keys=keys)