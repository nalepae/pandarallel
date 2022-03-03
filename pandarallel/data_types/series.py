from typing import Any, Callable, Dict, Iterable, Iterator

import pandas as pd

from ..utils import chunk
from .generic import DataType


class Series:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: pd.Series, **kwargs
        ) -> Iterator[pd.Series]:
            for chunk_ in chunk(data.size, nb_workers):
                yield data[chunk_]

        @staticmethod
        def work(
            data: pd.Series,
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> pd.Series:
            return data.apply(
                user_defined_function,
                *user_defined_function_args,
                **user_defined_function_kwargs
            )

        @staticmethod
        def reduce(datas: Iterable[pd.Series], extra: Dict[str, Any]) -> pd.Series:
            return pd.concat(datas, copy=False)

    class Map(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: pd.Series, **kwargs
        ) -> Iterator[pd.Series]:
            for chunk_ in chunk(data.size, nb_workers):
                yield data[chunk_]

        @staticmethod
        def work(
            data: pd.Series,
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> pd.Series:
            return data.map(
                user_defined_function,
                *user_defined_function_args,
                **user_defined_function_kwargs
            )

        @staticmethod
        def reduce(datas: Iterable[pd.Series], extra: Dict[str, Any]) -> pd.Series:
            return pd.concat(datas, copy=False)
