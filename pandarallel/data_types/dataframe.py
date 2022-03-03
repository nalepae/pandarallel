from typing import Any, Callable, Dict, Iterable, Iterator

import pandas as pd

from ..utils import chunk
from .generic import DataType


class DataFrame:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: pd.DataFrame, **kwargs
        ) -> Iterator[pd.DataFrame]:
            user_defined_function_kwargs = kwargs["user_defined_function_kwargs"]
            axis = user_defined_function_kwargs.get("axis", 0)

            if axis not in {0, 1, "index", "columns"}:
                raise ValueError(f"No axis named {axis} for object type DataFrame")

            axis_int = {0: 0, 1: 1, "index": 0, "columns": 1}[axis]
            opposite_axis_int = 1 - axis_int

            for chunk_ in chunk(data.shape[opposite_axis_int], nb_workers):
                yield data.iloc[chunk_] if axis_int == 1 else data.iloc[:, chunk_]

        @staticmethod
        def work(
            data: pd.DataFrame,
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> pd.DataFrame:
            return data.apply(
                user_defined_function,
                *user_defined_function_args,
                **user_defined_function_kwargs,
            )

        @staticmethod
        def reduce(
            datas: Iterable[pd.DataFrame], extra: Dict[str, Any]
        ) -> pd.DataFrame:
            return pd.concat(datas, copy=False)

    class ApplyMap(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: pd.DataFrame, **kwargs
        ) -> Iterator[pd.DataFrame]:
            for chunk_ in chunk(data.shape[0], nb_workers):
                yield data.iloc[chunk_]

        @staticmethod
        def work(
            data: pd.DataFrame,
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> pd.DataFrame:
            return data.applymap(user_defined_function)

        @staticmethod
        def reduce(
            datas: Iterable[pd.DataFrame], extra: Dict[str, Any]
        ) -> pd.DataFrame:
            return pd.concat(datas, copy=False)
