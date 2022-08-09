from typing import Any, Callable, Dict, Iterable, Iterator
from types import GeneratorType

import pandas as pd

from ..utils import chunk, get_axis_int
from .generic import DataType


class DataFrame:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: pd.DataFrame, **kwargs
        ) -> Iterator[pd.DataFrame]:
            user_defined_function_kwargs = kwargs["user_defined_function_kwargs"]

            axis_int = get_axis_int(user_defined_function_kwargs)
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
        def get_reduce_extra(
            data: Any, user_defined_function_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"axis": get_axis_int(user_defined_function_kwargs)}

        @staticmethod
        def reduce(
            datas: Iterable[pd.DataFrame], extra: Dict[str, Any]
        ) -> pd.DataFrame:
            if isinstance(datas, GeneratorType):
                datas = list(datas)
            axis = 0 if isinstance(datas[0], pd.Series) else 1 - extra["axis"]
            return pd.concat(datas, copy=False, axis=axis)

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
