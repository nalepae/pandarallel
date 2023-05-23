import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union, cast

import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy as PandasDataFrameGroupBy

from ..utils import chunk, df_indexed_like, get_pandas_version
from .generic import DataType


class DataFrameGroupBy:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, dataframe_groupby: PandasDataFrameGroupBy, **kwargs
        ) -> Iterator[List[Tuple[int, pd.DataFrame]]]:
            chunks = chunk(dataframe_groupby.ngroups, nb_workers)
            iterator = iter(dataframe_groupby)

            for chunk_ in chunks:
                yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

        @staticmethod
        def work(
            data: List[Tuple[int, pd.DataFrame]],
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
                mutated = not df_indexed_like(result, df.axes)
                return key, result, mutated

            return [compute_result(key, df) for key, df in data]

        @staticmethod
        def get_reduce_extra(
            data: PandasDataFrameGroupBy, user_defined_function_kwargs: Dict[str, Any]
        ) -> Dict[str, Any]:
            return {"df_groupby": data}

        @staticmethod
        def reduce(
            datas: Iterable[List[Tuple[int, pd.DataFrame, bool]]], extra: Dict[str, Any]
        ) -> pd.Series:
            def get_args(
                keys: List[int],
                values: List[pd.DataFrame],
                df_groupby: PandasDataFrameGroupBy,
            ) -> Union[
                Tuple[List[int], List[pd.DataFrame]],
                Tuple[pd.DataFrame, List[int], List[pd.DataFrame]],
                Tuple[pd.DataFrame, List[pd.DataFrame]],
            ]:
                pandas_version = get_pandas_version()

                if pandas_version < (1, 3):
                    return keys, values
                elif pandas_version < (1, 4):
                    return df_groupby._selected_obj, keys, values
                else:
                    return df_groupby._selected_obj, values

            df_groupby: PandasDataFrameGroupBy = extra["df_groupby"]

            results = itertools.chain.from_iterable(datas)
            keys, values, mutated = zip(*results)

            keys = cast(List[int], keys)
            values = cast(List[pd.DataFrame], values)
            mutated = cast(List[bool], mutated)

            args = get_args(keys, values, df_groupby)
  
            return df_groupby._wrap_applied_output(*args, not_indexed_same=mutated)
