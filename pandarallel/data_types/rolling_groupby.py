import multiprocessing
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple

import pandas as pd
from pandas.core.window.rolling import RollingGroupby as PandasRollingGroupby

from ..utils import WorkerStatus, chunk, get_pandas_version
from .generic import DataType


class RollingGroupBy:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, data: PandasRollingGroupby, *args, **kwargs
        ) -> Iterator[List[Tuple[int, pd.DataFrame]]]:
            pandas_version = get_pandas_version()

            nb_items = (
                len(data._groupby) if pandas_version < (1, 3) else data._grouper.ngroups
            )

            chunks = chunk(nb_items, nb_workers)

            iterator = (
                iter(data._groupby)
                if pandas_version < (1, 3)
                else data._grouper.get_iterator(data.obj)
            )

            for chunk_ in chunks:
                yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

        @staticmethod
        def get_work_extra(data: PandasRollingGroupby):
            attributes = {
                attribute: getattr(data, attribute) for attribute in data._attributes
            }

            return {"attributes": attributes}

        @staticmethod
        def work(
            data: List[Tuple[int, pd.DataFrame]],
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> List[pd.DataFrame]:
            show_progress_bars: bool = extra["show_progress_bars"]
            master_workers_queue: multiprocessing.Queue = extra["master_workers_queue"]
            worker_index: int = extra["worker_index"]

            def compute_result(
                iteration: int,
                attributes: Dict[str, Any],
                index: int,
                df: pd.DataFrame,
                user_defined_function: Callable,
                user_defined_function_args: tuple,
                user_defined_function_kwargs: Dict[str, Any],
            ) -> pd.DataFrame:
                item = df.rolling(**attributes).apply(
                    user_defined_function,
                    *user_defined_function_args,
                    **user_defined_function_kwargs
                )

                item.index = pd.MultiIndex.from_product([[index], item.index])

                if show_progress_bars:
                    master_workers_queue.put_nowait(
                        (worker_index, WorkerStatus.Running, iteration)
                    )

                return item

            attributes = extra["attributes"]
            attributes.pop("_grouper", None)

            dfs = (
                compute_result(
                    iteration,
                    attributes,
                    index,
                    df,
                    user_defined_function,
                    user_defined_function_args,
                    user_defined_function_kwargs,
                )
                for iteration, (index, df) in enumerate(data)
            )

            return pd.concat(dfs)

        @staticmethod
        def reduce(datas: Iterable[pd.DataFrame], extra: Dict[str, Any]) -> pd.Series:
            return pd.concat(datas, copy=False)
