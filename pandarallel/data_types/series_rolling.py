from typing import Any, Callable, Dict, Iterable, Iterator

import pandas as pd
from pandas.core.window.rolling import Rolling

from ..utils import chunk
from .generic import DataType


class SeriesRolling:
    class Apply(DataType):
        @staticmethod
        def get_chunks(
            nb_workers: int, rolling: Rolling, **kwargs
        ) -> Iterator[pd.Series]:
            chunks = chunk(rolling.obj.size, nb_workers, rolling.window)

            for chunk_ in chunks:
                yield rolling.obj[chunk_]

        @staticmethod
        def get_work_extra(data: Rolling) -> Dict[str, Any]:
            return {
                "attributes": {
                    attribute: getattr(data, attribute)
                    for attribute in data._attributes
                }
            }

        @staticmethod
        def work(
            data: pd.Series,
            user_defined_function: Callable,
            user_defined_function_args: tuple,
            user_defined_function_kwargs: Dict[str, Any],
            extra: Dict[str, Any],
        ) -> pd.Series:
            attributes: Dict[str, Any] = extra["attributes"]
            worker_index: int = extra["worker_index"]

            result = data.rolling(**attributes).apply(
                user_defined_function,
                *user_defined_function_args,
                **user_defined_function_kwargs
            )

            return result if worker_index == 0 else result[attributes["window"] :]

        @staticmethod
        def reduce(datas: Iterable[pd.Series], extra: Dict[str, Any]) -> pd.Series:
            return pd.concat(datas, copy=False)
