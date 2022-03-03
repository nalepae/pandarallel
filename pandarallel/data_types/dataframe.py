from itertools import count
from time import time as pandarallel_time
from time import time

import pandas as pd
from pandarallel.utils.tools import chunk


class ProgressState:
    def __init__(self, chunk_size):
        self.last_put_iteration = 0
        self.next_put_iteration = max(chunk_size // 100, 1)
        self.last_put_time = pandarallel_time()


class DataFrame:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    class Apply:
        @staticmethod
        def get_chunks(nb_workers, df, *_, **kwargs):
            axis = kwargs.get("axis", 0)

            if axis == "index":
                axis = 0

            if axis == "columns":
                axis = 1

            opposite_axis = 1 - axis

            for chunk_ in chunk(df.shape[opposite_axis], nb_workers):
                if axis == 1:
                    yield df.iloc[chunk_]
                else:
                    yield df.iloc[:, chunk_]

        @staticmethod
        def worker(
            df,
            _index,
            _meta_args,
            _progress_bar,
            _queue,
            func,
            *func_args,
            **func_kwargs
        ):
            func.__globals__["counter"] = count()
            func.__globals__["state"] = ProgressState(len(df))
            func.__globals__["pandarallel_time"] = time
            func.__globals__["queue"] = _progress_bar
            return df.apply(func, *func_args, **func_kwargs)

    class ApplyMap:
        @staticmethod
        def get_chunks(nb_workers, df, *_):
            for chunk_ in chunk(df.shape[0], nb_workers):
                yield df.iloc[chunk_]

        @staticmethod
        def worker(df, _index, _meta_args, _progress_bar, _queue, func, *_):
            return df.applymap(func)
