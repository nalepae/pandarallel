import itertools
import pandas as pd
from pandarallel.utils.tools import chunk


class DataFrameGroupBy:
    @staticmethod
    def get_reduce_meta_args(df_grouped):
        return df_grouped

    @staticmethod
    def reduce(results, df_grouped):
        results = itertools.chain.from_iterable(results)
        keys, values, mutated = zip(*results)
        mutated = any(mutated)
        return df_grouped._wrap_applied_output(
            keys, values, not_indexed_same=df_grouped.mutated or mutated
        )

    @staticmethod
    def get_chunks(nb_workers, df_grouped, *args, **kwargs):
        chunks = chunk(len(df_grouped), nb_workers)
        iterator = iter(df_grouped)

        for chunk_ in chunks:
            yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

    @staticmethod
    def worker(
        tuples, _index, _meta_args, _progress_bar, _queue, func, *args, **kwargs
    ):
        keys, results, mutated = [], [], []
        for key, df in tuples:
            res = func(df, *args, **kwargs)
            results.append(res)
            mutated.append(not pd.core.groupby.ops._is_indexed_like(res, df.axes))
            keys.append(key)

        return zip(keys, results, mutated)
