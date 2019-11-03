import itertools
import pandas as pd
from pandarallel.utils.tools import chunk


class DataFrameGroupBy:
    @staticmethod
    def get_index(df_grouped):
        if len(df_grouped.grouper.shape) == 1:
            # One element in "by" argument
            if type(df_grouped.keys) == list:
                # "by" argument is a list with only one element
                keys = df_grouped.keys[0]
            else:
                keys = df_grouped.keys

            return pd.Series(list(df_grouped.grouper), name=keys)

        # A list in "by" argument
        return pd.MultiIndex.from_tuples(
            list(df_grouped.grouper), names=df_grouped.keys
        )

    @staticmethod
    def reduce(results, index):
        return pd.DataFrame(
            list(itertools.chain.from_iterable([result for result in results])),
            index=index,
        ).squeeze()

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
        return [func(df, *args, **kwargs) for _, df in tuples]
