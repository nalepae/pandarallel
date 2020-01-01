import pandas as pd
from pandarallel.utils.tools import chunk


class DataFrame:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    class Apply:
        @staticmethod
        def get_chunks(nb_workers, df, *args, **kwargs):
            axis = kwargs.get("axis", 0)
            if axis == "index":
                axis = 0
            elif axis == "columns":
                axis = 1

            opposite_axis = 1 - axis

            for chunk_ in chunk(df.shape[opposite_axis], nb_workers):
                if axis == 1:
                    yield df.iloc[chunk_]
                else:
                    yield df.iloc[:, chunk_]

        @staticmethod
        def worker(
            df, _index, _meta_args, _progress_bar, _queue, func, *args, **kwargs
        ):
            return df.apply(func, *args, **kwargs)

    class ApplyMap:
        @staticmethod
        def get_chunks(nb_workers, df, *_):
            for chunk_ in chunk(df.shape[0], nb_workers):
                yield df.iloc[chunk_]

        @staticmethod
        def worker(df, _index, _meta_args, _progress_bar, _queue, func, *_):
            return df.applymap(func)
