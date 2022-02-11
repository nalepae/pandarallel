import pandas as pd
from pandarallel.utils.tools import chunk


def _parse_axis_from_kwargs(kwargs):
    axis = kwargs.get("axis", 0)
    if axis == "index":
        axis = 0
    elif axis == "columns":
        axis = 1
    return axis


class DataFrame:

    class Apply:
        @staticmethod
        def get_chunks(nb_workers, df, *args, **kwargs):
            axis = _parse_axis_from_kwargs(kwargs)

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

        @staticmethod
        def get_reduce_meta_args(_, kwargs):
            axis = _parse_axis_from_kwargs(kwargs)
            return abs(axis - 1)

        @staticmethod
        def reduce(results, axis):
            if all([isinstance(ir, pd.Series) for ir in results]):
                axis = 0
            return pd.concat(results, copy=False, axis=axis)

    class ApplyMap:
        @staticmethod
        def get_chunks(nb_workers, df, *_):
            for chunk_ in chunk(df.shape[0], nb_workers):
                yield df.iloc[chunk_]

        @staticmethod
        def worker(df, _index, _meta_args, _progress_bar, _queue, func, *_):
            return df.applymap(func)

        @staticmethod
        def reduce(results, _):
            return pd.concat(results, copy=False)