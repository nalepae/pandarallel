import pandas as pd
from pandarallel.utils.tools import chunk


class SeriesRolling:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    @staticmethod
    def get_chunks(nb_workers, rolling, *args, **kwargs):
        chunks = chunk(rolling.obj.size, nb_workers, rolling.window)

        for chunk_ in chunks:
            yield rolling.obj[chunk_]

    @staticmethod
    def att2value(rolling):
        return {
            attribute: getattr(rolling, attribute) for attribute in rolling._attributes
        }

    @staticmethod
    def worker(
        series, index, attribue2value, _progress_bar, _queue, func, *args, **kwargs
    ):
        result = series.rolling(**attribue2value).apply(func, *args, **kwargs)

        return result if index == 0 else result[attribue2value["window"] :]
