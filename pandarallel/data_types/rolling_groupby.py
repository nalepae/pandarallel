import itertools
import pandas as pd
from pandarallel.utils.tools import chunk, PROGRESSION


class RollingGroupBy:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    @staticmethod
    def get_chunks(nb_workers, rolling_groupby, *args, **kwargs):
        chunks = chunk(len(rolling_groupby._groupby), nb_workers)
        iterator = iter(rolling_groupby._groupby)

        for chunk_ in chunks:
            yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

    @staticmethod
    def att2value(rolling):
        return {
            attribute: getattr(rolling, attribute) for attribute in rolling._attributes
        }

    @staticmethod
    def worker(
        tuples, index, attribute2value, queue, progress_bar, func, *args, **kwargs
    ):
        # TODO: See if this pd.concat is avoidable
        results = []

        for iteration, (name, df) in enumerate(tuples):
            item = df.rolling(**attribute2value).apply(func, *args, **kwargs)
            item.index = pd.MultiIndex.from_product([[name], item.index])
            results.append(item)

            if progress_bar:
                queue.put_nowait((PROGRESSION, (index, iteration)))

        return pd.concat(results)
