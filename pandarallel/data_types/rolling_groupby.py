from datetime import timedelta

import pandas as pd
from pandas.tseries.frequencies import to_offset

from pandarallel.utils.tools import chunk, PROGRESSION


class RollingGroupBy:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    @staticmethod
    def get_chunks(nb_workers, rolling_groupby, *args, **kwargs):
        pandas_version = tuple((int(item) for item in pd.__version__.split(".")))

        nb_items = (
            rolling_groupby._grouper.ngroups
            if pandas_version >= (1, 3)
            else len(rolling_groupby._groupby)
        )

        chunks = chunk(nb_items, nb_workers)

        iterator = (
            rolling_groupby._grouper.get_iterator(rolling_groupby.obj)
            if pandas_version > (1, 3)
            else iter(rolling_groupby._groupby)
        )

        for chunk_ in chunks:
            yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

    @staticmethod
    def att2value(rolling):
        attributes = {
            attribute: getattr(rolling, attribute) for attribute in rolling._attributes
        }

        # Fix window for win_type = freq, because then it was defined by the user in a format like '1D' and refers
        # to a time window rolling
        if "win_type" in attributes and attributes["win_type"] == "freq":
            window = to_offset(timedelta(microseconds=int(attributes["window"] / 1000)))
            attributes["window"] = window
            attributes.pop("win_type")

        return attributes

    @staticmethod
    def worker(
        tuples, index, attribute2value, queue, progress_bar, func, *args, **kwargs
    ):
        # TODO: See if this pd.concat is avoidable
        results = []

        attribute2value.pop("_grouper", None)

        for iteration, (name, df) in enumerate(tuples):
            item = df.rolling(**attribute2value).apply(func, *args, **kwargs)
            item.index = pd.MultiIndex.from_product([[name], item.index])
            results.append(item)

            if progress_bar:
                queue.put_nowait((PROGRESSION, (index, iteration)))

        return pd.concat(results)
