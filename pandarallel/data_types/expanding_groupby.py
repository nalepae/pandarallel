import pandas as pd

from pandarallel.utils.tools import chunk, PROGRESSION


class ExpandingGroupBy:
    @staticmethod
    def reduce(results, _):
        return pd.concat(results, copy=False)

    @staticmethod
    def get_chunks(nb_workers, expanding_groupby, *args, **kwargs):
        pandas_version = tuple((int(item) for item in pd.__version__.split(".")))

        nb_items = (
            expanding_groupby._grouper.ngroups
            if pandas_version >= (1, 3)
            else len(expanding_groupby._groupby)
        )

        chunks = chunk(nb_items, nb_workers)

        iterator = (
            expanding_groupby._grouper.get_iterator(expanding_groupby.obj)
            if pandas_version > (1, 3)
            else iter(expanding_groupby._groupby)
        )

        for chunk_ in chunks:
            yield [next(iterator) for _ in range(chunk_.stop - chunk_.start)]

    @staticmethod
    def att2value(expanding):
        attributes = {
            attribute: getattr(expanding, attribute)
            for attribute in expanding._attributes
        }

        return attributes

    @staticmethod
    def worker(
        tuples, index, attribute2value, queue, progress_bar, func, *args, **kwargs
    ):
        # TODO: See if this pd.concat is avoidable
        results = []

        attribute2value.pop("_grouper", None)

        for iteration, (name, df) in enumerate(tuples):
            item = df.expanding(**attribute2value).apply(func, *args, **kwargs)
            item.index = pd.MultiIndex.from_product([[name], item.index])
            results.append(item)

            if progress_bar:
                queue.put_nowait((PROGRESSION, (index, iteration)))

        return pd.concat(results)
