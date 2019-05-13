import pyarrow.plasma as plasma
import pandas as pd
import itertools
from concurrent.futures import ProcessPoolExecutor
from .utils import parallel, chunk

class DataFrameGroupBy:
    @staticmethod
    def worker(plasma_store_name, object_id, groups_id, chunk,
               func, *args, **kwargs):
        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        groups = client.get(groups_id)[chunk]
        result = [
                    func(df.iloc[indexes], *args, **kwargs)
                    for _, indexes in groups
        ]

        return client.put(result)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client):
        @parallel(plasma_client)
        def closure(df_grouped, func, *args, **kwargs):
            groups = list(df_grouped.groups.items())
            chunks = chunk(len(groups), nb_workers)
            object_id = plasma_client.put(df_grouped.obj)
            groups_id = plasma_client.put(groups)

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                    executor.submit(DataFrameGroupBy.worker,
                                    plasma_store_name, object_id,
                                    groups_id, chunk, func, *args, **kwargs)
                    for chunk in chunks
                ]

            if len(df_grouped.grouper.shape) == 1:
                # One element in "by" argument
                if type(df_grouped.keys) == list:
                    # "by" argument is a list with only one element
                    keys = df_grouped.keys[0]
                else:
                    keys = df_grouped.keys

                index = pd.Series(list(df_grouped.grouper),
                                  name=keys)

            else:
                # A list in "by" argument
                index = pd.MultiIndex.from_tuples(list(df_grouped.grouper),
                                                  names=df_grouped.keys)

            result = pd.DataFrame(list(itertools.chain.from_iterable([
                                    plasma_client.get(future.result())
                                    for future in futures
                                   ])),
                                   index=index
                     ).squeeze()
            return result
        return closure
