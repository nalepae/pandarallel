import pyarrow.plasma as plasma
import pandas as pd
import itertools
from pathos.multiprocessing import ProcessingPool
from .utils import parallel, chunk


class DataFrameGroupBy:
    @staticmethod
    def worker(worker_args):
        (plasma_store_name, object_id, groups_id, chunk, func, args,
         kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        groups = client.get(groups_id)[chunk]
        result = [
            func(df.iloc[indexes], *args, **kwargs)
            for _, indexes in groups
        ]

        return client.put(result)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client, _1, _2):
        @parallel(plasma_client)
        def closure(df_grouped, func, *args, **kwargs):
            groups = list(df_grouped.groups.items())
            chunks = chunk(len(groups), nb_workers)
            object_id = plasma_client.put(df_grouped.obj)
            groups_id = plasma_client.put(groups)

            workers_args = [(plasma_store_name, object_id, groups_id, chunk,
                             func, args, kwargs) for chunk in chunks]

            with ProcessingPool(nb_workers) as pool:
                result_workers = pool.map(
                    DataFrameGroupBy.worker, workers_args)

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
                plasma_client.get(result_worker)
                for result_worker in result_workers
            ])),
                index=index
            ).squeeze()
            return result
        return closure
