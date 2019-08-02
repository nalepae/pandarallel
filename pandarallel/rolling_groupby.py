import pyarrow.plasma as plasma
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from .utils import parallel, chunk


class RollingGroupby:
    @staticmethod
    def worker(worker_args):
        (plasma_store_name, object_id, groups_id, attribute2value, chunk,
         func, args, kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        groups = client.get(groups_id)[chunk]

        results = []
        for name, indexes in groups:
            item = (df.iloc[indexes].rolling(**attribute2value)
                                    .apply(func, *args, **kwargs))

            item.index = pd.MultiIndex.from_product([[name], item.index])

            results.append(item)

        return client.put(pd.concat(results))

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client, _1, _2):
        @parallel(plasma_client)
        def closure(rolling_groupby, func, *args, **kwargs):
            groups = list(rolling_groupby._groupby.groups.items())
            chunks = chunk(len(groups), nb_workers)
            object_id = plasma_client.put(rolling_groupby.obj)
            groups_id = plasma_client.put(groups)

            attribute2value = {attribute: getattr(rolling_groupby, attribute)
                               for attribute in rolling_groupby._attributes}

            worker_args = [(plasma_store_name, object_id, groups_id,
                            attribute2value, chunk, func, args, kwargs)
                           for chunk in chunks]

            with ProcessingPool(nb_workers) as pool:
                result_workers = pool.map(RollingGroupby.worker, worker_args)

            result = pd.concat([
                plasma_client.get(result_worker)
                for result_worker in result_workers
            ], copy=False)

            return result
        return closure
