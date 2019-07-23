import pyarrow.plasma as plasma
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from .utils import parallel, chunk


class Series:
    @staticmethod
    def worker_map(worker_args):
        (plasma_store_name, object_id, chunk, arg, progress_bar,
         kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

        map_func = "progress_map" if progress_bar else "map"

        if progress_bar:
            # This following print is a workaround for this issue:
            # https://github.com/tqdm/tqdm/issues/485
            print(' ', end='', flush=True)

        res = getattr(series[chunk], map_func)(arg, **kwargs)

        return client.put(res)

    @staticmethod
    def map(plasma_store_name, nb_workers, plasma_client, progress_bar):
        @parallel(plasma_client)
        def closure(data, arg, **kwargs):
            chunks = chunk(data.size, nb_workers)
            object_id = plasma_client.put(data)

            workers_args = [(plasma_store_name, object_id, chunk, arg,
                             progress_bar, kwargs) for chunk in chunks]

            with ProcessingPool(nb_workers) as pool:
                result_workers = pool.map(Series.worker_map, workers_args)

            result = pd.concat([
                plasma_client.get(result_worker)
                for result_worker in result_workers
            ], copy=False)

            return result
        return closure

    @staticmethod
    def worker_apply(worker_args):
        (plasma_store_name, object_id, chunk, func,
         progress_bar, args, kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

        apply_func = "progress_apply" if progress_bar else "apply"

        if progress_bar:
            # This following print is a workaround for this issue:
            # https://github.com/tqdm/tqdm/issues/485
            print(' ', end='', flush=True)

        res = getattr(series[chunk], apply_func)(func, *args, **kwargs)

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client, progress_bar):
        @parallel(plasma_client)
        def closure(series, func, *args, **kwargs):
            chunks = chunk(series.size, nb_workers)
            object_id = plasma_client.put(series)

            workers_args = [(plasma_store_name, object_id,
                             chunk, func, progress_bar,
                             args, kwargs) for chunk in chunks]

            with ProcessingPool(nb_workers) as pool:
                results_workers = pool.map(Series.worker_apply, workers_args)

            result = pd.concat([
                plasma_client.get(result_worker)
                for result_worker in results_workers
            ], copy=False)

            return result
        return closure
