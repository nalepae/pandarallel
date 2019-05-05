import pyarrow.plasma as plasma
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .utils import parallel, chunk

class Series:
    @staticmethod
    def worker_map(plasma_store_name, object_id, chunk, arg, progress_bar,
                   **kwargs):
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

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(Series.worker_map,
                                            plasma_store_name, object_id,
                                            _chunk, arg, progress_bar,
                                            **kwargs)
                            for _chunk in chunks
                        ]

            result = pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure

    @staticmethod
    def worker_apply(plasma_store_name, object_id, chunk, func,
                     progress_bar, *args, **kwargs):
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

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(Series.worker_apply,
                                            plasma_store_name, object_id,
                                            chunk, func, progress_bar,
                                            *args, **kwargs)
                            for chunk in chunks
                        ]

            result = pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure
