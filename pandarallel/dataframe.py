import pyarrow.plasma as plasma
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .utils import parallel, chunk

class DataFrame:
    @staticmethod
    def worker_apply(plasma_store_name, object_id,  axis_chunk, func,
                     progress_bar, *args, **kwargs):
        axis = kwargs.get("axis", 0)
        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        apply_func = "progress_apply" if progress_bar else "apply"

        if axis == 1:
            if progress_bar:
                # This following print is a workaround for this issue:
                # https://github.com/tqdm/tqdm/issues/485
                print(' ', end='', flush=True)
            res = getattr(df[axis_chunk], apply_func)(func, *args, **kwargs)
        else:
            chunk = slice(0, df.shape[0]), df.columns[axis_chunk]
            res = getattr(df.loc[chunk], apply_func)(func, *args, **kwargs)

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client,
              progress_bar=False):
        @parallel(plasma_client)
        def closure(df, func, *args, **kwargs):
            axis = kwargs.get("axis", 0)
            if axis == 'index':
                axis = 0
            elif axis == 'columns':
                axis = 1

            opposite_axis = 1 - axis
            chunks = chunk(df.shape[opposite_axis], nb_workers)

            object_id = plasma_client.put(df)

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                    executor.submit(DataFrame.worker_apply,
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

    @staticmethod
    def worker_applymap(plasma_store_name, object_id,  axis_chunk, func,
                        progress_bar):
        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        applymap_func = "progress_applymap" if progress_bar else "applymap"

        if progress_bar:
            # This following print is a workaround for this issue:
            # https://github.com/tqdm/tqdm/issues/485
            print(' ', end='', flush=True)
        res = getattr(df[axis_chunk], applymap_func)(func)

        return client.put(res)

    @staticmethod
    def applymap(plasma_store_name, nb_workers, plasma_client,
                 progress_bar=False):
        @parallel(plasma_client)
        def closure(df, func):
            chunks = chunk(df.shape[0], nb_workers)
            object_id = plasma_client.put(df)

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                    executor.submit(DataFrame.worker_applymap,
                                    plasma_store_name, object_id,
                                    chunk, func, progress_bar)
                    for index, chunk in enumerate(chunks)
                ]

            result = pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure
