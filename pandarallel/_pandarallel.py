import pandas as _pd
import pyarrow.plasma as _plasma
from pyarrow.lib import PlasmaStoreFull as _PlasmaStoreFull
import multiprocessing as _multiprocessing
import itertools as _itertools
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
import functools
from tqdm._tqdm_notebook import tqdm_notebook

SHM_SIZE_MO = int(2e3) # 2 Go
NB_WORKERS = _multiprocessing.cpu_count()
PROGRESS_BAR = False

def _chunk(nb_elem, nb_chunks):
    if nb_elem <= nb_chunks:
        return [slice(idx, idx + 1) for idx in range(nb_elem)]

    quotient = nb_elem // nb_chunks
    remainder = nb_elem % nb_chunks

    quotients = [quotient] * nb_chunks
    remainders = [1] * remainder + [0] * (nb_chunks - remainder)

    nb_elems_per_chunk = [
                            quotient + remainder for quotient, remainder
                            in zip(quotients, remainders)
                        ]
    accumulated = list(_itertools.accumulate(nb_elems_per_chunk))
    shifted_accumulated = accumulated.copy()
    shifted_accumulated.insert(0, 0)
    shifted_accumulated.pop()

    return [
            slice(begin, end) for begin, end
            in zip(shifted_accumulated, accumulated)
        ]

def _parallel(nb_workers, client):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except _PlasmaStoreFull:
                msg = f"The pandarallel shared memory is too small to allow \
parallel computation. \
Just after pandarallel import, please write: \
pandarallel.initialize(<size of memory in Mo>), and retry."

                raise Exception(msg)

            finally:
                client.delete(client.list().keys())

        return wrapper
    return decorator

class _DataFrame:
    @staticmethod
    def worker(plasma_store_name, object_id,  axis_chunk, func,
               progress_bar, *args, **kwargs):
        axis = kwargs.get("axis", 0)
        client = _plasma.connect(plasma_store_name)
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

        @_parallel(nb_workers, plasma_client)
        def closure(df, func, *args, **kwargs):
            axis = kwargs.get("axis", 0)
            if axis == 'index':
                axis = 0
            elif axis == 'columns':
                axis = 1

            opposite_axis = 1 - axis
            chunks = _chunk(df.shape[opposite_axis], nb_workers)

            object_id = plasma_client.put(df)

            with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                    executor.submit(_DataFrame.worker,
                                    plasma_store_name, object_id,
                                    chunk, func, progress_bar,
                                    *args, **kwargs)
                    for index, chunk in enumerate(chunks)
                ]

            result = _pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure

class _DataFrameGroupBy:
    @staticmethod
    def worker(plasma_store_name, object_id, func):
        client = _plasma.connect(plasma_store_name)
        df = client.get(object_id)
        return client.put(func(df))

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client):
        @_parallel(nb_workers, plasma_client)
        def closure(data, func, **kwargs):
            keys = data.groups.keys()

            with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(_DataFrameGroupBy.worker,
                                            plasma_store_name,
                                            plasma_client.put(data.get_group(key)),
                                            func)
                            for key in keys
                        ]

            result = _pd.DataFrame([
                                    plasma_client.get(future.result())
                                    for future in futures
                                ], index=_pd.Series(list(data.grouper),
                                name=data.keys))

            return result
        return closure

class _Series:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, func):
        client = _plasma.connect(plasma_store_name)
        series = client.get(object_id)
        return client.put(series[chunk].map(func))

    @staticmethod
    def map(plasma_store_name, nb_workers, plasma_client):
        @_parallel(nb_workers, plasma_client)
        def closure(data, func):
            chunks = _chunk(data.size, nb_workers)
            object_id = plasma_client.put(data)

            with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(_Series.worker, plasma_store_name,
                                            object_id, _chunk, func)
                            for _chunk in chunks
                        ]

            result = _pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure

class pandarallel:
    @classmethod
    def initialize(cls, shm_size_mo=SHM_SIZE_MO, nb_workers=NB_WORKERS,
                   progress_bar=False):
        print(f"New pandarallel memory created - Size: {shm_size_mo} Mo")
        print(f"Pandarallel will run on {nb_workers} workers")

        if progress_bar:
            print("WARNING: Progress bar is an experimental feature which \
can lead to a sensitive performance lost")
            tqdm_notebook().pandas()

        cls.__store_ctx = _plasma.start_plasma_store(int(shm_size_mo * 1e6))
        plasma_store_name, _ = cls.__store_ctx.__enter__()

        plasma_client = _plasma.connect(plasma_store_name)

        args = plasma_store_name, nb_workers, plasma_client

        _pd.DataFrame.parallel_apply = _DataFrame.apply(*args, progress_bar)
        _pd.Series.parallel_map = _Series.map(*args)
        _pd.core.groupby.DataFrameGroupBy.parallel_apply = _DataFrameGroupBy.apply(*args)
