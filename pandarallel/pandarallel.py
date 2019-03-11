import pandas as _pd
import pyarrow.plasma as _plasma
from pyarrow.lib import PlasmaStoreFull as _PlasmaStoreFull
import multiprocessing as _multiprocessing
import itertools as _itertools
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor

__all__ = ['shm_size', 'nb_workers']

_plasma_store_ctx = None
_plasma_store_name = None

shm_size = int(2e9) # 2 Go
nb_workers = _multiprocessing.cpu_count()

def _chunk(nb_elem, nb_chunks):
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

def _parallel(func):
    def wrapper(*args, **kwargs):
        try:
            global _plasma_store_ctx
            global _plasma_store_name

            if not _plasma_store_ctx:
                mem_mo = round(shm_size / 1e6, 2)
                msg = f"New pandarallel shared memory created - \
Size: {mem_mo} Mo"
                print(msg)
                _plasma_store_ctx = _plasma.start_plasma_store(shm_size)
                _plasma_store_name, _ = _plasma_store_ctx.__enter__()

            print(f"Running task on {nb_workers} workers")

            return func(*args, **kwargs)

        except _PlasmaStoreFull:
            msg = f"The pandarallel shared memory: \
{round(shm_size / 1e6, 2)} Mo is too small to allow parallel computation. \
Just after pandarallel import, please write: \
pandarallel.shm_size = <size in bytes>, then restart your Python \
kernel and retry."

            raise Exception(msg)

    return wrapper

class _Series:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, func):
        client = _plasma.connect(plasma_store_name)
        series = client.get(object_id)
        return client.put(series[chunk].map(func))

    @staticmethod
    @_parallel
    def map(data, func):
        client = _plasma.connect(_plasma_store_name)
        chunks = _chunk(data.size, nb_workers)
        object_id = client.put(data)

        with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(_Series.worker, _plasma_store_name,
                                        object_id, _chunk, func)
                        for _chunk in chunks
                    ]

        result = _pd.concat([
                            client.get(future.result())
                            for future in futures
                        ], copy=False)

        client.delete(client.list().keys())

        return result

_pd.Series.parallel_map = _Series.map

class _DataFrameGroupBy:
    @staticmethod
    def worker(plasma_store_name, object_id, func):
        client = _plasma.connect(plasma_store_name)
        df = client.get(object_id)
        return client.put(func(df))

    @staticmethod
    @_parallel
    def apply(data, func):
        client = _plasma.connect(_plasma_store_name)
        keys = data.groups.keys()

        with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(_DataFrameGroupBy.worker,
                                        _plasma_store_name,
                                        client.put(data.get_group(key)),
                                        func)
                        for key in keys
                    ]
            
        result = _pd.DataFrame([
                                client.get(future.result())
                                for future in futures
                            ], index=_pd.Series(list(data.grouper),
                               name=data.keys))

        client.delete(client.list().keys())

        return result

_pd.core.groupby.DataFrameGroupBy.parallel_apply = _DataFrameGroupBy.apply

class _DataFrame:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, func, **kwargs):
        client = _plasma.connect(plasma_store_name)
        df = client.get(object_id)
        return client.put(df[chunk].apply(func, **kwargs))

    @staticmethod
    @_parallel
    def apply(data, func, **kwargs):
        axis = kwargs.get("axis", 0)
        if axis == 0:
            msg = "dataframe.parallel_apply is only implemented for axis=1. \
For axis=0, please use at the moment standard dataframe.apply. \
Implementation of dataframe.parallel_apply with axis=0 will come soon."
            raise NotImplementedError(msg)

        client = _plasma.connect(_plasma_store_name)
        chunks = _chunk(data.shape[0], nb_workers)
        object_id = client.put(data)

        with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(_DataFrame.worker, _plasma_store_name,
                                        object_id, _chunk, func, **kwargs)
                        for _chunk in chunks
                    ]

        result = _pd.concat([
                            client.get(future.result())
                            for future in futures
                        ], copy=False)

        client.delete(client.list().keys())

        return result        


_pd.DataFrame.parallel_apply = _DataFrame.apply
