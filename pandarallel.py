import pandas as pd
import pyarrow.plasma as plasma
from pyarrow.lib import PlasmaStoreFull
import multiprocessing
import itertools
from concurrent.futures import ProcessPoolExecutor

plasma_store_ctx = None
plasma_store_name = None

shm_size = int(2e9) # 2 Go
nb_workers = multiprocessing.cpu_count()

def chunk(nb_elem, nb_chunks):
    quotient = nb_elem // nb_chunks
    remainder = nb_elem % nb_chunks

    quotients = [quotient] * nb_chunks
    remainders = [1] * remainder + [0] * (nb_chunks - remainder)

    nb_elems_per_chunk = [
                            quotient + remainder for quotient, remainder
                            in zip(quotients, remainders)
                        ]
    accumulated = list(itertools.accumulate(nb_elems_per_chunk))
    shifted_accumulated = accumulated.copy()
    shifted_accumulated.insert(0, 0)
    shifted_accumulated.pop()

    return [
            slice(begin, end) for begin, end
            in zip(shifted_accumulated, accumulated)
        ]

def parallel(func):
    def wrapper(*args, **kwargs):
        try:
            global plasma_store_ctx
            global plasma_store_name

            if not plasma_store_ctx:
                mem_mo = round(shm_size / 1e6, 2)
                msg = f"New pandarallel shared memory created - \
Size: {mem_mo} Mo"
                print(msg)
                plasma_store_ctx = plasma.start_plasma_store(shm_size)
                plasma_store_name, _ = plasma_store_ctx.__enter__()

            print(f"Running task on {nb_workers} workers")

            return func(*args, **kwargs)

        except PlasmaStoreFull:
            msg = f"The pandarallel shared memory: \
{round(shm_size / 1e6, 2)} Mo is too small to allow parallel computation. \
Just after pandarallel import, please write: \
pandarallel.shm_size = <size in bytes>, then restart your Python \
kernel and retry."

            raise Exception(msg)

    return wrapper

class Series:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, func):
        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)
        return client.put(series[chunk].map(func))

    @staticmethod
    @parallel
    def map(data, func):
        client = plasma.connect(plasma_store_name)
        chunks = chunk(data.size, nb_workers)
        object_id = client.put(data)

        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(Series.worker, plasma_store_name,
                                        object_id, chunk, func)
                        for chunk in chunks
                    ]

        result = pd.concat([
                            client.get(future.result())
                            for future in futures
                        ], copy=False)

        client.delete(client.list().keys())

        return result

pd.Series.parallel_map = Series.map

class DataFrameGroupBy:
    @staticmethod
    def worker(plasma_store_name, object_id, func):
        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        return client.put(func(df))

    @staticmethod
    @parallel
    def apply(data, func):
        client = plasma.connect(plasma_store_name)
        keys = data.groups.keys()

        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(DataFrameGroupBy.worker,
                                        plasma_store_name,
                                        client.put(data.get_group(key)),
                                        func)
                        for key in keys
                    ]
            
        result = pd.DataFrame([
                                client.get(future.result())
                                for future in futures
                            ], index=pd.Series(list(data.grouper),
                               name=data.keys))

        client.delete(client.list().keys())

        return result

pd.core.groupby.DataFrameGroupBy.parallel_apply = DataFrameGroupBy.apply

class DataFrame:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, func, **kwargs):
        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        return client.put(df[chunk].apply(func, **kwargs))

    @staticmethod
    @parallel
    def apply(data, func, **kwargs):
        axis = kwargs.get("axis", 0)
        if axis == 0:
            msg = "dataframe.parallel_apply is only implemented for axis=1. \
For axis=0, please use at the moment standard dataframe.apply. \
Implementation of dataframe.parallel_apply with axis=0 will come soon."
            raise NotImplementedError(msg)

        client = plasma.connect(plasma_store_name)
        chunks = chunk(data.shape[0], nb_workers)
        object_id = client.put(data)

        with ProcessPoolExecutor(max_workers=nb_workers) as executor:
            futures = [
                        executor.submit(DataFrame.worker, plasma_store_name,
                                        object_id, chunk, func, **kwargs)
                        for chunk in chunks
                    ]

        result = pd.concat([
                            client.get(future.result())
                            for future in futures
                        ], copy=False)

        client.delete(client.list().keys())

        return result        


pd.DataFrame.parallel_apply = DataFrame.apply