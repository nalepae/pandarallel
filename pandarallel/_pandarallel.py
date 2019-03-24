import itertools
import pandas as _pd
import numpy as np
import pyarrow.plasma as _plasma
from pyarrow.lib import PlasmaStoreFull as _PlasmaStoreFull
import multiprocessing as _multiprocessing
import itertools as _itertools
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from tqdm._tqdm_notebook import tqdm_notebook

SHM_SIZE_MO = int(2e3) # 2 Go
NB_WORKERS = _multiprocessing.cpu_count()
PROGRESS_BAR = False

def _chunk(nb_item, nb_chunks):
    """
    Return `nb_chunks` slices of approximatively `nb_item / nb_chunks` each.

    Parameters
    ----------
    nb_item : int
        Total number of items

    nb_chunks : int
        Number of chunks to return

    Returns
    -------
    A list of slices


    Examples
    --------
    >>> chunks = _pandarallel._chunk(103, 4)
    >>> chunks
    [slice(0, 26, None), slice(26, 52, None), slice(52, 78, None),
     slice(78, 103, None)]
    """
    if nb_item <= nb_chunks:
        return [slice(idx, idx + 1) for idx in range(nb_item)]

    quotient = nb_item // nb_chunks
    remainder = nb_item % nb_chunks

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
            """Please see the docstring of this method without `parallel`"""
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
    def worker(plasma_store_name, object_id, groups_id, chunk,
               func, *args, **kwargs):
        client = _plasma.connect(plasma_store_name)
        df = client.get(object_id)
        groups = client.get(groups_id)[chunk]
        result = [
                    func(df.iloc[indexes], *args, **kwargs)
                    for _, indexes in groups
        ]

        return client.put(result)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client):
        @_parallel(nb_workers, plasma_client)
        def closure(df_grouped, func, *args, **kwargs):
            groups = list(df_grouped.groups.items())
            chunks = _chunk(len(groups), nb_workers)
            object_id = plasma_client.put(df_grouped.obj)
            groups_id = plasma_client.put(groups)

            with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                    executor.submit(_DataFrameGroupBy.worker,
                                    plasma_store_name, object_id,
                                    groups_id, chunk, func, *args, **kwargs)
                    for chunk in chunks
                ]

            result = _pd.DataFrame(list(itertools.chain.from_iterable([
                                    plasma_client.get(future.result())
                                    for future in futures
                                   ])),
                                   index=_pd.Series(list(df_grouped.grouper),
                                   name=df_grouped.keys)
                     ).squeeze()
            return result
        return closure

class _Series:
    @staticmethod
    def worker(plasma_store_name, object_id, chunk, arg, progress_bar,
               **kwargs):
        client = _plasma.connect(plasma_store_name)
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
        @_parallel(nb_workers, plasma_client)
        def closure(data, arg, **kwargs):
            chunks = _chunk(data.size, nb_workers)
            object_id = plasma_client.put(data)

            with _ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(_Series.worker, plasma_store_name,
                                            object_id, _chunk, arg,
                                            progress_bar, **kwargs)
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
        """
        Initialize Pandarallel shared memory.

        Parameters
        ----------
        shm_size_mo : int, optional
            Size of Pandarallel shared memory

        nb_workers : int, optional
            Number of worker used for parallelisation

        progress_bar : bool, optional
            Display a progress bar
            WARNING: Progress bar is an experimental feature.
                     This can lead to a sensitive performance loss.
        """

        print(f"New pandarallel memory created - Size: {shm_size_mo} Mo")
        print(f"Pandarallel will run on {nb_workers} workers")

        if progress_bar:
            print("WARNING: Progress bar is an experimental feature. This \
can lead to a sensitive performance loss")
            tqdm_notebook().pandas()

        cls.__store_ctx = _plasma.start_plasma_store(int(shm_size_mo * 1e6))
        plasma_store_name, _ = cls.__store_ctx.__enter__()

        plasma_client = _plasma.connect(plasma_store_name)

        args = plasma_store_name, nb_workers, plasma_client

        _pd.DataFrame.parallel_apply = _DataFrame.apply(*args, progress_bar)
        _pd.Series.parallel_map = _Series.map(*args, progress_bar)
        _pd.core.groupby.DataFrameGroupBy.parallel_apply = _DataFrameGroupBy.apply(*args)
