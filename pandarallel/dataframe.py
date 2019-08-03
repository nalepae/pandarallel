from time import time
from ctypes import c_double, c_uint64
from multiprocessing import Manager
import pyarrow.plasma as plasma
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from .utils import (parallel, chunk, ProgressBarsConsole,
                    ProgressBarsNotebookLab)

REFRESH_PROGRESS_TIME = 0.25  # s


class DataFrame:
    @staticmethod
    def worker_apply(worker_args):
        (plasma_store_name, object_id, axis_chunk, func, progress_bar, queue,
         index, args, kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)

        counter = c_uint64(0)
        last_push_time = c_double(time())

        def with_progress(func):
            def decorator(*args, **kwargs):
                counter.value += 1

                cur_time = time()

                if cur_time - last_push_time.value >= REFRESH_PROGRESS_TIME:
                    queue.put_nowait((index, counter.value, False))
                    last_push_time.value = cur_time

                return func(*args, **kwargs)

            return decorator

        axis = kwargs.get("axis", 0)
        func_to_apply = with_progress(func) if progress_bar else func

        if axis == 1:
            res = df[axis_chunk].apply(func_to_apply, *args, **kwargs)
        else:
            chunk = slice(0, df.shape[0]), df.columns[axis_chunk]
            res = df.loc[chunk].apply(func_to_apply, *args, **kwargs)

        if progress_bar:
            queue.put((index, counter.value, True))

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client,
              display_progress_bar, in_notebook_lab):
        @parallel(plasma_client)
        def closure(df, func, *args, **kwargs):
            pool = ProcessingPool(nb_workers)
            manager = Manager()
            queue = manager.Queue()

            ProgressBars = (ProgressBarsNotebookLab if in_notebook_lab
                            else ProgressBarsConsole)

            axis = kwargs.get("axis", 0)
            if axis == 'index':
                axis = 0
            elif axis == 'columns':
                axis = 1

            opposite_axis = 1 - axis
            chunks = chunk(df.shape[opposite_axis], nb_workers)

            maxs = [chunk.stop - chunk.start for chunk in chunks]
            values = [0] * nb_workers
            finished = [False] * nb_workers

            if display_progress_bar:
                progress_bar = ProgressBars(maxs)

            object_id = plasma_client.put(df)

            workers_args = [(plasma_store_name, object_id, chunk, func,
                             display_progress_bar, queue, index, args, kwargs)
                            for index, chunk in enumerate(chunks)]

            result_workers = pool.amap(DataFrame.worker_apply, workers_args)

            if display_progress_bar:
                while not all(finished):
                    for _ in range(finished.count(False)):
                        index, value, status = queue.get()
                        values[index] = value
                        finished[index] = status

                    progress_bar.update(values)

            result = pd.concat([
                plasma_client.get(result_worker)
                for result_worker in result_workers.get()
            ], copy=False)

            return result
        return closure

    @staticmethod
    def worker_applymap(worker_args):
        (plasma_store_name, object_id, axis_chunk, func,
         progress_bar, queue, index) = worker_args

        client = plasma.connect(plasma_store_name)
        df = client.get(object_id)
        nb_columns_1 = df.shape[1] + 1

        counter = c_uint64(0)
        last_push_time = c_double(time())

        def with_progress(func):
            def decorator(arg):
                counter.value += 1

                cur_time = time()

                if(cur_time - last_push_time.value >= REFRESH_PROGRESS_TIME):
                    if(counter.value % nb_columns_1 == 0):
                        queue.put_nowait((index,
                                          counter.value // nb_columns_1,
                                          False))
                        last_push_time.value = cur_time

                return func(arg)

            return decorator

        func_to_apply = with_progress(func) if progress_bar else func

        res = df[axis_chunk].applymap(func_to_apply)

        if progress_bar:
            row_counter = counter.value // nb_columns_1
            queue.put((index, row_counter, True))

        return client.put(res)

    @staticmethod
    def applymap(plasma_store_name, nb_workers, plasma_client,
                 display_progress_bar, in_notebook_lab):
        @parallel(plasma_client)
        def closure(df, func):
            pool = ProcessingPool(nb_workers)
            manager = Manager()
            queue = manager.Queue()

            ProgressBars = (ProgressBarsNotebookLab if in_notebook_lab
                            else ProgressBarsConsole)

            chunks = chunk(df.shape[0], nb_workers)

            maxs = [chunk.stop - chunk.start for chunk in chunks]
            values = [0] * nb_workers
            finished = [False] * nb_workers

            if display_progress_bar:
                progress_bar = ProgressBars(maxs)

            object_id = plasma_client.put(df)

            worker_args = [(plasma_store_name, object_id, chunk, func,
                            display_progress_bar, queue, index)
                           for index, chunk in enumerate(chunks)]

            result_workers = pool.amap(DataFrame.worker_applymap, worker_args)

            if display_progress_bar:
                while not all(finished):
                    for _ in range(finished.count(False)):
                        index, value, status = queue.get()
                        values[index] = value
                        finished[index] = status

                    progress_bar.update(values)

            result = pd.concat([
                plasma_client.get(result_worker)
                for result_worker in result_workers.get()
            ], copy=False)

            return result
        return closure
