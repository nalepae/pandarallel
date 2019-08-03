from time import time
from ctypes import c_uint64, c_double
from multiprocessing import Manager
import pyarrow.plasma as plasma
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from .utils import (parallel, chunk, ProgressBarsConsole,
                    ProgressBarsNotebookLab)

REFRESH_PROGRESS_TIME = 0.25  # s


class Series:
    @staticmethod
    def worker_map(worker_args):
        (plasma_store_name, object_id, chunk, func, progress_bar, queue, index,
         kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

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

        func_to_apply = with_progress(func) if progress_bar else func

        res = series[chunk].map(func_to_apply, **kwargs)

        if progress_bar:
            queue.put((index, counter.value, True))

        return client.put(res)

    @staticmethod
    def map(plasma_store_name, nb_workers, plasma_client,
            display_progress_bar, in_notebook_lab):
        @parallel(plasma_client)
        def closure(data, func, **kwargs):
            pool = ProcessingPool(nb_workers)
            manager = Manager()
            queue = manager.Queue()

            ProgressBars = (ProgressBarsNotebookLab if in_notebook_lab
                            else ProgressBarsConsole)

            chunks = chunk(data.size, nb_workers)

            maxs = [chunk.stop - chunk.start for chunk in chunks]
            values = [0] * nb_workers
            finished = [False] * nb_workers

            if display_progress_bar:
                progress_bar = ProgressBars(maxs)

            object_id = plasma_client.put(data)

            workers_args = [(plasma_store_name, object_id, chunk, func,
                             display_progress_bar, queue, index, kwargs)
                            for index, chunk in enumerate(chunks)]

            result_workers = pool.amap(Series.worker_map, workers_args)

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
    def worker_apply(worker_args):
        (plasma_store_name, object_id, chunk, func, progress_bar, queue, index,
         args, kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

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

        func_to_apply = with_progress(func) if progress_bar else func

        res = series[chunk].apply(func_to_apply, *args, **kwargs)

        if progress_bar:
            queue.put((index, counter.value, True))

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client,
              display_progress_bar, in_notebook_lab):
        @parallel(plasma_client)
        def closure(series, func, *args, **kwargs):
            pool = ProcessingPool(nb_workers)
            manager = Manager()
            queue = manager.Queue()

            ProgressBars = (ProgressBarsNotebookLab if in_notebook_lab
                            else ProgressBarsConsole)

            chunks = chunk(series.size, nb_workers)

            maxs = [chunk.stop - chunk.start for chunk in chunks]
            values = [0] * nb_workers
            finished = [False] * nb_workers

            if display_progress_bar:
                progress_bar = ProgressBars(maxs)

            object_id = plasma_client.put(series)

            workers_args = [(plasma_store_name, object_id, chunk, func,
                             display_progress_bar, queue, index,
                             args, kwargs)
                            for index, chunk in enumerate(chunks)]

            result_workers = pool.amap(Series.worker_apply, workers_args)

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
