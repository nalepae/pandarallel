from time import time_ns
from ctypes import c_int64
from multiprocessing import Manager
import pyarrow.plasma as plasma
import pandas as pd
from pathos.multiprocessing import ProcessingPool
from .utils import (parallel, chunk, ProgressBarsConsole,
                    ProgressBarsNotebookLab)

REFRESH_PROGRESS_TIME = int(2.5e8)  # 250 ms


class SeriesRolling:
    @staticmethod
    def worker(worker_args):
        (plasma_store_name, object_id, chunk, func, progress_bar, queue, index,
         attribute2value, args, kwargs) = worker_args

        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

        counter = c_int64(0)
        last_push_time_ns = c_int64(time_ns())

        def with_progress(func):
            def decorator(*args, **kwargs):
                counter.value += 1

                current_time_ns = time_ns()
                delta = current_time_ns - last_push_time_ns.value

                if delta >= REFRESH_PROGRESS_TIME:
                    queue.put_nowait((index, counter.value, False))
                    last_push_time_ns.value = current_time_ns

                return func(*args, **kwargs)

            return decorator

        func_to_apply = with_progress(func) if progress_bar else func

        series_chunk_rolling = series[chunk].rolling(**attribute2value)

        res = series_chunk_rolling.apply(func_to_apply, *args, **kwargs)

        res = res if index == 0 else res[attribute2value['window']:]

        if progress_bar:
            queue.put((index, counter.value, True))

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client,
              display_progress_bar, in_notebook_lab):
        @parallel(plasma_client)
        def closure(rolling, func, *args, **kwargs):
            pool = ProcessingPool(nb_workers)
            manager = Manager()
            queue = manager.Queue()

            ProgressBars = (ProgressBarsNotebookLab if in_notebook_lab
                            else ProgressBarsConsole)

            series = rolling.obj
            window = rolling.window
            chunks = chunk(len(series), nb_workers, window)

            maxs = [chunk.stop - chunk.start for chunk in chunks]
            values = [0] * nb_workers
            finished = [False] * nb_workers

            if display_progress_bar:
                progress_bar = ProgressBars(maxs)

            object_id = plasma_client.put(series)

            attribute2value = {attribute: getattr(rolling, attribute)
                               for attribute in rolling._attributes}

            workers_args = [(plasma_store_name, object_id, chunk, func,
                             display_progress_bar, queue, index,
                             attribute2value, args, kwargs)
                            for index, chunk in enumerate(chunks)]

            result_workers = pool.amap(SeriesRolling.worker, workers_args)

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
