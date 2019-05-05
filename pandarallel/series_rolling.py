import pyarrow.plasma as plasma
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .utils import parallel, chunk

class SeriesRolling:
    @staticmethod
    def worker(plasma_store_name, num, object_id, attribute2value, chunk, func,
               progress_bar, *args, **kwargs):
        client = plasma.connect(plasma_store_name)
        series = client.get(object_id)

        apply_func = "progress_apply" if progress_bar else "apply"

        if progress_bar:
            # This following print is a workaround for this issue:
            # https://github.com/tqdm/tqdm/issues/485
            print(' ', end='', flush=True)

        series_chunk_rolling = series[chunk].rolling(**attribute2value)

        res = getattr(series_chunk_rolling, apply_func)(func, *args, **kwargs)

        res = res if num == 0 else res[attribute2value['window']:]

        return client.put(res)

    @staticmethod
    def apply(plasma_store_name, nb_workers, plasma_client, progress_bar):
        @parallel(plasma_client)
        def closure(rolling, func, *args, **kwargs):
            series = rolling.obj
            window = rolling.window
            chunks = chunk(len(series), nb_workers, window)
            object_id = plasma_client.put(series)

            attribute2value = {attribute: getattr(rolling, attribute)
                               for attribute in rolling._attributes}

            with ProcessPoolExecutor(max_workers=nb_workers) as executor:
                futures = [
                            executor.submit(SeriesRolling.worker,
                                            plasma_store_name, num, object_id,
                                            attribute2value,
                                            chunk, func, progress_bar,
                                            *args, **kwargs)
                            for num, chunk in enumerate(chunks)
                        ]

            result = pd.concat([
                                plasma_client.get(future.result())
                                for future in futures
                            ], copy=False)

            return result
        return closure
