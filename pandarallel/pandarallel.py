import os
import pickle
from itertools import count
from multiprocessing import Manager, Pool, cpu_count
from tempfile import NamedTemporaryFile
from time import time

import dill
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import Rolling, RollingGroupby

from pandarallel.data_types.dataframe import DataFrame as DF
from pandarallel.data_types.dataframe_groupby import DataFrameGroupBy as DFGB
from pandarallel.data_types.rolling_groupby import RollingGroupBy as RGB
from pandarallel.data_types.series import Series as S
from pandarallel.data_types.series_rolling import SeriesRolling as SR
from pandarallel.utils.inliner import inline
from pandarallel.utils.progress_bars import get_progress_bars
from pandarallel.utils.tools import ERROR, INPUT_FILE_READ, PROGRESSION, VALUE

NB_WORKERS = cpu_count()
PREFIX = "pandarallel_"
PREFIX_INPUT = PREFIX + "input_"
PREFIX_OUTPUT = PREFIX + "output_"
SUFFIX = ".pickle"
MEMORY_FS_ROOT = "/dev/shm"

NO_PROGRESS, PROGRESS_IN_WORKER, PROGRESS_IN_FUNC, PROGRESS_IN_FUNC_MUL = list(range(4))

_func = None


class ProgressState:
    last_put_iteration = None
    next_put_iteration = None
    last_put_time = None


def worker_init(func):
    global _func
    _func = func


def global_worker(x):
    return _func(x)


def is_memory_fs_available():
    return os.path.exists(MEMORY_FS_ROOT)


def prepare_worker(use_memory_fs):
    def closure(function):
        def wrapper(worker_args):
            if use_memory_fs:
                (
                    input_file_path,
                    output_file_path,
                    index,
                    meta_args,
                    queue,
                    progress_bar,
                    dilled_func,
                    args,
                    kwargs,
                ) = worker_args

                try:
                    with open(input_file_path, "rb") as file:
                        data = pickle.load(file)
                        queue.put((INPUT_FILE_READ, index))

                    result = function(
                        data,
                        index,
                        meta_args,
                        queue,
                        progress_bar,
                        dill.loads(dilled_func),
                        *args,
                        **kwargs
                    )

                    with open(output_file_path, "wb") as file:
                        pickle.dump(result, file)

                    queue.put((VALUE, index))

                except Exception:
                    queue.put((ERROR, index))
                    raise
            else:
                (
                    data,
                    index,
                    meta_args,
                    queue,
                    progress_bar,
                    dilled_func,
                    args,
                    kwargs,
                ) = worker_args

                try:
                    result = function(
                        data,
                        index,
                        meta_args,
                        queue,
                        progress_bar,
                        dill.loads(dilled_func),
                        *args,
                        **kwargs
                    )
                    queue.put((VALUE, index))

                    return result

                except Exception:
                    queue.put((ERROR, index))
                    raise

        return wrapper

    return closure


def create_temp_files(nb_files):
    return [
        NamedTemporaryFile(prefix=PREFIX_INPUT, suffix=SUFFIX, dir=MEMORY_FS_ROOT)
        for _ in range(nb_files)
    ]


def progress_pre_func(queue, index, counter, progression, state, time):
    iteration = next(counter)

    if iteration == state.next_put_iteration:
        time_now = time()
        queue.put_nowait((progression, (index, iteration)))

        delta_t = time_now - state.last_put_time
        delta_i = iteration - state.last_put_iteration

        state.next_put_iteration += max(int((delta_i / delta_t) * 0.25), 1)
        state.last_put_iteration = iteration
        state.last_put_time = time_now


def progress_wrapper(progress_bar, queue, index, chunk_size):
    counter = count()
    state = ProgressState()
    state.last_put_iteration = 0
    state.next_put_iteration = max(chunk_size // 100, 1)
    state.last_put_time = time()

    def wrapper(func):
        if progress_bar:
            wrapped_func = inline(
                progress_pre_func,
                func,
                dict(
                    queue=queue,
                    index=index,
                    counter=counter,
                    progression=PROGRESSION,
                    state=state,
                    time=time,
                ),
            )
            return wrapped_func

        return func

    return wrapper


def get_workers_args(
    use_memory_fs,
    nb_workers,
    progress_bar,
    chunks,
    worker_meta_args,
    queue,
    func,
    args,
    kwargs,
):
    def dump_and_get_lenght(chunk, input_file):
        with open(input_file.name, "wb") as file:
            pickle.dump(chunk, file)

        return len(chunk)

    if use_memory_fs:
        input_files = create_temp_files(nb_workers)
        output_files = create_temp_files(nb_workers)

        chunk_lengths = [
            dump_and_get_lenght(chunk, input_file)
            for chunk, input_file in zip(chunks, input_files)
        ]

        workers_args = [
            (
                input_file.name,
                output_file.name,
                index,
                worker_meta_args,
                queue,
                progress_bar == PROGRESS_IN_WORKER,
                dill.dumps(
                    progress_wrapper(
                        progress_bar >= PROGRESS_IN_FUNC, queue, index, chunk_length
                    )(func)
                ),
                args,
                kwargs,
            )
            for index, (input_file, output_file, chunk_length) in enumerate(
                zip(input_files, output_files, chunk_lengths)
            )
        ]

        return workers_args, chunk_lengths, input_files, output_files

    else:
        workers_args, chunk_lengths = zip(
            *[
                (
                    (
                        chunk,
                        index,
                        worker_meta_args,
                        queue,
                        progress_bar,
                        dill.dumps(
                            progress_wrapper(
                                progress_bar == PROGRESS_IN_FUNC,
                                queue,
                                index,
                                len(chunk),
                            )(func)
                        ),
                        args,
                        kwargs,
                    ),
                    len(chunk),
                )
                for index, chunk in enumerate(chunks)
            ]
        )

        return workers_args, chunk_lengths, [], []


def get_workers_result(
    use_memory_fs,
    nb_workers,
    show_progress_bar,
    nb_columns,
    queue,
    chunk_lengths,
    input_files,
    output_files,
    map_result,
):
    if show_progress_bar:
        if show_progress_bar == PROGRESS_IN_FUNC_MUL:
            chunk_lengths = [
                chunk_length * (nb_columns + 1) for chunk_length in chunk_lengths
            ]

        progress_bars = get_progress_bars(chunk_lengths)

        progresses = [0] * nb_workers

    finished_workers = [False] * nb_workers

    generation = count()

    while not all(finished_workers):
        message_type, message = queue.get()

        if message_type is INPUT_FILE_READ:
            file_index = message
            input_files[file_index].close()

        elif message_type is PROGRESSION:
            worker_index, progression = message
            progresses[worker_index] = progression

            if next(generation) % nb_workers == 0:
                progress_bars.update(progresses)

        elif message_type is VALUE:
            worker_index = message
            finished_workers[worker_index] = VALUE

            if show_progress_bar:
                progresses[worker_index] = chunk_lengths[worker_index]
                progress_bars.update(progresses)

        elif message_type is ERROR:
            worker_index = message
            finished_workers[worker_index] = ERROR

            if show_progress_bar:
                progress_bars.set_error(worker_index)
                progress_bars.update(progresses)

    results = map_result.get()

    return (
        [pickle.load(output_files) for output_files in output_files]
        if use_memory_fs
        else results
    )


def parallelize(
    nb_workers,
    use_memory_fs,
    progress_bar,
    get_chunks,
    worker,
    reduce,
    get_worker_meta_args=lambda _: dict(),
    get_reduce_meta_args=lambda _: dict(),
):
    def closure(data, func, *args, **kwargs):
        chunks = get_chunks(nb_workers, data, *args, **kwargs)
        nb_columns = len(data.columns) if progress_bar == PROGRESS_IN_FUNC_MUL else None
        worker_meta_args = get_worker_meta_args(data)
        reduce_meta_args = get_reduce_meta_args(data)
        manager = Manager()
        queue = manager.Queue()

        workers_args, chunk_lengths, input_files, output_files = get_workers_args(
            use_memory_fs,
            nb_workers,
            progress_bar,
            chunks,
            worker_meta_args,
            queue,
            func,
            args,
            kwargs,
        )
        try:
            pool = Pool(
                nb_workers, worker_init, (prepare_worker(use_memory_fs)(worker),)
            )

            map_result = pool.map_async(global_worker, workers_args)

            results = get_workers_result(
                use_memory_fs,
                nb_workers,
                progress_bar,
                nb_columns,
                queue,
                chunk_lengths,
                input_files,
                output_files,
                map_result,
            )

            return reduce(results, reduce_meta_args)

        finally:
            if use_memory_fs:
                for file in input_files + output_files:
                    file.close()

    return closure


class pandarallel:
    @classmethod
    def initialize(
        cls,
        shm_size_mb=None,
        nb_workers=NB_WORKERS,
        progress_bar=False,
        verbose=2,
        use_memory_fs=None,
    ):
        """
        Initialize Pandarallel shared memory.

        Parameters
        ----------
        shm_size_mb: int, optional
            Deprecated

        nb_workers: int, optional
            Number of workers used for parallelisation
            If not set, all available CPUs will be used.

        progress_bar: bool, optional
            Display progress bars if set to `True`

        verbose: int, optional
            The verbosity level
            0 - Don't display any logs
            1 - Display only warning logs
            2 - Display all logs

        use_memory_fs: bool, optional
            If set to None and if memory file system is available, Pandaralllel will use
            it to tranfer data between the main process and workers. If memory file
            system is not available, Pandarallel will default on multiprocessing data
            transfer (pipe).

            If set to True, Pandarallel will use memory file system to tranfer data
            between the main process and workers and will raise a SystemError if memory
            file system is not available.

            If set to False, Pandarallel will use multiprocessing data transfer
            (pipe) to tranfer data between the main process and workers.

            Using memory file system reduces data transfer time between the main
            process and workers, especially for big data.

            Memory file system is considered as available only if the
            directory `/dev/shm` exists and if the user has read an write
            rights on it.

            Basicaly memory file system is only available on some Linux
            distributions (including Ubuntu)
        """

        memory_fs_available = is_memory_fs_available()
        use_memory_fs = use_memory_fs or use_memory_fs is None and memory_fs_available

        if use_memory_fs and not memory_fs_available:
            raise SystemError("Memory file system is not available")

        if verbose >= 2:
            print("INFO: Pandarallel will run on", nb_workers, "workers.")

            if use_memory_fs:
                print(
                    "INFO: Pandarallel will use Memory file system to transfer data",
                    "between the main process and workers.",
                    sep=" ",
                )
            else:
                print(
                    "INFO: Pandarallel will use standard multiprocessing data tranfer",
                    "(pipe) to transfer data between the main",
                    "process and workers.",
                    sep=" ",
                )

        nbw = nb_workers

        progress_in_func = PROGRESS_IN_FUNC * progress_bar
        progress_in_func_mul = PROGRESS_IN_FUNC_MUL * progress_bar
        progress_in_worker = PROGRESS_IN_WORKER * progress_bar

        bargs_prog_func = (nbw, use_memory_fs, progress_in_func)
        bargs_prog_func_mul = (nbw, use_memory_fs, progress_in_func_mul)

        bargs_prog_worker = (nbw, use_memory_fs, progress_in_worker)

        # DataFrame
        args = bargs_prog_func + (DF.Apply.get_chunks, DF.Apply.worker, DF.reduce)
        DataFrame.parallel_apply = parallelize(*args)

        args = bargs_prog_func_mul + (
            DF.ApplyMap.get_chunks,
            DF.ApplyMap.worker,
            DF.reduce,
        )

        DataFrame.parallel_applymap = parallelize(*args)

        # Series
        args = bargs_prog_func + (S.get_chunks, S.Apply.worker, S.reduce)
        Series.parallel_apply = parallelize(*args)

        args = bargs_prog_func + (S.get_chunks, S.Map.worker, S.reduce)
        Series.parallel_map = parallelize(*args)

        # Series Rolling
        args = bargs_prog_func + (SR.get_chunks, SR.worker, SR.reduce)
        kwargs = dict(get_worker_meta_args=SR.att2value)
        Rolling.parallel_apply = parallelize(*args, **kwargs)

        # DataFrame GroupBy
        args = bargs_prog_func + (DFGB.get_chunks, DFGB.worker, DFGB.reduce)
        kwargs = dict(get_reduce_meta_args=DFGB.get_index)
        DataFrameGroupBy.parallel_apply = parallelize(*args, **kwargs)

        # Rolling GroupBy
        args = bargs_prog_worker + (RGB.get_chunks, RGB.worker, RGB.reduce)
        kwargs = dict(get_worker_meta_args=SR.att2value)
        RollingGroupby.parallel_apply = parallelize(*args, **kwargs)
