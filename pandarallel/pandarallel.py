"""Main Pandarallel file"""

from ast import Call
from asyncio import Queue
from audioop import mul
import multiprocessing
from multiprocessing.pool import MapResult
import os
from pathlib import Path
import pickle
from itertools import count
from multiprocessing import get_context
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from time import time as pandarallel_time
from typing import Any, Callable, List, Tuple

import dill
from pandas import DataFrame, Series
from pandas.core.groupby import DataFrameGroupBy
from pandas.core.window import ExpandingGroupby, Rolling, RollingGroupby
from prometheus_client import Enum

from pandarallel.data_types.dataframe import DataFrame as DF
from pandarallel.data_types.dataframe_groupby import DataFrameGroupBy as DFGB
from pandarallel.data_types.expanding_groupby import ExpandingGroupBy as EGB
from pandarallel.data_types.rolling_groupby import RollingGroupBy as RGB
from pandarallel.data_types.series import Series as S
from pandarallel.data_types.series_rolling import SeriesRolling as SR
from pandarallel.utils.inliner import inline, ProgressState
from pandarallel.utils.progress_bars import get_progress_bars, is_notebook_lab
from pandarallel.utils.tools import ERROR, INPUT_FILE_READ, PROGRESSION, VALUE

# Python 3.8 on MacOS by default uses "spawn" instead of "fork" as start method for new
# processes, which is incompatible with pandarallel. We force it to use "fork" method.
CONTEXT = get_context("fork")

# By default, Pandarallel use all available CPUs
NB_WORKERS = CONTEXT.cpu_count()

# Prefix and suffix for files used with Memory File System
PREFIX = "pandarallel_"
PREFIX_INPUT = PREFIX + "input_"
PREFIX_OUTPUT = PREFIX + "output_"
SUFFIX = ".pickle"

# Root of Memory File System
MEMORY_FS_ROOT = "/dev/shm"


class ProgressType(int, Enum):
    NoProgress = 0
    ProgressInWorker = 1
    ProgressInFunc = 2
    ProgressInFuncMul = 3


# The goal of this part is to let Pandarallel to serialize functions which are not defined
# at the top level of the module (like DataFrame.Apply.worker). This trick is inspired by
# this article: https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
# Warning: In this article, the trick is presented to be able to serialize lambda functions.
# Even if Pandarallel is able to serialize lambda functions, it is only thanks to `dill`.
_func = None


def worker_init(func: Callable) -> None:
    global _func
    _func = func


def global_worker(x):
    return _func(x)


def is_memory_fs_available() -> bool:
    """Check if Memory File System is available"""
    return Path(MEMORY_FS_ROOT).exists()


WorkerArgsMemoryFS = Tuple[
    Path, Path, int, dict, multiprocessing.Queue, bool, Callable, list, dict
]

WorkerArgsMemoryPipe = Tuple[
    Any, int, dict, multiprocessing.Queue, bool, bytes, list, dict
]


def wrap_worker_memory_fs(
    worker_function: Callable[
        [Any, int, dict, multiprocessing.Queue, bool, Callable, list, dict], Any
    ]
):
    def closure(worker_args: WorkerArgsMemoryFS) -> None:
        (
            input_file_path,
            output_file_path,
            index,
            meta_args,
            queue,
            progress_bar_enabled,
            func,
            func_args,
            func_kwargs,
        ) = worker_args

        try:
            with open(input_file_path, "rb") as file:
                data = pickle.load(file)
                queue.put((INPUT_FILE_READ, index))

            result = worker_function(
                data,
                index,
                meta_args,
                queue,
                progress_bar_enabled,
                func,
                *func_args,
                **func_kwargs,
            )

            with open(output_file_path, "wb") as file:
                pickle.dump(result, file)

            queue.put((VALUE, index))

        except Exception:
            queue.put((ERROR, index))
            raise

    return closure


def wrap_worker_pipe(
    worker_function: Callable[
        [Any, int, dict, multiprocessing.Queue, bool, bytes, list, dict], Any
    ]
):
    def closure(worker_args: WorkerArgsMemoryPipe) -> None:
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
            result = worker_function(
                data,
                index,
                meta_args,
                queue,
                progress_bar,
                dill.loads(dilled_func),
                *args,
                **kwargs,
            )

            queue.put((VALUE, index))

            return result

        except Exception:
            queue.put((ERROR, index))
            raise

    return closure


def prepare_worker(use_memory_fs: bool):
    """
    use_memory_fs: True Memory file system has to be used, else false
    """

    def closure(function):
        def wrapper(worker_args):
            """This function runs on WORKERS.

            If Memory File System is used:
            1. Load all pickled files (previously dumped by the MASTER) in the
               Memory File System
            2. Undill the function to apply (for lambda functions)
            3. Tell to the MASTER the input file has been read (so the MASTER can remove it
               from the memory
            4. Apply the function
            5. Pickle the result in the Memory File System (so the Master can read it)
            6. Tell the master task is finished

            If Memory File System is not used, steps are the same except 1. and 5. which are
            skipped.
            """
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
                        **kwargs,
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
                        **kwargs,
                    )
                    queue.put((VALUE, index))

                    return result

                except Exception:
                    queue.put((ERROR, index))
                    raise

        return wrapper

    return closure


def create_temp_files(nb_files: int) -> List[_TemporaryFileWrapper]:
    """Create temporary files in Memory File System."""
    return [
        NamedTemporaryFile(prefix=PREFIX_INPUT, suffix=SUFFIX, dir=MEMORY_FS_ROOT)
        for _ in range(nb_files)
    ]


def progress_pre_func(
    queue: multiprocessing.Queue,
    index: int,
    counter: count,
    progression: int,
    state: ProgressState,
) -> None:
    """Send progress to the MASTER about every 250 ms.

    The estimation system is implemented to avoid to call time() to often,
    which is time consuming.
    """
    iteration = next(counter)

    if iteration == state.next_put_iteration:
        time_now = pandarallel_time()
        queue.put_nowait((progression, (index, iteration)))

        delta_t = time_now - state.last_put_time
        delta_i = iteration - state.last_put_iteration

        state.next_put_iteration += max(int((delta_i / delta_t) * 0.25), 1)
        state.last_put_iteration = iteration
        state.last_put_time = time_now


def progress_wrapper(
    progress_bar: bool, queue: multiprocessing.Queue, index: int, chunk_size: int
) -> Callable:
    """Wrap the function to apply in a function which monitor the part of work already done.

    inline is used instead of traditional wrapping system to avoid unnecessary function call
    (and context switch) which is time consuming.
    """
    counter = count()
    state = ProgressState(chunk_size)

    def wrapper(func: Callable) -> Callable:
        if not progress_bar:
            return func

        wrapped_func = inline(
            progress_pre_func,
            func,
            dict(
                index=index,
                progression=PROGRESSION,
            ),
            dict(
                counter=counter,
                queue=queue,
                state=state,
            ),
        )

        return wrapped_func

    return wrapper


# def get_workers_args_and_metadata_for_memory_fs(
#     progress_bar: ProgressType,
#     chunks: List[Any],
#     worker_meta_args: dict,
#     queue: multiprocessing.Queue,
#     func: Callable,
#     func_args: list,
#     func_kwargs: dict,
# ) -> Tuple[
#     List[WorkerArgsMemoryFS],
#     List[int],
#     List[_TemporaryFileWrapper],
#     List[_TemporaryFileWrapper],
# ]:
#     """This function runs on the MASTER.

#     1. Create temporary files in Memory File System
#     2. Dump chunked input files into Memory File System
#        (So they can be read by workers)
#     3. Break input data into several chunks
#     4. Wrap the function to apply to display progress bars
#     5. Dill the function to apply (to handle lambda functions)
#     6. Return the function to be sent to workers and path of files
#        in the Memory File System

#     progress_bar    : True is the progress bar has to be used, else False
#     chunks          : The list if chunked DataFrames
#     worker_meta_args: Worker meta arguments
#     queue           : The communication queue between master and workers
#     func            : The pandas user defined function
#     func_args       : Arguments of `func`
#     func_kwargs     : Keywork arguments of `func`
#     """

#     def dump_and_get_lenght(chunk: Any, input_file: _TemporaryFileWrapper) -> int:
#         with Path(input_file.name).open("wb") as file_descriptor:
#             pickle.dump(chunk, file_descriptor)

#         return len(chunk)

#     nb_chunks = len(chunks)
#     input_files = create_temp_files(nb_chunks)

#     try:
#         chunk_lengths = [
#             dump_and_get_lenght(chunk, input_file)
#             for chunk, input_file in zip(chunks, input_files)
#         ]

#         nb_chunks = len(chunk_lengths)
#         output_files = create_temp_files(nb_chunks)

#     except OSError:
#         link = "https://stackoverflow.com/questions/58804022/how-to-resize-dev-shm"
#         msg = " ".join(
#             (
#                 "It seems you use Memory File System and you don't have enough",
#                 "available space in `dev/shm`. You can either call",
#                 "pandarallel.initalize with `use_memory_fs=False`, or you can ",
#                 "increase the size of `dev/shm` as described here:",
#                 link,
#                 ".",
#                 " Please also remove all files beginning with 'pandarallel_' in the",
#                 "`/dev/shm` directory. If you have troubles with your web browser,",
#                 "these troubles should disappear after cleaning `/dev/shm`.",
#             )
#         )
#         raise OSError(msg)

#     workers_args = [
#         (
#             Path(input_file.name),
#             Path(output_file.name),
#             index,
#             worker_meta_args,
#             queue,
#             progress_bar == ProgressType.ProgressInWorker,
#             dill.dumps(
#                 progress_wrapper(
#                     progress_bar >= ProgressType.ProgressInFunc,
#                     queue,
#                     index,
#                     chunk_length,
#                 )(func)
#             ),
#             func_args,
#             func_kwargs,
#         )
#         for index, (input_file, output_file, chunk_length) in enumerate(
#             zip(input_files, output_files, chunk_lengths)
#         )
#     ]

#     return workers_args, chunk_lengths, input_files, output_files


def get_workers_args_and_metadata_for_pipe(
    progress_bar: int,
    chunks: List[Any],
    worker_meta_args: dict,
    queue: multiprocessing.Queue,
    func: Callable,
    func_args: list,
    func_kwargs: dict,
) -> Tuple[List[WorkerArgsMemoryFS], List[int]]:
    """This function runs on the MASTER.

    1. Create temporary files in Memory File System
    2. Dump chunked input files into Memory File System
       (So they can be read by workers)
    3. Break input data into several chunks
    4. Wrap the function to apply to display progress bars
    5. Dill the function to apply (to handle lambda functions)
    6. Return the function to be sent to workers and path of files
       in the Memory File System

    nb_workers      : The number of workers
    progress_bar    : True is the progress bar has to be used, else False
    chunks          : The list if chunked DataFrames
    worker_meta_args: Worker meta arguments
    queue           : The communication queue between master and workers
    func            : The pandas user defined function
    func_args       : Arguments of `func`
    func_kwargs     : Keywork arguments of `func`
    """
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
                            progress_bar == ProgressType.ProgressInFunc,
                            queue,
                            index,
                            len(chunk),
                        )(func)
                    ),
                    func_args,
                    func_kwargs,
                ),
                len(chunk),
            )
            for index, chunk in enumerate(chunks)
        ]
    )

    return workers_args, chunk_lengths


def get_workers_result(
    show_progress_bar: ProgressType,
    nb_columns: int,
    queue: multiprocessing.Queue,
    chunk_lengths: List[int],
    input_files: List[_TemporaryFileWrapper],
    map_result: MapResult,
) -> Any:
    """Wait for the workers' results while eventually display progress bars."""
    if show_progress_bar:
        if show_progress_bar == ProgressType.ProgressInfuncMul:
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
                if is_notebook_lab():
                    progress_bars.set_error(worker_index)
                progress_bars.update(progresses)

    return map_result.get()


def parallelize_memory_fs(
    progress_bar: ProgressType,
    worker: Callable[
        [Any, int, dict, multiprocessing.Queue, bool, Callable, list, dict], Any
    ],
    chunks: list,
    worker_meta_args: dict,
    queue: multiprocessing.Queue,
    func: Callable,
    func_args: list,
    func_kwargs: dict,
) -> Any:
    def dump_and_get_lenght(chunk: Any, input_file: _TemporaryFileWrapper) -> int:
        try:
            with Path(input_file.name).open("wb") as file_descriptor:
                pickle.dump(chunk, file_descriptor)

            return len(chunk)
        except OSError as error:
            link = "https://stackoverflow.com/questions/58804022/how-to-resize-dev-shm"

            msg = (
                "It seems you use Memory File System and you don't have enough"
                "available space in `dev/shm`. You can either call"
                "pandarallel.initalize with `use_memory_fs=False`, or you can "
                f"increase the size of `dev/shm` as described here: {link}."
                " Please also remove all files beginning with 'pandarallel_' in the"
                "`/dev/shm` directory. If you have troubles with your web browser,"
                "these troubles should disappear after cleaning `/dev/shm`."
            )
            raise OSError(msg) from error

    nb_chunks = len(chunks)

    input_files = create_temp_files(nb_chunks)
    output_files = create_temp_files(nb_chunks)

    chunk_lengths = [
        dump_and_get_lenght(chunk, input_file)
        for chunk, input_file in zip(chunks, input_files)
    ]

    workers_args = [
        (
            Path(input_file.name),
            Path(output_file.name),
            index,
            worker_meta_args,
            queue,
            progress_bar == ProgressType.ProgressInWorker,
            progress_wrapper(
                progress_bar >= ProgressType.ProgressInFunc,
                queue,
                index,
                chunk_length,
            )(func),
            func_args,
            func_kwargs,
        )
        for index, (input_file, output_file, chunk_length) in enumerate(
            zip(input_files, output_files, chunk_lengths)
        )
    ]

    try:
        wrapped_worker = wrap_worker_memory_fs(worker)

        pool = CONTEXT.Pool(
            nb_chunks,
            worker_init,
            (wrapped_worker,),
        )

        map_result = pool.map_async(global_worker, workers_args)
        pool.close()

        # results = get_workers_result(
        #     progress_bar,
        #     nb_columns,
        #     queue,
        #     chunk_lengths,
        #     input_files,
        #     output_files,
        #     map_result,
        # )

        # return (
        #     [pickle.load(output_files) for output_files in output_files]
        #     if use_memory_fs
        #     else results
        # )

        # return reduce(results, reduce_meta_args)
    finally:
        for file in input_files + output_files:
            file.close()


# def parallelize(
#     nb_requested_workers: int,
#     use_memory_fs: bool,
#     progress_bar: ProgressType,
#     get_chunks: Callable,
#     worker: Callable[
#         [Any, int, dict, bool, multiprocessing.Queue, Callable, list, dict], Any
#     ],
#     reduce: Callable,
#     get_worker_meta_args: Callable[[Any], dict] = lambda _: dict(),
#     get_reduce_meta_args: Callable[[Any], dict] = lambda _: dict(),
# ):
#     """Master function.
#     1. Split data into chunks
#     2. Send chunks to workers
#     3. Wait for the workers' results (while displaying a progress bar if needed)
#     4. Once results are available, combine them
#     5. Return combined results to the user
#     """

#     def closure(data, func, *args, **kwargs):

#         chunks = get_chunks(nb_requested_workers, data, *args, **kwargs)

#         nb_columns = (
#             len(data.columns)
#             if progress_bar == ProgressType.ProgressInFuncMul
#             else None
#         )

#         worker_meta_args = get_worker_meta_args(data)
#         reduce_meta_args = get_reduce_meta_args(data)
#         manager = CONTEXT.Manager()
#         queue: multiprocessing.Queue = manager.Queue()

#         (
#             workers_args,
#             chunk_lengths,
#             input_files,
#             output_files,
#         ) = get_workers_args_and_metadata(
#             use_memory_fs,
#             nb_requested_workers,
#             progress_bar,
#             chunks,
#             worker_meta_args,
#             queue,
#             func,
#             args,
#             kwargs,
#         )

#         nb_workers = len(chunk_lengths)

#         try:
#             pool = CONTEXT.Pool(
#                 nb_workers,
#                 worker_init,
#                 (prepare_worker(use_memory_fs)(worker),),
#             )

#             map_result = pool.map_async(global_worker, workers_args)
#             pool.close()

#             results = get_workers_result(
#                 use_memory_fs,
#                 nb_workers,
#                 progress_bar,
#                 nb_columns,
#                 queue,
#                 chunk_lengths,
#                 input_files,
#                 output_files,
#                 map_result,
#             )

#             return reduce(results, reduce_meta_args)

#         finally:
#             if use_memory_fs:
#                 for file in input_files + output_files:
#                     file.close()

#     return closure


def parallelize(
    nb_requested_workers: int,
    use_memory_fs: bool,
    progress_bar: ProgressType,
    get_chunks: Callable,
    worker: Callable[
        [Any, int, dict, bool, multiprocessing.Queue, Callable, list, dict], Any
    ],
    reduce: Callable,
    get_worker_meta_args: Callable[[Any], dict] = lambda _: dict(),
    get_reduce_meta_args: Callable[[Any], dict] = lambda _: dict(),
):
    """Master function.
    1. Split data into chunks
    2. Send chunks to workers
    3. Wait for the workers' results (while displaying a progress bar if needed)
    4. Once results are available, combine them
    5. Return combined results to the user
    """

    def closure(data, func, *args, **kwargs):

        chunks = get_chunks(nb_requested_workers, data, *args, **kwargs)

        nb_columns = (
            len(data.columns)
            if progress_bar == ProgressType.ProgressInFuncMul
            else None
        )

        worker_meta_args = get_worker_meta_args(data)
        reduce_meta_args = get_reduce_meta_args(data)
        manager = CONTEXT.Manager()
        queue: multiprocessing.Queue = manager.Queue()

        (
            workers_args,
            chunk_lengths,
            input_files,
            output_files,
        ) = get_workers_args_and_metadata(
            use_memory_fs,
            nb_requested_workers,
            progress_bar,
            chunks,
            worker_meta_args,
            queue,
            func,
            args,
            kwargs,
        )

        nb_workers = len(chunk_lengths)

        try:
            pool = CONTEXT.Pool(
                nb_workers,
                worker_init,
                (prepare_worker(use_memory_fs)(worker),),
            )

            map_result = pool.map_async(global_worker, workers_args)
            pool.close()

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
            it to transfer data between the main process and workers. If memory file
            system is not available, Pandarallel will default on multiprocessing data
            transfer (pipe).

            If set to True, Pandarallel will use memory file system to transfer data
            between the main process and workers and will raise a SystemError if memory
            file system is not available.

            If set to False, Pandarallel will use multiprocessing data transfer
            (pipe) to transfer data between the main process and workers.

            Using memory file system reduces data transfer time between the main
            process and workers, especially for big data.

            Memory file system is considered as available only if the
            directory `/dev/shm` exists and if the user has read and write
            permission on it.

            Basically memory file system is only available on some Linux
            distributions (including Ubuntu)
        """

        memory_fs_available = is_memory_fs_available()
        use_memory_fs = use_memory_fs or use_memory_fs is None and memory_fs_available

        if shm_size_mb:
            print(
                "WARNING: `shm_size_mb` is a deprecated argument. "
                "It will be removed in `pandarallel 2.0.0`."
            )

        if use_memory_fs and not memory_fs_available:
            raise SystemError("Memory file system is not available")

        if verbose >= 2:
            print("INFO: Pandarallel will run on", nb_workers, "workers.")

            if use_memory_fs:
                print(
                    (
                        "INFO: Pandarallel will use Memory file system to transfer data"
                        "between the main process and workers."
                    )
                )
            else:
                print(
                    (
                        "INFO: Pandarallel will use standard multiprocessing data "
                        "transfer (pipe) to transfer data between the main process and"
                        "workers."
                    )
                )

        bargs_prog_func = (nb_workers, use_memory_fs, ProgressType.ProgressInFunc)
        # bargs_prog_func_mul = (nb_workers, use_memory_fs, progress_in_func_mul)

        # bargs_prog_worker = (nb_workers, use_memory_fs, progress_in_worker)

        # DataFrame
        DataFrame.parallel_apply = parallelize(
            nb_workers,
            use_memory_fs,
            ProgressType.ProgressInFunc,
            DF.Apply.get_chunks,
            DF.Apply.worker,
            DF.reduce,
        )

        # args = bargs_prog_func_mul + (
        #     DF.ApplyMap.get_chunks,
        #     DF.ApplyMap.worker,
        #     DF.reduce,
        # )

        # DataFrame.parallel_applymap = parallelize(*args)

        # # Series
        # args = bargs_prog_func + (S.get_chunks, S.Apply.worker, S.reduce)
        # Series.parallel_apply = parallelize(*args)

        # args = bargs_prog_func + (S.get_chunks, S.Map.worker, S.reduce)
        # Series.parallel_map = parallelize(*args)

        # # Series Rolling
        # args = bargs_prog_func + (SR.get_chunks, SR.worker, SR.reduce)
        # kwargs = dict(get_worker_meta_args=SR.att2value)
        # Rolling.parallel_apply = parallelize(*args, **kwargs)

        # # DataFrame GroupBy
        # args = bargs_prog_func + (DFGB.get_chunks, DFGB.worker, DFGB.reduce)
        # kwargs = dict(get_reduce_meta_args=DFGB.get_reduce_meta_args)
        # DataFrameGroupBy.parallel_apply = parallelize(*args, **kwargs)

        # # Rolling GroupBy
        # args = bargs_prog_worker + (RGB.get_chunks, RGB.worker, RGB.reduce)
        # kwargs = dict(get_worker_meta_args=RGB.att2value)
        # RollingGroupby.parallel_apply = parallelize(*args, **kwargs)

        # # Expanding GroupBy
        # args = bargs_prog_worker + (EGB.get_chunks, EGB.worker, EGB.reduce)
        # kwargs = dict(get_worker_meta_args=EGB.att2value)
        # ExpandingGroupby.parallel_apply = parallelize(*args, **kwargs)
