import multiprocessing
import pickle
from itertools import count
from multiprocessing.managers import SyncManager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Type, cast

import dill
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy as PandaDataFrameGroupBy
from pandas.core.window.expanding import ExpandingGroupby as PandasExpandingGroupby
from pandas.core.window.rolling import RollingGroupby as PandasRollingGroupby

from .data_types import (
    DataFrame,
    DataFrameGroupBy,
    DataType,
    ExpandingGroupBy,
    RollingGroupBy,
    Series,
    SeriesRolling,
)
from .progress_bars import ProgressBarsType, get_progress_bars, progress_wrapper
from .utils import WorkerStatus

# Python 3.8 on MacOS by default uses "spawn" instead of "fork" as start method for new
# processes, which is incompatible with pandarallel. We force it to use "fork" method.
CONTEXT = multiprocessing.get_context("fork")

# Root of Memory File System
MEMORY_FS_ROOT = "/dev/shm"

# By default, Pandarallel use all available CPUs
NB_WORKERS = max(CONTEXT.cpu_count() // 2, 1)

# Prefix and suffix for files used with Memory File System
PREFIX = "pandarallel"
PREFIX_INPUT = f"{PREFIX}_input_"
PREFIX_OUTPUT = f"{PREFIX}_output_"
SUFFIX = ".pickle"

# The goal of this part is to let Pandarallel to serialize functions which are not defined
# at the top level of the module (like DataFrame.Apply.worker). This trick is inspired by
# this article: https://medium.com/@yasufumy/python-multiprocessing-c6d54107dd55
# Warning: In this article, the trick is presented to be able to serialize lambda functions.
# Even if Pandarallel is able to serialize lambda functions, it is only thanks to `dill`.
_func = None


def worker_init(func: Callable) -> None:
    global _func
    _func = func


def global_worker(*args, **kwargs):
    return _func(*args, **kwargs)


def wrap_work_function_for_file_system(
    work_function: Callable[
        [Any, Callable, tuple, Dict[str, Any], Dict[str, Any]], Any
    ],
) -> Callable[
    [
        Path,
        Path,
        ProgressBarsType,
        int,
        multiprocessing.Queue,
        bytes,
        tuple,
        Dict[str, Any],
        Dict[str, Any],
    ],
    None,
]:
    def closure(
        input_file_path: Path,
        output_file_path: Path,
        progress_bars_type: ProgressBarsType,
        worker_index: int,
        master_workers_queue: multiprocessing.Queue,
        dilled_user_defined_function: bytes,
        user_defined_function_args: tuple,
        user_defined_function_kwargs: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> None:
        try:
            # Load dataframe from input file
            with input_file_path.open("rb") as file_descriptor:
                data = pickle.load(file_descriptor)

            # Delete input file since we don't need it any more. It will free some RAM
            # since the input file is stored into Shared Memory.
            input_file_path.unlink()

            data_size = len(data)
            user_defined_function: Callable = dill.loads(dilled_user_defined_function)

            progress_wrapped_user_defined_function = progress_wrapper(
                user_defined_function, master_workers_queue, worker_index, data_size
            )

            used_user_defined_function = (
                progress_wrapped_user_defined_function
                if progress_bars_type
                in (
                    ProgressBarsType.InUserDefinedFunction,
                    ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns,
                )
                else user_defined_function
            )

            result = work_function(
                data,
                used_user_defined_function,
                user_defined_function_args,
                user_defined_function_kwargs,
                extra,
            )

            with output_file_path.open("wb") as file_descriptor:
                pickle.dump(result, file_descriptor)

            master_workers_queue.put((worker_index, WorkerStatus.Success, None))

        except:
            master_workers_queue.put((worker_index, WorkerStatus.Error, None))
            raise

    return closure


def wrap_work_function_for_pipe(
    work_function: Callable[
        [
            Any,
            Callable,
            tuple,
            Dict[str, Any],
            Dict[str, Any],
        ],
        Any,
    ],
) -> Callable[
    [
        Any,
        ProgressBarsType,
        int,
        multiprocessing.Queue,
        bytes,
        tuple,
        Dict[str, Any],
        Dict[str, Any],
    ],
    Any,
]:
    def closure(
        data: Any,
        progress_bars_type: ProgressBarsType,
        worker_index: int,
        master_workers_queue: multiprocessing.Queue,
        dilled_user_defined_function: bytes,
        user_defined_function_args: tuple,
        user_defined_function_kwargs: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> Any:
        try:
            data_size = len(data)
            user_defined_function: Callable = dill.loads(dilled_user_defined_function)

            progress_wrapped_user_defined_function = progress_wrapper(
                user_defined_function, master_workers_queue, worker_index, data_size
            )

            used_user_defined_function = (
                progress_wrapped_user_defined_function
                if progress_bars_type
                in (
                    ProgressBarsType.InUserDefinedFunction,
                    ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns,
                )
                else user_defined_function
            )

            results = work_function(
                data,
                used_user_defined_function,
                user_defined_function_args,
                user_defined_function_kwargs,
                extra,
            )

            master_workers_queue.put((worker_index, WorkerStatus.Success, None))

            return results

        except:
            master_workers_queue.put((worker_index, WorkerStatus.Error, None))
            raise

    return closure


def wrap_reduce_function_for_file_system(
    reduce_function: Callable[[Iterator, Dict[str, Any]], Any]
) -> Callable[[Iterator[Path], Dict[str, Any]], Any]:
    """This wrapper transforms a `reduce` function which takes as input:
    - A list of pandas Dataframe
    - An user defined function
    and which returns a pandas Dataframe, into a `reduct` function which takes as input:
    - A list of paths where  pandas Dataframe are pickled
    which returns a pandas Dataframe.
    """

    def closure(output_file_paths: Iterator[Path], extra: Dict[str, Any]) -> Any:
        def get_dataframe_and_delete_file(file_path: Path) -> Any:
            with file_path.open("rb") as file_descriptor:
                data = pickle.load(file_descriptor)

            file_path.unlink()
            return data

        dfs = (
            get_dataframe_and_delete_file(output_file_path)
            for output_file_path in output_file_paths
        )

        return reduce_function(dfs, extra)

    return closure


def parallelize_with_memory_file_system(
    nb_requested_workers: int,
    data_type: Type[DataType],
    progress_bars_type: ProgressBarsType,
):
    def closure(
        data: Any,
        user_defined_function: Callable,
        *user_defined_function_args: tuple,
        **user_defined_function_kwargs: Dict[str, Any],
    ):
        wrapped_work_function = wrap_work_function_for_file_system(data_type.work)
        wrapped_reduce_function = wrap_reduce_function_for_file_system(data_type.reduce)

        chunks = list(
            data_type.get_chunks(
                nb_requested_workers,
                data,
                user_defined_function_kwargs=user_defined_function_kwargs,
            )
        )

        nb_workers = len(chunks)

        multiplicator_factor = (
            len(cast(pd.DataFrame, data).columns)
            if progress_bars_type
            == ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns
            else 1
        )

        progresses_length = [len(chunk_) * multiplicator_factor for chunk_ in chunks]

        work_extra = data_type.get_work_extra(data)
        reduce_extra = data_type.get_reduce_extra(data)

        show_progress_bars = progress_bars_type != ProgressBarsType.No

        progress_bars = get_progress_bars(progresses_length, show_progress_bars)
        progresses = [0] * nb_workers
        workers_status = [WorkerStatus.Running] * nb_workers

        input_files = [
            NamedTemporaryFile(
                prefix=PREFIX_INPUT, suffix=SUFFIX, dir=MEMORY_FS_ROOT, delete=False
            )
            for _ in range(nb_workers)
        ]

        output_files = [
            NamedTemporaryFile(
                prefix=PREFIX_OUTPUT, suffix=SUFFIX, dir=MEMORY_FS_ROOT, delete=False
            )
            for _ in range(nb_workers)
        ]

        try:
            for chunk, input_file in zip(chunks, input_files):
                with Path(input_file.name).open("wb") as file_descriptor:
                    pickle.dump(chunk, file_descriptor)

            dilled_user_defined_function = dill.dumps(user_defined_function)
            manager: SyncManager = CONTEXT.Manager()
            master_workers_queue = manager.Queue()

            work_args_list = [
                (
                    Path(input_file.name),
                    Path(output_file.name),
                    progress_bars_type,
                    worker_index,
                    master_workers_queue,
                    dilled_user_defined_function,
                    user_defined_function_args,
                    user_defined_function_kwargs,
                    {
                        **work_extra,
                        **{
                            "master_workers_queue": master_workers_queue,
                            "show_progress_bars": show_progress_bars,
                            "worker_index": worker_index,
                        },
                    },
                )
                for worker_index, (
                    input_file,
                    output_file,
                ) in enumerate(zip(input_files, output_files))
            ]

            pool = CONTEXT.Pool(nb_workers, worker_init, (wrapped_work_function,))
            pool.starmap_async(global_worker, work_args_list)
            pool.close()

            generation = count()

            while any(
                (
                    worker_status == WorkerStatus.Running
                    for worker_status in workers_status
                )
            ):
                message: Tuple[int, WorkerStatus, Any] = master_workers_queue.get()
                worker_index, worker_status, payload = message
                workers_status[worker_index] = worker_status

                if worker_status == WorkerStatus.Success:
                    progresses[worker_index] = progresses_length[worker_index]
                    progress_bars.update(progresses)
                elif worker_status == WorkerStatus.Running:
                    progress = cast(int, payload)
                    progresses[worker_index] = progress

                    if next(generation) % nb_workers == 0:
                        progress_bars.update(progresses)
                elif worker_status == WorkerStatus.Error:
                    progress_bars.set_error(worker_index)
                    progress_bars.update(progresses)

            return wrapped_reduce_function(
                (Path(output_file.name) for output_file in output_files),
                reduce_extra,
            )

        finally:
            for output_file in output_files:
                # When pandarallel stop supporting Python 3.7 and older, replace this
                # try/except clause by:
                # Path(output_file.name).unlink(missing_ok=True)
                try:
                    Path(output_file.name).unlink()
                except FileNotFoundError:
                    # Do nothing, this is the nominal case.
                    pass

    return closure


def parallelize_with_pipe(
    nb_requested_workers: int,
    data_type: Type[DataType],
    progress_bars_type: ProgressBarsType,
):
    def closure(
        data: Any,
        user_defined_function: Callable,
        *user_defined_function_args: tuple,
        **user_defined_function_kwargs: Dict[str, Any],
    ):
        wrapped_work_function = wrap_work_function_for_pipe(data_type.work)
        dilled_user_defined_function = dill.dumps(user_defined_function)
        manager: SyncManager = CONTEXT.Manager()
        master_workers_queue = manager.Queue()

        chunks = list(
            data_type.get_chunks(
                nb_requested_workers,
                data,
                user_defined_function_kwargs=user_defined_function_kwargs,
            )
        )

        nb_workers = len(chunks)

        multiplicator_factor = (
            len(cast(pd.DataFrame, data).columns)
            if progress_bars_type
            == ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns
            else 1
        )

        progresses_length = [len(chunk_) * multiplicator_factor for chunk_ in chunks]

        work_extra = data_type.get_work_extra(data)
        reduce_extra = data_type.get_reduce_extra(data)

        show_progress_bars = progress_bars_type != ProgressBarsType.No

        progress_bars = get_progress_bars(progresses_length, show_progress_bars)
        progresses = [0] * nb_workers
        workers_status = [WorkerStatus.Running] * nb_workers

        work_args_list = [
            (
                chunk,
                progress_bars_type,
                worker_index,
                master_workers_queue,
                dilled_user_defined_function,
                user_defined_function_args,
                user_defined_function_kwargs,
                {
                    **work_extra,
                    **{
                        "master_workers_queue": master_workers_queue,
                        "show_progress_bars": show_progress_bars,
                        "worker_index": worker_index,
                    },
                },
            )
            for worker_index, chunk in enumerate(chunks)
        ]

        pool = CONTEXT.Pool(nb_workers, worker_init, (wrapped_work_function,))
        results_promise = pool.starmap_async(global_worker, work_args_list)
        pool.close()

        generation = count()

        while any(
            (worker_status == WorkerStatus.Running for worker_status in workers_status)
        ):
            message: Tuple[int, WorkerStatus, Any] = master_workers_queue.get()
            worker_index, worker_status, payload = message
            workers_status[worker_index] = worker_status

            if worker_status == WorkerStatus.Success:
                progresses[worker_index] = progresses_length[worker_index]
                progress_bars.update(progresses)
            elif worker_status == WorkerStatus.Running:
                progress = cast(int, payload)
                progresses[worker_index] = progress

                if next(generation) % nb_workers == 0:
                    progress_bars.update(progresses)
            elif worker_status == WorkerStatus.Error:
                progress_bars.set_error(worker_index)

        results = results_promise.get()

        return data_type.reduce(results, reduce_extra)

    return closure


class pandarallel:
    @classmethod
    def initialize(
        cls,
        shm_size_mb=None,
        nb_workers=NB_WORKERS,
        progress_bar=False,
        verbose=2,
        use_memory_fs: Optional[bool] = None,
    ) -> None:
        show_progress_bars = progress_bar
        is_memory_fs_available = Path(MEMORY_FS_ROOT).exists()

        use_memory_fs = (
            use_memory_fs if use_memory_fs is not None else is_memory_fs_available
        )

        parallelize = (
            parallelize_with_memory_file_system
            if use_memory_fs
            else parallelize_with_pipe
        )

        if use_memory_fs and not is_memory_fs_available:
            raise SystemError("Memory file system is not available")

        if verbose >= 2:
            print(f"INFO: Pandarallel will run on {nb_workers} workers.")

            message = (
                (
                    "INFO: Pandarallel will use Memory file system to transfer data "
                    "between the main process and workers."
                )
                if use_memory_fs
                else (
                    "INFO: Pandarallel will use standard multiprocessing data transfer "
                    "(pipe) to transfer data between the main process and workers."
                )
            )

            print(message)

        progress_bars_in_user_defined_function = (
            ProgressBarsType.InUserDefinedFunction
            if show_progress_bars
            else ProgressBarsType.No
        )

        progress_bars_in_user_defined_function_multiply_by_number_of_columns = (
            ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns
            if show_progress_bars
            else ProgressBarsType.No
        )

        progress_bars_in_work_function = (
            ProgressBarsType.InWorkFunction
            if show_progress_bars
            else ProgressBarsType.No
        )

        # DataFrame
        pd.DataFrame.parallel_apply = parallelize(
            nb_workers, DataFrame.Apply, progress_bars_in_user_defined_function
        )
        pd.DataFrame.parallel_applymap = parallelize(
            nb_workers,
            DataFrame.ApplyMap,
            progress_bars_in_user_defined_function_multiply_by_number_of_columns,
        )

        # DataFrame GroupBy
        PandaDataFrameGroupBy.parallel_apply = parallelize(
            nb_workers, DataFrameGroupBy.Apply, progress_bars_in_user_defined_function
        )

        # Expanding GroupBy
        PandasExpandingGroupby.parallel_apply = parallelize(
            nb_workers, ExpandingGroupBy.Apply, progress_bars_in_work_function
        )

        # Rolling GroupBy
        PandasRollingGroupby.parallel_apply = parallelize(
            nb_workers, RollingGroupBy.Apply, progress_bars_in_work_function
        )

        # Series
        pd.Series.parallel_apply = parallelize(
            nb_workers, Series.Apply, progress_bars_in_user_defined_function
        )
        pd.Series.parallel_map = parallelize(nb_workers, Series.Map, show_progress_bars)

        # Series Rolling
        pd.core.window.Rolling.parallel_apply = parallelize(
            nb_workers, SeriesRolling.Apply, progress_bars_in_user_defined_function
        )
