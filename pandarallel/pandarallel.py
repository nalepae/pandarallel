import pandas as pd
import pyarrow.plasma as plasma
import multiprocessing as multiprocessing

from .dataframe import DataFrame
from .series import Series
from .series_rolling import SeriesRolling
from .rolling_groupby import RollingGroupby
from .dataframe_groupby import DataFrameGroupBy
from .plasma_store import start_plasma_store

SHM_SIZE_MB = int(2e3)  # 2 GB
NB_WORKERS = multiprocessing.cpu_count()
PROGRESS_BAR = False


def is_notebook_lab():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook/lab or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


class pandarallel:
    @classmethod
    def initialize(cls, shm_size_mb=SHM_SIZE_MB, nb_workers=NB_WORKERS,
                   progress_bar=False, verbose=2):
        """
        Initialize Pandarallel shared memory.

        Parameters
        ----------
        shm_size_mb: int, optional
            Size of Pandarallel shared memory

        nb_workers: int, optional
            Number of worker used for parallelisation

        progress_bar: bool, optional
            Display a progress bar
            WARNING: Progress bar is an experimental feature.
                        This can lead to a considerable performance loss.

        verbose: int, optional
            If verbose >= 2, display all logs
            If verbose == 1, display only initialization logs
            If verbose < 1, display no log
        """
        if progress_bar:
            print("WARNING: Progress bar is an experimental feature. This \
can lead to a considerable performance loss.")

        verbose_store = verbose >= 2

        i_am_in_notebook_lab = is_notebook_lab()

        if hasattr(cls, "proc"):
            cls.proc.kill()

        if verbose >= 1:
            print("New pandarallel memory created - Size:", shm_size_mb, "MB")
            print("Pandarallel will run on", nb_workers, "workers")

        plasma_store_name, cls.proc = start_plasma_store(int(shm_size_mb * 1e6),
                                                         verbose=verbose_store)

        plasma_client = plasma.connect(plasma_store_name)

        args = (plasma_store_name, nb_workers, plasma_client, progress_bar,
                i_am_in_notebook_lab)

        pd.DataFrame.parallel_apply = DataFrame.apply(*args)
        pd.DataFrame.parallel_applymap = DataFrame.applymap(*args)

        pd.Series.parallel_map = Series.map(*args)
        pd.Series.parallel_apply = Series.apply(*args)

        pd.core.window.Rolling.parallel_apply = SeriesRolling.apply(*args)

        dfgb_a = DataFrameGroupBy.apply(*args)
        pd.core.groupby.DataFrameGroupBy.parallel_apply = dfgb_a

        rgb_a = RollingGroupby.apply(*args)
        pd.core.window.RollingGroupby.parallel_apply = rgb_a
