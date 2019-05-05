import pandas as pd
import pyarrow.plasma as plasma
import multiprocessing as multiprocessing
from tqdm._tqdm_notebook import tqdm_notebook as tqdm_notebook

from .dataframe import DataFrame
from .series import Series
from .series_rolling import SeriesRolling
from .rolling_groupby import RollingGroupby
from .dataframe_groupby import DataFrameGroupBy

SHM_SIZE_MB = int(2e3) # 2 GB
NB_WORKERS = multiprocessing.cpu_count()
PROGRESS_BAR = False

class pandarallel:
    @classmethod
    def initialize(cls, shm_size_mb=SHM_SIZE_MB, nb_workers=NB_WORKERS,
                   progress_bar=False):
        """
        Initialize Pandarallel shared memory.

        Parameters
        ----------
        shm_size_mb : int, optional
            Size of Pandarallel shared memory

        nb_workers : int, optional
            Number of worker used for parallelisation

        progress_bar : bool, optional
            Display a progress bar
            WARNING: Progress bar is an experimental feature.
                     This can lead to a considerable performance loss.
        """

        print("New pandarallel memory created - Size:", shm_size_mb, "MB")
        print("Pandarallel will run on", nb_workers, "workers")

        if progress_bar:
            print("WARNING: Progress bar is an experimental feature. This \
can lead to a considerable performance loss.")
            tqdm_notebook().pandas()

        cls.__store_ctx = plasma.start_plasma_store(int(shm_size_mb * 1e6))
        plasma_store_name, _ = cls.__store_ctx.__enter__()

        plasma_client = plasma.connect(plasma_store_name)

        args = plasma_store_name, nb_workers, plasma_client

        pd.DataFrame.parallel_apply = DataFrame.apply(*args, progress_bar)
        pd.DataFrame.parallel_applymap = DataFrame.applymap(*args, progress_bar)

        pd.Series.parallel_map = Series.map(*args, progress_bar)
        pd.Series.parallel_apply = Series.apply(*args, progress_bar)

        pd.core.window.Rolling.parallel_apply = SeriesRolling.apply(*args, progress_bar)

        pd.core.groupby.DataFrameGroupBy.parallel_apply = DataFrameGroupBy.apply(*args)

        pd.core.window.RollingGroupby.parallel_apply = RollingGroupby.apply(*args)
