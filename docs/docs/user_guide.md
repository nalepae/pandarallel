## Installation

```bash
pip install pandarallel [--upgrade] [--user]
```

## Usage

First, you have to import `pandarallel`:

```python
from pandarallel import pandarallel
```

Then, you have to initialize it.

```python
pandarallel.initialize()
```

This method takes 5 optional parameters:

- `shm_size_mb`: Deprecated - Do not use.
- `nb_workers`: Number of workers used for parallelization. (int)
  If not set, default to the number of cores available.
- `progress_bar`: Display progress bars if set to `True`. (bool, `False` by default)
- `verbose`: The verbosity level (int, `2` by default)
    * `0` - don't display any logs
    * `1` - display only warning logs
    * `2` - display all logs
- `use_memory_fs`: (bool, `None` by default)
    * If set to None and if memory file system is available, `pandarallel` will use it to
    transfer data between the main process and workers. If memory file system is not
    available, `pandarallel` will default on multiprocessing data transfer (pipe).
    * If set to `True`, `pandarallel` will use memory file system to transfer data between
    the main process and workers and will raise a `SystemError` if memory file system is
    not available.
    * If set to `False`, `pandarallel` will use multiprocessing data transfer (pipe) to
    transfer data between the main process and workers.
- `leave`: (bool, `True` by default): whether to keep the progress bar after running on a notebook or not.
    * If set to `True`, will leave the notebook progress bar after running.
    * If set to `False`, will delete the notebook progress bar after running.

Using memory file system reduces data transfer time between the main process and
workers, especially for big data.

Memory file system is considered as available only if the directory `/dev/shm` exists
and if the user has read and write rights on it.

Basically, memory file system is only available on some Linux distributions (including
Ubuntu).
