# Pandaral·lel

[![PyPI version fury.io](https://badge.fury.io/py/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)
[![PyPI license](https://img.shields.io/pypi/l/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)
[![PyPI download month](https://img.shields.io/pypi/dm/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)

| Without parallelisation  | ![Without Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_apply.gif)       |
| :----------------------: | -------------------------------------------------------------------------------------------------------- |
| **With parallelisation** | ![With Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_parallel_apply.gif) |

## Installation

`$ pip install pandarallel [--upgrade] [--user]`

## Requirements

On Windows, **Pandaral·lel** will works only if the Python session (`python`, `ipython`, `jupyter notebook`, `jupyter lab`, ...) is executed from [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10).

On Linux & macOS, nothing special has to be done.

## Warning

- Parallelization has a cost (instanciating new processes, sending data via shared memory, etc ...), so parallelization is efficiant only if the amount of calculation to parallelize is high enough. For very little amount of data, using parallezation not always worth it.

## Examples

An example of each API is available [here](https://github.com/nalepae/pandarallel/blob/master/docs/examples.ipynb).

## Benchmark

For [some examples](https://github.com/nalepae/pandarallel/blob/master/docs/examples.ipynb), here is the comparative benchmark with and without using Pandaral·lel.

Computer used for this benchmark:

- OS: Linux Ubuntu 16.04
- Hardware: Intel Core i7 @ 3.40 GHz - 4 cores

![Benchmark](https://github.com/nalepae/pandarallel/blob/3d470139d409fc2cf61bab085298011fefe638c0/docs/standard_vs_parallel_4_cores.png)

For those given examples, parallel operations run approximatively 4x faster than the standard operations (except for `series.map` which runs only 3.2x faster).

## API

First, you have to import `pandarallel`:

```python
from pandarallel import pandarallel
```

Then, you have to initialize it.

```python
pandarallel.initialize()
```

This method takes 5 optional parameters:

- `shm_size_mb`: Deprecated
- `nb_workers`: Number of workers used for parallelisation. (int)
                If not set, all available CPUs will be used.
- `progress_bar`: Display progress bars if set to `True`. (bool)
- `verbose`: The verbosity level (int)
   - 0 - Don't display any logs
   - 1 - Display only warning logs
   - 2 - Display all logs
- `use_memory_fs`: (bool)
   - If set to None and if memory file system is available, Pandarallel will use it to
transfer data between the main process and workers. If memory file system is not
available, Pandarallel will default on multiprocessing data transfer (pipe).
   - If set to True, Pandaralllel will use memory file system to transfer data between
the main process and workers and will raise a SystemError if memory file system is not available.
   - If set to False, Pandarallel will use multiprocessing data transfer (pipe) to
transfer data between the main process and workers.

Using memory file system reduces data transfer time between the main process and
workers, especially for big data.

Memory file system is considered as available only if the directory `/dev/shm` exists
and if the user has read an writerights on it.

Basicaly memory file system is only available on some Linux distributions (including
Ubuntu)

With `df` a pandas DataFrame, `series` a pandas Series, `func` a function to
apply/map, `args1`, `args2` some arguments & `col_name` a column name:

| Without parallelisation                                 | With parallelisation                                             |
| ------------------------------------------------------- | ---------------------------------------------------------------- |
| `df.apply(func)`                                        | `df.parallel_apply(func)`                                        |
| `df.applymap(func)`                                     | `df.parallel_applymap(func)`                                     |
| `df.groupby(args).apply(func)`                          | `df.groupby(args).parallel_apply(func)`                          |
| `df.groupby(args1).col_name.rolling(args2).apply(func)` | `df.groupby(args1).col_name.rolling(args2).parallel_apply(func)` |
| `series.map(func)`                                      | `series.parallel_map(func)`                                      |
| `series.apply(func)`                                    | `series.parallel_apply(func)`                                    |
| `series.rolling(args).apply(func)`                      | `series.rolling(args).parallel_apply(func)`                      |

You will find a complete example [here](https://github.com/nalepae/pandarallel/blob/master/docs/examples.ipynb) for each line of this table.

## Troubleshooting

_I have 8 CPUs but `parallel_apply` speeds up computation only about x4. Why ?_

Actually **Pandarallel** can only speed up computation until about the number of **cores** your computer has. The majority of recent CPUs (like Intel core-i7) uses hyperthreading. For example, a 4 cores hyperthreaded CPU will show 8 CPUs to the Operating System, but will **really** have only 4 physical computation units.

On **Ubuntu**, you can get the number of cores with `$ grep -m 1 'cpu cores' /proc/cpuinfo`.

--------------------------------

_I use **Jupyter lab** and instead of progress bars I have this kind of things:_
`VBox(children=(HBox(children=(IntProgress(value=0, description='0.00%', max=625000), Label(value='0 / 625000')…`

Run this 3 following lines, and you should be able to see the progress bars:
```
$ pip install ipywidgets 
$ jupyter nbextension enable --py widgetsnbextension
$ jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
(You may also have to install `nodejs` if asked)
