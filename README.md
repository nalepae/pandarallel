# Pandaral·lel

 | Without parallelisation  | ![Without Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_apply.gif)       |
 | :----------------------: | -------------------------------------------------------------------------------------------------------- |
 | **With parallelisation** | ![With Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_parallel_apply.gif) |

<table>
<tr>
  <td>Latest Release</td>
  <td>
    <a href="https://pypi.org/project/pandarallel/">
    <img src="https://img.shields.io/pypi/v/pandarallel.svg" alt="latest release" />
    </a>
  </td>
</tr>
<tr>
  <td>License</td>
  <td>
    <a href="https://github.com/nalepae/pandarallel/blob/master/LICENSE">
    <img src="https://img.shields.io/pypi/l/pandarallel.svg" alt="license" />
    </a>
  </td>
</tr>
</table>

## Installation
`$ pip install pandarallel [--user]`


## Requirements
  - Linux or macOS (Windows is not supported at the moment)


## Warnings
  - Parallelization has a cost (instanciating new processes, sending data via shared memory, etc ...), so parallelization is efficiant only if the amount of calculation to parallelize is high enough. For very little amount of data, using parallezation not always worth it.
  - Functions applied should NOT be lambda functions.

 ```python
 from pandarallel import pandarallel
 from math import sin

 pandarallel.initialize()

 # FORBIDDEN
 df.parallel_apply(lambda x: sin(x**2), axis=1)

 # ALLOWED
 def func(x):
     return sin(x**2)

 df.parallel_apply(func, axis=1)
  ```

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
 This method takes 3 optional parameters:
 - `shm_size_mb`: The size of the Pandarallel shared memory in MB. If the
 default one is too small, it is possible to set a larger one. By default,
 it is set to 2 GB. (int)
 - `nb_workers`: The number of workers. By default, it is set to the number
 of cores your operating system sees. (int)
 - `progress_bar`: Put it to `True` to display a progress bar.
 
 **WARNING**: Progress bar is an experimental feature. This can lead to a
 considerable performance loss.
 Not available for `DataFrameGroupy.parallel_apply`.

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
*I have 8 CPUs but `parallel_apply` speeds up computation only about x4. Why ?*

Actually **Pandarallel** can only speed up computation until about the number of **cores** your computer has. The majority of recent CPUs (like Intel core-i7) uses hyperthreading. For example, a 4 cores hyperthreaded CPU will show 8 CPUs to the Operating System, but will **really** have only 4 physical computation units.

On **Ubuntu**, you can get the number of cores with `$ grep -m 1 'cpu cores' /proc/cpuinfo`.

*When I run `from pandarallel import pandarallel`, I get `ModuleNotFoundError: No module named 'pyarrow._plasma`. Why?*

Are you using Windows? **Pandarallel** relies on the **Pyarrow Plasma** shared memory to work. Currently, **Pyarrow Plasma** works only on Linux & macOS (Windows in not supported). So sorry, but for now **Pandarallel** is supported only on Linux & macOS ...
