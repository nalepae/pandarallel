# pandaral·lel
An easy to use library to speed up computation (by parallelizing on multi CPUs) with [pandas](https://pandas.pydata.org/).

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
 - [pandas](https://pypi.org/project/pandas/)
 - [pyarrow](https://pypi.org/project/pyarrow/)
 
 
## Warnings
  - The V1.0 of this library is not yet released. API is able to change at any time.
  - Parallelization has a cost (instanciating new processes, transmitting data via shared memory, etc ...), so parallelization is efficiant only if the amount of computation to parallelize is high enough. For very little amount of data, using parallezation not always worth it.
  - Functions applied should NOT be lambda functions.

 ```python
 import pandarallel
 from math import sin
 
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
 For the `Dataframe.apply` example [here](https://github.com/nalepae/pandarallel/blob/master/docs/examples.ipynb), here is the comparative benchmark with "standard" `apply` and with `parallel_apply` (error bars are too small to be displayed).
 Computer used for this benchmark:
 - OS: Linux Ubuntu 16.04
 - Hardware: Intel Core i7 @ 3.40 GHz (4 cores)
 - Number of workers (parallel processes) used: 4
 
 ![Benchmark](https://github.com/nalepae/pandarallel/blob/master/docs/apply_vs_parallel_apply.png)
 
 For this given example, `parallel_apply` runs approximatively 3.7 faster than the "standard" `apply`.
 
 
 ## API
 First, you have to import `pandarallel`:
 ```python
 from pandarallel import pandarallel
 ```
 
 With `df` a pandas DataFrame, `series` a pandas Series, `col_name` the name of a pandas Dataframe column & `func` a function to apply/map,
 
 | Without parallelisation            | With parallelisation                        |
 | ---------------------------------- | ------------------------------------------- |
 | `df.apply(func, axis=1)`           | `df.parallel_apply(func, axis=1)`           |
 | `series.map(func)`                 | `series.parallel_apply(func)`               |
 | `df.groupby(col_name).apply(func)` | `df.groupby(col_name).parallel_apply(func)` |

  _Note: ``apply`` on DataFrane with ``axis=0`` is not yet implemented._
 
