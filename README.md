# pandaral·lel
An easy to use library to speed up computation (by parallelizing on multi CPUs) with [pandas](https://pandas.pydata.org/).

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
 An example of each API is available in [examples.ipynb](https://github.com/nalepae/pandarallel/blob/master/examples.ipynb).
 
 ## Benchmark
 For the `Dataframe.apply` example in [examples.ipynb](https://github.com/nalepae/pandarallel/blob/master/examples.ipynb), here is the comparative benchmark with "standard" `apply` and with `progress_apply` (error bars are too small to be displayed).
 Computer used for this benchmark:
 - OS: Linux Ubuntu 16.04
 - Hardware: Intel Core i7 @ 3.40 GHz (8 cores, **but 4 "truely parallelizable" CPUs**)
 - Number of workers (parallel processes) used: 4
 
 ![Benchmark](https://github.com/nalepae/pandarallel/blob/master/docs/apply_vs_parallel_apply.png)
 
 For this given example, `parallel_apply` runs approximatively 3.7 faster than the "standard" `apply`.
 
 
 ## API
 First, you have to import `pandarallel` (don't forget the double _l_):
 ```python
 import pandarallel
 ```
 ### DataFrame.parallel_apply
 
 If `df` is a pandas DataFrame, and `func` a function to apply to this DataFrame, replace
 ```python
 df.apply(func, axis=1)
 ```
 by
 ```python
 df.parallel_apply(func, axis=1)
 ```
 
  _Note: ``apply`` with ``axis=0`` is not yet implemented._
 
 ### Series.parallel_map
 If `series` is a pandas Series (aka a DataFrame column), and `func` a function to apply to this Series, replace
 ```python
 series.map(func)
 ```
 by
 ```python
 series.parallel_map(func)
 ```
 
 ### DataFrame.groupby.parallel_apply
 If `df` is a pandas DataFrame, `col_name` is the name of a column of this DataFrame and `func` a function to apply to this column, replace
 ```python
 df.groupby(col_name).apply(func)
 ```
 by
 ```python
 df.groupby(col_name).parallel_apply(func)
 ```
