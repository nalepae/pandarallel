# Pandaral·lel

[![PyPI version fury.io](https://badge.fury.io/py/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)
[![PyPI license](https://img.shields.io/pypi/l/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)
[![PyPI download month](https://img.shields.io/pypi/dm/pandarallel.svg)](https://pypi.python.org/pypi/pandarallel/)

| Without parallelization  | ![Without Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_apply.gif?raw=true)       |
| :----------------------: | ----------------------------------------------------------------------------------------------------------------- |
| **With parallelization** | ![With Pandarallel](https://github.com/nalepae/pandarallel/blob/master/docs/progress_parallel_apply.gif?raw=true) |

**Pandaral.lel** provides a simple way to parallelize your pandas operations on all your
CPUs by changing only one line of code. It also displays progress bars.

## Installation

```bash
pip install pandarallel [--upgrade] [--user]`
```

## Quickstart

```python
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

# df.apply(func)
df.parallel_apply(func)
```

## Usage

Be sure to check out the [documentation](https://nalepae.github.io/pandarallel).

## Examples

An example of each available `pandas` API is available [here](https://github.com/nalepae/pandarallel/blob/master/docs/examples.ipynb).
