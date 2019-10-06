import os
import shutil
import sys

from ipywidgets import HBox, VBox, IntProgress, Label
from IPython.display import display

import itertools as _itertools

try:
    # Pyarrow version > 0.14
    from pyarrow.plasma import PlasmaStoreFull as _PlasmaStoreFull
except ImportError:
    # Pyarrow version <= 0.14
    from pyarrow.lib import PlasmaStoreFull as _PlasmaStoreFull

MINIMUM_TERMINAL_WIDTH = 72


def chunk(nb_item, nb_chunks, start_offset=0):
    """
    Return `nb_chunks` slices of approximatively `nb_item / nb_chunks` each.

    Parameters
    ----------
    nb_item : int
        Total number of items

    nb_chunks : int
        Number of chunks to return

    start_offset : int
        Shift start of slice by this amount

    Returns
    -------
    A list of slices


    Examples
    --------
    >>> chunks = _pandarallel._chunk(103, 4)
    >>> chunks
    [slice(0, 26, None), slice(26, 52, None), slice(52, 78, None),
     slice(78, 103, None)]
    """
    if nb_item <= nb_chunks:
        return [
            slice(max(0, idx - start_offset), idx + 1)
            for idx in range(nb_item)
        ]

    quotient = nb_item // nb_chunks
    remainder = nb_item % nb_chunks

    quotients = [quotient] * nb_chunks
    remainders = [1] * remainder + [0] * (nb_chunks - remainder)

    nb_elems_per_chunk = [
        quotient + remainder for quotient, remainder
        in zip(quotients, remainders)
    ]

    accumulated = list(_itertools.accumulate(nb_elems_per_chunk))
    shifted_accumulated = accumulated.copy()
    shifted_accumulated.insert(0, 0)
    shifted_accumulated.pop()

    return [
        slice(max(0, begin - start_offset), end) for begin, end
        in zip(shifted_accumulated, accumulated)
    ]


def parallel(client):
    def decorator(func):
        def wrapper(*args, **kwargs):
            """Please see the docstring of this method without `parallel`"""
            try:
                return func(*args, **kwargs)

            except _PlasmaStoreFull:
                msg = "The pandarallel shared memory is too small to allow \
parallel computation. \
Just after pandarallel import, please write: \
pandarallel.initialize(<size of memory in MB>), and retry."

                raise Exception(msg)

            finally:
                client.delete(client.list().keys())

        return wrapper
    return decorator


class ProgressBarsConsole():
    def __init__(self, maxs):
        self.__bars = [[0, max] for max in maxs]
        self.__width = self.__get_width()

        self.__update_lines()

        sys.stdout.write("\n".join(self.__lines))
        sys.stdout.flush()

    def __get_width(self):
        try:
            columns = shutil.get_terminal_size().columns
            return max(MINIMUM_TERMINAL_WIDTH, columns - 1)
        except AttributeError:
            # Python 2
            pass

        try:
            columns = int(os.popen('stty size', 'r').read().split()[1])
            return max(MINIMUM_TERMINAL_WIDTH, columns - 1)
        except:
            return MINIMUM_TERMINAL_WIDTH

    def __remove_displayed_lines(self):
        if len(self.__bars) >= 1:
            sys.stdout.write('\b'*len(self.__lines[-1]))

        if len(self.__bars) >= 2:
            sys.stdout.write('\033M'*(len(self.__lines) - 1))

        self.__lines = []

    def __update_line(self, done, total):
        percent = done / total
        bar = (':' * int(percent * 40)).ljust(40, " ")
        percent = round(percent * 100, 2)
        format = ' {percent:6.2f}% {bar:s} | {done:8d} / {total:8d} |'
        ret = format.format(percent=percent, bar=bar, done=done, total=total)
        return ret[:self.__width].ljust(self.__width, ' ')

    def __update_lines(self):
        self.__lines = [
            self.__update_line(value, max)
            for value, max in self.__bars
        ]

    def update(self, values):
        """Update a bar value.

        Positional arguments:
        values - The new values of each bar
        """
        for index, value in enumerate(values):
            self.__bars[index][0] = value

        self.__remove_displayed_lines()
        self.__update_lines()

        sys.stdout.write("\n".join(self.__lines))
        sys.stdout.flush()


class ProgressBarsNotebookLab():
    def __init__(self, maxs):
        """Initialization.

        Positional argument:
        maxs - List containing the max value of each progress bar
        """
        self.__bars = [
            HBox([
                IntProgress(0, 0, max, description='{:.2f}%'.format(0)),
                Label("{} / {}".format(0, max))
            ])
            for max in maxs
        ]

        display(VBox(self.__bars))

    def update(self, values):
        """Update a bar value.

        Positional arguments:
        values - The new values of each bar
        """
        for index, value in enumerate(values):
            bar, label = self.__bars[index].children

            bar.value = value
            bar.description = '{:.2f}%'.format(value/bar.max * 100)
            bar.bar_style = 'success' if value >= bar.max else ''

            label.value = "{} / {}".format(value, bar.max)
