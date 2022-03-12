import multiprocessing
import os
import shutil
import sys
from abc import ABC, abstractmethod
from enum import Enum
from itertools import count
from time import time
from typing import Callable, List, Union

from .utils import WorkerStatus

MINIMUM_TERMINAL_WIDTH = 72


class ProgressBarsType(int, Enum):
    No = 0
    InUserDefinedFunction = 1
    InUserDefinedFunctionMultiplyByNumberOfColumns = 2
    InWorkFunction = 3


class ProgressBars(ABC):
    @abstractmethod
    def __init__(self, maxs: List[int], show: bool) -> None:
        ...

    @abstractmethod
    def update(self, values: List[int]) -> None:
        ...

    def set_error(self, index: int) -> None:
        pass


class ProgressState:
    def __init__(self, chunk_size: int) -> None:
        self.last_put_iteration = 0
        self.next_put_iteration = max(chunk_size // 100, 1)
        self.last_put_time = time()


def is_notebook_lab() -> bool:
    try:
        shell: str = get_ipython().__class__.__name__  # type: ignore

        # Shell: Google Colab
        # TerminalInteractiveShell: Terminal running IPython
        # ZMQInteractiveShell: Jupyter notebook/lab or qtconsole
        return shell in {"Shell", "ZMQInteractiveShell"}
    except NameError:
        # Probably standard Python interpreter
        return False


class ProgressBarsConsole(ProgressBars):
    def __init__(self, maxs: List[int], show: bool) -> None:
        self.__show = show
        self.__bars = [[0, max] for max in maxs]
        self.__width = self.__get_width()

        self.__lines = self.__update_lines()

        if show:
            sys.stdout.write("\n".join(self.__lines))
            sys.stdout.flush()

    def __get_width(self) -> int:
        try:
            columns = shutil.get_terminal_size().columns
            return max(MINIMUM_TERMINAL_WIDTH, columns - 1)
        except AttributeError:
            # Python 2
            pass

        try:
            columns = int(os.popen("stty size", "r").read().split()[1])
            return max(MINIMUM_TERMINAL_WIDTH, columns - 1)
        except:
            return MINIMUM_TERMINAL_WIDTH

    def __remove_displayed_lines(self) -> None:
        if len(self.__bars) >= 1:
            sys.stdout.write("\b" * len(self.__lines[-1]))

        if len(self.__bars) >= 2:
            sys.stdout.write("\033M" * (len(self.__lines) - 1))

        self.__lines = []

    def __update_line(self, done: int, total: int) -> str:
        percent = done / total
        bar = (":" * int(percent * 40)).ljust(40, " ")
        percent = round(percent * 100, 2)
        format = " {percent:6.2f}% {bar:s} | {done:8d} / {total:8d} |"
        ret = format.format(percent=percent, bar=bar, done=done, total=total)
        return ret[: self.__width].ljust(self.__width, " ")

    def __update_lines(self) -> List[str]:
        return [self.__update_line(value, max) for value, max in self.__bars]

    def update(self, values: List[int]) -> None:
        """Update a bar value.
        Positional arguments:
        values - The new values of each bar
        """
        if not self.__show:
            return

        for index, value in enumerate(values):
            self.__bars[index][0] = value

        self.__remove_displayed_lines()
        self.__lines = self.__update_lines()

        sys.stdout.write("\n".join(self.__lines))
        sys.stdout.flush()


class ProgressBarsNotebookLab(ProgressBars):
    def __init__(self, maxs: List[int], show: bool) -> None:
        """Initialization.
        Positional argument:
        maxs - List containing the max value of each progress bar
        """
        self.__show = show

        if not show:
            return

        from IPython.display import display
        from ipywidgets import HBox, IntProgress, Label, VBox

        self.__bars = [
            HBox(
                [
                    IntProgress(0, 0, max, description="{:.2f}%".format(0)),
                    Label("{} / {}".format(0, max)),
                ]
            )
            for max in maxs
        ]

        display(VBox(self.__bars))

    def update(self, values: List[int]) -> None:
        """Update a bar value.
        Positional arguments:
        values - The new values of each bar
        """
        if not self.__show:
            return

        for index, value in enumerate(values):
            bar, label = self.__bars[index].children

            bar.value = value
            bar.description = "{:.2f}%".format(value / bar.max * 100)

            if value >= bar.max:
                bar.bar_style = "success"

            label.value = "{} / {}".format(value, bar.max)

    def set_error(self, index: int) -> None:
        """Set a bar on error"""
        if not self.__show:
            return

        bar, _ = self.__bars[index].children
        bar.bar_style = "danger"


def get_progress_bars(
    maxs: List[int], show
) -> Union[ProgressBarsNotebookLab, ProgressBarsConsole]:
    return (
        ProgressBarsNotebookLab(maxs, show)
        if is_notebook_lab()
        else ProgressBarsConsole(maxs, show)
    )


def progress_wrapper(
    user_defined_function: Callable,
    master_workers_queue: multiprocessing.Queue,
    index: int,
    chunk_size: int,
) -> Callable:
    """Wrap the function to apply in a function which monitor the part of work already
    done.
    """
    counter = count()
    state = ProgressState(chunk_size)

    def closure(*user_defined_function_args, **user_defined_functions_kwargs):
        iteration = next(counter)

        if iteration == state.next_put_iteration:
            time_now = time()
            master_workers_queue.put_nowait((index, WorkerStatus.Running, iteration))

            delta_t = time_now - state.last_put_time
            delta_i = iteration - state.last_put_iteration

            state.next_put_iteration += max(int((delta_i / delta_t) * 0.25), 1)
            state.last_put_iteration = iteration
            state.last_put_time = time_now

        return user_defined_function(
            *user_defined_function_args, **user_defined_functions_kwargs
        )

    return closure
