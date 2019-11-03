import shutil
import sys

MINIMUM_TERMINAL_WIDTH = 72


def is_notebook_lab():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Jupyter notebook/lab or qtconsole
            return True
        elif shell == "TerminalInteractiveShell":
            # Terminal running IPython
            return False
        else:
            # Other type (?)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False


def get_progress_bars(maxs):
    return (
        ProgressBarsNotebookLab(maxs)
        if is_notebook_lab()
        else ProgressBarsConsole(maxs)
    )


class ProgressBarsConsole:
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
            columns = int(os.popen("stty size", "r").read().split()[1])
            return max(MINIMUM_TERMINAL_WIDTH, columns - 1)
        except:
            return MINIMUM_TERMINAL_WIDTH

    def __remove_displayed_lines(self):
        if len(self.__bars) >= 1:
            sys.stdout.write("\b" * len(self.__lines[-1]))

        if len(self.__bars) >= 2:
            sys.stdout.write("\033M" * (len(self.__lines) - 1))

        self.__lines = []

    def __update_line(self, done, total):
        percent = done / total
        bar = (":" * int(percent * 40)).ljust(40, " ")
        percent = round(percent * 100, 2)
        format = " {percent:6.2f}% {bar:s} | {done:8d} / {total:8d} |"
        ret = format.format(percent=percent, bar=bar, done=done, total=total)
        return ret[: self.__width].ljust(self.__width, " ")

    def __update_lines(self):
        self.__lines = [self.__update_line(value, max) for value, max in self.__bars]

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


class ProgressBarsNotebookLab:
    def __init__(self, maxs):
        """Initialization.
        Positional argument:
        maxs - List containing the max value of each progress bar
        """
        from IPython.display import display
        from ipywidgets import HBox, VBox, IntProgress, Label

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

    def update(self, values):
        """Update a bar value.
        Positional arguments:
        values - The new values of each bar
        """
        for index, value in enumerate(values):
            bar, label = self.__bars[index].children

            bar.value = value
            bar.description = "{:.2f}%".format(value / bar.max * 100)

            if value >= bar.max:
                bar.bar_style = "success"

            label.value = "{} / {}".format(value, bar.max)

    def set_error(self, index):
        """Set a bar on error"""
        bar, _ = self.__bars[index].children
        bar.bar_style = "danger"
