"""
Timing of the steps of a reduction: the time spent in each step is summed
over the runs a process reduces, and the totals of all the processes are
combined into one table with the mean time per run.
"""

import json
import time
from contextlib import contextmanager


class Timer:
    """
    Wall-clock time spent in named steps of a reduction, summed over runs.

    Steps are timed either as laps (the time since the previous lap or run),
    the outermost steps, or as sections (a ``with`` block), which are nested
    in the lap they are part of and may be nested in each other. Times
    measured elsewhere, e.g. reported by an algorithm, can be added too.

    Parameters
    ----------
    enabled : bool, optional
        Record timings. The default is True.

    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.runs = 0
        # step -> [seconds, count, depth, parent], in the order first timed
        self.totals = {}
        self._depth = 0
        self._stack = []
        # steps in the current lap, whose parent is the lap
        self._pending = []
        self._last = time.perf_counter()

    def set_run(self, run=None):
        """
        Start timing a run: the next lap starts now.

        Parameters
        ----------
        run : int or str, optional
            Run being reduced.

        """

        self.runs += 1
        self._last = time.perf_counter()

    def lap(self, name):
        """
        Record the time since the previous lap or run, including any
        sections in it.

        Parameters
        ----------
        name : str
            Step name.

        """

        now = time.perf_counter()
        self._record(name, now - self._last, 0, parent=None)
        for child in self._pending:
            if child in self.totals and self.totals[child][3] is None:
                self.totals[child][3] = name
        self._pending = []
        self._last = now

    @contextmanager
    def section(self, name):
        """
        Record the time spent in a ``with`` block.

        Parameters
        ----------
        name : str
            Step name.

        """

        start = time.perf_counter()
        self._depth += 1
        self._stack.append(name)
        try:
            yield
        finally:
            self._stack.pop()
            parent = self._stack[-1] if self._stack else None
            self._record(
                name, time.perf_counter() - start, self._depth, parent=parent
            )
            if parent is None:
                self._pending.append(name)
            self._depth -= 1

    def add(self, name, seconds, count=1):
        """
        Record a time measured elsewhere (e.g. reported by an algorithm),
        nested in the current lap or section.

        Parameters
        ----------
        name : str
            Step name.
        seconds : float
            Time spent.
        count : int, optional
            How many items (e.g. peaks) the time covers. The default is 1.

        """

        parent = self._stack[-1] if self._stack else None
        self._record(name, seconds, self._depth + 1, count, parent=parent)
        if parent is None:
            self._pending.append(name)

    def _record(self, name, seconds, depth, count=1, parent=None):
        if not self.enabled:
            return

        if name not in self.totals:
            self.totals[name] = [0.0, 0, depth, parent]
        self.totals[name][0] += seconds
        self.totals[name][1] += count

    def to_dict(self):
        return {
            "runs": self.runs,
            "steps": [[name] + values for name, values in self.totals.items()],
        }

    def save(self, filename):
        """
        Save the totals so far, to be combined with those of other processes.

        Parameters
        ----------
        filename : str
            JSON file.

        """

        if self.enabled:
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f)


def combine_timings(timings):
    """
    Sum the totals of several timers.

    Parameters
    ----------
    timings : list of dict
        Totals, as from ``Timer.to_dict``.

    Returns
    -------
    combined : dict
        Summed totals, with the steps in the order first timed.

    """

    runs = 0
    steps = {}
    for timing in timings:
        runs += timing["runs"]
        for name, seconds, count, depth, parent in timing["steps"]:
            if name not in steps:
                steps[name] = [0.0, 0, depth, parent]
            steps[name][0] += seconds
            steps[name][1] += count

    return {
        "runs": runs,
        "steps": [[name] + values for name, values in steps.items()],
    }


def write_timing_table(timing, filename):
    """
    Write a table of the total time of each step over all runs, its mean
    per run, and its fraction of the total time of the outermost steps
    (nested steps are part of the one around them), each step followed by
    the steps in it, indented.

    Parameters
    ----------
    timing : dict
        Totals, as from ``Timer.to_dict`` or ``combine_timings``.
    filename : str
        Text file.

    """

    runs = timing["runs"]
    steps = timing["steps"]
    total = sum(step[1] for step in steps if step[3] == 0)

    # each step followed by the steps in it
    children = {}
    for step in steps:
        children.setdefault(step[4], []).append(step)

    ordered = []

    def visit(parent):
        for step in children.get(parent, []):
            ordered.append(step)
            visit(step[0])

    visit(None)
    listed = {step[0] for step in ordered}
    ordered += [step for step in steps if step[0] not in listed]

    rows = [
        (
            "  " * depth + name,
            "{:.3f}".format(seconds),
            "{:.3f}".format(seconds / runs if runs > 0 else seconds),
            "{:.1f}%".format(100 * seconds / total if total > 0 else 0.0),
            str(count),
        )
        for name, seconds, count, depth, _ in ordered
    ]
    header = ("step", "total (s)", "per run (s)", "fraction", "count")

    # the step column as wide as the longest step, the others right-aligned
    widths = [max(len(row[i]) for row in rows + [header]) for i in range(5)]

    def line(row):
        return "  ".join(
            [row[0].ljust(widths[0])]
            + [value.rjust(width) for value, width in zip(row[1:], widths[1:])]
        )

    with open(filename, "w") as f:
        f.write("{} runs, {:.1f} s in total\n\n".format(runs, total))
        f.write(line(header).rstrip() + "\n")
        f.write("-" * len(line(header)) + "\n")
        for row in rows:
            f.write(line(row).rstrip() + "\n")
