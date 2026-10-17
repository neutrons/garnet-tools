"""
Timing of the segments of a reduction, written to a text file as each one
finishes, so that a long reduction can be followed (e.g. with ``tail -f``)
and the slow steps found afterwards.
"""

import os
import time
import resource
from contextlib import contextmanager


class Timer:
    """
    Wall-clock timing of named segments of a reduction.

    Each segment is appended to a tab-separated text file as it finishes,
    with the run, the elapsed seconds and the peak memory of the process so
    far. Segments are timed either as laps (the time since the previous lap
    or run), the outermost segments, or as sections (a ``with`` block), which
    are nested in the lap they are part of and may be nested in each other.

    Parameters
    ----------
    filename : str or None
        Text file for the timings; None keeps them only in memory.
    enabled : bool, optional
        Record timings. The default is True.
    label : str, optional
        Label for each line, e.g. the process. The default is "".

    """

    def __init__(self, filename=None, enabled=True, label=""):
        self.filename = filename
        self.enabled = enabled
        self.label = label
        self.run = ""
        self.totals = {}
        self._depth = 0
        self._last = time.perf_counter()

        if self.enabled and self.filename is not None:
            directory = os.path.dirname(self.filename)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.filename, "w") as f:
                f.write(
                    "# time\tlabel\trun\tsegment\tseconds\tmax_memory_MB\n"
                )

    def set_run(self, run):
        """
        Start timing a run: later lines are labelled with it, and the next
        lap starts now.

        Parameters
        ----------
        run : int or str
            Run being reduced.

        """

        self.run = run
        self._last = time.perf_counter()

    def lap(self, name):
        """
        Record the time since the previous lap or run, including any sections
        in it.

        Parameters
        ----------
        name : str
            Segment name.

        """

        now = time.perf_counter()
        self._record(name, now - self._last, 0)
        self._last = now

    @contextmanager
    def section(self, name):
        """
        Record the time spent in a ``with`` block.

        Parameters
        ----------
        name : str
            Segment name.

        """

        start = time.perf_counter()
        self._depth += 1
        try:
            yield
        finally:
            self._record(name, time.perf_counter() - start, self._depth)
            self._depth -= 1

    def add(self, name, seconds, count=1):
        """
        Record a time measured elsewhere (e.g. reported by an algorithm),
        nested in the current lap or section.

        Parameters
        ----------
        name : str
            Segment name.
        seconds : float
            Time spent.
        count : int, optional
            How many items the time covers, counted in the summary. The
            default is 1.

        """

        self._record(name, seconds, self._depth + 1, count)

    def _record(self, name, seconds, depth, count=1):
        if not self.enabled:
            return

        total_count, total, _ = self.totals.get(name, (0, 0.0, depth))
        self.totals[name] = (total_count + count, total + seconds, depth)

        if self.filename is None:
            return

        # peak resident memory of this process so far (kB on Linux)
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

        line = "{}\t{}\t{}\t{}{}\t{:.3f}\t{:.0f}\n".format(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            self.label,
            self.run,
            "  " * depth,
            name,
            seconds,
            memory,
        )
        with open(self.filename, "a") as f:
            f.write(line)

    def write_summary(self, filename):
        """
        Write the total time of each segment, summed over runs, slowest
        first, with its fraction of the total time of the laps (sections are
        part of the lap around them).

        Parameters
        ----------
        filename : str
            Text file for the summary.

        """

        if not self.enabled:
            return

        rows = sorted(self.totals.items(), key=lambda item: -item[1][1])
        grand_total = sum(total for _, (_, total, depth) in rows if depth == 0)

        with open(filename, "w") as f:
            f.write("# segment\tcount\ttotal_s\tmean_s\tfraction\n")
            for name, (count, total, _) in rows:
                fraction = total / grand_total if grand_total > 0 else 0
                f.write(
                    "{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(
                        name, count, total, total / count, fraction
                    )
                )
