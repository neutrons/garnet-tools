import os
import time

from garnet.reduction.timing import (
    Timer,
    combine_timings,
    write_timing_table,
)
from garnet.reduction.plan import ReductionPlan
from garnet.reduction.integration import Integration


def read_table(filename):
    """The rows of a timing table, as [step, total, per run, fraction, count]."""
    with open(filename) as f:
        lines = f.read().splitlines()[4:]
    rows = []
    for line in lines:
        fields = line.split()
        name_end = len(line) - len(line.lstrip()) + len(" ".join(fields[:-4]))
        rows.append([line[:name_end]] + fields[-4:])
    return rows


def test_laps_and_sections_are_summed_over_runs():
    timer = Timer()
    for run in [1, 2]:
        timer.set_run(run)
        time.sleep(0.01)
        timer.lap("load")
        with timer.section("inner"):
            time.sleep(0.01)
        timer.lap("work")

    assert timer.runs == 2
    assert list(timer.totals) == ["load", "inner", "work"]
    load, inner, work = (timer.totals[n] for n in ["load", "inner", "work"])
    assert load[1] == 2 and load[2] == 0
    assert inner[2] == 1
    assert load[0] >= 0.02
    # a lap includes the sections in it
    assert work[0] >= inner[0]


def test_add_nests_external_times():
    timer = Timer()
    timer.set_run(1)
    with timer.section("algorithm"):
        timer.add("fits", 1.5, count=10)
    timer.lap("step")
    assert timer.totals["fits"] == [1.5, 10, 2, "algorithm"]
    assert timer.totals["algorithm"][3] == "step"


def test_combine_and_table(tmp_path):
    a = {"runs": 2, "steps": [["load", 4.0, 2, 0, None]]}
    b = {
        "runs": 3,
        "steps": [
            ["load", 6.0, 3, 0, None],
            ["fits", 2.0, 30, 1, "save"],
            ["save", 10.0, 3, 0, None],
        ],
    }
    combined = combine_timings([a, b])
    assert combined["runs"] == 5
    assert combined["steps"] == [
        ["load", 10.0, 5, 0, None],
        ["fits", 2.0, 30, 1, "save"],
        ["save", 10.0, 3, 0, None],
    ]

    filename = tmp_path / "timing.txt"
    write_timing_table(combined, str(filename))
    rows = read_table(filename)
    # each step followed by the steps in it
    assert [row[0] for row in rows] == ["load", "save", "  fits"]
    # total, mean per run, and fraction of the outermost steps
    assert rows[0][1:4] == ["10.000", "2.000", "50.0%"]
    assert rows[2][1:5] == ["2.000", "0.400", "10.0%", "30"]
    with open(filename) as f:
        lines = f.read().splitlines()
    assert lines[0] == "5 runs, 20.0 s in total"
    # the columns line up
    assert len({len(line) for line in lines[2:4]}) == 1
    starts = {line.index("%") for line in lines[4:]}
    assert len(starts) == 1


def test_processes_combined_into_one_file(tmp_path):
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["OutputPath"] = str(tmp_path)

    for proc in [0, 1]:
        plan = dict(rp.plan)
        plan["ProcName"] = "_p{}".format(proc)
        worker = Integration(plan)
        worker.timer.set_run(proc)
        worker.timer.lap("load")
        worker.save_timing()

    main = Integration(rp.plan)
    main.write_timing()

    output = tmp_path / main.output
    assert (output / "test_timing.txt").exists()
    rows = read_table(output / "test_timing.txt")
    assert rows[0][0] == "load"
    with open(output / "test_timing.txt") as f:
        assert f.readline().startswith("2 runs")
    # no per-process files are left
    assert not any("_p" in name for name in os.listdir(output))
    assert not any(
        name.startswith("timing")
        for name in os.listdir(main.get_diagnostic_path())
    )
