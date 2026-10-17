import time

from garnet.reduction.timing import Timer
from garnet.reduction.plan import ReductionPlan
from garnet.reduction.integration import Integration


def read_rows(filename):
    with open(filename) as f:
        return [
            line.rstrip("\n").split("\t")
            for line in f
            if not line.startswith("#")
        ]


def test_laps_and_sections(tmp_path):
    filename = tmp_path / "timing.txt"
    timer = Timer(str(filename), label="_p0")

    timer.set_run(123)
    time.sleep(0.02)
    timer.lap("load")
    with timer.section("inner"):
        time.sleep(0.02)
    time.sleep(0.01)
    timer.lap("work")

    rows = read_rows(filename)
    assert [row[3] for row in rows] == ["load", "  inner", "work"]
    assert all(row[1] == "_p0" and row[2] == "123" for row in rows)

    seconds = {row[3].strip(): float(row[4]) for row in rows}
    assert seconds["load"] >= 0.02
    assert seconds["inner"] >= 0.02
    # a lap includes the sections in it
    assert seconds["work"] >= seconds["inner"] + 0.01

    summary = tmp_path / "summary.txt"
    timer.write_summary(str(summary))
    lines = read_rows(summary)
    fractions = {line[0]: float(line[4]) for line in lines}
    # fractions are of the laps alone
    assert abs(fractions["load"] + fractions["work"] - 1) < 1e-2
    assert lines[0][0] == "work"


def test_repeated_segments_are_summed(tmp_path):
    timer = Timer(str(tmp_path / "timing.txt"))
    for run in [1, 2, 3]:
        timer.set_run(run)
        timer.lap("load")
    count, total, depth = timer.totals["load"]
    assert count == 3
    assert depth == 0


def test_disabled_timer_writes_nothing(tmp_path):
    filename = tmp_path / "timing.txt"
    timer = Timer(str(filename), enabled=False)
    timer.lap("load")
    with timer.section("inner"):
        pass
    timer.write_summary(str(tmp_path / "summary.txt"))
    assert not filename.exists()
    assert not (tmp_path / "summary.txt").exists()


def test_plan_timer_file_per_process(tmp_path):
    rp = ReductionPlan()
    rp.generate_plan("TOPAZ")
    rp.plan["OutputName"] = "test"
    rp.plan["OutputPath"] = str(tmp_path)
    rp.plan["ProcName"] = "_p2"

    integration = Integration(rp.plan)
    integration.timer.lap("load")
    integration.write_timing_summary()

    output = tmp_path / integration.output
    assert (output / "test_p2_timing.txt").exists()
    assert (output / "test_p2_timing_summary.txt").exists()

    rp.plan["Timing"] = False
    quiet = Integration(rp.plan)
    assert not quiet.timer.enabled


def test_add_nests_external_times(tmp_path):
    filename = tmp_path / "timing.txt"
    timer = Timer(str(filename))
    with timer.section("algorithm"):
        timer.add("fits", 1.5, count=10)
    timer.lap("step")

    rows = read_rows(filename)
    assert [row[3] for row in rows] == ["    fits", "  algorithm", "step"]
    count, total, depth = timer.totals["fits"]
    assert (count, total, depth) == (10, 1.5, 2)
