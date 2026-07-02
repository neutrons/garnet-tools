import os
import sys
import subprocess

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import pprint

from garnet.reduction.plan import ReductionPlan
from garnet.reduction.normalization import Normalization
from garnet.reduction.parametrization import Parametrization

SLICEVIEW = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "sliceview.py"
)


def get_result_file(plan, mode):
    if mode == "normalization":
        instance = Normalization(plan)
    else:
        instance = Parametrization(plan)

    output_file = instance.get_output_file()
    result_file = instance.get_file(output_file, "")
    return instance, result_file


def assemble(instance, plan, mode):
    output_path = instance.get_output_path()
    output_name = plan["OutputName"]

    if mode == "normalization":
        suffix = "_normalization"
    else:
        suffix = "_parametrization"

    # Collect partial files written by parallel workers (_p0, _p1, ...)
    partial_files = sorted(
        os.path.join(output_path, f)
        for f in os.listdir(output_path)
        if f.startswith(output_name)
        and "_p" in f
        and f.endswith(".nxs")
        and "_data" not in f
        and "_norm" not in f
        and "_bkg" not in f
    )

    if not partial_files:
        print("No partial files found to assemble.")
        return None

    print(f"Assembling from {len(partial_files)} partial file(s):")
    for f in partial_files:
        print(" ", f)

    result_file = instance.get_file(instance.get_output_file(), "")
    instance.combine(partial_files)
    return result_file


def view(result_file):
    try:
        process = subprocess.Popen(
            ["python", SLICEVIEW, result_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate()
        if process.returncode != 0:
            raise subprocess.SubprocessError(err.decode().strip())
        print("Opened:", result_file)
    except (FileNotFoundError, subprocess.SubprocessError):
        subprocess.Popen(["python", SLICEVIEW, result_file])


if __name__ == "__main__":
    filename = sys.argv[1]

    rp = ReductionPlan()
    rp.load_plan(filename)
    params = rp.plan

    pprint.pp(params)

    # Detect which mode is present
    has_norm = "Normalization" in params and params["Normalization"]
    has_param = "Parametrization" in params and params["Parametrization"]

    if not has_norm and not has_param:
        print("No Normalization or Parametrization section found in plan.")
        sys.exit(1)

    # Prefer parametrization if both present; user can override with second arg
    if len(sys.argv) > 2:
        mode = sys.argv[2].lower()
        if mode not in ("normalization", "parametrization"):
            print(
                "Usage: view.py <config.yaml> [normalization|parametrization]"
            )
            sys.exit(1)
    elif has_param:
        mode = "parametrization"
    else:
        mode = "normalization"

    print(f"Mode: {mode}")

    instance, result_file = get_result_file(params, mode)

    print(f"Expected result file: {result_file}")

    if os.path.exists(result_file):
        print("File found — opening SliceViewer.")
        view(result_file)
    else:
        print("File not found — attempting to assemble from partial files.")
        result_file = assemble(instance, params, mode)
        if result_file and os.path.exists(result_file):
            view(result_file)
        else:
            print("Assembly failed or result file still missing.")
            sys.exit(1)
