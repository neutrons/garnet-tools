import argparse
import os
import sys
from glob import glob

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

from mantid import config

config["Q.convention"] = "Crystallography"

from garnet.reduction.integration import Integration
from garnet.reduction.plan import ReductionPlan


def find_partial_files(instance):
    pattern = os.path.join(
        instance.get_output_path(), f"{instance.plan['OutputName']}_p*.nxs"
    )
    return sorted(glob(pattern))


def combine(filename):
    plan = ReductionPlan()
    plan.load_plan(filename)

    instance = Integration(plan.plan)
    files = find_partial_files(instance)

    if len(files) == 0:
        raise FileNotFoundError(
            "No partial integration files were found for the requested plan."
        )

    instance.combine(files)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Combine partial integration peak files for a plan."
    )
    parser.add_argument("yaml", help="Path to the reduction YAML file.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    combine(os.path.abspath(args.yaml))
