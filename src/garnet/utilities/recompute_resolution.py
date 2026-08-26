import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import argparse

import yaml

from mantid.simpleapi import LoadNexus

from garnet.reduction.resolution import ResolutionEllipsoid
from garnet.utilities.peaks import Peaks

# Datasets used to characterize the per-instrument resolution model.
# Each entry re-fits an already-reduced peaks.nxs (no raw-data reduction
# needed) and writes an updated resolution.txt next to it, one directory
# above the peaks/ folder alongside the run's other top-level config
# files. Edit/add entries here when a new characterization run is
# collected -- e.g. after a new resolution.py fitting change, or a new
# reference standard measurement.
CASES = {
    "MANDI": {
        "peaks_nxs": "/SNS/MANDI/IPTS-34720/shared/2026B/peaks/peaks.nxs",
        "r_cut": 0.1,
        "output": "/SNS/MANDI/IPTS-34720/shared/2026B/resolution.txt",
    },
    "CORELLI": {
        "peaks_nxs": "/SNS/CORELLI/IPTS-31429/shared/detcal/peaks/peaks.nxs",
        "r_cut": 0.1,
        "output": "/SNS/CORELLI/IPTS-31429/shared/detcal/resolution.txt",
    },
    "TOPAZ": {
        "peaks_nxs": (
            "/SNS/TOPAZ/IPTS-31856/shared/2026B_garnet_cal/peaks/peaks.nxs"
        ),
        "r_cut": 0.15,
        "output": (
            "/SNS/TOPAZ/IPTS-31856/shared/2026B_garnet_cal/resolution.txt"
        ),
    },
}


def recompute(instrument, peaks_nxs, r_cut, output):
    ws_name = "peaks_{}".format(instrument)
    LoadNexus(Filename=peaks_nxs, OutputWorkspace=ws_name)

    res = ResolutionEllipsoid(ws_name, r_cut=r_cut)
    res.fit()

    if res.model is None:
        print(
            "[{}] fit failed (res.model is None) -- skipping".format(
                instrument
            )
        )
        return

    res.write_resolution_parameters(output)
    print("[{}] wrote {}".format(instrument, output))

    csv_output = os.path.splitext(output)[0] + ".csv"
    res.write_diagnostics_csv(csv_output)
    print("[{}] wrote {}".format(instrument, csv_output))

    plot_output = os.path.splitext(output)[0] + ".pdf"
    res.plot_diagnostics(plot_output)
    print("[{}] wrote {}".format(instrument, plot_output))


def run_peaks_yaml(config_file):
    """
    Run the full peak-finding pipeline (garnet.utilities.peaks.Peaks) from
    a config file -- same as `python -m garnet.utilities.peaks <config>` --
    then recompute the resolution model (with CSV/plot diagnostics) from
    the peaks.nxs it produces, unlike `Peaks.finalize_and_save`'s own
    (isotropic, diagnostics-free) fit.
    """
    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_file))
    name = os.path.splitext(os.path.basename(config_file))[0]

    output_folder = os.path.join(config_dir, name)
    os.makedirs(output_folder, exist_ok=True)

    params["OutputFolder"] = output_folder

    peaks = Peaks(params)
    peaks.run()

    instrument = params.get("Instrument", "TOPAZ")
    peaks_nxs = os.path.join(output_folder, "peaks.nxs")
    r_cut = params.get("PeakRadius", 0.25)
    output = os.path.join(output_folder, "resolution.txt")

    recompute(instrument, peaks_nxs, r_cut, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-instrument resolution-model parameters "
            "(with uncertainties and a correlation matrix) from an "
            "already-saved peaks.nxs, without re-running the full "
            "calibration pipeline."
        )
    )
    parser.add_argument(
        "--instrument",
        choices=sorted(CASES),
        help="Only recompute this instrument (default: all of them)",
    )
    parser.add_argument(
        "--peaks-yaml",
        action="append",
        metavar="CONFIG",
        help=(
            "Run the full peak-finding pipeline from this "
            "garnet.utilities.peaks config file first (re-reduces the raw "
            "runs), then recompute resolution from the resulting "
            "peaks.nxs. May be given multiple times; combines with "
            "--instrument/CASES if both are given."
        ),
    )
    args = parser.parse_args()

    if args.peaks_yaml:
        for config_file in args.peaks_yaml:
            run_peaks_yaml(config_file)

    if not args.peaks_yaml or args.instrument:
        instruments = [args.instrument] if args.instrument else sorted(CASES)

        for instrument in instruments:
            case = CASES[instrument]
            recompute(
                instrument, case["peaks_nxs"], case["r_cut"], case["output"]
            )
