import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import argparse

from mantid.simpleapi import LoadNexus

from garnet.reduction.resolution import ResolutionEllipsoid

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
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else sorted(CASES)

    for instrument in instruments:
        case = CASES[instrument]
        recompute(instrument, case["peaks_nxs"], case["r_cut"], case["output"])
