import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import argparse

import numpy as np
import yaml

from mantid.simpleapi import LoadNexus, mtd

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


def _percentile_sig_noise_cut(instrument, ws_name, percentile, label):
    ws = mtd[ws_name]
    res_tmp = ResolutionEllipsoid(ws_name)
    sig_noise = []
    for i in range(ws.getNumberPeaks()):
        radii, _, _ = res_tmp._get_peak_shape(ws, i)
        if np.all(np.isfinite(radii)):
            sig_noise.append(ws.getPeak(i).getIntensityOverSigma())
    sig_noise_cut = np.nanpercentile(sig_noise, percentile)
    print(
        "[{}] {} sig_noise {}th percentile cutoff: {:.4f} ({} of {} "
        "peaks retained)".format(
            instrument,
            label,
            percentile,
            sig_noise_cut,
            int(np.sum(np.array(sig_noise) > sig_noise_cut)),
            len(sig_noise),
        )
    )
    return sig_noise_cut


def recompute(
    instrument,
    peaks_nxs,
    r_cut,
    output,
    sig_noise_cut=5.0,
    sig_noise_percentile=None,
    weighting="sn_over_q2",
    sn_correction_percentile=None,
):
    """
    Parameters
    ----------
    sig_noise_cut : float, optional
        Flat signal/noise cutoff for ResolutionEllipsoid. Default 5.0.
        Ignored if `sig_noise_percentile` is given.
    sig_noise_percentile : float or None, optional
        If given, use peaks above this percentile of the observed
        signal/noise distribution (among peaks with a valid shape) as
        ResolutionEllipsoid's sig_noise_cut instead of `sig_noise_cut`
        -- e.g. 80 keeps only the strongest 20% of peaks. None
        (default) uses `sig_noise_cut` directly.
    weighting : str, optional
        Passed through to ResolutionEllipsoid.fit() -- "sn_over_q2"
        (default), "sn", "sn2", or "none".
    sn_correction_percentile : float or None, optional
        If given, run a second stage after the main (strong-peak)
        calibration fit: hold that fit's instrumental model fixed and
        fit ResolutionEllipsoid.fit_sn_correction's (a, s0)
        bias-correction parameters against peaks above this
        (typically lower) percentile of the S/N distribution instead
        -- e.g. 50 to use the upper half. None (default) skips this
        stage entirely.

    """
    ws_name = "peaks_{}".format(instrument)
    LoadNexus(Filename=peaks_nxs, OutputWorkspace=ws_name)

    if sig_noise_percentile is not None:
        sig_noise_cut = _percentile_sig_noise_cut(
            instrument, ws_name, sig_noise_percentile, "calibration"
        )

    res = ResolutionEllipsoid(
        ws_name, r_cut=float("inf"), sig_noise_cut=sig_noise_cut
    )
    res.fit(weighting=weighting)

    if res.model is None:
        print(
            "[{}] fit failed (res.model is None) -- skipping".format(
                instrument
            )
        )
        return

    if sn_correction_percentile is not None:
        wide_cut = _percentile_sig_noise_cut(
            instrument, ws_name, sn_correction_percentile, "sn-correction"
        )
        res_wide = ResolutionEllipsoid(
            ws_name, r_cut=float("inf"), sig_noise_cut=wide_cut
        )
        result = res_wide.fit_sn_correction(res.model, weighting=weighting)
        if result is None:
            print(
                "[{}] sn-correction fit failed -- skipping".format(instrument)
            )
        else:
            a, s0, sn_residual_norm, lsq_result, sn_used = result
            cov = res_wide._parameter_covariance(lsq_result)
            if cov is not None:
                a_stderr, s0_stderr = np.sqrt(np.clip(np.diag(cov), 0, None))
            else:
                a_stderr, s0_stderr = None, None
            res.model["sn_bias_a"] = a
            res.model["sn_bias_a_stderr"] = a_stderr
            res.model["sn_bias_s0"] = s0
            res.model["sn_bias_s0_stderr"] = s0_stderr
            print(
                "[{}] sn-correction: a={:.4f} +/- {}, s0={:.4f} +/- {}, "
                "residual_norm={:.4e}, {} peaks used".format(
                    instrument,
                    a,
                    a_stderr,
                    s0,
                    s0_stderr,
                    sn_residual_norm,
                    len(sn_used),
                )
            )

    res.write_resolution_parameters(output)
    print("[{}] wrote {}".format(instrument, output))

    csv_output = os.path.splitext(output)[0] + ".csv"
    res.write_diagnostics_csv(csv_output)
    print("[{}] wrote {}".format(instrument, csv_output))

    plot_output = os.path.splitext(output)[0] + ".pdf"
    res.plot_diagnostics(plot_output)
    print("[{}] wrote {}".format(instrument, plot_output))


def run_peaks_yaml(
    config_file,
    sig_noise_cut=5.0,
    sig_noise_percentile=None,
    weighting="sn_over_q2",
    sn_correction_percentile=None,
    n_proc=10,
):
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
    peaks.run(n_proc=n_proc)

    instrument = params.get("Instrument", "TOPAZ")
    peaks_nxs = os.path.join(output_folder, "peaks.nxs")
    r_cut = params.get("PeakRadius", 0.25)
    output = os.path.join(output_folder, "resolution.txt")

    recompute(
        instrument,
        peaks_nxs,
        r_cut,
        output,
        sig_noise_cut=sig_noise_cut,
        sig_noise_percentile=sig_noise_percentile,
        weighting=weighting,
        sn_correction_percentile=sn_correction_percentile,
    )


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
    parser.add_argument(
        "--sig-noise-cut",
        type=float,
        default=5.0,
        metavar="SN",
        help=(
            "Flat signal/noise cutoff for the fit. Default: 5.0. "
            "Ignored if --sig-noise-percentile is given."
        ),
    )
    parser.add_argument(
        "--sig-noise-percentile",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Restrict the fit to peaks above this percentile of the "
            "observed signal/noise distribution instead of "
            "--sig-noise-cut -- e.g. 80 keeps only the strongest 20%% "
            "of peaks. Default: no percentile cutoff."
        ),
    )
    parser.add_argument(
        "--weighting",
        choices=("sn_over_q2", "sn", "sn2", "none"),
        default="sn_over_q2",
        help="Row weighting passed to ResolutionEllipsoid.fit().",
    )
    parser.add_argument(
        "--sn-correction-percentile",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Run a second stage that holds the main calibration fit's "
            "instrumental model fixed and fits "
            "ResolutionEllipsoid.fit_sn_correction's S/N bias-correction "
            "(a, s0) against peaks above this (typically lower) "
            "percentile of the S/N distribution -- e.g. 50 to use the "
            "upper half. Default: no sn-correction stage."
        ),
    )
    parser.add_argument(
        "--n-proc",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of processes for the --peaks-yaml raw-data reduction "
            "pipeline (Peaks.run). Default: 10."
        ),
    )
    args = parser.parse_args()

    if args.peaks_yaml:
        for config_file in args.peaks_yaml:
            run_peaks_yaml(
                config_file,
                sig_noise_cut=args.sig_noise_cut,
                sig_noise_percentile=args.sig_noise_percentile,
                weighting=args.weighting,
                sn_correction_percentile=args.sn_correction_percentile,
                n_proc=args.n_proc,
            )

    if not args.peaks_yaml or args.instrument:
        instruments = [args.instrument] if args.instrument else sorted(CASES)

        for instrument in instruments:
            case = CASES[instrument]
            recompute(
                instrument,
                case["peaks_nxs"],
                case["r_cut"],
                case["output"],
                sig_noise_cut=args.sig_noise_cut,
                sig_noise_percentile=args.sig_noise_percentile,
                weighting=args.weighting,
                sn_correction_percentile=args.sn_correction_percentile,
            )
