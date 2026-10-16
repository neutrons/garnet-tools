"""
Compare IntegratePeaksShapeMD's flux/solid-angle normalization with the
garnet MDNorm pipeline on real data, for the same peaks and shapes.

For each peak, the reference bins the events with MDNorm on a fine local
Q_sample grid, and from the voxels whose centers are inside the peak
ellipsoid and inside its background shell computes

- the garnet intensity: the summed normalized signal times the voxel volume
  over the peak, minus the shell's scaled to the peak volume, and
- the binned ratio of sums: V (N_peak / Norm_peak - N_shell / Norm_shell)
  from the binned counts and normalization, which isolates the comparison of
  the normalization from how the binned signal is summed.

The new intensity is IntegratePeaksShapeMD with SolidAngleWorkspace and
FluxWorkspace (count mode, and optionally profile mode) on the same events.

The garnet intensity needs voxels larger than the spacing between detector
trajectories in Q: voxels crossed by no trajectory have no signal, and those
crossed by a short piece of one have a noisy one. The fraction of the peak's
voxels without normalization is reported for each peak, and ellipsoids
thinner than a voxel are skipped.

Use the solid angle and flux from vanadium, as garnet does: ones made from
the sample's own events contain its Bragg peaks, which the exposure model
then mistakes for exposure. Profile mode reads the peak radii as standard
deviations, so for shapes whose radii are boundaries (e.g. 3 sigma, as from
IntegrateEllipsoids) pass --profile-radius-scale.

Reports the distributions of the ratios new / reference, their outliers
(with their bank and coverage flags), and the wall time of both methods, and
writes a CSV with one row per peak. Not part of any test suite.

Example (TOPAZ or MANDI, files from a garnet reduction):

    python validate_integrate_peaks_shape_norm.py \\
        --events TOPAZ_12345.nxs.h5 --ub TOPAZ_12345.mat \\
        --solid-angle solid_angle.nxs --flux flux.nxs \\
        --peaks TOPAZ_12345_peaks.nxs --k-min 1.8 --k-max 12.5 \\
        --region-radius 0.25 --step 0.02 --profile --output comparison.csv
"""

import argparse
import csv
import json
import time

import numpy as np
from mantid.dataobjects import PeakShapeEllipsoid
from mantid.kernel import SpecialCoordinateSystem, V3D
from mantid.simpleapi import (
    ConvertToMD,
    ConvertUnits,
    CropWorkspaceForMDNorm,
    IntegratePeaksShapeMD,
    Load,
    LoadIsawPeaks,
    LoadIsawUB,
    LoadNexus,
    MaskDetectors,
    MDNorm,
    mtd,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--events", required=True, help="raw event file of one run"
    )
    parser.add_argument(
        "--solid-angle",
        required=True,
        help="solid angle (integrated vanadium) workspace file",
    )
    parser.add_argument(
        "--flux", required=True, help="integrated flux workspace file"
    )
    parser.add_argument(
        "--peaks",
        required=True,
        help="peaks file whose peaks have ellipsoidal shapes",
    )
    parser.add_argument("--ub", help="ISAW UB file, if the events have no UB")
    parser.add_argument(
        "--k-min",
        type=float,
        required=True,
        help="minimum momentum, as in garnet",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        required=True,
        help="maximum momentum, as in garnet",
    )
    parser.add_argument(
        "--region-radius",
        type=float,
        default=0.25,
        help="IntegratePeaksShapeMD RegionRadius",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="MDNorm voxel size for the reference, 1/A",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also run IntegratePeaksShapeMD in profile mode",
    )
    parser.add_argument(
        "--profile-radius-scale",
        type=float,
        default=1.0,
        help="profile mode reads the peak radii as standard deviations; scale them by this for the profile fit, "
        "e.g. 1/3 for shapes whose radii are 3 sigma boundaries",
    )
    parser.add_argument(
        "--max-peaks", type=int, help="only compare the first this many peaks"
    )
    parser.add_argument(
        "--outlier-mads",
        type=float,
        default=5.0,
        help="outliers differ from the median by this many MADs",
    )
    parser.add_argument(
        "--output",
        default="comparison.csv",
        help="CSV file with one row per peak",
    )
    parser.add_argument(
        "--plot", help="optional PNG file for a histogram of the ratios"
    )
    return parser.parse_args()


def load_inputs(args):
    """Load and crop the events as garnet does, and load the normalization and peaks."""
    Load(Filename=args.events, OutputWorkspace="events")
    if args.ub:
        LoadIsawUB(InputWorkspace="events", Filename=args.ub)
    LoadNexus(Filename=args.solid_angle, OutputWorkspace="solid_angle")
    LoadNexus(Filename=args.flux, OutputWorkspace="flux")
    # leave out the detectors without vanadium, as MDNorm needs
    MaskDetectors(Workspace="events", MaskedWorkspace="solid_angle")
    ConvertUnits(
        InputWorkspace="events", OutputWorkspace="events", Target="Momentum"
    )
    CropWorkspaceForMDNorm(
        InputWorkspace="events",
        XMin=args.k_min,
        XMax=args.k_max,
        OutputWorkspace="events",
    )

    if args.peaks.endswith(".peaks") or args.peaks.endswith(".integrate"):
        LoadIsawPeaks(Filename=args.peaks, OutputWorkspace="peaks")
    else:
        LoadNexus(Filename=args.peaks, OutputWorkspace="peaks")
    peaks = mtd["peaks"]
    if args.max_peaks is not None:
        for i in reversed(range(args.max_peaks, peaks.getNumberPeaks())):
            peaks.removePeak(i)
    return mtd["events"], peaks


def shape_parameters(peak):
    """Directions (as rows), peak/inner/outer radii, frame and translation of a peak's ellipsoid."""
    shape = json.loads(peak.getPeakShape().toJSON())
    if shape.get("shape") != "ellipsoid":
        raise RuntimeError(
            "Peak {} has no ellipsoidal shape".format(peak.getHKL())
        )
    directions = np.array(
        [
            [float(v) for v in shape["direction{}".format(i)].split()]
            for i in range(3)
        ]
    )
    radii = np.array([shape["radius{}".format(i)] for i in range(3)])
    inner = np.array(
        [shape["background_inner_radius{}".format(i)] for i in range(3)]
    )
    outer = np.array(
        [shape["background_outer_radius{}".format(i)] for i in range(3)]
    )
    translation = np.array(
        [shape.get("translation{}".format(i), 0.0) for i in range(3)]
    )
    return directions, radii, inner, outer, shape["frame"], translation


def shapes_to_q_lab(peaks, R):
    """IntegratePeaksShapeMD needs shapes in Q_lab; rotate any in Q_sample by the goniometer R."""
    for i in range(peaks.getNumberPeaks()):
        peak = peaks.getPeak(i)
        directions, radii, inner, outer, frame, _ = shape_parameters(peak)
        if frame == int(SpecialCoordinateSystem.QLab):
            continue
        lab = directions @ R.T  # rows R d
        peak.setPeakShape(
            PeakShapeEllipsoid(
                [V3D(*d) for d in lab],
                list(radii),
                list(inner),
                list(outer),
                SpecialCoordinateSystem.QLab,
            )
        )


def inside(points, center, directions, radii):
    """Whether each point (rows) is inside the ellipsoid."""
    projected = (points - center) @ directions.T / radii
    return np.sum(projected**2, axis=1) <= 1.0


def reference_intensities(events, peaks, R, step):
    """garnet's approach: bin each peak with MDNorm in Q_sample and sum over the voxels in its ellipsoids."""
    ConvertToMD(
        InputWorkspace=events,
        QDimensions="Q3D",
        dEAnalysisMode="Elastic",
        Q3DFrames="Q_sample",
        MinValues="-50,-50,-50",
        MaxValues="50,50,50",
        OutputWorkspace="md",
    )
    n = peaks.getNumberPeaks()
    garnet = np.full(n, np.nan)
    ratio_of_sums = np.full(n, np.nan)
    empty_fraction = np.full(n, np.nan)
    for i in range(n):
        peak = peaks.getPeak(i)
        directions, radii, inner, outer, _, _ = shape_parameters(peak)
        if np.min(outer - inner) < step or np.min(radii) < step:
            continue  # thinner than a voxel
        # the shapes are now in Q_lab; MDNorm bins in Q_sample = R^-1 Q_lab
        center = R.T @ np.array(peak.getQLabFrame())
        directions = directions @ R
        extent = np.max(outer) + step
        binning = {
            "Dimension{}Binning".format(d): "{},{},{}".format(
                center[d] - extent, step, center[d] + extent
            )
            for d in range(3)
        }
        MDNorm(
            InputWorkspace="md",
            SolidAngleWorkspace="solid_angle",
            FluxWorkspace="flux",
            RLU=False,
            OutputWorkspace="reference",
            OutputDataWorkspace="reference_data",
            OutputNormalizationWorkspace="reference_norm",
            **binning,
        )
        normalized = mtd["reference"]
        signal = normalized.getSignalArray().ravel(order="F")
        counts = mtd["reference_data"].getSignalArray().ravel(order="F")
        norm = mtd["reference_norm"].getSignalArray().ravel(order="F")
        dims = [normalized.getDimension(d) for d in range(3)]
        widths = [
            (dim.getMaximum() - dim.getMinimum()) / dim.getNBins()
            for dim in dims
        ]
        axes = [
            dim.getMinimum() + (np.arange(dim.getNBins()) + 0.5) * w
            for dim, w in zip(dims, widths)
        ]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
            -1, 3, order="F"
        )
        voxel = np.prod(widths)
        in_peak = inside(grid, center, directions, radii)
        in_shell = inside(grid, center, directions, outer) & ~inside(
            grid, center, directions, inner
        )
        finite = np.isfinite(signal)
        peak_volume = 4.0 / 3.0 * np.pi * np.prod(radii)

        empty_fraction[i] = np.count_nonzero(in_peak & ~finite) / max(
            np.count_nonzero(in_peak), 1
        )
        shell_volume = np.count_nonzero(in_shell & finite) * voxel
        background = (
            signal[in_shell & finite].sum()
            * voxel
            * peak_volume
            / shell_volume
            if shell_volume > 0
            else 0.0
        )
        garnet[i] = signal[in_peak & finite].sum() * voxel - background

        peak_norm, shell_norm = norm[in_peak].sum(), norm[in_shell].sum()
        if peak_norm > 0:
            rate = counts[in_peak].sum() / peak_norm
            if shell_norm > 0:
                rate -= counts[in_shell].sum() / shell_norm
            ratio_of_sums[i] = peak_volume * rate
    return garnet, ratio_of_sums, empty_fraction


def scaled_shapes(peaks, scale):
    """A copy of peaks with every peak radius scaled, for the profile fit."""
    scaled = peaks.clone()
    for i in range(scaled.getNumberPeaks()):
        peak = scaled.getPeak(i)
        directions, radii, inner, outer, _, _ = shape_parameters(peak)
        peak.setPeakShape(
            PeakShapeEllipsoid(
                [V3D(*d) for d in directions],
                list(radii * scale),
                list(inner),
                list(outer),
                SpecialCoordinateSystem.QLab,
            )
        )
    return scaled


def integrate(events, peaks, region_radius, profile):
    result, diagnostics = IntegratePeaksShapeMD(
        InputWorkspace=events,
        PeaksWorkspace=peaks,
        RegionRadius=region_radius,
        ProfileFit=profile,
        SolidAngleWorkspace="solid_angle",
        FluxWorkspace="flux",
        OutputWorkspace=(
            "normalized_profile" if profile else "normalized_counts"
        ),
        OutputDiagnosticsWorkspace=(
            "diagnostics_profile" if profile else "diagnostics_counts"
        ),
    )
    intensities = np.array(
        [
            result.getPeak(i).getIntensity()
            for i in range(result.getNumberPeaks())
        ]
    )
    sigmas = np.array(
        [
            result.getPeak(i).getSigmaIntensity()
            for i in range(result.getNumberPeaks())
        ]
    )
    return intensities, sigmas, diagnostics


def summarize(name, ratio, peaks, diagnostics, outlier_mads):
    good = np.isfinite(ratio)
    if not np.any(good):
        print("{}: no peaks to compare".format(name))
        return
    median = np.median(ratio[good])
    mad = 1.4826 * np.median(np.abs(ratio[good] - median))
    low, high = np.percentile(ratio[good], [16, 84])
    print(
        "{}: {} peaks, median ratio {:.4f}, 16-84% [{:.4f}, {:.4f}], MAD {:.4f}".format(
            name, np.count_nonzero(good), median, low, high, mad
        )
    )
    outliers = np.flatnonzero(
        good & (np.abs(ratio - median) > outlier_mads * mad)
    )
    print("  {} outliers beyond {} MADs:".format(len(outliers), outlier_mads))
    banks = peaks.column("BankName")
    for i in outliers:
        i = int(i)
        print(
            "    HKL {} bank {} ratio {:.3f} pixels {} clipped {} masked {}".format(
                peaks.getPeak(i).getHKL(),
                banks[i],
                ratio[i],
                diagnostics.cell("NumPixels", i),
                diagnostics.cell("ClippedByMomentumRange", i),
                diagnostics.cell("MaskedDetectors", i),
            )
        )


def main():
    args = parse_args()
    events, peaks = load_inputs(args)
    R = np.array(events.run().getGoniometer().getR())
    shapes_to_q_lab(peaks, R)

    start = time.perf_counter()
    counts, count_sigmas, count_diagnostics = integrate(
        events, peaks, args.region_radius, False
    )
    count_time = time.perf_counter() - start

    if args.profile:
        start = time.perf_counter()
        profile_peaks = (
            scaled_shapes(peaks, args.profile_radius_scale)
            if args.profile_radius_scale != 1.0
            else peaks
        )
        profiles, profile_sigmas, _ = integrate(
            events, profile_peaks, args.region_radius, True
        )
        profile_time = time.perf_counter() - start

    start = time.perf_counter()
    garnet, ratio_of_sums, empty_fraction = reference_intensities(
        events, peaks, R, args.step
    )
    reference_time = time.perf_counter() - start

    with np.errstate(divide="ignore", invalid="ignore"):
        count_ratio = counts / garnet
        count_ratio_of_sums = counts / ratio_of_sums
        profile_ratio = profiles / garnet if args.profile else None
        profile_to_count = profiles / counts if args.profile else None

    skipped = np.count_nonzero(np.isnan(ratio_of_sums))
    if skipped:
        print(
            "{} peaks skipped: ellipsoids thinner than a voxel, or no normalization".format(
                skipped
            )
        )
    print(
        "median fraction of peak voxels without normalization: {:.3f}".format(
            np.nanmedian(empty_fraction)
        )
    )
    summarize(
        "count mode / binned ratio of sums",
        count_ratio_of_sums,
        peaks,
        count_diagnostics,
        args.outlier_mads,
    )
    summarize(
        "count mode / garnet",
        count_ratio,
        peaks,
        count_diagnostics,
        args.outlier_mads,
    )
    if args.profile:
        summarize(
            "profile mode / garnet",
            profile_ratio,
            peaks,
            count_diagnostics,
            args.outlier_mads,
        )
        summarize(
            "profile mode / count mode",
            profile_to_count,
            peaks,
            count_diagnostics,
            args.outlier_mads,
        )
    print(
        "wall time: IntegratePeaksShapeMD count mode {:.1f} s".format(
            count_time
        )
    )
    if args.profile:
        print(
            "           IntegratePeaksShapeMD profile mode {:.1f} s".format(
                profile_time
            )
        )
    print(
        "           MDNorm binning (ConvertToMD + one MDNorm per peak) {:.1f} s".format(
            reference_time
        )
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "h",
            "k",
            "l",
            "bank",
            "garnet",
            "binned_ratio_of_sums",
            "empty_voxel_fraction",
            "count",
            "count_sigma",
            "count_ratio",
            "count_ratio_of_sums",
        ]
        if args.profile:
            header += ["profile", "profile_sigma", "profile_ratio"]
        header += ["num_pixels", "clipped", "masked"]
        writer.writerow(header)
        banks = peaks.column("BankName")
        for i in range(peaks.getNumberPeaks()):
            row = list(peaks.getPeak(i).getHKL()) + [
                banks[i],
                garnet[i],
                ratio_of_sums[i],
                empty_fraction[i],
                counts[i],
                count_sigmas[i],
                count_ratio[i],
                count_ratio_of_sums[i],
            ]
            if args.profile:
                row += [profiles[i], profile_sigmas[i], profile_ratio[i]]
            row += [
                count_diagnostics.cell("NumPixels", i),
                count_diagnostics.cell("ClippedByMomentumRange", i),
                count_diagnostics.cell("MaskedDetectors", i),
            ]
            writer.writerow(row)
    print("wrote {}".format(args.output))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.hist(
            count_ratio[np.isfinite(count_ratio)],
            bins=50,
            histtype="step",
            label="count mode",
        )
        if args.profile:
            ax.hist(
                profile_ratio[np.isfinite(profile_ratio)],
                bins=50,
                histtype="step",
                label="profile mode",
            )
        ax.set_xlabel("IntegratePeaksShapeMD / MDNorm")
        ax.set_ylabel("peaks")
        ax.legend()
        fig.savefig(args.plot, dpi=150)
        print("wrote {}".format(args.plot))


if __name__ == "__main__":
    main()
