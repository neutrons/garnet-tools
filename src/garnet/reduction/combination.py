import os
import gc
import traceback
import numpy as np

from mantid.simpleapi import mtd
from mantid import config

config["Q.convention"] = "Crystallography"

config["MultiThreaded.MaxCores"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TBB_THREAD_ENABLED"] = "0"

from garnet.config.instruments import beamlines
from garnet.reduction.ub import lattice_group
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.ellipsoid import PeakEllipsoid
from garnet.reduction.data import DataModel
from garnet.reduction.intensity import (
    IntegrationModel,
    revert_ellipsoid_parameters,
    voxel_weights,
)
from garnet.reduction.parallel import ParallelProcessor
from garnet.plots.peaks import PeakPlot


def _integrate_peak(kv):
    """Fit and integrate one peak. Module-level for pickling in ParallelProcessor."""
    key, value = kv

    (
        Q0,
        Q1,
        Q2,
        d,
        n,
        dQ,
        Q,
        shape,
        projections,
        c,
        neighbors,
        index,
        total,
        peak_file,
        hkl,
        d_spacing,
        wavelength,
        angles,
        goniometer,
    ) = value

    print("combination 2/2 {:}/{:}".format(index, total))

    weights = voxel_weights(Q0, Q1, Q2, c, neighbors)

    ellipsoid = PeakEllipsoid()
    ellipsoid.update_constraints(Q0, Q1, Q2, dQ)
    ellipsoid.update_estimate(shape)
    ellipsoid.lamda_center = 0

    args = (Q0, Q1, Q2, d, n, dQ, Q, weights)
    fit_params = ellipsoid.fit(*args)

    if fit_params is not None:
        intens_params = ellipsoid.extract_result(*fit_params, Q)

        if intens_params is None:
            print("Cannot extract fit")
            return key, None

        c_fit, S, *best_fit = ellipsoid.best_fit

        shape_out = revert_ellipsoid_parameters(intens_params, projections)

        norm_params = Q0, Q1, Q2, d, n, c_fit, S

        try:
            intens, sig = ellipsoid.integrate(*norm_params)
        except Exception as e:
            print("Exception extracting intensity: {}".format(e))
            print(traceback.format_exc())
            return key, None

        info = ellipsoid.info

        peak_plot = PeakPlot()

        peak_plot.add_ellipsoid_fit(best_fit)
        peak_plot.add_profile_fit(ellipsoid.best_prof)
        peak_plot.add_projection_fit(ellipsoid.best_proj)
        peak_plot.add_ellipsoid(c_fit, S)
        peak_plot.add_estimated_ellipsoid(*ellipsoid.estimated_fit)
        peak_plot.update_envelope(*ellipsoid.peak_background_mask)
        peak_plot.add_peak_info(hkl, d_spacing, wavelength, angles, goniometer)
        peak_plot.add_peak_stats(
            ellipsoid.reddev, ellipsoid.intensity, ellipsoid.sigma
        )
        peak_plot.add_data_norm_fit(*ellipsoid.data_norm_fit)
        peak_plot.add_integral_fit(ellipsoid.integral)
        peak_plot.add_filter(ellipsoid.filter)

        try:
            peak_plot.save_plot(peak_file)
        except Exception as e:
            print("Exception saving figure: {}".format(e))
            print(traceback.format_exc())

        peak_plot.close()

        return key, (intens, sig, shape_out, info)

    return key, None


class Combination(IntegrationModel):
    def __init__(self, plan):
        super(Combination, self).__init__(plan)

        self.params = plan["Integration"]
        self.output = plan["OutputName"] + "_combination"

        self.validate_params()

    def validate_params(self):
        self.check(
            self.params["Cell"], "in", lattice_group.keys(), "Invalid Cell"
        )
        self.check(
            self.params["Centering"],
            "in",
            centering_reflection.keys(),
            "Invalid Centering",
        )
        self.check(self.params["MinD"], ">", 0, "Invalid minimum d-spacing")
        self.check(self.params["Radius"], ">", 0, "Invalid radius")

        for key in ("ModVec1", "ModVec2", "ModVec3"):
            if self.params.get(key) is None:
                self.params[key] = [0, 0, 0]
            self.check(
                len(self.params[key]), "==", 3, f"{key} must have 3 components"
            )

        if self.params.get("MaxOrder") is None:
            self.params["MaxOrder"] = 0
        if self.params.get("CrossTerms") is None:
            self.params["CrossTerms"] = False
        if self.params.get("ProfileFit") is None:
            self.params["ProfileFit"] = True
        if self.params.get("NumProcesses") is None:
            self.params["NumProcesses"] = 1

        self.check(
            self.params["MaxOrder"], ">=", 0, "MaxOrder must be non-negative"
        )
        self.check(
            type(self.params["CrossTerms"]),
            "is",
            bool,
            "CrossTerms must be a boolean",
        )

    def combine(self):
        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        runs = self.plan["Runs"]

        r_cut = self.params["Radius"]
        d_min = self.params["MinD"]
        centering = self.params["Centering"]
        n_proc = self.params["NumProcesses"]

        # --- Phase 1: per-run load, predict, convert, delete events ---

        md_names = []

        for run in runs:
            data.load_data("data", self.plan["IPTS"], run)

            data.load_generate_normalization(
                self.plan["VanadiumFile"], self.plan.get("FluxFile")
            )

            data.apply_calibration(
                "data",
                self.plan.get("DetectorCalibration"),
                self.plan.get("TubeCalibration"),
                self.plan.get("GoniometerCalibration"),
            )

            data.preprocess_detectors("data")

            data.crop_for_normalization("data")

            data.apply_mask("data", self.plan.get("MaskFile"))

            data.load_clear_UB(self.plan["UBFile"], "data", run)

            lamda_min, lamda_max = data.wavelength_band

            peaks.predict_peaks(
                "data",
                "peaks_{}".format(run),
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            if self.params["MaxOrder"] > 0:
                sat_min_d = d_min
                if self.params.get("SatMinD") is not None:
                    sat_min_d = self.params["SatMinD"]

                peaks.predict_satellite_peaks(
                    "peaks_{}".format(run),
                    "data",
                    centering,
                    lamda_min,
                    lamda_max,
                    sat_min_d,
                    self.params["ModVec1"],
                    self.params["ModVec2"],
                    self.params["ModVec3"],
                    self.params["MaxOrder"],
                    self.params["CrossTerms"],
                )

            peaks.combine_peaks("peaks_{}".format(run), "peaks")
            peaks.delete_peaks("peaks_{}".format(run))

            data.convert_to_Q_sample(
                "data", "md_{}".format(run), lorentz_corr=False
            )

            data.delete_workspace("data")

            md_names.append("md_{}".format(run))

        # --- Phase 2: merge all MD workspaces ---

        data.combine_Q_sample(md_names, "md")

        # --- Phase 3: keep one representative per unique (h,k,l,m,n,p) ---

        peaks.filter_unique_hkl("peaks", "unique_peaks")
        peaks.delete_peaks("peaks")

        # --- Phase 4: normalize (serial, MDNorm) then fit (parallel) ---

        self.data = data

        peak_data = self._extract_all("unique_peaks", r_cut)

        results = ParallelProcessor(n_proc).process_dict(
            peak_data, _integrate_peak
        )

        del peak_data

        self._update_peak_info("unique_peaks", results)

        del results
        gc.collect()

        # --- Phase 5: sort and save ---

        peaks.sort_peaks_by_d("unique_peaks")

        peaks.save_peaks(output_file, "unique_peaks")

        hkl_file = os.path.splitext(output_file)[0] + ".hkl"

        peaks.save_hkl_cw(hkl_file, "unique_peaks")

        data.delete_workspace("md")

        mtd.clear()

        return output_file

    def _extract_all(self, peaks_ws, r_cut):
        """
        Normalize each unique peak against the merged MD and collect bin arrays.

        Returns a dict mapping peak index → tuple of numpy arrays and metadata
        ready to pass to the parallel fitting step.
        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        UB = peak.get_UB()

        hkls = [peak.get_hkl(i) for i in range(n_peak)]

        peak_data = {}

        for j, i in enumerate(range(n_peak)):
            print("combination 1/2  #{:}/{:}".format(j, n_peak))

            d_spacing = peak.get_d_spacing(i)

            Q = 2 * np.pi / d_spacing

            hkl = peak.get_hkl(i)

            lamda = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            goniometer = peak.get_goniometer_angles(i)

            peak.set_peak_intensity(i, 0, 0)

            R = peak.get_goniometer_matrix(i)

            shape = peak.get_peak_shape(i, r_cut=r_cut / 3)

            dQ = data.get_resolution_in_Q(lamda, two_theta)

            (
                bins,
                extents,
                projections,
                transform,
                conversion,
            ) = self.bin_extent(
                UB, hkl, lamda, R, two_theta, az_phi, shape, dQ
            )

            center = conversion @ np.array(hkl)

            neighbors = [
                conversion @ np.array(hkl_k)
                for k, hkl_k in enumerate(hkls)
                if k != j
            ]

            data.normalize_to_hkl("md", transform, extents, bins)

            d, _, Q0, Q1, Q2 = data.extract_bin_info("md_data")
            n, _, Q0, Q1, Q2 = data.extract_bin_info("md_norm")

            data.clear_norm("md")

            params = self.project_ellipsoid_parameters(shape, projections)

            peak_name = peak.get_peak_name(i, merge=True)

            peak_file = self.get_plot_file(peak_name)

            os.makedirs(os.path.dirname(peak_file), exist_ok=True)

            peak_data[i] = (
                Q0,
                Q1,
                Q2,
                d,
                n,
                dQ,
                Q,
                params,
                projections,
                center,
                neighbors,
                j,
                n_peak,
                peak_file,
                hkl,
                d_spacing,
                lamda,
                angles,
                goniometer,
            )

        return peak_data

    def _update_peak_info(self, peaks_ws, results):
        peak = PeakModel(peaks_ws)

        for i, value in results.items():
            if value is not None:
                intens, sig, shape, info = value

                peak.set_peak_intensity(i, intens, sig)

                peak.set_peak_shape(i, *shape)

                Qx, Qy, Qz = shape[:3]
                info["Qx"] = Qx
                info["Qy"] = Qy
                info["Qz"] = Qz

                peak.add_diagnostic_info(i, info)

            else:
                peak.set_peak_intensity(i, 0, 0)
