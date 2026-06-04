import os
import gc
import subprocess
import traceback
import time
import numpy as np

import scipy.spatial.transform
import scipy.special
import scipy.stats

from lmfit import Minimizer, Parameters, fit_report

from itertools import permutations

from mantid.simpleapi import mtd
from mantid import config

config["Q.convention"] = "Crystallography"

config["MultiThreaded.MaxCores"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TBB_THREAD_ENABLED"] = "0"

from garnet.plots.peaks import PeakPlot
from garnet.config.instruments import beamlines
from garnet.reduction.ub import UBModel, Optimization, Reorient, lattice_group
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.resolution import ResolutionEllipsoid
from garnet.reduction.data import DataModel
from garnet.reduction.plan import SubPlan

INTEGRATION = os.path.abspath(__file__)
directory = os.path.dirname(INTEGRATION)

filename = os.path.join(directory, "../utilities/structure.py")
REFLECTIONS = os.path.abspath(filename)

assert os.path.exists(REFLECTIONS)


class Integration(SubPlan):
    def __init__(self, plan):
        super(Integration, self).__init__(plan)

        self.params = plan["Integration"]
        self.output = plan["OutputName"] + "_integration"

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

        self.check(
            self.params["MaxOrder"], ">=", 0, "MaxOrder must be non-negative"
        )
        self.check(
            type(self.params["CrossTerms"]),
            "is",
            bool,
            "CrossTerms must be a boolean",
        )

    @staticmethod
    def integrate_parallel(plan, runs, proc):
        plan["Runs"] = runs
        plan["ProcName"] = "_p{}".format(proc)

        instance = Integration(plan)
        instance.proc = proc
        instance.n_proc = 1

        return instance.integrate()

    @staticmethod
    def combine_parallel(plan, files):
        instance = Integration(plan)

        return instance.combine(files)

    def combine(self, files):
        output_file = self.get_output_file()
        result_file = self.get_file(output_file, "")

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        self.data = data

        self.make_plot = False

        peaks = PeaksModel()

        for file in files:
            peaks.load_peaks(file, "tmp")
            peaks.combine_peaks("tmp", "combine")

        for file in files:
            os.remove(file)
            os.remove(os.path.splitext(file)[0] + ".mat")

        peaks.reset_satellites("combine")

        if mtd.doesExist("combine"):
            peaks.save_peaks(result_file, "combine")

            opt = Optimization("combine")
            opt.optimize_lattice(self.params["Cell"])

            ub_file = os.path.splitext(result_file)[0] + ".mat"

            ub = UBModel("combine")
            ub.save_UB(ub_file)

        self.cleanup()
        self.write(result_file)

    def write(self, result_file):
        process = subprocess.Popen(
            ["python", REFLECTIONS, self.plan["YAML"]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = process.communicate()
        if process.returncode == 0:
            print("First command succeeded:", out.decode().strip())

    def integrate(self):
        output_file = self.get_output_file()

        data = DataModel(beamlines[self.plan["Instrument"]])
        data.update_raw_path(self.plan)

        peaks = PeaksModel()

        self.make_plot = True

        runs = self.plan["Runs"]

        self.run = 0
        self.runs = len(runs)

        result_file = self.get_file(output_file, "")

        for run in runs:
            self.run += 1

            self.status = "{}: {:}/{:}".format(self.proc, self.run, len(runs))

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

            # data.load_background(self.plan.get("BackgroundFile"), "data")

            data.load_clear_UB(self.plan["UBFile"], "data", run)

            lamda_min, lamda_max = data.wavelength_band

            d_min = self.params["MinD"]

            centering = self.params["Centering"]

            cell = self.params["Cell"]

            self.cntrt = data.get_counting_rate("data")

            data.convert_to_Q_sample("data", "md", lorentz_corr=True)

            r_cut = self.params["Radius"]

            if self.params.get("Recalibrate"):
                ub = UBModel("data")

                const = ub.get_lattice_parameters()

                min_d, max_d = ub.get_primitive_cell_length_range(centering)

                const = ub.convert_conventional_to_primitive(*const, centering)

                peaks.find_peaks("md", "peaks", max_d)

                peaks.integrate_peaks(
                    "md",
                    "peaks",
                    r_cut / np.cbrt(3),
                    method="ellipsoid",
                    centroid=False,
                )

                peaks.remove_weak_peaks("peaks", 20)

                ub = UBModel("peaks")
                ub.determine_UB_with_lattice_parameters(*const)
                ub.index_peaks()
                ub.transform_primitive_to_conventional(centering)
                ub.refine_UB_with_constraints(cell)

                Reorient("peaks", cell)

                ub.copy_UB("data")

                ub_file = self.get_diagnostic_file("run#{}_ub".format(run))
                ub_file = os.path.splitext(ub_file)[0] + ".mat"

                ub.save_UB(ub_file)

                data.load_clear_UB(ub_file, "data", run)

            peaks.predict_peaks(
                "data",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            md_file = self.get_diagnostic_file("run#{}_data".format(run))
            md_file = os.path.splitext(md_file)[0] + ".nxs"

            data.save_histograms(md_file, "md")

            ub = UBModel("peaks")
            self.P = ub.centering_matrix(centering)

            self.peaks, self.data = peaks, data

            self.r_cut = r_cut

            self.predict_add_satellite_peaks(lamda_min, lamda_max)

            peaks.integrate_peaks(
                "md",
                "peaks",
                r_cut / np.cbrt(3),
                method="ellipsoid",
                centroid=True,
                update=False,
            )

            pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))

            peaks.save_peaks(pk_file, "peaks")

            res_file = self.get_plot_file("run#{}_res".format(run))

            res = ResolutionEllipsoid("peaks", r_cut=np.inf)

            res.fit()
            res.plot_diagnostics(res_file)
            res.apply()

            pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))

            peaks.save_peaks(pk_file, "peaks")

            data.delete_workspace("md")

            fit = self.params["ProfileFit"]

            banks = peaks.get_bank_names("peaks")

            for bank in banks:
                if self.make_plot:
                    self.peak_plot = PeakPlot()

                data.mask_to_bank("data", bank)

                data.preprocess_detector_banks(bank)

                data.convert_to_Q_sample(bank, bank, False, bank + "_dets")

                peak_dict = self.extract_peak_info(
                    "peaks", r_cut, True, fit, bank
                )

                data.delete_workspace(bank)

                data.delete_workspace(bank + "_dets")

                results = self.integrate_peaks(peak_dict)

                del peak_dict

                self.update_peak_info("peaks", results)

                del results

                gc.collect()

                if self.make_plot:
                    self.peak_plot.close()

            peaks.update_scale_factor("peaks", data.monitor)

            peaks.combine_peaks("peaks", "combine")

            pk_file = self.get_diagnostic_file("run#{}_integrate".format(run))

            peaks.save_peaks(pk_file, "peaks")

            data.delete_workspace("data")

            data.delete_workspace("peaks")

        peaks.remove_weak_peaks("combine", -100)

        peaks.save_peaks(result_file, "combine")

        # ---

        if mtd.doesExist("combine"):
            opt = Optimization("combine")
            opt.optimize_lattice(cell)

            ub_file = os.path.splitext(result_file)[0] + ".mat"

            ub = UBModel("combine")
            ub.save_UB(ub_file)

        mtd.clear()

        return result_file

    def predict_add_satellite_peaks(self, lamda_min, lamda_max):
        if self.params["MaxOrder"] > 0:
            sat_min_d = self.params["MinD"]
            if self.params.get("SatMinD") is not None:
                sat_min_d = self.params["SatMinD"]

            self.peaks.predict_satellite_peaks(
                "peaks",
                "md",
                self.params["Centering"],
                lamda_min,
                lamda_max,
                sat_min_d,
                self.params["ModVec1"],
                self.params["ModVec2"],
                self.params["ModVec3"],
                self.params["MaxOrder"],
                self.params["CrossTerms"],
            )

    def get_file(self, file, ws=""):
        """
        Update filename with identifier name and optional workspace name.

        Parameters
        ----------
        file : str
            Original file name.
        ws : str, optional
            Name of workspace. The default is ''.

        Returns
        -------
        output_file : str
            File with updated name for identifier and workspace name.

        """

        if len(ws) > 0:
            ws = "_" + ws

        return self.append_name(file).replace(".nxs", ws + ".nxs")

    def append_name(self, file):
        """
        Update filename with identifier name.

        Parameters
        ----------
        file : str
            Original file name.

        Returns
        -------
        output_file : str
            File with updated name for identifier name.

        """

        append = (
            self.cell_centering_name()
            + self.modulation_name()
            + self.resolution_name()
        )

        name, ext = os.path.splitext(file)

        return name + append + ext

    def cell_centering_name(self):
        """
        Lattice and reflection condition.

        Returns
        -------
        lat_ref : str
            Underscore separated strings.

        """

        cell = self.params["Cell"]
        centering = self.params["Centering"]

        return "_" + cell + "_" + centering

    def modulation_name(self):
        """
        Modulation vectors.

        Returns
        -------
        mod : str
            Underscore separated vectors and max order

        """

        mod = ""

        max_order = self.params.get("MaxOrder")
        mod_vec_1 = self.params.get("ModVec1")
        mod_vec_2 = self.params.get("ModVec2")
        mod_vec_3 = self.params.get("ModVec3")
        cross_terms = self.params.get("CrossTerms")

        if max_order > 0:
            for vec in [mod_vec_1, mod_vec_2, mod_vec_3]:
                if np.linalg.norm(vec) > 0:
                    mod += "_({},{},{})".format(*vec)
            if cross_terms:
                mod += "_mix"

        return mod

    def resolution_name(self):
        """
        Minimum d-spacing and starting radii

        Returns
        -------
        res_rad : str
            Underscore separated strings.

        """

        min_d = self.params["MinD"]
        max_r = self.params["Radius"]

        return "_d(min)={:.2f}".format(min_d) + "_r(max)={:.2f}".format(max_r)

    def unit_key(self, v, tol=1e-2):
        v = np.asarray(v)
        v = v / np.linalg.norm(v)
        return tuple(np.round(v / tol).astype(int).tolist())

    def voxel_weights(
        self, Q0, Q1, Q2, c, neighbors, t_max=0.95, k_nearest=12
    ):
        weights = np.ones_like(Q2)

        if neighbors:
            dx = np.stack([Q0 - c[0], Q1 - c[1], Q2 - c[2]], axis=-1)

            box_radius = np.sqrt(np.einsum("...i,...i->...", dx, dx).max())
            deltas = np.array([q_j - c for q_j in neighbors])
            dist_sqs = np.einsum("ji,ji->j", deltas, deltas)

            nearby = dist_sqs < (2.0 * box_radius) ** 2
            if nearby.any():
                deltas = deltas[nearby]
                dist_sqs = dist_sqs[nearby]

                if len(dist_sqs) > k_nearest:
                    idx = np.argpartition(dist_sqs, k_nearest)[:k_nearest]
                    deltas = deltas[idx]
                    dist_sqs = dist_sqs[idx]

                dots = np.einsum("...i,ji->...j", dx, deltas)
                t_all = np.clip(dots / (0.5 * dist_sqs), 0.0, t_max)
                weights = (1.0 - t_all).min(axis=-1)

        return weights

    def integrate_peaks(self, data):
        result = {}

        for key, value in data.items():
            data_info, peak_info, index = value

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
            ) = data_info

            (
                peak_file,
                hkl,
                d_spacing,
                wavelength,
                angles,
                goniometer,
            ) = peak_info

            print(self.status + " 2/2 {:}/{:}".format(index, self.total))

            weights = self.voxel_weights(Q0, Q1, Q2, c, neighbors)

            ellipsoid = PeakEllipsoid()
            ellipsoid.update_constraints(Q0, Q1, Q2, dQ)
            ellipsoid.update_estimate(shape)

            args = (Q0, Q1, Q2, d, n, dQ, Q, weights)
            fit_params = ellipsoid.fit(*args)

            intens_params = None
            if fit_params is not None:
                intens_params = ellipsoid.extract_result(*fit_params, Q)

                if intens_params is None:
                    result[key] = None
                    print("Cannot extract fit")
                    assert False

                c, S, *best_fit = ellipsoid.best_fit

                shape = self.revert_ellipsoid_parameters(
                    intens_params, projections
                )

                norm_params = Q0, Q1, Q2, d, n, c, S

                try:
                    intens, sig = ellipsoid.integrate(*norm_params)
                except Exception as e:
                    print("Exception extracting intensity: {}".format(e))
                    print(traceback.format_exc())
                    result[key] = None

                info = ellipsoid.info
                best_prof = ellipsoid.best_prof
                best_proj = ellipsoid.best_proj
                data_norm_fit = ellipsoid.data_norm_fit
                redchi2 = ellipsoid.redchi2
                intensity = ellipsoid.intensity
                sigma = ellipsoid.sigma
                peak_background_mask = ellipsoid.peak_background_mask
                integral = ellipsoid.integral
                matched_filter = ellipsoid.filter
                estimated_fit = ellipsoid.estimated_fit

                if self.make_plot:
                    self.peak_plot.add_ellipsoid_fit(best_fit)

                    self.peak_plot.add_profile_fit(best_prof)

                    self.peak_plot.add_projection_fit(best_proj)

                    self.peak_plot.add_ellipsoid(c, S)

                    self.peak_plot.add_estimated_ellipsoid(*estimated_fit)

                    self.peak_plot.update_envelope(*peak_background_mask)

                    self.peak_plot.add_peak_info(
                        hkl, d_spacing, wavelength, angles, goniometer
                    )

                    self.peak_plot.add_peak_stats(redchi2, intensity, sigma)

                    self.peak_plot.add_data_norm_fit(*data_norm_fit)

                    self.peak_plot.add_integral_fit(integral)

                    self.peak_plot.add_filter(matched_filter)

                    try:
                        self.peak_plot.save_plot(peak_file)
                    except Exception as e:
                        print("Exception saving figure: {}".format(e))
                        print(traceback.format_exc())

                extra_info = [*info, *shape[:3], self.cntrt]

                result[key] = intens, sig, shape, extra_info, hkl

            del ellipsoid

        return result

    def pad_to_shape(self, x, shape, fill=0):
        out = np.full(shape, fill, dtype=np.result_type(x, fill))
        sx, sy, sz = x.shape
        out[:sx, :sy, :sz] = x
        return out

    def add_with_padding(self, a, b, fill=0):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("Both arrays must be 3D")

        shape = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))
        a2 = self.pad_to_shape(a, shape, fill=fill)
        b2 = self.pad_to_shape(b, shape, fill=fill)
        return a2 + b2

    def extract_peak_info(
        self, peaks_ws, r_cut, norm=False, fit=True, bank=None
    ):
        """
        Obtain peak information for envelope determination.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r_cut : list or float
            Cutoff radius parameter(s).

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        UB = peak.get_UB()

        peak_dict = {}

        indices = range(n_peak)

        if bank is not None:
            inds = []
            for i in indices:
                if peak.get_bank_name(i) == bank:
                    inds.append(i)
            indices = inds

        self.total = len(indices)
        self.bank = bank

        hkls = []
        for i in indices:
            hkls.append(peak.get_hkl(i))

        for j, i in enumerate(indices):
            print(
                self.status
                + " 1/2  #{:}: {:}/{:}".format(self.bank, j, self.total)
            )

            d_spacing = peak.get_d_spacing(i)

            Q = 2 * np.pi / d_spacing

            hkl = peak.get_hkl(i)

            lamda = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            peak.set_peak_intensity(i, 0, 0)

            goniometer = peak.get_goniometer_angles(i)

            peak_name = peak.get_peak_name(i)

            dQ = data.get_resolution_in_Q(lamda, two_theta)

            R = peak.get_goniometer_matrix(i)

            bank_name = peak.get_bank_name(i)

            shape = peak.get_peak_shape(i)

            bin_params = UB, hkl, lamda, R, two_theta, az_phi, shape, dQ

            bin_extent = self.bin_extent(*bin_params)

            bins, extents, projections, transform, conversion = bin_extent

            center = conversion @ hkl

            neighbors = [
                conversion @ hkl for k, hkl in enumerate(hkls) if k != j
            ]

            data.normalize_to_hkl(bank_name, transform, extents, bins)

            d, _, Q0, Q1, Q2 = data.extract_bin_info(bank_name + "_data")
            n, _, Q0, Q1, Q2 = data.extract_bin_info(bank_name + "_norm")

            data.check_volume_preservation(bank_name + "_result")

            peak_file = self.get_diagnostic_file(peak_name)

            directory = os.path.dirname(peak_file)

            os.makedirs(directory, exist_ok=True)

            data_file = self.get_diagnostic_file(peak_name + "_data")
            norm_file = self.get_diagnostic_file(peak_name + "_norm")

            data.save_histograms(data_file, bank_name + "_data")
            data.save_histograms(norm_file, bank_name + "_norm")

            data.clear_norm(bank_name)

            params = self.project_ellipsoid_parameters(shape, projections)

            data_info = (
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
            )

            peak_file = self.get_plot_file(peak_name)

            directory = os.path.dirname(peak_file)

            os.makedirs(directory, exist_ok=True)

            peak_info = (peak_file, hkl, d_spacing, lamda, angles, goniometer)

            peak_dict[i] = data_info, peak_info, j

        return peak_dict

    def update_peak_offsets(self, peaks_ws, offsets, peak_dict):
        peak = PeakModel(peaks_ws)

        c0, c1, c2, Q = offsets

        for i, value in peak_dict.items():
            if value is not None:
                data_info, peak_info = peak_dict[i]

                projections = data_info[7]

                W = np.column_stack(projections)

                vec = [c0[i] + Q[i], c1[i], c2[i]]

                if np.isfinite(vec).all():
                    Q0, Q1, Q2 = np.dot(W, vec)

                    peak.set_peak_center(i, Q0, Q1, Q2)

    def update_peak_info(self, peaks_ws, peak_dict):
        peak = PeakModel(peaks_ws)

        for i, value in peak_dict.items():
            if value is not None:
                I, sigma, shape, info, hkl = value

                peak.set_peak_intensity(i, I, sigma)

                peak.set_peak_shape(i, *shape)

                peak.add_diagonstic_info(i, info)

            else:
                peak.set_peak_intensity(i, 0, 0)

    def bin_axes(self, R, two_theta, az_phi):
        two_theta = np.deg2rad(two_theta)
        az_phi = np.deg2rad(az_phi)

        kf_hat = np.array(
            [
                np.sin(two_theta) * np.cos(az_phi),
                np.sin(two_theta) * np.sin(az_phi),
                np.cos(two_theta),
            ]
        )

        ki_hat = np.array([0, 0, 1])

        n = kf_hat - ki_hat
        n /= np.linalg.norm(n)

        u = kf_hat + ki_hat
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        return R.T @ n, R.T @ u, R.T @ v

    def project_ellipsoid_parameters(self, params, projections):
        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W.T, [c0, c1, c2]), r0, r1, r2, *np.dot(W.T, V).T

    def revert_ellipsoid_parameters(self, params, projections):
        W = np.column_stack(projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        V = np.column_stack([v0, v1, v2])

        return *np.dot(W, [c0, c1, c2]), r0, r1, r2, *np.dot(W, V).T

    def transform_Q(self, Q0, Q1, Q2, projections):
        W = np.column_stack(projections)

        return np.einsum("ij,j...->i...", W, [Q0, Q1, Q2])

    def bin_extent(
        self,
        UB,
        hkl,
        lamda,
        R,
        two_theta,
        az_phi,
        shape,
        dQ,
        bin_min=21,
        bin_max=21,
    ):
        n, u, v = self.bin_axes(R, two_theta, az_phi)
        projections = [n, u, v]

        params = self.project_ellipsoid_parameters(shape, projections)

        c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

        U = np.column_stack([v0, v1, v2])
        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)
        r_cut = 3 * np.sqrt(np.diag(S))

        r_cut = np.where(r_cut < 3 * dQ, 3 * dQ, r_cut)

        W = np.column_stack(projections)
        A = 2 * np.pi * (W.T @ UB)

        miller_half_width = 0.5 * np.sum(np.abs(A), axis=1)
        r_cut = np.minimum(r_cut, miller_half_width)

        bins = np.clip(1 + np.floor(r_cut / dQ).astype(int), bin_min, bin_max)

        Wp = np.linalg.inv(W.T @ (2 * np.pi * UB)).T
        transform = Wp.tolist()

        h, k, l = hkl

        Q0, Q1, Q2 = A @ np.array([h, k, l])

        extents = np.array(
            [
                [Q0 - r_cut[0], Q0 + r_cut[0]],
                [Q1 - r_cut[1], Q1 + r_cut[1]],
                [Q2 - r_cut[2], Q2 + r_cut[2]],
            ]
        )

        conversion = A.copy()

        return bins, extents, projections, transform, conversion


class PeakEllipsoid:
    def __init__(self):
        self.params = Parameters()

        self.lamda_center = 0.01 / 3
        self.lamda_cov = 0.01 / 6
        self.prior_center_mean = np.zeros(3)
        self.prior_center_sigma = np.ones(3)
        self.prior_cov = np.eye(3)
        self.prior_cov_sigma = np.ones(6)

    def vech6(self, M):
        return np.array(
            [M[0, 0], M[1, 1], M[2, 2], M[1, 2], M[0, 2], M[0, 1]],
            dtype=float,
        )

    def update_constraints(self, x0, x1, x2, dx):
        dx0 = x0[:, 0, 0][1] - x0[:, 0, 0][0]
        dx1 = x1[0, :, 0][1] - x1[0, :, 0][0]
        dx2 = x2[0, 0, :][1] - x2[0, 0, :][0]

        r0_max = (x0[:, 0, 0][-1] - x0[:, 0, 0][0]) / 2
        r1_max = (x1[0, :, 0][-1] - x1[0, :, 0][0]) / 2
        r2_max = (x2[0, 0, :][-1] - x2[0, 0, :][0]) / 2

        c0, c1, c2 = 0, 0, 0

        c0_min, c1_min, c2_min = (
            c0 - r0_max,
            c1 - r1_max,
            c2 - r2_max,
        )
        c0_max, c1_max, c2_max = (
            c0 + r0_max,
            c1 + r1_max,
            c2 + r2_max,
        )

        r0 = 4 * dx0
        r1 = 4 * dx1
        r2 = 4 * dx2

        dx = np.min([dx0, dx1, dx2])

        self.params.add("c0", value=c0, min=c0_min, max=c0_max)
        self.params.add("c1", value=c1, min=c1_min, max=c1_max)
        self.params.add("c2", value=c2, min=c2_min, max=c2_max)

        self.params.add("r0", value=r0, min=dx, max=2 * r0_max)
        self.params.add("r1", value=r1, min=dx, max=2 * r1_max)
        self.params.add("r2", value=r2, min=dx, max=2 * r2_max)

        self.params.add("u0", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("u1", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("u2", value=0.0, min=-np.pi, max=np.pi)

        self.combine_params = None

    def copy_combine(self):
        self.combine_params = self.params.copy()

    def best_ordered_shape(self, r0, r1, r2, v0, v1, v2):
        rs = [r0, r1, r2]
        vs = [v0, v1, v2]

        signs = [
            (+1, +1, +1),
            (+1, -1, -1),
            (-1, +1, -1),
            (-1, -1, +1),
        ]

        best = None

        for p in permutations(range(3)):
            r_try = [rs[i] for i in p]
            v_try = [vs[i] for i in p]

            for s in signs:
                V = [si * vi for si, vi in zip(s, v_try)]
                U = np.column_stack(V)

                if np.linalg.det(U) < 0:
                    continue

                u = scipy.spatial.transform.Rotation.from_matrix(U).as_rotvec()
                score = np.linalg.norm(u)

                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "r": r_try,
                        "v": V,
                        "U": U,
                        "u": u,
                    }

        return best

    def update_estimate(self, shape):
        c0, c1, c2, r0, r1, r2, v0, v1, v2 = shape

        best = self.best_ordered_shape(r0, r1, r2, v0, v1, v2)

        r0, r1, r2 = best["r"]
        u0, u1, u2 = best["u"]

        self.params["c0"].set(value=0)
        self.params["c1"].set(value=0)
        self.params["c2"].set(value=0)

        for name, value in zip(["r0", "r1", "r2"], [r0, r1, r2]):
            p = self.params[name]
            if p.min < value < p.max:
                p.set(value=value)
            else:
                p.set(value=2 * p.min)

        self.params["u0"].set(value=u0)
        self.params["u1"].set(value=u1)
        self.params["u2"].set(value=u2)

        self.params["u0"].set(min=-np.pi, max=np.pi)
        self.params["u1"].set(min=-np.pi, max=np.pi)
        self.params["u2"].set(min=-np.pi, max=np.pi)

        self.estimated_fit = (
            np.array([0.0, 0.0, 0.0]),
            self.S_matrix(r0, r1, r2, u0, u1, u2),
        )

        self.initialize_priors(best)

    def initialize_priors(self, best):
        r0, r1, r2 = np.asarray(best["r"], dtype=float)
        u0, u1, u2 = best["u"]
        center_scale = np.mean([r0, r1, r2])

        self.prior_center_mean = np.zeros(3)
        self.prior_center_sigma = np.full(3, center_scale, dtype=float)

        self.prior_cov = self.S_matrix(r0, r1, r2, u0, u1, u2)
        v0 = self.vech6(self.prior_cov)
        floor = 0.25 * np.nanmax(np.abs(np.diag(self.prior_cov)))
        self.prior_cov_sigma = np.maximum(np.abs(v0), floor)

    def prior_residual(self, params):
        terms = []

        for name, mean, sigma in zip(
            ["c0", "c1", "c2"], self.prior_center_mean, self.prior_center_sigma
        ):
            terms.append(
                np.sqrt(self.lamda_center)
                * (params[name].value - mean)
                / sigma
            )

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        S = self.S_matrix(r0, r1, r2, u0, u1, u2)
        dS = self.vech6(S - self.prior_cov)
        terms += (np.sqrt(self.lamda_cov) * dS / self.prior_cov_sigma).tolist()

        return np.asarray(terms, dtype=float)

    def prior_jacobian(self, params, S, dr, du):
        names = ["c0", "c1", "c2", "S00", "S11", "S22", "S12", "S02", "S01"]
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), len(names)), dtype=float)

        for col, (name, sigma) in enumerate(
            zip(["c0", "c1", "c2"], self.prior_center_sigma)
        ):
            jac[params_list.index(name), col] = (
                np.sqrt(self.lamda_center) / sigma
            )

        d_inv_S = {
            "r0": dr[0],
            "r1": dr[1],
            "r2": dr[2],
            "u0": du[0],
            "u1": du[1],
            "u2": du[2],
        }

        cov_scale = np.sqrt(self.lamda_cov) / self.prior_cov_sigma
        for name in ["r0", "r1", "r2", "u0", "u1", "u2"]:
            dS = -S @ d_inv_S[name] @ S
            jac[params_list.index(name), 3:] = cov_scale * self.vech6(dS)

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]
        return jac[ind]

    def S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):
        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def det_S(self, r0, r1, r2, u0, u1, u2):
        S = self.S_matrix(r0, r1, r2, u0, u1, u2)
        return np.linalg.det(S)

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):
        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S

    def data_norm(self, d, n, v, rel_err=30):
        mask = (n > 0) & np.isfinite(n)

        d[~mask] = np.nan
        n[~mask] = np.nan
        v[~mask] = np.nan

        y_int = d / n
        e_int = np.sqrt(d + np.nanpercentile(d, rel_err) + 1) / n

        return y_int, e_int

    def profile_project(self, x0, x1, x2, d, n, w, c, inv_S, mode="3d"):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == "1d_0":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="2d_0")[
                np.newaxis, :, :
            ]
        elif mode == "1d_1":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="2d_1")[
                :, np.newaxis, :
            ]
        elif mode == "1d_2":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="2d_2")[
                :, :, np.newaxis
            ]
        elif mode == "2d_0":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="1d_0")[
                :, np.newaxis, np.newaxis
            ]
        elif mode == "2d_1":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="1d_1")[
                np.newaxis, :, np.newaxis
            ]
        elif mode == "2d_2":
            p = self.gaussian(x0, x1, x2, c, inv_S, mode="1d_2")[
                np.newaxis, np.newaxis, :
            ]
        else:
            p = np.ones_like(d)

        p /= np.nanmean(p)

        if mode == "1d_0":
            d_int = np.nansum(d * p, axis=(1, 2))
            v_int = np.nansum(d * p**2, axis=(1, 2))
            n_int = np.nansum(n * p / dx1 / dx2, axis=(1, 2))
            w_int = np.nanmean(w, axis=(1, 2))
        elif mode == "1d_1":
            d_int = np.nansum(d * p, axis=(0, 2))
            v_int = np.nansum(d * p**2, axis=(0, 2))
            n_int = np.nansum(n * p / dx0 / dx2, axis=(0, 2))
            w_int = np.nanmean(w, axis=(0, 2))
        elif mode == "1d_2":
            d_int = np.nansum(d * p, axis=(0, 1))
            v_int = np.nansum(d * p**2, axis=(0, 1))
            n_int = np.nansum(n * p / dx0 / dx1, axis=(0, 1))
            w_int = np.nanmean(w, axis=(0, 1))
        elif mode == "2d_0":
            d_int = np.nansum(d * p, axis=0)
            v_int = np.nansum(d * p**2, axis=0)
            n_int = np.nansum(n * p / dx0, axis=0)
            w_int = np.nanmean(w, axis=0)
        elif mode == "2d_1":
            d_int = np.nansum(d * p, axis=1)
            v_int = np.nansum(d * p**2, axis=1)
            n_int = np.nansum(n * p / dx1, axis=1)
            w_int = np.nanmean(w, axis=1)
        elif mode == "2d_2":
            d_int = np.nansum(d * p, axis=2)
            v_int = np.nansum(d * p**2, axis=2)
            n_int = np.nansum(n * p / dx2, axis=2)
            w_int = np.nanmean(w, axis=2)
        elif mode == "3d":
            d_int = d.copy()
            v_int = d.copy()
            n_int = n.copy()
            w_int = w.copy()

        return d_int, n_int, v_int, w_int

    def normalize(
        self, x0, x1, x2, d, n, w, c, inv_S, mode="3d", rel_err=30, perc=99.7
    ):
        d_int, n_int, v_int, w_int = self.profile_project(
            x0, x1, x2, d, n, w, c, inv_S, mode=mode
        )

        y_int, e_int = self.data_norm(d_int, n_int, v_int)

        return y_int, e_int / w_int

    def ellipsoid_covariance(self, inv_S, mode="3d", perc=99.7):
        if mode == "3d":
            scale = scipy.stats.chi2.ppf(perc / 100, df=3)
            inv_var = inv_S * scale
        elif mode == "2d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[1:, 1:] * scale
        elif mode == "2d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[0::2, 0::2] * scale
        elif mode == "2d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[:2, :2] * scale
        elif mode == "1d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[0, 0] * scale
        elif mode == "1d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[1, 1] * scale
        elif mode == "1d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[2, 2] * scale

        return inv_var

    def coerce_weight(self, target, weight=None):
        target = np.asarray(target)

        if weight is None:
            return np.ones_like(target, dtype=float)
        if np.isscalar(weight):
            return np.full_like(target, weight, dtype=float)

        return np.asarray(weight, dtype=float)

    def coerce_weights(self, targets, weights=None):
        if weights is None:
            weights = [None] * len(targets)

        return [
            self.coerce_weight(target, weight)
            for target, weight in zip(targets, weights)
        ]

    def uniform_mode_weights(self, ys, es, val=1.0):
        n_valid = sum(
            np.count_nonzero(np.isfinite(y) & np.isfinite(e) & (e > 0))
            for y, e in zip(ys, es)
        )

        scale = np.sqrt(val) / np.sqrt(max(n_valid, 1))

        return [np.full_like(e, scale, dtype=float) for e in es]

    def chi_2_fit(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
            m = 11
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
            m = 7
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
            m = 7
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
            m = 7
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
            m = 4
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
            m = 4
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2
            m = 4
            k = 1

        mask = (d2 <= 2 ** (2 / k)) & np.isfinite(e) & (e > 0)

        n = np.sum(mask)

        dof = n - m

        if dof <= 0:
            return np.inf
        else:
            return np.nansum(((y_fit[mask] - y[mask]) / e[mask]) ** 2) / dof

    def estimate_intensity(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == "3d":
            dx = np.prod([dx0, dx1, dx2])
            # k = 3
        elif mode == "2d_0":
            dx = np.prod([dx1, dx2])
            # k = 2
        elif mode == "2d_1":
            dx = np.prod([dx0, dx2])
            # k = 2
        elif mode == "2d_2":
            dx = np.prod([dx0, dx1])
            # k = 2
        elif mode == "1d_0":
            dx = dx0
            # k = 1
        elif mode == "1d_1":
            dx = dx1
            # k = 1
        elif mode == "1d_2":
            dx = dx2
            # k = 1

        pk = (d2 <= 1**2) & np.isfinite(y) & (e > 0)
        # bkg = (d2 > 1**2) & (d2 < 2 ** (2 / k)) & (e > 0)

        b = self.params["B{}".format(mode)].value
        b_err = self.params["B{}".format(mode)].stderr

        if b_err is None:
            b_err = b

        I = np.nansum(y[pk] - b) * dx
        sig = np.sqrt(np.nansum(e[pk] ** 2 + b_err**2)) * dx

        return I, sig

    def gaussian(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2

        return np.exp(-0.5 * d2)

    def inv_S_deriv_r(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2 / r0**3, 0, 0]) @ U.T
        dinv_S1 = U @ np.diag([0, -2 / r1**3, 0]) @ U.T
        dinv_S2 = U @ np.diag([0, 0, -2 / r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_u(self, r0, r1, r2, u0, u1, u2):
        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        U = self.U_matrix(u0, u1, u2)
        dU0, dU1, dU2 = self.U_deriv_u(u0, u1, u2)

        dinv_S0 = dU0 @ V @ U.T + U @ V @ dU0.T
        dinv_S1 = dU1 @ V @ U.T + U @ V @ dU1.T
        dinv_S2 = dU2 @ V @ U.T + U @ V @ dU2.T

        return dinv_S0, dinv_S1, dinv_S2

    def U_deriv_u(self, u0, u1, u2, delta=1e-6):
        dU0 = self.U_matrix(u0 + delta, u1, u2) - self.U_matrix(
            u0 - delta, u1, u2
        )
        dU1 = self.U_matrix(u0, u1 + delta, u2) - self.U_matrix(
            u0, u1 - delta, u2
        )
        dU2 = self.U_matrix(u0, u1, u2 + delta) - self.U_matrix(
            u0, u1, u2 - delta
        )

        return 0.5 * dU0 / delta, 0.5 * dU1 / delta, 0.5 * dU2 / delta

    def gaussian_integral(self, inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        return np.sqrt((2 * np.pi) ** k * det)

    def gaussian_integral_jac_S(self, inv_S, d_inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        g = np.sqrt((2 * np.pi) ** k * det)

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
        elif mode == "2d_0":
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g0 = g1 * 0
        elif mode == "2d_1":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g1 = g2 * 0
        elif mode == "2d_2":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = g0 * 0
        elif mode == "1d_0":
            g0 = d_inv_var[0] * inv_var
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            g1 = d_inv_var[1] * inv_var
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            g2 = d_inv_var[2] * inv_var
            g0 = g1 = g2 * 0

        return 0.5 * g * np.array([g0, g1, g2])

    def gaussian_jac_c(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g0 = np.zeros_like(g1)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g1 = np.zeros_like(g0)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1 = np.einsum("ij,j...->i...", inv_var, dx)
            g2 = np.zeros_like(g0)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = inv_var * dx
            g1 = g2 = np.zeros_like(g0)
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = inv_var * dx
            g2 = g0 = np.zeros_like(g1)
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = inv_var * dx
            g0 = g1 = np.zeros_like(g2)

        g = np.exp(-0.5 * d2)

        return g * np.array([g0, g1, g2])

    def gaussian_jac_S(self, x0, x1, x2, c, inv_S, d_inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g0 = g1 * 0
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g1 = g2 * 0
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = g0 * 0
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = d_inv_var[0] * dx**2
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = d_inv_var[1] * dx**2
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = d_inv_var[2] * dx**2
            g0 = g1 = g2 * 0

        g = np.exp(-0.5 * d2)

        return -0.5 * g * np.array([g0, g1, g2])

    def residual_1d(
        self, params, x0, x1, x2, ys, es, ws=None, c=None, inv_S=None
    ):
        y0, y1, y2 = ys
        e0, e1, e2 = es
        w0, w1, w2 = self.coerce_weights(ys, ws)

        A0 = params["A1d_0"]
        A1 = params["A1d_1"]
        A2 = params["A1d_2"]

        B0 = params["B1d_0"]
        B1 = params["B1d_1"]
        B2 = params["B1d_2"]

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "1d_0")
        y1_gauss = self.gaussian(*args, "1d_1")
        y2_gauss = self.gaussian(*args, "1d_2")

        diff = []

        y0_fit = A0 * y0_gauss + B0
        y1_fit = A1 * y1_gauss + B1
        y2_fit = A2 * y2_gauss + B2

        res = (y0_fit - y0) / e0 * w0

        diff += res.flatten().tolist()

        res = (y1_fit - y1) / e1 * w1

        diff += res.flatten().tolist()

        res = (y2_fit - y2) / e2 * w2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_1d(
        self,
        params,
        x0,
        x1,
        x2,
        ys,
        es,
        ws=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        params_list = [name for name, par in params.items()]

        y0, y1, y2 = ys
        e0, e1, e2 = es
        w0, w1, w2 = self.coerce_weights(ys, ws)

        A0 = params["A1d_0"]
        A1 = params["A1d_1"]
        A2 = params["A1d_2"]

        # B0 = params["B1d_0"]
        # B1 = params["B1d_1"]
        # B2 = params["B1d_2"]

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "1d_0")
        y1_gauss = self.gaussian(*args, "1d_1")
        y2_gauss = self.gaussian(*args, "1d_2")

        dA0 = y0_gauss / e0 * w0
        dA1 = y1_gauss / e1 * w1
        dA2 = y2_gauss / e2 * w2

        dB0 = 1 / e0 * w0
        dB1 = 1 / e1 * w1
        dB2 = 1 / e2 * w2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_0")
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_1")
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="1d_2")

        dc0_0, dc1_0, dc2_0 = (A0 * yc0_gauss) / e0 * w0
        dc0_1, dc1_1, dc2_1 = (A1 * yc1_gauss) / e1 * w1
        dc0_2, dc1_2, dc2_2 = (A2 * yc2_gauss) / e2 * w2

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_0")
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_1")
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="1d_2")

        dr0_0, dr1_0, dr2_0 = (A0 * yr0_gauss) / e0 * w0
        dr0_1, dr1_1, dr2_1 = (A1 * yr1_gauss) / e1 * w1
        dr0_2, dr1_2, dr2_2 = (A2 * yr2_gauss) / e2 * w2

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_0")
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_1")
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="1d_2")

        du0_0, du1_0, du2_0 = (A0 * yu0_gauss) / e0 * w0
        du0_1, du1_1, du2_1 = (A1 * yu1_gauss) / e1 * w1
        du0_2, du1_2, du2_2 = (A2 * yu2_gauss) / e2 * w2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)

        n01 = n0 + n1
        n012 = n01 + n2

        jac = np.zeros((n_params, n012))

        jac[params_list.index("A1d_0"), :n0] = dA0.flatten()
        jac[params_list.index("B1d_0"), :n0] = dB0.flatten()
        jac[params_list.index("c0"), :n0] = dc0_0.flatten()
        jac[params_list.index("c1"), :n0] = dc1_0.flatten()
        jac[params_list.index("c2"), :n0] = dc2_0.flatten()
        jac[params_list.index("r0"), :n0] = dr0_0.flatten()
        jac[params_list.index("r1"), :n0] = dr1_0.flatten()
        jac[params_list.index("r2"), :n0] = dr2_0.flatten()
        jac[params_list.index("u0"), :n0] = du0_0.flatten()
        jac[params_list.index("u1"), :n0] = du1_0.flatten()
        jac[params_list.index("u2"), :n0] = du2_0.flatten()

        jac[params_list.index("A1d_1"), n0:n01] = dA1.flatten()
        jac[params_list.index("B1d_1"), n0:n01] = dB1.flatten()
        jac[params_list.index("c0"), n0:n01] = dc0_1.flatten()
        jac[params_list.index("c1"), n0:n01] = dc1_1.flatten()
        jac[params_list.index("c2"), n0:n01] = dc2_1.flatten()
        jac[params_list.index("r0"), n0:n01] = dr0_1.flatten()
        jac[params_list.index("r1"), n0:n01] = dr1_1.flatten()
        jac[params_list.index("r2"), n0:n01] = dr2_1.flatten()
        jac[params_list.index("u0"), n0:n01] = du0_1.flatten()
        jac[params_list.index("u1"), n0:n01] = du1_1.flatten()
        jac[params_list.index("u2"), n0:n01] = du2_1.flatten()

        jac[params_list.index("A1d_2"), n01:n012] = dA2.flatten()
        jac[params_list.index("B1d_2"), n01:n012] = dB2.flatten()
        jac[params_list.index("c0"), n01:n012] = dc0_2.flatten()
        jac[params_list.index("c1"), n01:n012] = dc1_2.flatten()
        jac[params_list.index("c2"), n01:n012] = dc2_2.flatten()
        jac[params_list.index("r0"), n01:n012] = dr0_2.flatten()
        jac[params_list.index("r1"), n01:n012] = dr1_2.flatten()
        jac[params_list.index("r2"), n01:n012] = dr2_2.flatten()
        jac[params_list.index("u0"), n01:n012] = du0_2.flatten()
        jac[params_list.index("u1"), n01:n012] = du1_2.flatten()
        jac[params_list.index("u2"), n01:n012] = du2_2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        diff = np.concatenate(
            [(w / e).flatten() for w, e in zip((w0, w1, w2), es)]
        )

        mask = np.isfinite(diff)

        return jac[ind][:, mask]

    def residual_2d(
        self, params, x0, x1, x2, ys, es, ws=None, c=None, inv_S=None
    ):
        y0, y1, y2 = ys
        e0, e1, e2 = es
        w0, w1, w2 = self.coerce_weights(ys, ws)

        A0 = params["A2d_0"]
        A1 = params["A2d_1"]
        A2 = params["A2d_2"]

        B0 = params["B2d_0"]
        B1 = params["B2d_1"]
        B2 = params["B2d_2"]

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "2d_0")
        y1_gauss = self.gaussian(*args, "2d_1")
        y2_gauss = self.gaussian(*args, "2d_2")

        diff = []

        y0_fit = A0 * y0_gauss + B0
        y1_fit = A1 * y1_gauss + B1
        y2_fit = A2 * y2_gauss + B2

        res = (y0_fit - y0) / e0 * w0

        diff += res.flatten().tolist()

        res = (y1_fit - y1) / e1 * w1

        diff += res.flatten().tolist()

        res = (y2_fit - y2) / e2 * w2

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_2d(
        self,
        params,
        x0,
        x1,
        x2,
        ys,
        es,
        ws=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        params_list = [name for name, par in params.items()]

        y0, y1, y2 = ys
        e0, e1, e2 = es
        w0, w1, w2 = self.coerce_weights(ys, ws)

        A0 = params["A2d_0"]
        A1 = params["A2d_1"]
        A2 = params["A2d_2"]

        # B0 = params["B2d_0"]
        # B1 = params["B2d_1"]
        # B2 = params["B2d_2"]

        args = x0, x1, x2, c, inv_S

        y0_gauss = self.gaussian(*args, "2d_0")
        y1_gauss = self.gaussian(*args, "2d_1")
        y2_gauss = self.gaussian(*args, "2d_2")

        dA0 = y0_gauss / e0 * w0
        dA1 = y1_gauss / e1 * w1
        dA2 = y2_gauss / e2 * w2

        dB0 = 1 / e0 * w0
        dB1 = 1 / e1 * w1
        dB2 = 1 / e2 * w2

        yc0_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_0")
        yc1_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_1")
        yc2_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="2d_2")

        dc0_0, dc1_0, dc2_0 = (A0 * yc0_gauss) / e0 * w0
        dc0_1, dc1_1, dc2_1 = (A1 * yc1_gauss) / e1 * w1
        dc0_2, dc1_2, dc2_2 = (A2 * yc2_gauss) / e2 * w2

        yr0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_0")
        yr1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_1")
        yr2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="2d_2")

        dr0_0, dr1_0, dr2_0 = (A0 * yr0_gauss) / e0 * w0
        dr0_1, dr1_1, dr2_1 = (A1 * yr1_gauss) / e1 * w1
        dr0_2, dr1_2, dr2_2 = (A2 * yr2_gauss) / e2 * w2

        yu0_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_0")
        yu1_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_1")
        yu2_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="2d_2")

        du0_0, du1_0, du2_0 = (A0 * yu0_gauss) / e0 * w0
        du0_1, du1_1, du2_1 = (A1 * yu1_gauss) / e1 * w1
        du0_2, du1_2, du2_2 = (A2 * yu2_gauss) / e2 * w2

        n0, n1, n2, n_params = y0.size, y1.size, y2.size, len(params)

        n01 = n0 + n1
        n012 = n01 + n2

        jac = np.zeros((n_params, n012))

        jac[params_list.index("A2d_0"), :n0] = dA0.flatten()
        jac[params_list.index("B2d_0"), :n0] = dB0.flatten()
        jac[params_list.index("c0"), :n0] = dc0_0.flatten()
        jac[params_list.index("c1"), :n0] = dc1_0.flatten()
        jac[params_list.index("c2"), :n0] = dc2_0.flatten()
        jac[params_list.index("r0"), :n0] = dr0_0.flatten()
        jac[params_list.index("r1"), :n0] = dr1_0.flatten()
        jac[params_list.index("r2"), :n0] = dr2_0.flatten()
        jac[params_list.index("u0"), :n0] = du0_0.flatten()
        jac[params_list.index("u1"), :n0] = du1_0.flatten()
        jac[params_list.index("u2"), :n0] = du2_0.flatten()

        jac[params_list.index("A2d_1"), n0:n01] = dA1.flatten()
        jac[params_list.index("B2d_1"), n0:n01] = dB1.flatten()
        jac[params_list.index("c0"), n0:n01] = dc0_1.flatten()
        jac[params_list.index("c1"), n0:n01] = dc1_1.flatten()
        jac[params_list.index("c2"), n0:n01] = dc2_1.flatten()
        jac[params_list.index("r0"), n0:n01] = dr0_1.flatten()
        jac[params_list.index("r1"), n0:n01] = dr1_1.flatten()
        jac[params_list.index("r2"), n0:n01] = dr2_1.flatten()
        jac[params_list.index("u0"), n0:n01] = du0_1.flatten()
        jac[params_list.index("u1"), n0:n01] = du1_1.flatten()
        jac[params_list.index("u2"), n0:n01] = du2_1.flatten()

        jac[params_list.index("A2d_2"), n01:n012] = dA2.flatten()
        jac[params_list.index("B2d_2"), n01:n012] = dB2.flatten()
        jac[params_list.index("c0"), n01:n012] = dc0_2.flatten()
        jac[params_list.index("c1"), n01:n012] = dc1_2.flatten()
        jac[params_list.index("c2"), n01:n012] = dc2_2.flatten()
        jac[params_list.index("r0"), n01:n012] = dr0_2.flatten()
        jac[params_list.index("r1"), n01:n012] = dr1_2.flatten()
        jac[params_list.index("r2"), n01:n012] = dr2_2.flatten()
        jac[params_list.index("u0"), n01:n012] = du0_2.flatten()
        jac[params_list.index("u1"), n01:n012] = du1_2.flatten()
        jac[params_list.index("u2"), n01:n012] = du2_2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        diff = np.concatenate(
            [(w / e).flatten() for w, e in zip((w0, w1, w2), es)]
        )

        mask = np.isfinite(diff)

        return jac[ind][:, mask]

    def residual_3d(
        self, params, x0, x1, x2, y, e, w=None, c=None, inv_S=None
    ):
        A = params["A3d"]
        B = params["B3d"]

        w = self.coerce_weight(y, w)

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, "3d")

        diff = []

        y_fit = A * y_gauss + B

        res = (y_fit - y) / e * w

        diff += res.flatten().tolist()

        # ---

        diff = np.array(diff)

        mask = np.isfinite(diff)

        return diff[mask]

    def jacobian_3d(
        self,
        params,
        x0,
        x1,
        x2,
        y,
        e,
        w=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        params_list = [name for name, par in params.items()]

        A = params["A3d"]
        # B = params['B3d']

        w = self.coerce_weight(y, w)

        args = x0, x1, x2, c, inv_S

        y_gauss = self.gaussian(*args, "3d")

        dA = y_gauss / e * w

        dB = 1 / e * w

        yc_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode="3d")

        dc0, dc1, dc2 = (A * yc_gauss) / e * w

        yr_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode="3d")

        dr0, dr1, dr2 = (A * yr_gauss) / e * w

        yu_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode="3d")

        du0, du1, du2 = (A * yu_gauss) / e * w

        n, n_params = y.size, len(params)
        jac = np.zeros((n_params, n))

        jac[params_list.index("A3d"), :n] = dA.flatten()
        jac[params_list.index("B3d"), :n] = dB.flatten()
        jac[params_list.index("c0"), :n] = dc0.flatten()
        jac[params_list.index("c1"), :n] = dc1.flatten()
        jac[params_list.index("c2"), :n] = dc2.flatten()
        jac[params_list.index("r0"), :n] = dr0.flatten()
        jac[params_list.index("r1"), :n] = dr1.flatten()
        jac[params_list.index("r2"), :n] = dr2.flatten()
        jac[params_list.index("u0"), :n] = du0.flatten()
        jac[params_list.index("u1"), :n] = du1.flatten()
        jac[params_list.index("u2"), :n] = du2.flatten()

        # ---

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]

        mask = np.isfinite((w / e).flatten())

        return jac[ind][:, mask]

    def residual(self, params, args_1d, args_2d, args_3d):
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value

        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        cost_1d = self.residual_1d(params, *args_1d, c, inv_S)
        cost_2d = self.residual_2d(params, *args_2d, c, inv_S)
        cost_3d = self.residual_3d(params, *args_3d, c, inv_S)
        cost_prior = self.prior_residual(params)

        cost = np.concatenate([cost_1d, cost_2d, cost_3d, cost_prior])
        cost = np.nan_to_num(cost, nan=0.0, posinf=1e16, neginf=-1e16)

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value

        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)
        S = np.linalg.inv(inv_S)

        jac_1d = self.jacobian_1d(params, *args_1d, c, inv_S, dr, du)
        jac_2d = self.jacobian_2d(params, *args_2d, c, inv_S, dr, du)
        jac_3d = self.jacobian_3d(params, *args_3d, c, inv_S, dr, du)
        jac_prior = self.prior_jacobian(params, S, dr, du)

        jac = np.column_stack([jac_1d, jac_2d, jac_3d, jac_prior])
        jac = np.nan_to_num(jac, nan=0.0, posinf=1e16, neginf=-1e16)

        return jac.T

    def collect_mode_fit_metrics(self, x0, x1, x2, c, inv_S, mode_data):
        args = x0, x1, x2, c, inv_S

        metrics = {}

        for mode, (y, e) in mode_data.items():
            A = self.params["A" + mode].value
            B = self.params["B" + mode].value

            y_fit = A * self.gaussian(*args, mode) + B

            if mode == "3d":
                valid = np.isfinite(y) & (e >= 0)
            else:
                valid = np.isfinite(y) & (e > 0)

            y_fit[~valid] = np.nan

            fit_triplet = (y_fit, y, e)

            chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, *fit_triplet, mode)
            I0, s0 = self.estimate_intensity(
                x0, x1, x2, c, inv_S, *fit_triplet, mode
            )

            metrics[mode] = [I0, s0, chi2, fit_triplet]

        return metrics

    def extract_result(self, args_1d, args_2d, args_3d, xmod):
        x0, x1, x2, y1d, e1d, w1d = args_1d
        x0, x1, x2, y2d, e2d, w2d = args_2d
        x0, x1, x2, y3d, e3d, w3d = args_3d

        y1d_0, y1d_1, y1d_2 = y1d
        y2d_0, y2d_1, y2d_2 = y2d

        e1d_0, e1d_1, e1d_2 = e1d
        e2d_0, e2d_1, e2d_2 = e2d

        self.redchi2 = []
        self.intensity = []
        self.sigma = []

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c0_err = self.params["c0"].stderr
        c1_err = self.params["c1"].stderr
        c2_err = self.params["c2"].stderr

        if c0_err is None:
            c0_err = c0
        if c1_err is None:
            c1_err = c1
        if c2_err is None:
            c2_err = c2

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c_err = c0_err, c1_err, c2_err

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        mode_1d = {
            "1d_0": (y1d_0, e1d_0),
            "1d_1": (y1d_1, e1d_1),
            "1d_2": (y1d_2, e1d_2),
        }
        metrics_1d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_1d
        )

        modes = ["1d_0", "1d_1", "1d_2"]

        y1 = [metrics_1d[mode][3] for mode in modes]
        self.redchi2.append([metrics_1d[mode][2] for mode in modes])
        self.intensity.append([metrics_1d[mode][0] for mode in modes])
        self.sigma.append([metrics_1d[mode][1] for mode in modes])

        # ---

        mode_2d = {
            "2d_0": (y2d_0, e2d_0),
            "2d_1": (y2d_1, e2d_1),
            "2d_2": (y2d_2, e2d_2),
        }
        metrics_2d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_2d
        )

        modes = ["2d_0", "2d_1", "2d_2"]

        y2 = [metrics_2d[mode][3] for mode in modes]
        self.redchi2.append([metrics_2d[mode][2] for mode in modes])
        self.intensity.append([metrics_2d[mode][0] for mode in modes])
        self.sigma.append([metrics_2d[mode][1] for mode in modes])

        # ---

        mode_3d = {"3d": (y3d, e3d)}
        metrics_3d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_3d
        )

        y3 = metrics_3d["3d"][3]
        self.redchi2.append(metrics_3d["3d"][2])
        self.intensity.append(metrics_3d["3d"][0])
        self.sigma.append(metrics_3d["3d"][1])

        self.fit_metrics = {
            **{
                mode: {
                    "I0": values[0],
                    "s0": values[1],
                    "chi2": values[2],
                }
                for mode, values in metrics_1d.items()
            },
            **{
                mode: {
                    "I0": values[0],
                    "s0": values[1],
                    "chi2": values[2],
                }
                for mode, values in metrics_2d.items()
            },
            "3d": {
                "I0": metrics_3d["3d"][0],
                "s0": metrics_3d["3d"][1],
                "chi2": metrics_3d["3d"][2],
            },
        }

        if not np.linalg.det(inv_S) > 0:
            print("Improper optimal covariance")
            return None

        S = np.linalg.inv(inv_S)

        V, W = np.linalg.eigh(S)

        c0, c1, c2 = c

        c0 += xmod
        c = c0, c1, c2

        self.estimated_fit[0][0] += xmod

        r0, r1, r2 = np.sqrt(V)

        v0, v1, v2 = W.T

        fitting = (x0 + xmod, x1, x2, *y3)

        self.peak_pos = c, c_err

        self.best_fit = c, S, *fitting

        self.best_prof = (
            (x0[:, 0, 0] + xmod, *y1[0]),
            (x1[0, :, 0], *y1[1]),
            (x2[0, 0, :], *y1[2]),
        )

        self.best_proj = (
            (x1[0, :, :], x2[0, :, :], *y2[0]),
            (x0[:, 0, :] + xmod, x2[:, 0, :], *y2[1]),
            (x0[:, :, 0] + xmod, x1[:, :, 0], *y2[2]),
        )

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def extract_amplitude_background(self):
        A0 = self.params["A1d_0"]
        A1 = self.params["A1d_1"]
        A2 = self.params["A1d_2"]

        B0 = self.params["B1d_0"]
        B1 = self.params["B1d_1"]
        B2 = self.params["B1d_2"]

        return np.array([A0, A1, A2]), np.array([B0, B1, B2])

    def estimate_center_weighted(
        self, d, n, kernel, frac=0.9, min_weight=1e-12
    ):
        d = np.asarray(d, dtype=float)
        n = np.asarray(n, dtype=float)
        k = np.asarray(kernel, dtype=float)

        valid = np.isfinite(d) & np.isfinite(n) & (n > 0)

        d0 = np.where(valid, d, 0.0)
        n0 = np.where(valid, n, 0.0)

        ybar = np.sum(d0) / np.sum(n0)

        r = d0 - ybar * n0

        num = scipy.signal.correlate(r, k, mode="same", method="direct")
        den = scipy.signal.correlate(n0, k**2, mode="same", method="direct")

        c = np.divide(
            num,
            np.sqrt(den),
            out=np.full_like(num, np.nan),
            where=den > min_weight,
        )

        i0 = len(d) // 2

        i = np.nanargmax(c)
        if i < len(d) // 8 or i > 7 * len(d) // 8:
            return i0

        return i

    def quick_gaussian(self, x0, x1, x2, d, n, c, inv_S, mode="3d"):
        mask = n > 0

        if mask.sum() <= 3:
            return None

        w = np.ones_like(n)

        d_int, n_int, *_ = self.profile_project(
            x0, x1, x2, d, n, w, c, inv_S, mode=mode
        )

        y = d / n
        y[~mask] = np.nan

        B = np.nanpercentile(y, 5)
        A = np.nanpercentile(y, 95) - B

        if A <= 0 or not np.isfinite(A):
            A = 1
        if B <= 0 or not np.isfinite(B):
            B = 1

        self.params.add("A" + mode, value=A, min=0, max=np.inf)
        self.params.add("B" + mode, value=B, min=0, max=np.inf)

        return A > B

    def estimate_envelope(
        self,
        x0,
        x1,
        x2,
        d_int,
        n_int,
        wgt,
        report_fit=False,
    ):
        d = d_int.copy()
        n = n_int.copy()

        e = np.sqrt(np.clip(d, 0, None) + 1) / n

        if self.combine_params is not None:
            self.params = self.combine_params.copy()

        if (np.array(e.shape) < 3).any():
            return None

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        y1d_0, e1d_0 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="1d_0"
        )
        y1d_1, e1d_1 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="1d_1"
        )
        y1d_2, e1d_2 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="1d_2"
        )

        est1d_0 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="1d_0")
        est1d_1 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="1d_1")
        est1d_2 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="1d_2")

        if est1d_0 is None or est1d_1 is None or est1d_2 is None:
            return None

        y1d = [y1d_0, y1d_1, y1d_2]
        e1d = [e1d_0, e1d_1, e1d_2]
        w1d = self.uniform_mode_weights(y1d, e1d, 0.05)

        args_1d = [x0, x1, x2, y1d, e1d, w1d]

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        y2d_0, e2d_0 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="2d_0"
        )
        y2d_1, e2d_1 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="2d_1"
        )
        y2d_2, e2d_2 = self.normalize(
            x0, x1, x2, d, n, wgt, c, inv_S, mode="2d_2"
        )

        est2d_0 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="2d_0")
        est2d_1 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="2d_1")
        est2d_2 = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="2d_2")

        if est2d_0 is None or est2d_1 is None or est2d_2 is None:
            return None

        y2d = [y2d_0, y2d_1, y2d_2]
        e2d = [e2d_0, e2d_1, e2d_2]
        w2d = self.uniform_mode_weights(y2d, e2d, 0.2)

        args_2d = [x0, x1, x2, y2d, e2d, w2d]

        y3d, e3d = self.normalize(x0, x1, x2, d, n, wgt, c, inv_S, mode="3d")

        w3d = self.uniform_mode_weights([y3d], [e3d])[0]

        est3d = self.quick_gaussian(x0, x1, x2, d, n, c, inv_S, mode="3d")

        if est3d is None:
            return None

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        args_3d = [x0, x1, x2, y3d, e3d, w3d]

        protocol = [False] * 9
        n_iter = 10

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        lamda_1d, lamda_2d = self.update_adaptive_prior(
            args_1d, args_2d, args_3d
        )

        w1d = self.uniform_mode_weights(y1d, e1d, lamda_1d)
        w2d = self.uniform_mode_weights(y2d, e2d, lamda_2d)

        args_1d = [x0, x1, x2, y1d, e1d, w1d]
        args_2d = [x0, x1, x2, y2d, e2d, w2d]

        # ---

        protocol = [True] * 3 + [False] * 3 + [False] * 3
        n_iter = 20

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        protocol = [False] * 3 + [True] * 3 + [False] * 3
        n_iter = 20

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        amplitude, background = self.estimate_peak_strength()

        if np.all(amplitude > background):
            protocol = [False] * 3 + [True] * 3 + [True] * 3
            n_iter = 20

            self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        return args_1d, args_2d, args_3d

    def estimate_peak_strength(self):
        amplitude, background = self.extract_amplitude_background()

        amplitude = np.array([param for param in amplitude], dtype=float)
        background = np.array([param for param in background], dtype=float)

        return amplitude, background

    def update_adaptive_prior(self, args_1d, args_2d, args_3d):
        x0, x1, x2, y1d, e1d, w1d = args_1d
        x0, x1, x2, y2d, e2d, w2d = args_2d
        x0, x1, x2, y3d, e3d, w3d = args_3d

        y1d_0, y1d_1, y1d_2 = y1d
        y2d_0, y2d_1, y2d_2 = y2d

        e1d_0, e1d_1, e1d_2 = e1d
        e2d_0, e2d_1, e2d_2 = e2d

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        mode_1d = {
            "1d_0": (y1d_0, e1d_0),
            "1d_1": (y1d_1, e1d_1),
            "1d_2": (y1d_2, e1d_2),
        }
        metrics_1d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_1d
        )

        mode_2d = {
            "2d_0": (y2d_0, e2d_0),
            "2d_1": (y2d_1, e2d_1),
            "2d_2": (y2d_2, e2d_2),
        }
        metrics_2d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_2d
        )

        mode_3d = {"3d": (y3d, e3d)}
        metrics_3d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_3d
        )

        I1d_0, e1d_0, chi2_1d_0, _ = metrics_1d["1d_0"]
        I1d_1, e1d_1, chi2_1d_1, _ = metrics_1d["1d_1"]
        I1d_2, e1d_2, chi2_1d_2, _ = metrics_1d["1d_2"]

        I2d_0, e2d_0, chi2_2d_0, _ = metrics_2d["2d_0"]
        I2d_1, e2d_1, chi2_2d_1, _ = metrics_2d["2d_1"]
        I2d_2, e2d_2, chi2_2d_2, _ = metrics_2d["2d_2"]

        I3d, e3d, chi2_3d, _ = metrics_3d["3d"]

        scale_1d = chi2_3d / np.nanmedian([chi2_1d_0, chi2_1d_1, chi2_1d_2])
        scale_2d = chi2_3d / np.nanmedian([chi2_2d_0, chi2_2d_1, chi2_2d_2])

        sn_1d = np.nanmedian([I1d_0 / e1d_0, I1d_1 / e1d_1, I1d_2 / e1d_2])
        sn_2d = np.nanmedian([I2d_0 / e2d_0, I2d_1 / e2d_1, I2d_2 / e2d_2])
        sn_3d = I3d / e3d

        scale = 1 / (1 + np.log10(sn_3d))

        self.lamda_center = 1 if sn_1d < 5 else scale
        self.lamda_cov = 1 if sn_2d < 5 else scale

        self.lamda_center /= 3
        self.lamda_cov /= 6

        return scale_1d, scale_2d

    def sweep(
        self, args_1d, args_2d, args_3d, protocol, n_iter=50, report_fit=False
    ):
        self.params["c0"].set(vary=protocol[0])
        self.params["c1"].set(vary=protocol[1])
        self.params["c2"].set(vary=protocol[2])

        self.params["r0"].set(vary=protocol[3])
        self.params["r1"].set(vary=protocol[4])
        self.params["r2"].set(vary=protocol[5])

        self.params["u0"].set(vary=protocol[6])
        self.params["u1"].set(vary=protocol[7])
        self.params["u2"].set(vary=protocol[8])

        out = Minimizer(
            self.residual,
            self.params,
            fcn_args=(args_1d, args_2d, args_3d),
            nan_policy="omit",
        )

        result = out.minimize(
            method="least_squares",
            jac=self.jacobian,
            max_nfev=n_iter,
        )

        if report_fit:
            print(fit_report(result))

        self.params = result.params

    def calculate_intensity(self, A, H, r0, r1, r2, u0, u1, u2, mode="3d"):
        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)
        g = self.gaussian_integral(inv_S, mode)

        return A * g

    def voxels(self, x0, x1, x2):
        return (
            x0[1, 0, 0] - x0[0, 0, 0],
            x1[0, 1, 0] - x1[0, 0, 0],
            x2[0, 0, 1] - x2[0, 0, 0],
        )

    def voxel_volume(self, x0, x1, x2):
        return np.prod(self.voxels(x0, x1, x2))

    def fit(
        self,
        x0_prof,
        x1_proj,
        x2_proj,
        d,
        n,
        dx,
        xmod,
        voxel_weights,
    ):
        fit_start = time.perf_counter()

        def log_fit_time(outcome):
            elapsed = time.perf_counter() - fit_start
            print("fit [{}] {:.3f}s".format(outcome, elapsed))

        x0 = x0_prof - xmod
        x1 = x1_proj.copy()
        x2 = x2_proj.copy()

        d_val = d.copy()
        n_val = n.copy()

        y = d_val / n_val
        e = np.sqrt(np.clip(d_val, 0, None) + 1) / n_val

        if (np.array(y.shape) <= 3).any():
            log_fit_time("small-shape")
            return None

        y_max = np.nanmax(y)

        det_mask = n_val > 0

        if y_max <= 0 or np.sum(det_mask) < 11:
            log_fit_time("insufficient-data")
            return None

        coords = np.argwhere(det_mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0) + 1

        y = y[i0:j0, i1:j1, i2:j2].copy()
        e = e[i0:j0, i1:j1, i2:j2].copy()

        d_val = d_val[i0:j0, i1:j1, i2:j2].copy()
        n_val = n_val[i0:j0, i1:j1, i2:j2].copy()

        if (np.array(y.shape) <= 3).any():
            log_fit_time("small-cropped-shape")
            return None

        x0 = x0[i0:j0, i1:j1, i2:j2].copy()
        x1 = x1[i0:j0, i1:j1, i2:j2].copy()
        x2 = x2[i0:j0, i1:j1, i2:j2].copy()

        voxel_weights = voxel_weights[i0:j0, i1:j1, i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print("Invalid data")
            log_fit_time("invalid-data")
            return None

        weights = None
        try:
            weights = self.estimate_envelope(
                x0, x1, x2, d_val, n_val, voxel_weights
            )
        except Exception as e:
            print("Exception estimating envelope: {}".format(e))
            log_fit_time("estimate-error")
            return None

        if weights is None:
            print("Invalid weight estimate")
            log_fit_time("invalid-weights")
            return None

        self.copy_combine()

        log_fit_time("ok")
        return weights

    def peak_roi(self, x0, x1, x2, c, S, scale, p=0.997):
        c0, c1, c2 = c

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        x = np.array([x0 - c0, x1 - c1, x2 - c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum("ij,jklm,iklm->klm", S_inv, x, x)

        mask = ellipsoid <= 1

        structure = np.ones((3, 3, 3), dtype=bool)

        pk = scipy.ndimage.binary_dilation(mask, structure=structure)

        # bound = np.zeros_like(pk, dtype=bool)
        # bound[0, :, :] = True
        # bound[-1, :, :] = True
        # bound[:, 0, :] = True
        # bound[:, -1, :] = True
        # bound[:, :, 0] = True
        # bound[:, :, -1] = True

        # clip = pk & (~bound)
        # pk = clip.copy()

        mask = scipy.ndimage.binary_dilation(pk, structure=structure)

        n_pk = np.sum(pk)

        for i in range(3):
            bkg = mask & (~pk)

            n_bkg = np.sum(bkg)

            if n_bkg < 2 * n_pk:
                mask = scipy.ndimage.binary_dilation(mask, structure=structure)

        scale = scipy.stats.chi2.ppf(p, df=3)

        inv_var = scale * S_inv

        d2 = np.einsum("i...,ij,j...->...", x, inv_var, x)

        det = 1 / np.linalg.det(inv_var)

        y = np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** 3 * det)

        return pk, bkg, mask, y

    def extract_raw_intensity(self, counts, pk, bkg):
        d = counts.copy()
        d[np.isinf(d)] = np.nan

        d_pk = d[pk].copy()
        d_bkg = d[bkg].copy()

        pk_intens = np.nansum(d_pk)
        pk_err = np.sqrt(pk_intens)

        bkg_intens = np.nansum(d_bkg)
        bkg_err = np.sqrt(bkg_intens)

        vol_pk = pk.sum()
        vol_bkg = bkg.sum()

        ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

        intens = pk_intens - ratio * bkg_intens
        sig = np.sqrt(pk_err**2 + ratio**2 * bkg_err**2)

        if not sig > 0:
            sig = float("inf")

        return intens, sig

    def extract_intensity(self, d, n, pk, bkg, kernel):
        core = pk & (n > 0)
        shell = bkg & (n > 0)

        # d_min, d_max = 0, np.inf
        # if np.sum(shell) > 0:
        #     d_med = np.nanmedian(d[shell])
        #     d_mad = np.nanmedian(np.abs(d[shell] - d_med))
        #     k = 1 / scipy.stats.norm.ppf(0.75)
        #     d_min, d_max = d_med - k * d_mad, d_med + k * d_mad

        shell = bkg & (n >= 0)  # & (d >= d_min) & (d <= d_max)

        d_pk = d[core].copy()
        d_bkg = d[shell].copy()

        n_pk = n[pk].copy()
        n_bkg = n[bkg].copy()

        k_pk = kernel[pk]
        k_bkg = kernel[bkg]

        bkg_cnts = np.nansum(d_bkg)
        bkg_norm = np.nansum(n_bkg * k_bkg)

        if bkg_cnts == 0.0:
            bkg_cnts = float("nan")
        if bkg_norm == 0.0:
            bkg_norm = float("nan")

        b = bkg_cnts / bkg_norm
        b_err = np.sqrt(bkg_cnts) / bkg_norm

        if not np.isfinite(b):
            b = 0
        if not np.isfinite(b_err):
            b_err = 0

        pk_cnts = np.nansum(d_pk)
        pk_norm = np.nansum(n_pk * k_pk)

        if pk_cnts == 0.0:
            pk_cnts = float("nan")
        if pk_cnts == 0.0:
            pk_norm = float("nan")

        vol_pk = float(core.sum())
        vol_bkg = float(shell.sum())

        vol = k_pk.sum()

        ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

        pk_intens = pk_cnts
        bkg_intens = bkg_cnts

        pk_err = np.sqrt(pk_cnts)
        bkg_err = np.sqrt(bkg_cnts)

        raw_intens = pk_intens - ratio * bkg_intens
        raw_sig = np.sqrt(pk_err**2 + ratio**2 * bkg_err**2)

        intens = vol * raw_intens / pk_norm
        sig = vol * raw_sig / pk_norm

        if not sig > 0:
            sig = float("-inf")

        data_norm = pk_cnts, pk_norm, bkg_cnts, bkg_norm, ratio

        return intens, sig, b, b_err, vol, *data_norm

    def matched_filter(self, d, n, v, pk, bkg, kernel, rel_error=0.15):
        mask = (pk | bkg) & (n > 0)
        p = kernel.copy()
        p[~mask] = np.nan

        b = np.nanquantile(d[mask], rel_error)

        e = np.sqrt(v + b)
        e[np.isclose(d, 0)] = 1

        q = n * p
        w = 1 / e**2

        Sqq = np.nansum(w * q**2)
        Sqn = np.nansum(w * q * n)
        Snn = np.nansum(w * n**2)

        Tq = np.nansum(w * q * d)
        Tn = np.nansum(w * n * d)

        den = Sqq * Snn - Sqn**2

        I = (Snn * Tq - Sqn * Tn) / den
        b = (Sqq * Tn - Sqn * Tq) / den
        sig = np.sqrt(Snn / den)

        A = I * np.nanmax(p)

        return I, sig, A, b

    def fitted_profile(self, x0, x1, x2, d, n, c, S, p=0.997, eta=0.5):
        scale = np.sqrt(scipy.stats.chi2.ppf(p, df=1))

        c0, c1, c2 = c

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        C = S.copy()

        sigma = np.sqrt(C[0, 0]) / scale

        S_inv = np.linalg.inv(C)

        weights = np.ones_like(n)

        d_int, n_int, v_int, _ = self.profile_project(
            x0, x1, x2, d, n, weights, c, S_inv, mode="1d_0"
        )

        y = d_int / n_int
        e = np.sqrt(np.clip(d_int, 0, None) + 1) / n_int

        x = x0[:, 0, 0] - c0

        for i in range(3):
            norm = np.sqrt(2 * np.pi) * sigma

            kernel = np.exp(-0.5 * (x / sigma) ** 2) / norm

            pk = np.abs(x) < scale * sigma
            bkg = np.abs(x) >= scale * sigma

            I, I_err, A, b = self.matched_filter(
                d_int, n_int, v_int, pk, bkg, kernel
            )

            w = np.clip(y - b, 0, None)

            wgt = np.nansum(w)

            if wgt > 0:
                sigma_hat = np.clip(np.nansum(w * x**2) / wgt, dx0, sigma)
                sigma = (1 - eta) * sigma + eta * sigma_hat

        y_fit = I * kernel + b

        return I, I_err, b, x, y_fit, y, e

    def integrate(self, x0, x1, x2, d, n, c, S):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d3x = self.voxel_volume(x0, x1, x2)

        d[np.isinf(d)] = np.nan
        n[np.isinf(n)] = np.nan

        pk, bkg, mask, kernel = self.peak_roi(x0, x1, x2, c, S, 1)

        result = self.extract_intensity(d, n, pk, bkg, kernel)

        intens, sig, b, b_err, N, *data_norm = result

        pk_data, pk_norm, bkg_data, bkg_norm, ratio = data_norm

        intens *= d3x
        sig *= d3x

        self.intensity.append(intens)
        self.sigma.append(sig)

        self.weights = (x0[pk], x1[pk], x2[pk]), d[pk].copy()

        self.info = [d3x, b, b_err]

        y = d / n
        e = np.sqrt(np.clip(d, 0, None) + 1) / n

        intens_raw, sig_raw = self.extract_raw_intensity(d, pk, bkg)

        self.info += [intens_raw, sig_raw]

        self.info += [N, pk_data, pk_norm, bkg_data, bkg_norm, ratio]

        if not np.isfinite(sig):
            sig = float("inf")

        xye = (x0, x1, x2), (dx0, dx1, dx2), y, e

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        self.peak_background_mask = x0, x1, x2, pk, bkg

        result = self.fitted_profile(x0, x1, x2, d, n, c, S)

        I, I_err, b, x, y_fit, y, e = result

        self.integral = x, y_fit, y, e

        result = self.matched_filter(d, n, d, pk, bkg, kernel)

        I_filt, sig_filt, A_filt, b_filt = result

        chi2_3d = self.redchi2[-1]

        if np.isfinite(chi2_3d) and chi2_3d > 1:
            sig_filt = sig_filt * np.sqrt(chi2_3d)
            result = I_filt, sig_filt, A_filt, b_filt

        self.filter = result

        self.intensity.append(I)
        self.sigma.append(I_err)

        return intens, sig if I_filt > sig_filt else intens
