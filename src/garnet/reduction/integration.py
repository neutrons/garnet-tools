import os
import gc
import subprocess
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

from garnet.plots.peaks import PeakPlot, ScanPlot
from garnet.config.instruments import beamlines
from garnet.reduction.ub import UBModel, Optimization, lattice_group
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.ellipsoid import PeakEllipsoid
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

            # ---

            r_cut = self.params["Radius"]

            data.convert_to_Q_sample("data", "md", lorentz_corr=True)

            peaks.predict_peaks(
                "data",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            ub = UBModel("peaks")

            Q_min, hkl_tol = ub.shortest_reciprocal_spacing(centering)

            result = peaks.scan_threshold("md", "peaks", Q_min)

            scan_file = self.get_plot_file("run#{}_scan".format(run))

            scan_plot = ScanPlot(*result)
            scan_plot.save_plot(scan_file)

            opt = Optimization("peaks", hkl_tol)
            opt.optimize_lattice(cell)

            ub_file = self.get_diagnostic_file("run#{}_ub".format(run))
            ub_file = os.path.splitext(ub_file)[0] + ".mat"

            ub = UBModel("peaks")
            ub.save_UB(ub_file)

            data.load_clear_UB(ub_file, "data", run)

            # ---

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

            self.res_sigma = res.estimate_prior_sigmas()

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
            ellipsoid.set_resolution_sigma(*self.res_sigma)

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
