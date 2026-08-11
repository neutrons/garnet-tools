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
from garnet.reduction.ub import (
    UBModel,
    Optimization,
    Reorient,
    lattice_group,
    write_ub_info,
)
from garnet.reduction.peaks import PeaksModel, PeakModel, centering_reflection
from garnet.reduction.ellipsoid import PeakEllipsoid
from garnet.reduction.resolution import (
    ResolutionEllipsoid,
    _plot_peak_shape_diagnostics,
)
from garnet.reduction.data import DataModel
from garnet.reduction.projection import PeakProjection

INTEGRATION = os.path.abspath(__file__)
directory = os.path.dirname(INTEGRATION)

filename = os.path.join(directory, "../utilities/structure.py")
REFLECTIONS = os.path.abspath(filename)

assert os.path.exists(REFLECTIONS)


class Integration(PeakProjection):
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
        if self.params.get("OptimizeUB") is None:
            self.params["OptimizeUB"] = False
        if self.params.get("OptimizePeaks") is None:
            self.params["OptimizePeaks"] = False
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
        self.check(
            type(self.params["ProfileFit"]),
            "is",
            bool,
            "ProfileFit must be a boolean",
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

            mat_file = os.path.splitext(file)[0] + ".mat"

            if os.path.exists(mat_file):
                os.remove(mat_file)

        peaks.reset_satellites("combine")

        if data.workspace_exists("combine"):
            peaks.save_peaks(result_file, "combine")

            opt = Optimization("combine")
            opt.optimize_lattice(self.params["Cell"])

            ub_file = os.path.splitext(result_file)[0] + ".mat"

            ub = UBModel("combine")
            ub.save_UB(ub_file)

            self.cleanup()
            self.write(result_file)
        else:
            self.cleanup()

    def write(self, result_file):
        try:
            process = subprocess.Popen(
                ["python", REFLECTIONS, self.plan["YAML"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = process.communicate()
            if process.returncode == 0:
                print("First command succeeded:", out.decode().strip())
            else:
                raise subprocess.SubprocessError(err.decode().strip())
        except (FileNotFoundError, subprocess.SubprocessError):
            subprocess.Popen(["python", REFLECTIONS, self.plan["YAML"]])

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

            data.load_clear_UB(self.plan["UBFile"], "data", run)

            data.load_background(
                self.plan.get("BackgroundFile"),
                "data",
                self.plan.get("DetectorCalibration"),
                self.plan.get("TubeCalibration"),
                remove=False,
            )

            lamda_min, lamda_max = data.wavelength_band

            d_min = self.params["MinD"]

            centering = self.params["Centering"]

            cell = self.params["Cell"]

            self.cntrt = data.get_counting_rate("data")

            # ---

            r_cut = self.params["Radius"]

            data.subtract_background("data")

            app = "_sub" if data.workspace_exists("data_sub") else ""

            data.convert_to_Q_sample("data" + app, "md", lorentz_corr=True)

            self.predict_all_peaks(
                "data" + app,
                "md",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            peak = PeakModel("peaks")

            self.orig_d = {
                peak.get_hklmnp(i): peak.get_d_from_ub(i)
                for i in range(peak.get_number_peaks())
            }

            if self.params["OptimizePeaks"]:
                self.optimize_peaks(
                    "data" + app, "md", "peaks", centering, cell, run
                )

            # md_file = self.get_diagnostic_file("run#{}_data".format(run))
            # data.save_histograms(md_file, "md")

            self.data = data

            div_params = beamlines[self.plan["Instrument"]].get(
                "DivergenceParams"
            )

            self.prior_res = None
            if div_params is not None:
                self.prior_res = ResolutionEllipsoid(
                    "peaks", r_cut=r_cut, mosaic="isotropic"
                )
                self.prior_res.set_variance_parameters_deg(div_params)

                peaks.stash_run_number("peaks")

                radii = self.prior_res.renumber_by_size(5)

                peaks.integrate_peaks_with_radii(
                    "md", "peaks", radii, centroid=True, update=True
                )

                peaks.restore_run_number("peaks")
            else:
                peaks.integrate_peaks(
                    "md",
                    "peaks",
                    r_cut / np.cbrt(3),
                    centroid=True,
                    update=True,
                )

            if self.params["OptimizeUB"]:
                self.optimize_ub("data" + app, "md", "peaks", cell, run)

            res = ResolutionEllipsoid("peaks", r_cut=r_cut, mosaic="isotropic")
            fixed_instrumental = (
                self.prior_res.model if self.prior_res is not None else None
            )
            res.fit(fixed_instrumental=fixed_instrumental)

            if res.model is not None:
                res_file = self.get_plot_file("run#{}_res".format(run))
                res.plot_diagnostics(res_file)

                self.plot_peak_shape_diagnostics("peaks", res, r_cut, 21)

            self.predict_all_peaks(
                "data" + app,
                "md",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            data.bin_Q_sample_to_hkl("md", "peaks", "hist", d_min=d_min)

            hist_file = self.get_diagnostic_file("run#{}_hist".format(run))
            data.save_histograms(hist_file, "hist")
            data.delete_workspace("hist")

            pk_file = self.get_diagnostic_file("run#{}_peaks".format(run))
            peaks.save_peaks(pk_file, "peaks")

            self.predict_all_peaks(
                "data" + app,
                "md",
                "peaks",
                centering,
                d_min,
                lamda_min,
                lamda_max,
            )

            if res.model is not None:
                res.apply()

            data.delete_workspace("md")

            ub = UBModel("peaks")
            self.P = ub.centering_matrix(centering)

            self.peaks, self.data = peaks, data
            self.r_cut = r_cut

            if self.params["ProfileFit"]:
                banks = peaks.get_bank_names("peaks")

                ub = UBModel("peaks")

                ub.copy_UB("data")

                for bank in banks:
                    if self.make_plot:
                        self.peak_plot = PeakPlot()

                    data.mask_to_bank("data", bank)

                    data.preprocess_detector_banks(bank)

                    data.convert_to_Q_sample(bank, bank, False, bank + "_dets")

                    peak_dict = self.extract_peak_info(
                        "peaks", r_cut, bank=bank
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
            else:
                data.convert_to_Q_sample("data", "md", lorentz_corr=False)

                peaks.stash_run_number("peaks")

                radii = res.renumber_by_size(5)

                peaks.integrate_peaks_with_radii(
                    "md",
                    "peaks",
                    radii,
                    centroid=False,
                    update=False,
                )

                peaks.restore_run_number("peaks")

                data.delete_workspace("md")

                self.stub_peak_info("peaks")
                self.correct_intensities("peaks")

            peaks.update_scale_factor("peaks", data.monitor)

            peaks.combine_peaks("peaks", "combine")

            pk_file = self.get_diagnostic_file("run#{}_integrate".format(run))

            peaks.save_peaks(pk_file, "peaks")

            data.delete_workspace("data")

            data.delete_workspace("data_sub")

            data.delete_workspace("peaks")

        peaks.remove_weak_peaks("combine", -100)

        peaks.save_peaks(result_file, "combine")

        # ---

        mtd.clear()

        return result_file

    def predict_add_satellite_peaks(
        self, peaks_ws, md_ws, lamda_min, lamda_max
    ):
        if self.params["MaxOrder"] > 0:
            sat_min_d = self.params["MinD"]
            if self.params.get("SatMinD") is not None:
                sat_min_d = self.params["SatMinD"]

            peaks = PeaksModel()
            peaks.predict_satellite_peaks(
                peaks_ws,
                md_ws,
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

    def predict_all_peaks(
        self, data, md, peaks_ws, centering, d_min, lamda_min, lamda_max
    ):
        peaks = PeaksModel()
        peaks.predict_peaks(
            data, peaks_ws, centering, d_min, lamda_min, lamda_max
        )

        self.predict_add_satellite_peaks(peaks_ws, md, lamda_min, lamda_max)

    def optimize_ub(self, data, md, peaks_ws, cell, run):
        opt = Optimization(peaks_ws, tol=0.15)
        opt.optimize_lattice(cell)

        ub_file = self.get_diagnostic_file("run#{}_ub".format(run))
        ub_file = os.path.splitext(ub_file)[0] + ".mat"

        ub = UBModel(peaks_ws)
        ub.save_UB(ub_file)

        info_file = self.get_diagnostic_file("run#{}_ub".format(run), ".txt")
        write_ub_info(info_file, run, self.params["MinD"], opt, ub)

        ub.copy_UB(data)
        ub.copy_UB(md)

    def optimize_peaks(
        self, data, md, peaks_ws, centering, cell, run, reindex=False
    ):
        ub = UBModel(peaks_ws)

        peaks = PeaksModel()

        UB = ub.get_UB()

        min_d, max_d = ub.get_primitive_cell_length_range(centering)

        Q_min, _ = ub.shortest_reciprocal_spacing(centering)

        result = peaks.scan_threshold(md, peaks_ws, Q_min)

        scan_file = self.get_plot_file("run#{}_scan".format(run))

        scan_plot = ScanPlot(*result)
        scan_plot.save_plot(scan_file)

        peaks.remove_duplicate_peaks(peaks_ws)

        ub = UBModel(peaks_ws)

        if reindex:
            ub.determine_UB_with_primitive_cell(min_d, max_d, tol=0.15)

            ub.select_type(cell, centering, 0.15)

            ub.index_peaks(0.15)

            ub.refine_UB_with_constraints(cell, 0.15)

            Reorient(peaks_ws, UB, cell)

            ub.copy_UB(data)
            ub.copy_UB(md)

        ub_file = self.get_diagnostic_file("run#{}_ub".format(run))
        ub_file = os.path.splitext(ub_file)[0] + ".mat"

        ub.save_UB(ub_file)

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
                shape,
                projections,
                c,
                neighbors,
                b,
                m,
                sigma_c,
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
            ellipsoid.update_estimate(shape, sigma_c)

            args = (Q0, Q1, Q2, d, n, dQ, c, weights)
            fit_params = ellipsoid.fit(*args, b=b, m=m)

            intens_params = None
            if fit_params is not None:
                intens_params = ellipsoid.extract_result(*fit_params, c)

                if intens_params is None:
                    result[key] = None
                    print("Cannot extract fit")
                    del ellipsoid
                    continue

                c, S, *best_fit = ellipsoid.best_fit

                shape = self.revert_ellipsoid_parameters(
                    intens_params, projections
                )

                norm_params = Q0, Q1, Q2, d, n, c, S

                try:
                    intens, sig = ellipsoid.integrate(*norm_params, b=b, m=m)
                except Exception as e:
                    print("Exception extracting intensity: {}".format(e))
                    print(traceback.format_exc())
                    result[key] = None

                info = ellipsoid.info
                best_prof = ellipsoid.best_prof
                best_proj = ellipsoid.best_proj
                data_norm_fit = ellipsoid.data_norm_fit
                reddev = ellipsoid.reddev
                intensity = ellipsoid.intensity
                sigma = ellipsoid.sigma
                peak_background_mask = ellipsoid.peak_background_mask
                profile_iterations = ellipsoid.profile_iterations
                estimated_fit = ellipsoid.estimated_fit
                bkg_prof = ellipsoid.best_bkg_prof

                if self.make_plot:
                    self.peak_plot.add_ellipsoid_fit(best_fit)

                    self.peak_plot.add_profile_fit(best_prof)

                    self.peak_plot.add_profile_bkg(bkg_prof)

                    self.peak_plot.add_projection_fit(best_proj)

                    self.peak_plot.add_ellipsoid(c, S)

                    self.peak_plot.add_estimated_ellipsoid(*estimated_fit)

                    self.peak_plot.update_envelope(*peak_background_mask)

                    self.peak_plot.add_peak_info(
                        hkl, d_spacing, wavelength, angles, goniometer
                    )

                    self.peak_plot.add_peak_stats(reddev, intensity, sigma)

                    self.peak_plot.add_data_norm_fit(*data_norm_fit)

                    self.peak_plot.add_profile_iterations(profile_iterations)

                    try:
                        self.peak_plot.save_plot(peak_file)
                    except Exception as e:
                        print("Exception saving figure: {}".format(e))
                        print(traceback.format_exc())

                result[key] = intens, sig, shape, info, hkl

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

        ub = UBModel(peaks_ws)

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

            hkl = peak.get_hkl(i)

            sigma_c = ub.get_center_uncertainty(hkl)

            lamda = peak.get_wavelength(i)

            angles = peak.get_angles(i)

            two_theta, az_phi = angles

            peak.set_peak_intensity(i, 0, 0)

            goniometer = peak.get_goniometer_angles(i)

            orig_d = self.orig_d.get(peak.get_hklmnp(i))

            peak_name = peak.get_peak_name(i, d=orig_d)

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

            b, m = None, None
            if data.workspace_exists(bank_name + "_bkg_data"):
                b, *_ = data.extract_bin_info(bank_name + "_bkg_data")
                m, *_ = data.extract_bin_info(bank_name + "_bkg_norm")

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
                params,
                projections,
                center,
                neighbors,
                b,
                m,
                sigma_c,
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

                Qx, Qy, Qz = shape[:3]
                info["Qx"] = Qx
                info["Qy"] = Qy
                info["Qz"] = Qz
                info["cntrt"] = self.cntrt

                peak.add_diagnostic_info(i, info)

            else:
                peak.set_peak_intensity(i, 0, 0)

    def stub_peak_info(self, peaks_ws):
        """
        Populate placeholder diagnostic info.
        """
        peak = PeakModel(peaks_ws)

        for i in range(peak.get_number_peaks()):
            intens_raw, sig_raw = peak.get_peak_intensity(i)

            info = {
                "d3x": 0.0,
                "bkg": 0.0,
                "bkg_err": 0.0,
                "n_vox": 0,
                "vol_frac": 1.0,
                "intens_raw": intens_raw,
                "sig_raw": sig_raw,
                "pk_data": 1.0,
                "pk_norm": 1.0,
                "bkg_data": 0.0,
                "bkg_norm": 1.0,
                "ratio": 1.0,
                "cntrt": self.cntrt,
            }

            peak.add_diagnostic_info(i, info)

    def correct_intensities(self, peaks_ws):
        """
        Apply the delta-function absolute-scale normalization.
        """
        peak = PeakModel(peaks_ws)

        proton_charge = self.data.monitor

        for i in range(peak.get_number_peaks()):
            lamda = peak.get_wavelength(i)
            two_theta, _ = peak.get_angles(i)
            det_ID = peak.get_detector_id(i)

            norm = self.data.approximate_norm(
                lamda, two_theta, det_ID, proton_charge
            )

            if not (np.isfinite(norm) and norm > 0):
                continue

            intens, sig = peak.get_peak_intensity(i)

            intens /= norm
            sig /= norm

            peak.set_peak_intensity(i, intens, sig)

    def extract_peak_counts(self, peaks_ws, r_cut, n_bins, peak_indices=None):
        """
        Obtain raw event counts in an axis-aligned Q-sample box around
        each peak, without normalization or ellipsoid fitting.

        Each peak's binned box is also saved to the diagnostic directory
        (as "..._counts.nxs", same peak-named-subdirectory convention as
        `extract_peak_info`'s "_data"/"_norm" files) for manual
        inspection of what's actually inside the box.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        r_cut : float or ndarray, shape (n_peaks,)
            Box half-width along each Q-sample axis. A scalar applies
            the same half-width to every peak; a per-peak array (e.g.
            from `ResolutionEllipsoid.predict_roi_radii`) sizes each
            peak's box individually.
        n_bins : int
            Number of bins along each Q-sample axis.
        peak_indices : iterable of int, optional
            Only bin these peaks. Defaults to all peaks. Useful for
            pulling a handful of boxes (e.g. for a diagnostic plot)
            without re-binning the whole peaks table.

        Returns
        -------
        peak_dict : dict
            Keyed by peak index, each value a dict with "counts",
            "Q", "R", "lamda", "two_theta", "az_phi". "counts" is
            signal from "md", which is Lorentz-corrected (see
            `convert_to_Q_sample`'s `lorentz_corr`) -- a real-valued
            weighted sum, not raw event counts.

        """

        data = self.data

        peak = PeakModel(peaks_ws)

        n_peak = peak.get_number_peaks()

        r_cuts = np.broadcast_to(r_cut, n_peak)

        bins = np.full(3, n_bins, dtype=int)
        projections = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        Qs = [np.array(peak.get_sample_Q(i)) for i in range(n_peak)]

        peak_dict = {}

        indices = range(n_peak) if peak_indices is None else peak_indices

        for i in indices:
            Q = Qs[i]

            lamda = peak.get_wavelength(i)
            two_theta, az_phi = peak.get_angles(i)
            R = peak.get_goniometer_matrix(i)

            extents = [[q - r_cuts[i], q + r_cuts[i]] for q in Q]

            counts, _, Q0, Q1, Q2 = data.bin_in_Q(
                "md", extents, bins.copy(), projections
            )

            orig_d = self.orig_d.get(peak.get_hklmnp(i))
            peak_name = peak.get_peak_name(i, d=orig_d)

            counts_file = self.get_diagnostic_file(peak_name + "_counts")

            directory = os.path.dirname(counts_file)

            os.makedirs(directory, exist_ok=True)

            data.save_histograms(counts_file, "md_bin")

            data.delete_workspace("md_bin")

            peak_dict[i] = {
                "counts": counts,
                "Q0": Q0,
                "Q1": Q1,
                "Q2": Q2,
                "Q": Q,
                "R": R,
                "lamda": lamda,
                "two_theta": two_theta,
                "az_phi": az_phi,
            }

        return peak_dict

    def plot_peak_shape_diagnostics(self, peaks_ws, res, r_cut, n_bins):
        """
        Per-peak diagnostic figure: raw counts box (from
        `extract_peak_counts`) with the observed (whatever ellipsoid
        shape is currently stored on the peak, e.g. from
        `PeaksModel.integrate_peaks`) and model-predicted (`res`)
        ellipse cross-sections overlaid on all three 2D projections.

        When `self.prior_res` is set (an instrument prior model was
        used to size the adaptive per-peak integration radii, see
        `integrate`), a second row is prepended comparing that prior
        model's prediction against the peak's stored shape -- i.e. two
        rows per peak: prior vs. observed, then observed vs. the
        now-updated population model `res`.

        One PNG per peak, saved next to that peak's other diagnostic
        files (`extract_peak_counts`'s "..._counts.nxs", same
        peak-named subdirectory). Complements `res.plot_diagnostics`'s
        population-level obs-vs-pred scatter with a per-peak, spatial
        check of the same thing -- whether the peak's stored shape
        actually matches the counts around it, and whether the box
        (`r_cut`) used for the box display looks well-sized.

        Parameters
        ----------
        peaks_ws : str
            Peaks table.
        res : ResolutionEllipsoid
            Fitted resolution model (`fit()` already called).
        r_cut : float or ndarray, shape (n_peaks,)
            Box half-width(s) for the counts box shown behind the
            overlay -- purely for display, independent of whatever
            produced the peak's stored shape.
        n_bins : int
            Number of bins along each Q-sample axis.

        """
        peak = PeakModel(peaks_ws)

        peak_dict = self.extract_peak_counts(peaks_ws, r_cut, n_bins)

        for i, info in peak_dict.items():
            c0, c1, c2, r0, r1, r2, v0, v1, v2 = peak.get_peak_shape(
                i, r_cut=r_cut
            )

            if not np.all(np.isfinite([r0, r1, r2])):
                continue

            V_obs = np.column_stack([v0, v1, v2])
            S_obs = V_obs @ np.diag(np.array([r0, r1, r2]) ** 2) @ V_obs.T

            S_pred = res.predict_sample_S(i)

            sig_noise = peak.get_signal_to_noise(i)

            orig_d = self.orig_d.get(peak.get_hklmnp(i))
            peak_name = peak.get_peak_name(i, d=orig_d)

            label = "peak {} (S/N={:.1f})".format(i, sig_noise)

            grid = {
                "Q0": info["Q0"],
                "Q1": info["Q1"],
                "Q2": info["Q2"],
                "counts": info["counts"],
            }

            obs_label = "observed"
            pred_label = (
                "updated resolution model" if self.prior_res else "predicted"
            )

            samples = []

            if self.prior_res is not None:
                S_prior = self.prior_res.predict_sample_S(i)

                samples.append(
                    {
                        "label": label + " -- prior vs. observed",
                        "ellipses": [
                            {
                                "center": (c0, c1, c2),
                                "S": S_prior,
                                "label": "instrument config guess",
                                "color": "gold",
                                "linestyle": "--",
                            },
                            {
                                "center": (c0, c1, c2),
                                "S": S_obs,
                                "label": obs_label,
                                "color": "red",
                                "linestyle": "-",
                            },
                        ],
                        **grid,
                    }
                )

            has_prior = self.prior_res is not None

            samples.append(
                {
                    "label": label + " -- observed vs. population model",
                    "ellipses": [
                        {
                            "center": (c0, c1, c2),
                            "S": S_obs,
                            "label": obs_label,
                            "color": "red" if has_prior else "white",
                            "linestyle": "-" if has_prior else "--",
                        },
                        {
                            "center": (c0, c1, c2),
                            "S": S_pred,
                            "label": pred_label,
                            "color": "cyan" if has_prior else "red",
                            "linestyle": "-." if has_prior else "-",
                        },
                    ],
                    **grid,
                }
            )

            shape_file = self.get_diagnostic_file(
                peak_name + "_shape", ext=".png"
            )

            os.makedirs(os.path.dirname(shape_file), exist_ok=True)

            _plot_peak_shape_diagnostics(samples, shape_file)
