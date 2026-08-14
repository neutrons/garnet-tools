import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

import yaml

import numpy as np

import scipy.linalg
from scipy.spatial.transform import Rotation

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from mantid.simpleapi import (
    Load,
    LoadNexus,
    SaveNexus,
    LoadEmptyInstrument,
    ExtractMonitors,
    LoadIsawDetCal,
    LoadIsawUB,
    LoadParameterFile,
    ApplyCalibration,
    ConvertUnits,
    CompressEvents,
    CropWorkspace,
    SetGoniometer,
    CreatePeaksWorkspace,
    SetUB,
    IndexPeaks,
    FindPeaksMD,
    ConvertToMD,
    SaveMD,
    LoadMD,
    ConvertQtoHKLMDHisto,
    PreprocessDetectorsToMD,
    CloneWorkspace,
    DeleteWorkspace,
    CombinePeaksWorkspaces,
    mtd,
)

from garnet.reduction.ub import UBModel, Optimization, Reorient
from garnet.reduction.peaks import PeaksModel
from garnet.reduction.resolution import ResolutionEllipsoid


def scan_threshold(
    md, peaks, min_Q, max_peaks, max_threshold=1e5, min_found=50
):
    """
    Scan the peak-finding density threshold and select one that yields
    a balanced number of found peaks.

    Parameters
    ----------
    md : str
        Name of Q-sample MD workspace.
    peaks : str
        Name of output peaks table.
    min_Q : float
        Minimum Q-spacing enforcing lower limit of peak spacing.
    max_peaks : int
        Maximum number of peaks to find.
    max_threshold : float, optional
        Upper bound of the density threshold scan. The default is 1e5.
    min_found : int, optional
        Minimum number of found peaks desired for UB determination.
        The default is 50.

    Returns
    -------
    threshold : float
        Selected density threshold.

    """

    thresholds = np.logspace(1, np.log10(max_threshold), 20)
    found = []

    for threshold in thresholds:
        FindPeaksMD(
            InputWorkspace=md,
            MaxPeaks=max_peaks,
            PeakDistanceThreshold=min_Q,
            DensityThresholdFactor=threshold,
            OutputWorkspace=peaks,
        )
        found.append(mtd[peaks].getNumberPeaks())

    found = np.array(found)

    valid = np.argwhere(found < 0.9 * max_peaks).ravel()

    ind = valid[0] if valid.size > 0 else np.argmin(np.abs(found - min_found))
    threshold = thresholds[ind]

    FindPeaksMD(
        InputWorkspace=md,
        MaxPeaks=max_peaks,
        PeakDistanceThreshold=min_Q,
        DensityThresholdFactor=threshold,
        OutputWorkspace=peaks,
    )

    return threshold


def _B_matrix(a, b, c, alpha, beta, gamma):
    alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
        ]
    )

    return scipy.linalg.cholesky(np.linalg.inv(G), lower=False)


def _max_predicted_peaks(
    data_ws,
    peaks,
    centering,
    d_min,
    lamda_min,
    lamda_max,
    B,
    n_orient,
    seed=1234,
):
    """
    Predict peaks over random trial orientations and return the largest
    predicted count, for use as a robust MaxPeaks bound when the true
    orientation is unknown -- the number of reflections that fall within
    the accessible Q-coverage depends on orientation, so a single guess
    can under-estimate it.

    Parameters
    ----------
    data_ws : str
        Name of workspace to predict peaks with UB.
    peaks : str
        Name of output peaks table (overwritten each trial).
    centering : str
        Lattice centering that provides the reflection condition.
    d_min : float
        The lower d-spacing resolution to predict peaks.
    lamda_min, lamda_max : float
        The wavelength band over which to predict peaks.
    B : 2d-array
        B-matrix from the known lattice parameters.
    n_orient : int
        Number of random trial orientations to test.
    seed : int, optional
        Random seed for reproducibility. The default is 1234.

    Returns
    -------
    max_peaks : int
        Largest number of predicted peaks over all trial orientations.

    """

    peaks_model = PeaksModel()

    max_peaks = 0

    for U in Rotation.random(n_orient, random_state=seed).as_matrix():
        SetUB(Workspace=data_ws, UB=U @ B)
        peaks_model.predict_peaks(
            data_ws, peaks, centering, d_min, lamda_min, lamda_max
        )
        max_peaks = max(max_peaks, mtd[peaks].getNumberPeaks())

    return max_peaks


def _process_run(config, ipts, run, idx, tol):
    instrument = config["instrument"]
    file_folder = config["file_folder"]
    file_name = config["file_name"]
    output_folder = config["output_folder"]
    wavelength_band = config["wavelength_band"]
    gon_axis = config["gon_axis"]
    Q_max = config["Q_max"]
    Q_min = config["Q_min"]
    d_min = config["d_min"]
    max_threshold = config["max_threshold"]
    peak_radius = config["peak_radius"]
    a = config["a"]
    b = config["b"]
    c = config["c"]
    alpha = config["alpha"]
    beta = config["beta"]
    gamma = config["gamma"]
    centering = config["centering"]
    cell_type = config["cell_type"]
    crystal_system = config["crystal_system"]
    lattice_system = config["lattice_system"]
    UB_ref = config["UB_ref"]
    has_ub_ref = config["has_ub_ref"]
    n_orient = config["n_orient"]
    tube_calibration = config["tube_calibration"]
    detector_calibration = config["detector_calibration"]

    file_to_load = os.path.join(
        file_folder.format(instrument, ipts), file_name.format(instrument, run)
    )

    data_ws = "data"
    md_ws = "md"
    strong_ws = "strong"
    combine_ws = "combine"

    Load(Filename=file_to_load, OutputWorkspace=data_ws, NumberOfBins=1)

    if tube_calibration is not None:
        LoadNexus(
            Filename=tube_calibration,
            OutputWorkspace="tube_table",
        )
        ApplyCalibration(Workspace=data_ws, CalibrationTable="tube_table")
        DeleteWorkspace(Workspace="tube_table")

    if detector_calibration is not None:
        ext = os.path.splitext(detector_calibration)[1]
        if ext == ".xml":
            LoadParameterFile(
                Workspace=data_ws,
                Filename=detector_calibration,
            )
        else:
            LoadIsawDetCal(
                InputWorkspace=data_ws,
                Filename=detector_calibration,
            )

    ConvertUnits(
        InputWorkspace=data_ws, OutputWorkspace=data_ws, Target="Wavelength"
    )

    CompressEvents(
        InputWorkspace=data_ws,
        Tolerance=1e-2,
        OutputWorkspace=data_ws,
    )

    CropWorkspace(
        InputWorkspace=data_ws,
        OutputWorkspace=data_ws,
        XMin=wavelength_band[0],
        XMax=wavelength_band[1],
    )

    SetGoniometer(
        Workspace=data_ws,
        Goniometers="None, Specify Individually",
        Axis0=gon_axis[0],
        Axis1=gon_axis[1],
        Axis2=gon_axis[2],
        Axis3=gon_axis[3],
        Axis4=gon_axis[4],
        Axis5=gon_axis[5],
        Average=True,
    )

    peaks_model = PeaksModel()

    if has_ub_ref:
        SetUB(Workspace=data_ws, UB=UB_ref)
        peaks_model.predict_peaks(
            data_ws,
            strong_ws,
            centering,
            d_min,
            wavelength_band[0],
            wavelength_band[1],
        )
        max_peaks = mtd[strong_ws].getNumberPeaks()
    else:
        B = _B_matrix(a, b, c, alpha, beta, gamma)
        max_peaks = _max_predicted_peaks(
            data_ws,
            strong_ws,
            centering,
            d_min,
            wavelength_band[0],
            wavelength_band[1],
            B,
            n_orient,
        )

    ConvertToMD(
        InputWorkspace=data_ws,
        QDimensions="Q3D",
        dEAnalysisMode="Elastic",
        Q3DFrames="Q_sample",
        LorentzCorrection=True,
        MinValues=[-Q_max] * 3,
        MaxValues=[+Q_max] * 3,
        OutputWorkspace=md_ws,
    )

    scan_threshold(md_ws, strong_ws, Q_min, max_peaks, max_threshold)

    ub = UBModel(strong_ws)

    min_d, max_d = ub.get_primitive_cell_length_range(centering)

    ub.determine_UB_with_primitive_cell(min_d, max_d, tol)

    ub.index_peaks(tol)

    peaks_model.remove_unindexed_peaks(strong_ws)

    peaks_model.integrate_ellipsoids(data_ws, strong_ws, peak_radius)

    peaks_model.remove_weak_peaks(strong_ws, 20)

    ub.select_type(cell_type, centering, tol)

    ub.index_peaks(tol)

    ub.refine_UB_with_constraints(cell_type, tol)

    Reorient(strong_ws, UB_ref, crystal_system, lattice_system)

    DeleteWorkspace(Workspace=data_ws)

    mat_file = os.path.join(output_folder, "{}.mat".format(run))

    ub.save_UB(mat_file)

    peaks_file = os.path.join(output_folder, "peaks_{}.nxs".format(run))
    SaveNexus(InputWorkspace=strong_ws, Filename=peaks_file)

    extents = [
        -config["h_max"] / 2,
        +config["h_max"] / 2,
        -config["k_max"] / 2,
        +config["k_max"] / 2,
        -config["l_max"] / 2,
        +config["l_max"] / 2,
    ]

    bins = [256, 256, 256]

    ConvertQtoHKLMDHisto(
        InputWorkspace=md_ws,
        PeaksWorkspace=strong_ws,
        Extents=extents,
        Bins=bins,
        OutputWorkspace=combine_ws,
    )

    DeleteWorkspace(Workspace=md_ws)
    DeleteWorkspace(Workspace=strong_ws)

    md_filename = os.path.join(output_folder, "mdhkl_{}.nxs".format(run))
    SaveMD(
        InputWorkspace=combine_ws,
        Filename=md_filename,
        SaveHistory=False,
        SaveLogs=False,
        SaveInstrument=False,
    )

    DeleteWorkspace(Workspace=combine_ws)


from garnet.config.instruments import beamlines
from garnet.reduction.data import _read_signal_error_squared


class Peaks:
    def __init__(self, config):
        defaults = {
            "Instrument": "TOPAZ",
            "InstrumentDefinition": None,
            "IPTS": 31856,
            "Runs": None,
            "PeaksTable": None,
            "OutputFolder": "",
            "UnitCellLengths": [5.431, 5.431, 5.431],
            "UnitCellAngles": [90, 90, 90],
            "Centering": "P",
            "CrystalSystem": "Cubic",
            "LatticeSystem": "Cubic",
            "UBFile": None,
            "MaxThreshold": 1e5,
            "PeakRadius": 0.25,
        }
        defaults.update(config)

        self.instrument = defaults.get("Instrument")
        self.instrument_definition = defaults.get("InstrumentDefinition")

        self.ipts = defaults.get("IPTS")
        self.nos = defaults.get("Runs")

        self.output_folder = defaults.get("OutputFolder")

        self.detector_calibration = defaults.get("DetectorCalibration")
        self.tube_calibration = defaults.get("TubeCalibration")

        self.file_folder = "/SNS/{}/IPTS-{}/nexus/"
        self.file_name = "{}_{}.nxs.h5"
        self.calibration_folder = "/SNS/{}/shared/calibration"

        self.a, self.b, self.c = defaults.get("UnitCellLengths")
        self.alpha, self.beta, self.gamma = defaults.get("UnitCellAngles")

        self.centering = defaults.get("Centering")

        self.crystal_system = defaults.get("CrystalSystem")
        self.lattice_system = defaults.get("LatticeSystem")

        if self.crystal_system != "Trigonal":
            self.lattice_system = None

        self.cell_type = (
            self.lattice_system
            if self.crystal_system == "Trigonal"
            else self.crystal_system
        )

        self.ub_file = defaults.get("UBFile")

        self.max_threshold = defaults.get("MaxThreshold")
        self.peak_radius = defaults.get("PeakRadius")

        inst_config = beamlines[self.instrument]

        self.gon_axis = 6 * [None]
        gon = inst_config.get("Goniometer")
        gon_axis_names = inst_config.get("GoniometerAxisNames")
        if gon_axis_names is None:
            gon_axis_names = list(gon.keys())
        axes = list(gon.items())

        gon_ind = 0
        for i, name in enumerate(gon_axis_names):
            axis = axes[i][1]
            if name is not None:
                self.gon_axis[gon_ind] = ",".join(5 * ["{}"]).format(
                    name, *axis
                )
                gon_ind += 1

        self.wavelength_band = defaults.get(
            "Wavelength", inst_config["Wavelength"]
        )

    def _join(self, items):
        if isinstance(items, list):
            return ",".join(
                [
                    "{}-{}".format(*r) if isinstance(r, list) else str(r)
                    for r in items
                ]
            )
        else:
            return str(items)

    def load_instrument(self):
        LoadEmptyInstrument(
            Filename=self.instrument_definition,
            InstrumentName=self.instrument,
            OutputWorkspace=self.instrument,
        )
        ExtractMonitors(
            InputWorkspace=self.instrument, DetectorWorkspace=self.instrument
        )

    def calculate_limits(self):
        PreprocessDetectorsToMD(
            InputWorkspace=self.instrument, OutputWorkspace="detectors"
        )

        two_theta = max(mtd["detectors"].column("TwoTheta"))
        lamda = min(self.wavelength_band)
        self.Q_max = 4 * np.pi / lamda * np.sin(0.5 * two_theta)

        DeleteWorkspace(Workspace="detectors")

        CreatePeaksWorkspace(
            NumberOfPeaks=0,
            OutputType="LeanElasticPeak",
            OutputWorkspace="sample",
        )

        if self.ub_file:
            LoadIsawUB(InputWorkspace="sample", Filename=self.ub_file)
        else:
            SetUB(
                Workspace="sample",
                a=self.a,
                b=self.b,
                c=self.c,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

        ol = mtd["sample"].sample().getOrientedLattice()
        astar, bstar, cstar = ol.astar(), ol.bstar(), ol.cstar()

        self.UB_ref = ol.getUB().copy()

        ub = UBModel("sample")
        self.Q_min, _ = ub.shortest_reciprocal_spacing(self.centering)

        self.d_max = 2 * np.pi / self.Q_min
        self.d_min = 2 * np.pi / self.Q_max

        self.h_max = np.floor(1 / self.d_min / astar)
        self.k_max = np.floor(1 / self.d_min / bstar)
        self.l_max = np.floor(1 / self.d_min / cstar)

        DeleteWorkspace(Workspace="sample")

    def _runs_string_to_list(self, runs_str):
        """
        Convert runs string to list.

        Parameters
        ----------
        runs_str : str
            Condensed notation for run numbers.

        Returns
        -------
        runs : list
            Integer run numbers.

        """

        if type(runs_str) is not str:
            runs_str = str(runs_str)

        runs = []
        ranges = runs_str.split(",")

        for part in ranges:
            if ":" in part:
                range_part, *skip_part = part.split(";")
                start, end = map(int, range_part.split(":"))
                skip = int(skip_part[0]) if skip_part else 1

                if start > end or skip <= 0:
                    return None

                runs.extend(range(start, end + 1, skip))
            else:
                runs.append(int(part))

        return runs

    def load_convert_runs(self, ipts, run_nos, tol=0.25, n_proc=16):
        if not isinstance(run_nos, list):
            run_nos = self._runs_string_to_list(run_nos)

        config = {
            "instrument": self.instrument,
            "file_folder": self.file_folder,
            "file_name": self.file_name,
            "output_folder": self.output_folder,
            "wavelength_band": self.wavelength_band,
            "gon_axis": self.gon_axis,
            "Q_max": self.Q_max,
            "Q_min": self.Q_min,
            "d_min": self.d_min,
            "max_threshold": self.max_threshold,
            "peak_radius": self.peak_radius,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "centering": self.centering,
            "cell_type": self.cell_type,
            "crystal_system": self.crystal_system,
            "lattice_system": self.lattice_system,
            "UB_ref": self.UB_ref,
            "has_ub_ref": self.ub_file is not None,
            "n_orient": 10,
            "h_max": self.h_max,
            "k_max": self.k_max,
            "l_max": self.l_max,
            "tube_calibration": self.tube_calibration,
            "detector_calibration": self.detector_calibration,
        }

        args_list = [
            (config, ipts, run, i, tol) for i, run in enumerate(run_nos)
        ]

        multiprocessing.set_start_method("spawn", force=True)

        with multiprocessing.Pool(processes=n_proc) as pool:
            pool.starmap(_process_run, args_list)

    def finalize_and_save(self, tol=0.2, n_proc=None):
        md_files = [
            os.path.join(self.output_folder, f)
            for f in os.listdir(self.output_folder)
            if f.startswith("mdhkl_") and f.endswith(".nxs")
        ]

        if len(md_files) > 0:
            LoadMD(Filename=md_files[0], OutputWorkspace="merge")

            rest = md_files[1:]

            if len(rest) > 0:
                signal = mtd["merge"].getSignalArray().copy()
                error_sq = mtd["merge"].getErrorSquaredArray().copy()

                if n_proc is None or n_proc <= 1 or len(rest) == 1:
                    arrays = [_read_signal_error_squared(f) for f in rest]
                else:
                    ctx = multiprocessing.get_context("fork")
                    with ProcessPoolExecutor(
                        max_workers=min(n_proc, len(rest)), mp_context=ctx
                    ) as executor:
                        arrays = list(
                            executor.map(_read_signal_error_squared, rest)
                        )

                for sig, err_sq in arrays:
                    signal += sig
                    error_sq += err_sq

                mtd["merge"].setSignalArray(signal)
                mtd["merge"].setErrorSquaredArray(error_sq)

            for md in md_files:
                os.remove(md)

        peak_files = [
            os.path.join(self.output_folder, f)
            for f in os.listdir(self.output_folder)
            if f.startswith("peaks_") and f.endswith(".nxs")
        ]

        for i, sf in enumerate(peak_files):
            LoadNexus(Filename=sf, OutputWorkspace="tmp")
            if i == 0:
                CloneWorkspace(InputWorkspace="tmp", OutputWorkspace="peaks")
            else:
                CombinePeaksWorkspaces(
                    LHSWorkspace="tmp",
                    RHSWorkspace="peaks",
                    OutputWorkspace="peaks",
                )
            os.remove(sf)

        filename = os.path.join(self.output_folder, "peaks.nxs")
        SaveNexus(InputWorkspace="peaks", Filename=filename)

        filename = os.path.join(self.output_folder, "mdhkl.nxs")
        SaveMD(
            InputWorkspace="merge",
            Filename=filename,
            SaveHistory=False,
            SaveLogs=False,
            SaveInstrument=False,
        )

        IndexPeaks(PeaksWorkspace="peaks", Tolerance=tol, RoundHKLs=True)

        opt = Optimization("peaks", tol=tol)
        opt.optimize_lattice(self.cell_type)

        IndexPeaks(PeaksWorkspace="peaks", Tolerance=tol, RoundHKLs=True)

        filename = os.path.join(self.output_folder, "peaks.mat")
        ub = UBModel("peaks")
        ub.save_UB(filename)

        res = ResolutionEllipsoid("peaks", r_cut=self.peak_radius)
        res.fit()

        if res.model is not None:
            res_file = os.path.join(self.output_folder, "resolution.txt")
            res.write_resolution_parameters(res_file)

    def run(self, n_proc=10):
        self.load_instrument()
        self.calculate_limits()
        self.load_convert_runs(self.ipts, self.nos, n_proc=n_proc)
        self.finalize_and_save(n_proc=n_proc)


if __name__ == "__main__":
    config_file = sys.argv[1]

    with open(config_file, "r") as f:
        params = yaml.safe_load(f)

    config_dir = os.path.dirname(os.path.abspath(config_file))
    name = os.path.splitext(os.path.basename(config_file))[0]

    output_folder = os.path.join(config_dir, name)
    os.makedirs(output_folder, exist_ok=True)

    params["OutputFolder"] = output_folder

    peaks = Peaks(params)
    peaks.run()
