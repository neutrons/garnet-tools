import os
import sys

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

from mantid.simpleapi import (
    LoadNexus,
    FilterPeaks,
    SortPeaksWorkspace,
    StatisticsOfPeaksWorkspace,
    CountReflections,
    SaveHKL,
    SaveReflections,
    SaveIsawUB,
    LoadIsawUB,
    LoadIsawSpectrum,
    CloneWorkspace,
    SetGoniometer,
    SetSample,
    LoadSampleShape,
    mtd,
)

import re

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

np.random.seed(13)

import scipy.optimize
import scipy.interpolate
import scipy.stats

from sklearn.cluster import AgglomerativeClustering

from mantid.kernel import V3D

from mantid import config

config["Q.convention"] = "Crystallography"


from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import argparse

from garnet.config.instruments import beamlines
from garnet.reduction.data import DataModel
from garnet.reduction.ub import Optimization
from garnet.reduction.resolution import ResolutionEllipsoid
from garnet.utilities.absorption import AbsorptionEllipsoid

point_group_dict = {
    "-1": "-1 (Triclinic)",
    "1": "1 (Triclinic)",
    "2": "2 (Monoclinic, unique axis b)",
    "m": "m (Monoclinic, unique axis b)",
    "2/m": "2/m (Monoclinic, unique axis b)",
    "112": "112 (Monoclinic, unique axis c)",
    "11m": "11m (Monoclinic, unique axis c)",
    "112/m": "112/m (Monoclinic, unique axis c)",
    "222": "222 (Orthorhombic)",
    "2mm": "2mm (Orthorhombic)",
    "m2m": "m2m (Orthorhombic)",
    "mm2": "mm2 (Orthorhombic)",
    "mmm": "mmm (Orthorhombic)",
    "3": "3 (Trigonal - Hexagonal)",
    "32": "32 (Trigonal - Hexagonal)",
    "312": "312 (Trigonal - Hexagonal)",
    "321": "321 (Trigonal - Hexagonal)",
    "31m": "31m (Trigonal - Hexagonal)",
    "3m": "3m (Trigonal - Hexagonal)",
    "3m1": "3m1 (Trigonal - Hexagonal)",
    "-3": "-3 (Trigonal - Hexagonal)",
    "-31m": "-31m (Trigonal - Hexagonal)",
    "-3m": "-3m (Trigonal - Hexagonal)",
    "-3m1": "-3m1 (Trigonal - Hexagonal)",
    "3 r": "3 r (Trigonal - Rhombohedral)",
    "32 r": "32 r (Trigonal - Rhombohedral)",
    "3m r": "3m r (Trigonal - Rhombohedral)",
    "-3 r": "-3 r (Trigonal - Rhombohedral)",
    "-3m r": "-3m r (Trigonal - Rhombohedral)",
    "4": "4 (Tetragonal)",
    "4/m": "4/m (Tetragonal)",
    "4mm": "4mm (Tetragonal)",
    "422": "422 (Tetragonal)",
    "-4": "-4 (Tetragonal)",
    "-42m": "-42m (Tetragonal)",
    "-4m2": "-4m2 (Tetragonal)",
    "4/mmm": "4/mmm (Tetragonal)",
    "6": "6 (Hexagonal)",
    "6/m": "6/m (Hexagonal)",
    "6mm": "6mm (Hexagonal)",
    "622": "622 (Hexagonal)",
    "-6": "-6 (Hexagonal)",
    "-62m": "-62m (Hexagonal)",
    "-6m2": "-6m2 (Hexagonal)",
    "6/mmm": "6/mmm (Hexagonal)",
    "23": "23 (Cubic)",
    "m-3": "m-3 (Cubic)",
    "432": "432 (Cubic)",
    "-43m": "-43m (Cubic)",
    "m-3m": "m-3m (Cubic)",
}


def plot_normalization(data, filename):
    """
    Debug plot of the per-spectrum wavelength-dependent flux spectra.
    """
    fig, ax = plt.subplots(1, 1, layout="constrained")
    for y in data.spect_y:
        ax.plot(data.lamda_x, y)
    ax.set_xlabel(r"$\lambda$ [$\AA$]")
    ax.set_ylabel(r"Flux [counts/$\AA$]")
    ax.minorticks_on()
    fig.savefig(filename + "_flux.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, layout="constrained")
    sca = ax.scatter(
        np.rad2deg(data.sa_gamma),
        np.rad2deg(data.sa_nu),
        c=data.sa_y,
        s=4,
    )
    ax.set_xlabel(r"$\gamma$ [deg]")
    ax.set_ylabel(r"$\nu$ [deg]")
    fig.colorbar(sca, ax=ax, label="Solid angle efficiency")
    ax.minorticks_on()
    fig.savefig(filename + "_efficiency.pdf")
    plt.close(fig)


def error_volume(
    a, b, c, alpha, beta, gamma, e_a, e_b, e_c, e_alpha, e_beta, e_gamma
):
    """
    Propagate cell-edge and cell-angle esd's to a cell-volume esd by
    central finite differences, since OrientedLattice has no direct
    volume-esd accessor.
    """

    def volume(a, b, c, alpha, beta, gamma):
        ca = np.cos(np.radians(alpha))
        cb = np.cos(np.radians(beta))
        cg = np.cos(np.radians(gamma))
        return (
            a
            * b
            * c
            * np.sqrt(1.0 - ca**2 - cb**2 - cg**2 + 2.0 * ca * cb * cg)
        )

    v0 = volume(a, b, c, alpha, beta, gamma)

    d_alpha = volume(a, b, c, alpha + 0.5 * e_alpha, beta, gamma) - volume(
        a, b, c, alpha - 0.5 * e_alpha, beta, gamma
    )
    d_beta = volume(a, b, c, alpha, beta + 0.5 * e_beta, gamma) - volume(
        a, b, c, alpha, beta - 0.5 * e_beta, gamma
    )
    d_gamma = volume(a, b, c, alpha, beta, gamma + 0.5 * e_gamma) - volume(
        a, b, c, alpha, beta, gamma - 0.5 * e_gamma
    )

    return v0 * np.sqrt(
        (e_a / a) ** 2
        + (e_b / b) ** 2
        + (e_c / c) ** 2
        + (d_alpha / v0) ** 2
        + (d_beta / v0) ** 2
        + (d_gamma / v0) ** 2
    )


def format_value_esd(value, esd, max_precision=4):
    """
    Format ``value`` with its esd in the compact CIF convention,
    e.g. ``9.5432(7)``. Falls back to a plain fixed-precision number
    when no usable esd is available.
    """
    if esd is None or not np.isfinite(esd) or esd <= 0:
        return "{:.{p}f}".format(value, p=max_precision)

    precision = max_precision
    while precision > 0 and round(esd * 10**precision) >= 100:
        precision -= 1
    while (
        precision < max_precision and round(esd * 10 ** (precision + 1)) < 100
    ):
        precision += 1

    digits = int(round(esd * 10**precision))
    if digits == 0:
        return "{:.{p}f}".format(value, p=precision)

    return "{:.{p}f}({:d})".format(value, digits, p=precision)


class AbsorptionCorrection:
    def __init__(
        self,
        peaks,
        chemical_formula,
        z_parameter,
        u_vector=[0, 0, 1],
        v_vector=[1, 0, 0],
        params=None,
        filename=None,
    ):
        assert "PeaksWorkspace" in str(type(mtd[peaks]))

        self.peaks = peaks

        volume = mtd[self.peaks].sample().getOrientedLattice().volume()

        assert volume > 0

        self.volume = volume

        assert self.verify_chemical_formula(chemical_formula)

        self.chemical_formula = chemical_formula

        assert z_parameter > 0
        self.z_parameter = z_parameter

        assert len(u_vector) == 3
        assert len(v_vector) == 3

        assert not np.isclose(np.linalg.norm(np.cross(u_vector, v_vector)), 0)

        self.u_vector = u_vector
        self.v_vector = v_vector

        if params is not None:
            assert len(params) == 3

        self.params = params

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        self.set_shape()
        self.set_material()
        self.set_orientation()
        self.calculate_correction()
        self.write_absortion_parameters()

    def verify_chemical_formula(self, formula):
        pattern = (
            r"(?:\((?:[A-Z][a-z]?\d+)\)|[A-Z][a-z]?)(?:\d+(?:\.\d+)?|\.\d+)?"
        )

        parts = re.split(r"[-\s]+", formula.strip())

        return all(re.fullmatch(pattern, part) for part in parts)

    def save_ellipsoid_stl(self, params, filename="/tmp/ellipsoid.stl"):
        l_thickness, l_width, l_height = params
        sph = pv.Icosphere(radius=0.5)

        ell = sph.scale([l_width, l_height, l_thickness], inplace=False)
        ell.save(filename)

    def set_shape(self):
        self.UB = mtd[self.peaks].sample().getOrientedLattice().getUB().copy()

        u = np.dot(self.UB, self.u_vector)
        v = np.dot(self.UB, self.v_vector)

        u /= np.linalg.norm(u)

        w = np.cross(u, v)
        w /= np.linalg.norm(w)

        v = np.cross(w, u)

        T = np.column_stack([v, w, u])

        gon = mtd[self.peaks].run().getGoniometer()

        gon.setR(T)
        self.gamma, self.beta, self.alpha = gon.getEulerAngles("ZYX")

        if self.params is not None:
            self.shapestl = os.path.splitext(self.filename)[0] + ".stl"
            self.save_ellipsoid_stl(self.params, self.shapestl)

        self.absorption_ellipsoid = AbsorptionEllipsoid()

    def set_material(self):
        self.mat_dict = {
            "ChemicalFormula": self.chemical_formula,
            "ZParameter": float(self.z_parameter),
            "UnitCellVolume": self.volume,
        }

    def set_orientation(self):
        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="RunNumber",
            SortAscending=False,
        )

        Rs = [peak.getGoniometerMatrix() for peak in mtd[self.peaks]]
        matrix_dict = {}

        runs = []
        for peak in mtd[self.peaks]:
            R = peak.getGoniometerMatrix()

            matrix_tuple = tuple(R.flatten())

            if matrix_tuple in matrix_dict:
                run = matrix_dict[matrix_tuple]
            else:
                ind = np.isclose(Rs, R).all(axis=(1, 2))
                i = -1 if not np.any(ind) else ind.tolist().index(True)
                run = i + 1
                matrix_dict[matrix_tuple] = run

            runs.append(run)
            peak.setBinCount(peak.getRunNumber())
            peak.setRunNumber(run)

        self.runs = np.unique(runs).astype(int).tolist()
        self.Rs = Rs

    def calculate_correction(self):
        filename = os.path.splitext(self.filename)[0] + "_abs.pdf"

        self.apply_correction()

        with PdfPages(filename) as pdf:
            for i, (R, run) in enumerate(zip(self.Rs, self.runs)):
                FilterPeaks(
                    InputWorkspace=self.peaks,
                    FilterVariable="RunNumber",
                    FilterValue=run,
                    Operator="=",
                    OutputWorkspace="_tmp",
                )

                R = mtd["_tmp"].getPeak(0).getGoniometerMatrix()

                gon = mtd["_tmp"].run().getGoniometer()

                gon.setR(R)
                omega, chi, phi = gon.getEulerAngles("YZY")

                LoadSampleShape(
                    InputWorkspace="_tmp",
                    Filename=self.shapestl,
                    Scale="mm",
                    XDegrees=self.alpha,
                    YDegrees=self.beta,
                    ZDegrees=self.gamma,
                    OutputWorkspace="_tmp",
                )

                SetSample(
                    InputWorkspace="_tmp",
                    Material=self.mat_dict,
                )

                SetGoniometer(
                    Workspace="_tmp",
                    Axis0="{},0,1,0,1".format(omega),
                    Axis1="{},0,0,1,1".format(chi),
                    Axis2="{},0,1,0,1".format(phi),
                )

                hkl = np.eye(3)
                s = np.matmul(self.UB, hkl)

                reciprocal_lattice = np.matmul(R, s)

                shape = mtd["_tmp"].sample().getShape()
                mesh = shape.getMesh() * 1000

                mesh_polygon = Poly3DCollection(
                    mesh,
                    edgecolors="k",
                    facecolors="w",
                    alpha=0.5,
                    linewidths=0.1,
                )

                fig, ax = plt.subplots(
                    subplot_kw={"projection": "mantid3d", "proj_type": "persp"}
                )
                ax.add_collection3d(mesh_polygon)

                ax.set_title("run #{}".format(1 + i))
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                ax.set_zlabel("z [mm]")

                ax.set_mesh_axes_equal(mesh)
                ax.set_box_aspect((1, 1, 1))

                colors = ["r", "g", "b"]
                origin = (
                    ax.get_xlim3d()[1],
                    ax.get_ylim3d()[1],
                    ax.get_zlim3d()[1],
                )

                lims = ax.get_xlim3d()
                factor = (lims[1] - lims[0]) / 4
                origin = (lims[1] - lims[0]) / 4

                for j in range(3):
                    vector = reciprocal_lattice[:, j]
                    vector = vector / np.linalg.norm(vector)
                    ax.quiver(
                        origin,
                        origin,
                        origin,
                        vector[0],
                        vector[1],
                        vector[2],
                        length=factor,
                        color=colors[j],
                        linestyle="-",
                    )

                    ax.view_init(vertical_axis="y", elev=27, azim=50)

                pdf.savefig(fig, dpi=100, bbox_inches=None)
                plt.close(fig)
                plt.close("all")

    def write_absortion_parameters(self):
        mat = mtd["_tmp"].sample().getMaterial()

        sigma_a = mat.absorbXSection()
        sigma_s = mat.totalScatterXSection()

        M = mat.relativeMolecularMass()
        n = mat.numberDensityEffective  # A^-3
        N = mat.totalAtoms

        V = np.abs(np.prod(self.params) * 0.1**3)  # cm^3

        rho = (n / N) / 0.6022 * M
        m = rho * V * 1000  # mg
        r = np.cbrt(0.75 * np.pi * V)  # cm

        mu_s = n * sigma_s
        mu_a = n * sigma_a

        mu = mat.numberDensityEffective * (
            mat.totalScatterXSection() + mat.absorbXSection(1.8)
        )

        self.mu = mu  # linear absorption coefficient at 1.8 A, in cm^-1

        lines = [
            "{}\n".format(self.chemical_formula),
            "Z parameter: {:.4f}\n".format(self.z_parameter),
            "unit cell volume: {:.4f} A^3\n".format(self.volume),
            "absoption cross section: {:.4f} barn\n".format(sigma_a),
            "scattering cross section: {:.4f} barn\n".format(sigma_s),
            "linear absorption coefficient: {:.4f} 1/cm\n".format(mu_a),
            "linear scattering coefficient: {:.4f} 1/cm\n".format(mu_s),
            "absorption parameter: {:.4f} \n".format(mu * r),
            "total atoms: {:.4f}\n".format(N),
            "molar mass: {:.4f} g/mol\n".format(M),
            "number density: {:.4f} 1/A^3\n".format(n),
            "mass density: {:.4f} g/cm^3\n".format(rho),
            "volume: {:.4f} cm^3\n".format(V),
            "mass: {:.4f} mg\n".format(m),
            "equivalent-sphere: {:.4f} cm\n".format(r),
        ]

        for line in lines:
            print(line)

        if self.filename is not None:
            filename = os.path.splitext(self.filename)[0] + "_abs.txt"

            with open(filename, "w") as f:
                for line in lines:
                    f.write(line)

    def apply_correction(self):
        """
        Correct every peak's intensity for absorption.
        """
        SetSample(
            InputWorkspace=self.peaks,
            Material=self.mat_dict,
        )

        mat = mtd[self.peaks].sample().getMaterial()
        n = mat.numberDensityEffective

        thickness, width, height = np.array(self.params) / 10.0  # mm -> cm

        vol = np.pi / 6 * thickness * width * height
        ratio_b = width / thickness
        ratio_c = height / thickness

        coeffs = (self.alpha, self.beta, self.gamma, vol, ratio_b, ratio_c)

        lamdas, ri_hat, sf_hat = [], [], []

        for peak in mtd[self.peaks]:
            lamdas.append(peak.getWavelength())
            ri_hat.append(peak.getSourceDirectionSampleFrame())
            sf_hat.append(peak.getDetectorDirectionSampleFrame())

        lamdas = np.array(lamdas)

        mu = n * (
            mat.totalScatterXSection()
            + np.array([mat.absorbXSection(lamda) for lamda in lamdas])
        )

        T, Tbar = self.absorption_ellipsoid.correction(
            coeffs, mu, np.array(ri_hat), np.array(sf_hat)
        )

        self.T = T  # per-peak absorption transmission factor

        for i, peak in enumerate(mtd[self.peaks]):
            corr = 1 / T[i]

            print(
                "mu = {:4.2f} corr = {:4.2f} Tbar = {:4.2f}".format(
                    mu[i], corr, Tbar[i]
                )
            )

            peak.setBinCount(corr)
            peak.setIntensity(peak.getIntensity() * corr)
            peak.setSigmaIntensity(peak.getSigmaIntensity() * corr)
            peak.setAbsorptionWeightedPathLength(Tbar[i])


class Peaks:
    def __init__(self, peaks, filename, scale=None, point_group=None):
        self.peaks = peaks

        if filename is not None:
            assert type(filename) is str
            filename = os.path.abspath(filename)
            assert os.path.exists(os.path.dirname(filename))

        self.filename = filename

        if scale is not None:
            assert scale > 0

        self.scale = scale

        if point_group is not None:
            assert point_group in point_group_dict.keys()
            point_groups = [point_group]
        else:
            point_groups = list(point_group_dict.keys())

        self.point_groups = point_groups

        self.max_order = 0
        self.modUB = np.zeros((3, 3))
        self.modHKL = np.zeros((3, 3))

    def refine_ellipsoids(self, peaks):
        filename = os.path.splitext(self.filename)[0] + "_res.pdf"

        res = ResolutionEllipsoid(peaks, r_cut=np.inf)

        res.fit()
        res.plot_diagnostics(filename)

    def refine_UB(self, peaks):
        opt = Optimization(peaks)

        ol = mtd[peaks].sample().getOrientedLattice()

        a, b, c = ol.a(), ol.b(), ol.c()
        alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

        if np.allclose([a, b], c) and np.allclose([alpha, beta, gamma], 90):
            cell = "Cubic"
        elif np.allclose([a, b], c) and np.allclose([alpha, beta], gamma):
            cell = "Rhombohedral"
        elif np.isclose(a, b) and np.allclose([alpha, beta, gamma], 90):
            cell = "Tetragonal"
        elif (
            np.isclose(a, b)
            and np.allclose([alpha, beta], 90)
            and np.isclose(gamma, 120)
        ):
            cell = "Hexagonal"
        elif np.allclose([alpha, beta, gamma], 90):
            cell = "Orthorhombic"
        elif np.allclose([alpha, gamma], 90):
            cell = "Monoclinic"
        else:
            cell = "Triclinic"

        opt.optimize_lattice(cell)

    def rescale_intensities(self):
        maximal = 10000
        scale = 1 if self.scale is None else self.scale
        if mtd[self.peaks].getNumberPeaks() > 1 and self.scale is None:
            I = np.array(mtd[self.peaks].column("Intens"))
            I0 = np.nanpercentile(I, 95)
            scale = maximal / I0
            self.scale = scale
            self.maximal = maximal

        I_min, I_max = -maximal * 10, maximal * 10

        indices = np.arange(mtd[self.peaks].getNumberPeaks())
        for i, peak in zip(indices.tolist(), mtd[self.peaks]):
            peak.setIntensity(scale * peak.getIntensity())
            if peak.getIntensity() > I_max:
                peak.setSigmaIntensity(peak.getIntensity())
            else:
                peak.setSigmaIntensity(scale * peak.getSigmaIntensity())

            peak.setIntensity(
                float(np.clip(peak.getIntensity(), I_min, I_max))
            )
            peak.setSigmaIntensity(
                float(np.clip(peak.getSigmaIntensity(), I_min, I_max))
            )

            peak.setPeakNumber(peak.getRunNumber())
            peak.setBinCount(peak.getRunNumber())
            # peak.setRunNumber(1)

        filename = os.path.splitext(self.filename)[0] + "_scale.txt"
        with open(filename, "w") as f:
            f.write("{:.4e}".format(scale))

    def median_absolute_devation(self, arr):
        med = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr - med), axis=0)

        return med, mad

    def remove_off_centered(self):
        ol = mtd[self.peaks].sample().getOrientedLattice()

        peak_err = []

        for peak in mtd[self.peaks]:
            h, k, l = peak.getHKL()

            R = peak.getGoniometerMatrix()

            two_theta = peak.getScattering()
            az_phi = peak.getAzimuthal()

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

            u = np.cross(ki_hat, kf_hat)
            u /= np.linalg.norm(u)

            v = np.cross(n, u)
            v /= np.linalg.norm(v)

            Q0 = 2 * np.pi * R @ ol.getUB() @ np.array([h, k, l])
            Q = peak.getQLabFrame()

            W = np.column_stack([n, u, v])

            peak_err.append(W @ (Q - Q0))

        peak_err = np.array(peak_err)

        med_peak_err = np.median(peak_err, axis=0)
        mad_peak_err = np.median(np.abs(peak_err - med_peak_err), axis=0)

        mask = [
            np.abs(peak_err[:, i] - med_peak_err[i]) / mad_peak_err[i] < 4.5
            for i in range(3)
        ]

        for i in range(3):
            for j, peak in enumerate(mtd[self.peaks]):
                if not mask[i][j]:
                    peak.setSigmaIntensity(float("-inf"))

    def remove_volume_outliers(self, lo=0.1, hi=10.0):
        """
        Flag peaks whose 3D Gaussian profile-fit intensity (I_3d) disagrees
        with the peak's assigned intensity (peak.getIntensity()). Peaks
        whose I_3d / getIntensity() ratio falls outside [lo, hi] are
        considered outliers and flagged for removal.
        """

        run_info = mtd[self.peaks].run()
        run_keys = run_info.keys()

        if "peaks_I_3d" not in run_keys:
            return

        dh = run_info.getLogData("peaks_h").value
        dk = run_info.getLogData("peaks_k").value
        dl = run_info.getLogData("peaks_l").value
        dm = run_info.getLogData("peaks_m").value
        dn = run_info.getLogData("peaks_n").value
        dp = run_info.getLogData("peaks_p").value
        dr = run_info.getLogData("peaks_run").value
        d_I_3d = run_info.getLogData("peaks_I_3d").value

        I_3d_lookup = {}
        for j in range(len(dr)):
            key = (dr[j], dh[j], dk[j], dl[j], dm[j], dn[j], dp[j])
            I_3d_lookup[key] = d_I_3d[j]

        for peak in mtd[self.peaks]:
            h, k, l = [int(v) for v in peak.getIntHKL()]
            m, n, p = [int(v) for v in peak.getIntMNP()]
            key = (peak.getRunNumber(), h, k, l, m, n, p)

            I_3d = I_3d_lookup.get(key)
            if I_3d is None:
                continue

            intens = peak.getIntensity()
            if not intens:
                continue

            ratio = I_3d / intens

            if not (lo < ratio < hi):
                peak.setSigmaIntensity(float("-inf"))

    def remove_non_integrated(self):
        for peak in mtd[self.peaks]:
            shape = eval(peak.getPeakShape().toJSON())

            if shape["shape"] == "none":
                peak.setSigmaIntensity(peak.getIntensity())

            elif (
                shape["radius0"] == 0
                or shape["radius1"] == 0
                or shape["radius2"] == 0
            ):
                peak.setSigmaIntensity(float("-inf"))

    def remove_non_indexed(self, tol=0.1):
        UB = mtd[self.peaks].sample().getOrientedLattice().getUB()
        for peak in mtd[self.peaks]:
            hkl = np.array(peak.getHKL())
            Q = np.array(peak.getQSampleFrame())

            Q0 = 2 * np.pi * UB @ hkl

            diff = np.abs(Q / Q0 - 1).max()
            if diff > tol:
                peak.setSigmaIntensity(float("-inf"))

    def plot_signal_noise(self, peaks=None):
        if peaks is None:
            peaks = self.peaks

        filename = os.path.splitext(self.filename)[0]

        intens = np.array(mtd[peaks].column("Intens"))
        sig_noise = np.array(mtd[peaks].column("Intens/SigInt"))

        mask = (intens > 0) & (sig_noise > 0)

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.loglog(
            intens[mask], sig_noise[mask], ".", color="C0", rasterized=True
        )
        ax.axhline(2, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Signal/Noise")
        ax.minorticks_on()
        fig.savefig(filename + "_sn.pdf")
        plt.close(fig)

    def plot_volume_fraction(self, peaks=None):
        if peaks is None:
            peaks = self.peaks

        filename = os.path.splitext(self.filename)[0]

        vol_frac = []
        for peak in mtd[peaks]:
            h, k, l = [int(v) for v in peak.getIntHKL()]
            m, n, p = [int(v) for v in peak.getIntMNP()]
            key = (peak.getRunNumber(), h, k, l, m, n, p)

            info = self.info_dict.get(key)
            if info is not None:
                vol_frac.append(info["vol_frac"])

        vol_frac = np.array(vol_frac)

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.hist(vol_frac, bins=100, color="C0")
        ax.set_xlabel("Volume fraction")
        ax.set_ylabel("Count")
        ax.minorticks_on()
        fig.savefig(filename + "_volfrac.pdf")
        plt.close(fig)

    def remove_low_volume_fraction(self, cutoff=0.5):
        """
        Flag peaks whose fitted-ellipsoid volume fraction on the
        detector (peaks_vol_frac) falls below ``cutoff``. A low
        vol_frac means intensity/sigma were extrapolated from a small
        observed fraction of the peak (edge/gap clipping), which is
        amplified by 1/vol_frac and therefore unreliable.
        """
        for peak in mtd[self.peaks]:
            h, k, l = [int(v) for v in peak.getIntHKL()]
            m, n, p = [int(v) for v in peak.getIntMNP()]
            key = (peak.getRunNumber(), h, k, l, m, n, p)

            info = self.info_dict.get(key)
            if info is None:
                continue

            if info["vol_frac"] < cutoff:
                peak.setSigmaIntensity(float("-inf"))

    def load_spectrum(self, filename, instrument):
        LoadIsawSpectrum(
            SpectraFile=filename,
            OutputWorkspace="spectrum",
            InstrumentName=instrument,
        )

    def get_cell(self):
        ol = mtd[self.peaks].sample().getOrientedLattice()
        return ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()

    def peak_info(self):
        run_info = mtd[self.peaks].run()
        run_keys = run_info.keys()

        required = [
            "h",
            "k",
            "l",
            "m",
            "n",
            "p",
            "run",
            "intens_raw",
            "sig_raw",
            "d3x",
        ]

        info_dict = {}
        norm_dict = {}
        rate_dict = {}

        log_info = np.all(
            ["peaks_{}".format(item) in run_keys for item in required]
        )
        if log_info:
            h = run_info.getLogData("peaks_h").value
            k = run_info.getLogData("peaks_k").value
            l = run_info.getLogData("peaks_l").value
            m = run_info.getLogData("peaks_m").value
            n = run_info.getLogData("peaks_n").value
            p = run_info.getLogData("peaks_p").value
            run = run_info.getLogData("peaks_run").value

            cntrt = run_info.getLogData("peaks_cntrt").value

            n_vox = run_info.getLogData("peaks_n_vox").value
            d3x = run_info.getLogData("peaks_d3x").value
            vol_frac = run_info.getLogData("peaks_vol_frac").value
            pk_data = run_info.getLogData("peaks_pk_data").value
            pk_norm = run_info.getLogData("peaks_pk_norm").value
            bkg_data = run_info.getLogData("peaks_bkg_data").value
            bkg_norm = run_info.getLogData("peaks_bkg_norm").value
            ratio = run_info.getLogData("peaks_ratio").value

            intens = run_info.getLogData("peaks_intens_raw").value
            sig = run_info.getLogData("peaks_sig_raw").value

            for i in range(len(run)):
                key = (run[i], h[i], k[i], l[i], m[i], n[i], p[i])
                info_dict[key] = {
                    "n_vox": n_vox[i],
                    "d3x": d3x[i],
                    "vol_frac": vol_frac[i],
                    "pk_data": pk_data[i],
                    "pk_norm": pk_norm[i],
                    "bkg_data": bkg_data[i],
                    "bkg_norm": bkg_norm[i],
                    "ratio": ratio[i],
                }
                norm_dict[key] = (intens[i], sig[i])
                rate_dict[run[i]] = cntrt[i]

        return info_dict, norm_dict, rate_dict

    def update_monitor(self):
        stat_dict = {}
        for peak in mtd[self.peaks]:
            run = peak.getRunNumber()
            proton_charge = peak.getBinCount()
            peak.setMonitorCount(proton_charge)
            stat_dict[run] = proton_charge
        return stat_dict

    def load_peaks(self):
        LoadNexus(Filename=self.filename, OutputWorkspace=self.peaks)

        stat_dict = self.update_monitor()

        merge = self.filename.replace(".nxs", "_diagnostics/merge.nxs")
        if os.path.exists(merge):
            LoadNexus(Filename=merge, OutputWorkspace=self.peaks + "_merge")

        ub_file = self.filename.replace(".nxs", ".mat")

        if os.path.exists(ub_file):
            LoadIsawUB(Filename=ub_file, InputWorkspace=self.peaks)

        self.filename = re.sub(
            r"(_\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\))+",
            "",
            self.filename,
        )

        info_dict, norm_dict, rate_dict = self.peak_info()

        filename = os.path.splitext(self.filename)[0]

        self.info_dict = info_dict
        self.norm_dict = norm_dict

        x, y = [], []

        for key in stat_dict.keys():
            x.append(key)
            y.append(stat_dict[key])

        fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        ax.set_xlabel("")
        ax.plot(x, y, ".", rasterized=True)
        ax.minorticks_on()
        ax.set_ylabel("Monitor")
        fig.savefig(filename + "_stat.pdf")

        x, y = [], []

        for key in rate_dict.keys():
            x.append(key)
            y.append(rate_dict[key])

        fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        ax.set_xlabel("")
        ax.plot(x, y, ".", rasterized=True)
        ax.minorticks_on()
        ax.set_ylabel("Count rate")
        fig.savefig(filename + "_rate.pdf")

        self.plot_signal_noise()
        self.plot_volume_fraction()

        self.remove_non_integrated()
        self.remove_non_indexed()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=2,
            Operator=">=",
        )

        self.remove_off_centered()
        self.remove_low_volume_fraction()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=2,
            Operator=">=",
        )

        x, y = [], []

        for key in rate_dict.keys():
            x.append(key)
            y.append(rate_dict[key])

        fig, ax = plt.subplots(1, 1, sharex=True, layout="constrained")
        ax.set_xlabel("")
        ax.plot(x, y, ".", rasterized=True)
        ax.minorticks_on()
        ax.set_ylabel("Count rate")
        fig.savefig(filename + "_rate.pdf")

        rate_ave = np.mean([rate_dict.get(key) for key in rate_dict.keys()])

        for peak in mtd[self.peaks]:
            run = int(peak.getRunNumber())
            rate = rate_dict.get(run)
            if rate is None:
                scale = 0
            else:
                scale = rate_ave / rate
            peak.setIntensity(scale * peak.getIntensity())
            peak.setSigmaIntensity(scale * peak.getSigmaIntensity())

        lamda = np.array(mtd[self.peaks].column("Wavelength"))

        kde = scipy.stats.gaussian_kde(lamda)

        x = np.linspace(lamda.min(), lamda.max(), 1000)

        pdf = kde(x)

        cdf = scipy.integrate.cumulative_trapezoid(pdf, x, initial=0)
        cdf /= cdf[-1]

        lower_bound = x[np.searchsorted(cdf, 0.0001)]
        upper_bound = x[np.searchsorted(cdf, 0.9999)]

        filename = os.path.splitext(self.filename)[0]

        fig, ax = plt.subplots(layout="constrained")
        ax.hist(lamda, bins=100, density=True, color="C0")
        ax.set_xlabel("$\lambda$ [$\AA$]")
        ax.minorticks_on()
        ax.plot(x, pdf, color="C1")
        ax.axvline(lower_bound, color="k", linestyle="--", linewidth=1)
        ax.axvline(upper_bound, color="k", linestyle="--", linewidth=1)
        fig.savefig(filename + ".pdf")

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=lower_bound,
            Operator=">",
        )

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Wavelength",
            FilterValue=upper_bound,
            Operator="<",
        )

        self.remove_volume_outliers()

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=-1,
            Operator=">=",
        )

        self.reset_satellite()

        if mtd.doesExist(self.peaks + "_merge"):
            ol = mtd[self.peaks + "_merge"].sample().getOrientedLattice()
            ol.setMaxOrder(self.max_order)
            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)
            ol.setModUB(self.modUB)

        FilterPeaks(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            FilterVariable="Signal/Noise",
            FilterValue=-1,
            Operator=">=",
        )

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="DSpacing",
            SortAscending=False,
        )

        SortPeaksWorkspace(
            InputWorkspace=self.peaks,
            OutputWorkspace=self.peaks,
            ColumnNameToSortBy="Intens",
            SortAscending=False,
        )

    def renormalize_intensities(self, plan, vanadium_file, flux_file):
        """
        Reapply the flux/solid-angle/Lorentz normalization
        (``DataModel.approximate_norm``) to each peak's raw
        intensity/sigma (``self.norm_dict``) -- e.g. to redo the
        correction with an updated flux file without re-integrating.
        Builds the detector geometry from the first real run in the
        plan (``DataModel.update_raw_path``), the same way the real
        reduction does, rather than simulating an instrument.
        """
        data = DataModel(beamlines[plan["Instrument"]])
        data.update_raw_path(plan)

        data.load_generate_normalization(vanadium_file, flux_file)
        data.preprocess_detectors(data.instrument)

        norm_file = os.path.splitext(self.filename)[0] + "_normalization"
        plot_normalization(data, norm_file)

        for peak in mtd[self.peaks]:
            h, k, l = [int(val) for val in peak.getIntHKL()]
            m, n, p = [int(val) for val in peak.getIntMNP()]

            run = int(peak.getRunNumber())
            key = (run, h, k, l, m, n, p)
            raw_intens, raw_sig = self.norm_dict[key]

            lamda = peak.getWavelength()
            two_theta = np.rad2deg(peak.getScattering())
            det_ID = peak.getDetectorID()
            proton_charge = peak.getBinCount()

            norm = data.approximate_norm(
                lamda, two_theta, det_ID, proton_charge
            )

            if not (np.isfinite(norm) and norm > 0):
                continue

            peak.setIntensity(raw_intens / norm)
            peak.setSigmaIntensity(raw_sig / norm)

    def merge_intensities(self, name=None):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        filename = os.path.splitext(self.filename)[0] + app + "_merge"

        for peak in mtd[peaks + "_merge"]:
            peak.setIntensity(self.scale * peak.getIntensity())
            peak.setSigmaIntensity(self.scale * peak.getSigmaIntensity())

        for col in ["h", "k", "l", "DSpacing"]:
            SortPeaksWorkspace(
                InputWorkspace=peaks + "_merge",
                OutputWorkspace=peaks + "_merge",
                ColumnNameToSortBy=col,
                SortAscending=False,
            )

        SaveReflections(
            InputWorkspace=peaks + "_merge",
            Filename=filename + "_jana.int",
            Format="Jana",
        )

        SaveReflections(
            InputWorkspace=peaks + "_merge",
            Filename=filename + "_fullprof.int",
            Format="Fullprof",
        )

    def reset_satellite(self, peaks=None):
        mod_mnp = []
        mod_hkl = []
        if peaks is None:
            peaks = self.peaks
        for peak in mtd[peaks]:
            hkl = peak.getHKL()
            int_hkl = peak.getIntHKL()
            int_mnp = peak.getIntMNP()
            if int_mnp.norm2() > 0:
                mod_mnp.append(np.array(int_mnp))
                mod_hkl.append(np.array(hkl - int_hkl))

        ol = mtd[peaks].sample().getOrientedLattice()

        if len(mod_mnp) > 0:
            mod_vec = np.linalg.pinv(mod_mnp) @ np.array(mod_hkl)

            self.mod_vec_1 = V3D(*mod_vec[0])
            self.mod_vec_2 = V3D(*mod_vec[1])
            self.mod_vec_3 = V3D(*mod_vec[2])
            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)

            ol.setModUB(ol.getUB() @ ol.getModHKL())

            max_order = ol.getMaxOrder()

            self.max_order = max_order if max_order > 0 else 1
            self.modUB = ol.getModUB().copy()
            self.modHKL = ol.getModHKL().copy()

            ol.setMaxOrder(self.max_order)

        else:
            self.max_order = 0
            self.modUB = np.zeros((3, 3))
            self.modHKL = np.zeros((3, 3))
            self.mod_vec_1 = V3D(0, 0, 0)
            self.mod_vec_2 = V3D(0, 0, 0)
            self.mod_vec_3 = V3D(0, 0, 0)

            ol.setMaxOrder(self.max_order)

            ol.setModVec1(self.mod_vec_1)
            ol.setModVec2(self.mod_vec_2)
            ol.setModVec3(self.mod_vec_3)

            ol.setModUB(self.modUB)

    def save_peaks(
        self,
        name=None,
        fit_dict=None,
        mu=0.0,
        transmission=None,
        instrument=None,
        correction_type=None,
        control_software=None,
        crystal_size=None,
    ):
        if name is not None:
            peaks = name
            app = "_{}".format(name).replace(" ", "_")
        else:
            peaks = self.peaks
            app = ""

        filename = os.path.splitext(self.filename)[0] + app

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="Signal/Noise",
            FilterValue=2,
            Operator=">=",
        )

        self.rescale_intensities()

        FilterPeaks(
            InputWorkspace=peaks,
            OutputWorkspace=peaks,
            FilterVariable="Signal/Noise",
            FilterValue=2,
            Operator=">=",
        )

        SortPeaksWorkspace(
            InputWorkspace=peaks,
            ColumnNameToSortBy="PeakNumber",
            SortAscending=True,
            OutputWorkspace=peaks,
        )

        self.calculate_statistics(peaks, filename + "_symm.txt")

        self.renumber_peaks(peaks)

        SaveHKL(
            InputWorkspace=peaks,
            Filename=filename + ".hkl",
            DirectionCosines=True,
            ApplyAnvredCorrections=False,
            SortBy="RunNumber",
        )

        SaveReflections(
            InputWorkspace=peaks,
            Filename=filename + ".int",
            Format="Jana",
        )

        self.refine_UB(peaks)
        self.refine_ellipsoids(peaks)

        SaveIsawUB(InputWorkspace=peaks, Filename=filename + ".mat")

        self.write_cif_report(
            peaks,
            mu=mu,
            transmission=transmission,
            instrument=instrument,
            correction_type=correction_type,
            control_software=control_software,
            crystal_size=crystal_size,
            filename=filename + ".cif",
        )

        self.resort_hkl(peaks, filename + ".hkl")

        if self.max_order > 0:
            nuclear = peaks + "_nuc"
            satellite = peaks + "_sat"

            for peak in mtd[peaks]:
                peak.setRunNumber(1)
                peak.setPeakNumber(1)

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=nuclear,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator="=",
            )

            nuc_ol = mtd[nuclear].sample().getOrientedLattice()
            nuc_ol.setMaxOrder(0)
            nuc_ol.setModVec1(V3D(0, 0, 0))
            nuc_ol.setModVec2(V3D(0, 0, 0))
            nuc_ol.setModVec3(V3D(0, 0, 0))
            nuc_ol.setModUB(np.zeros((3, 3)))

            SaveHKL(
                InputWorkspace=nuclear,
                Filename=filename + "_nuc.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(InputWorkspace=nuclear, Filename=filename + "_nuc.mat")

            self.resort_hkl(nuclear, filename + "_nuc.hkl")

            FilterPeaks(
                InputWorkspace=peaks,
                OutputWorkspace=satellite,
                FilterVariable="m^2+n^2+p^2",
                FilterValue=0,
                Operator=">",
            )

            sat_ol = mtd[satellite].sample().getOrientedLattice()
            sat_ol.setMaxOrder(self.max_order)
            sat_ol.setModVec1(V3D(*self.modHKL[:, 0]))
            sat_ol.setModVec2(V3D(*self.modHKL[:, 1]))
            sat_ol.setModVec3(V3D(*self.modHKL[:, 2]))
            sat_ol.setModUB(self.modUB)

            SaveHKL(
                InputWorkspace=satellite,
                Filename=filename + "_sat.hkl",
                DirectionCosines=True,
                ApplyAnvredCorrections=False,
                SortBy="RunNumber",
            )

            SaveIsawUB(
                InputWorkspace=satellite, Filename=filename + "_sat.mat"
            )

            self.resort_hkl(satellite, filename + "_sat.hkl")

            if np.linalg.norm(self.modHKL[:, 1]) > 0:
                for i in range(3):
                    if np.linalg.norm(self.modHKL[:, i]) > 0:
                        CloneWorkspace(
                            InputWorkspace=peaks, OutputWorkspace="tmp"
                        )

                        for peak in mtd["tmp"]:
                            int_mnp = peak.getIntMNP()
                            if int_mnp[i] == 0:
                                peak.setIntensity(0)
                                peak.setSigmaIntensity(0)
                            else:
                                peak.setIntMNP(V3D(int_mnp[i], 0, 0))

                        ol = mtd["tmp"].sample().getOrientedLattice()
                        ol.setMaxOrder(self.max_order)
                        ol.setModVec1(V3D(*self.modHKL[:, i]))
                        ol.setModVec2(V3D(0, 0, 0))
                        ol.setModVec3(V3D(0, 0, 0))
                        ol.setModUB(ol.getUB() @ ol.getModHKL())

                        k = i + 1

                        SaveHKL(
                            InputWorkspace="tmp",
                            Filename=filename + "_sat_k={}.hkl".format(k),
                            DirectionCosines=True,
                            ApplyAnvredCorrections=False,
                            SortBy="RunNumber",
                        )

                        SaveIsawUB(
                            InputWorkspace="tmp",
                            Filename=filename + "_sat_k={}.mat".format(k),
                        )

                        self.resort_hkl(
                            "tmp", filename + "_sat_k={}.hkl".format(k)
                        )

    def _angle_distance_matrix(self, Rs):
        G = np.tensordot(Rs, Rs, axes=([1, 2], [1, 2]))
        cos_th = (G - 1.0) / 2.0
        np.clip(cos_th, -1.0, 1.0, out=cos_th)
        D = np.arccos(cos_th)
        np.fill_diagonal(D, 0.0)
        return D

    def renumber_peaks(self, peaks, k=60, decimals=6, linkage="average"):
        Rs_all = []
        for p in mtd[peaks]:
            Rs_all.append(p.getGoniometerMatrix())
        Rs_all = np.asarray(Rs_all)
        N = len(Rs_all)

        flat = np.round(Rs_all.reshape(N, 9), decimals=decimals)
        uniq_rows, inv = np.unique(flat, axis=0, return_inverse=True)
        Rs_unique = np.array(
            [Rs_all[np.where(inv == i)[0][0]] for i in range(len(uniq_rows))]
        )
        M = len(Rs_unique)

        D = self._angle_distance_matrix(Rs_unique)

        n_clusters = min(k, M) if M > 0 else 0
        if n_clusters <= 1:
            labels_unique = (
                np.zeros(M, dtype=int) if M else np.array([], dtype=int)
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage=linkage
            )
            labels_unique = model.fit_predict(D)

        old_ids = np.unique(labels_unique)
        new_id_map = {old: i + 1 for i, old in enumerate(sorted(old_ids))}
        new_labels_unique = np.array(
            [new_id_map[x] for x in labels_unique], dtype=int
        )

        labels_all = new_labels_unique[inv]

        for ind, (peak, lbl) in enumerate(zip(mtd[peaks], labels_all)):
            peak.setRunNumber(int(lbl))
            peak.setPeakNumber(ind)

    def calculate_statistics(self, name, filename):
        FilterPeaks(
            InputWorkspace=name,
            OutputWorkspace=name + "_stats",
            FilterVariable="Signal/Noise",
            FilterValue=2,
            Operator=">=",
        )

        point_groups, R_merge = [], []
        for point_group in self.point_groups:
            StatisticsOfPeaksWorkspace(
                InputWorkspace=name + "_stats",
                PointGroup=point_group_dict[point_group],
                OutputWorkspace="stats",
                EquivalentIntensities="Median",
                SigmaCritical=3,
                WeightedZScore=True,
            )

            R_merge.append(mtd["StatisticsTable"].toDict()["Rmerge"][0])
            point_groups.append(point_group)

        i = np.argmin(R_merge)
        point_group = point_groups[i]

        self.point_groups = [point_group]

        StatisticsOfPeaksWorkspace(
            InputWorkspace=name + "_stats",
            PointGroup=point_group_dict[point_group],
            OutputWorkspace="stats",
            EquivalentIntensities="Median",
            SigmaCritical=3,
            WeightedZScore=True,
        )

        ws = mtd["StatisticsTable"]

        self.point_group = point_group
        self.statistics = ws.row(0)

        column_names = ws.getColumnNames()
        col_widths = [max(len(str(name)), 8) for name in column_names]

        cols = [
            " ".join(
                name.ljust(col_widths[i])
                for i, name in enumerate(column_names)
            )
        ]

        for i in range(ws.rowCount()):
            row_values = []
            for j, val in enumerate(ws.row(i).values()):
                if isinstance(val, float):
                    val = "{:.2f}".format(val)
                row_values.append(str(val).ljust(col_widths[j]))

            cols.append(" ".join(row_values))

        table = "\n".join(cols)

        with open(filename, "w") as f:
            f.write("{}\n".format(point_group))
            f.write(table)

        # ---

        ol = mtd[name].sample().getOrientedLattice()
        d_max = np.max([ol.d(1, 0, 0), ol.d(0, 1, 0), ol.d(0, 0, 1)])
        d_min = np.min(mtd[name].column("DSpacing"))

        d = 1 / np.sqrt(np.linspace(1 / d_max**2, 1 / d_min**2, 5))

        column_names = "Resolution", "Completeness", "Redundancy", "Unique"
        col_widths = [max(len(str(name)), 12) for name in column_names]

        cols = [
            " ".join(
                name.ljust(col_widths[i])
                for i, name in enumerate(column_names)
            )
        ]

        for i in range(len(d) - 1):
            unique, completeness, redundancy, _ = CountReflections(
                InputWorkspace=name,
                PointGroup=point_group,
                LatticeCentering="P",
                MinDSpacing=d[i + 1],
                MaxDSpacing=d[i],
                MissingReflectionsWorkspace="",
            )

            shell = "{:.2f}-{:.2f}".format(d[i], d[i + 1])
            values = [shell, 100 * completeness, redundancy, unique]

            for i in range(len(values)):
                row_values = []
                for j, val in enumerate(values):
                    if isinstance(val, float):
                        val = "{:.2f}".format(val)
                    row_values.append(str(val).ljust(col_widths[j]))

                cols.append(" ".join(row_values))

        table = "\n".join(cols)

        with open(filename.replace("_symm.txt", "_stats.txt"), "w") as f:
            f.write("{}\n".format(point_group))
            f.write(table)

    def write_cif_report(
        self,
        peaks=None,
        mu=0.0,
        transmission=None,
        correction_type=None,
        instrument=None,
        facility=None,
        control_software=None,
        crystal_size=None,
        filename=None,
    ):
        """
        Write a minimal CIF-format summary for post-reduction checking:
        unit-cell constants with esd's, the measured wavelength/theta/
        d-spacing range, crystal size, a transmission range, and the
        merging statistics (Rint) from the last
        ``calculate_statistics`` call.

        ``transmission``, when given, is the actual per-peak
        absorption factor already computed for the correction that
        was applied (e.g. ``AbsorptionCorrection.T``) and is reported
        as-is. Leave at None to instead approximate it from ``mu``
        (the linear absorption coefficient in cm^-1, evaluated at a
        single reference wavelength) and each peak's
        absorption-weighted path length (Tbar); ``mu=0`` (default)
        reports T = 1, i.e. no absorption correction applied.

        ``crystal_size`` is the (thickness, width, height) full-axis
        lengths of the sample ellipsoid in mm, e.g. ``self.params``
        from ``AbsorptionCorrection`` or ``self.parameters`` from
        ``NuclearStructureRefinement``; leave at None to report those
        fields as unknown.
        """
        if peaks is None:
            peaks = self.peaks

        if not hasattr(self, "statistics"):
            raise RuntimeError(
                "Run calculate_statistics before write_cif_report."
            )

        if filename is None:
            filename = os.path.splitext(self.filename)[0] + ".cif"

        ol = mtd[peaks].sample().getOrientedLattice()

        a, b, c = ol.a(), ol.b(), ol.c()
        alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()
        volume = ol.volume()

        e_a, e_b, e_c = ol.errora(), ol.errorb(), ol.errorc()
        e_alpha, e_beta, e_gamma = (
            ol.erroralpha(),
            ol.errorbeta(),
            ol.errorgamma(),
        )

        e_volume = error_volume(
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            e_a,
            e_b,
            e_c,
            e_alpha,
            e_beta,
            e_gamma,
        )

        theta = np.array([peak.getScattering() for peak in mtd[peaks]])
        theta = np.degrees(theta) / 2.0

        wavelength = np.array(mtd[peaks].column("Wavelength"))
        d_spacing = np.array(mtd[peaks].column("DSpacing"))

        transmission_exact = transmission is not None

        if transmission is None:
            tbar = np.array(
                [peak.getAbsorptionWeightedPathLength() for peak in mtd[peaks]]
            )
            transmission = np.exp(-mu * tbar)
        else:
            transmission = np.asarray(transmission, dtype=float)

        if instrument is None:
            instrument = mtd[peaks].getInstrument().getName()

        if correction_type is None:
            correction_type = "none" if mu == 0 else "ellipsoid"

        if crystal_size is not None:
            size = np.sort(np.asarray(crystal_size, dtype=float))[::-1]
            size_max, size_mid, size_min = size

            a_cm, b_cm, c_cm = np.asarray(crystal_size, dtype=float) / 10.0
            volume_cm3 = (np.pi / 6.0) * a_cm * b_cm * c_cm
            size_rad = np.cbrt(0.75 * volume_cm3 / np.pi) * 10.0  # cm -> mm
        else:
            size_max = size_mid = size_min = size_rad = None

        def fmt_or_unknown(value, precision=4):
            if value is None:
                return "?"
            return "{:.{p}f}".format(value, p=precision)

        if mu > 0 or transmission_exact:
            absorpt_process_details = (
                "Absorption correction computed by Monte Carlo\n"
                "integration over an ellipsoidal sample shape."
            )
            if transmission_exact:
                absorpt_special_details = (
                    "Transmission for each reflection is the\n"
                    "ellipsoidal absorption factor computed for the\n"
                    "applied correction, evaluated at that\n"
                    "reflection's own wavelength. The value of\n"
                    "{:.4f} mm^-1 shown in\n"
                    "_exptl_absorpt_coefficient_mu is mu evaluated at\n"
                    "a reference wavelength of 1.8 Angstrom, for\n"
                    "reference only."
                ).format(mu / 10.0)
            else:
                absorpt_special_details = (
                    "The linear absorption coefficient mu is\n"
                    "wavelength dependent; the value of {:.4f} mm^-1\n"
                    "shown in _exptl_absorpt_coefficient_mu is\n"
                    "evaluated at a reference wavelength of 1.8\n"
                    "Angstrom. The transmission for each reflection\n"
                    "is approximated from mu and that reflection's\n"
                    "absorption-weighted path length (Tbar)."
                ).format(mu / 10.0)
        else:
            absorpt_process_details = "No absorption correction applied."
            absorpt_special_details = "No absorption correction applied."

        stats = self.statistics

        lines = [
            "data_{}".format(os.path.splitext(os.path.basename(filename))[0]),
            "",
            "_cell_length_a          {}".format(format_value_esd(a, e_a)),
            "_cell_length_b          {}".format(format_value_esd(b, e_b)),
            "_cell_length_c          {}".format(format_value_esd(c, e_c)),
            "_cell_angle_alpha       {}".format(
                format_value_esd(alpha, e_alpha)
            ),
            "_cell_angle_beta        {}".format(
                format_value_esd(beta, e_beta)
            ),
            "_cell_angle_gamma       {}".format(
                format_value_esd(gamma, e_gamma)
            ),
            "_cell_volume            {}".format(
                format_value_esd(volume, e_volume, max_precision=2)
            ),
            "",
            "_cell_measurement_reflns_used     {}".format(
                mtd[peaks].getNumberPeaks()
            ),
            "_cell_measurement_theta_min       {:.3f}".format(theta.min()),
            "_cell_measurement_theta_max       {:.3f}".format(theta.max()),
            "",
            "_diffrn_ambient_temperature          ?",
            '_diffrn_radiation_wavelength         "{:.2f}-{:.2f}"'.format(
                wavelength.min(), wavelength.max()
            ),
            '_diffrn_radiation_wavelength_details  "time-of-flight Laue"',
            "_diffrn_radiation_type               neutron",
            '_diffrn_source                       "{}"'.format(
                facility if facility is not None else "?"
            ),
            "_diffrn_measurement_device_type      {}".format(instrument),
            '_diffrn_measurement_method           "time-of-flight Laue"',
            "_diffrn_detector_area_resol_mean     ?",
            "",
            "_diffrn_reflns_theta_min             {:.3f}".format(theta.min()),
            "_diffrn_reflns_theta_max             {:.3f}".format(theta.max()),
            "_diffrn_reflns_min_d_Angs            {:.3f}".format(
                d_spacing.min()
            ),
            "",
            "_exptl_crystal_size_max              {}".format(
                fmt_or_unknown(size_max)
            ),
            "_exptl_crystal_size_mid              {}".format(
                fmt_or_unknown(size_mid)
            ),
            "_exptl_crystal_size_min              {}".format(
                fmt_or_unknown(size_min)
            ),
            "_exptl_crystal_size_rad              {}".format(
                fmt_or_unknown(size_rad)
            ),
            "",
            "_exptl_absorpt_correction_type       {}".format(correction_type),
            "_exptl_absorpt_coefficient_mu        {:.4f}".format(mu / 10.0),
            "_exptl_absorpt_correction_T_min      {:.4f}".format(
                transmission.min()
            ),
            "_exptl_absorpt_correction_T_max      {:.4f}".format(
                transmission.max()
            ),
            "_exptl_absorpt_process_details",
            ";",
            absorpt_process_details,
            ";",
            "_exptl_absorpt_special_details",
            ";",
            absorpt_special_details,
            ";",
            "",
            "_diffrn_reflns_number                {}".format(
                mtd[peaks].getNumberPeaks()
            ),
            "_reflns_number_total                 {}".format(
                stats["No. of Unique Reflections"]
            ),
            "_diffrn_reflns_av_R_equivalents      {:.4f}".format(
                stats["Rmerge"] / 100.0
            ),
            "_diffrn_reflns_av_Rpim               {:.4f}".format(
                stats["Rpim"] / 100.0
            ),
            "_diffrn_reflns_point_group           {}".format(self.point_group),
            "",
            '_computing_data_collection           "{}"'.format(
                control_software if control_software is not None else "?"
            ),
            '_computing_cell_refinement           "garnet-tools (Mantid)"',
            '_computing_data_reduction            "garnet-tools (Mantid)"',
            "",
        ]

        with open(filename, "w") as f:
            f.write("\n".join(lines))

        return filename

    def resort_hkl(self, peaks, filename):
        ol = mtd[peaks].sample().getOrientedLattice()

        UB = ol.getUB()

        mod_vec_1 = ol.getModVec(0)
        mod_vec_2 = ol.getModVec(1)
        mod_vec_3 = ol.getModVec(2)

        max_order = ol.getMaxOrder()

        hkl_widths = [4, 4, 4]
        info_widths = [8, 8, 4, 8, 8, 9, 9, 9, 9, 9, 9, 6, 7, 7, 4, 9, 8, 7, 7]

        if max_order > 0:
            hkl_widths += hkl_widths

        col_widths = hkl_widths + info_widths

        h, k, l, m, n, p = [], [], [], [], [], []

        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                start = 0
                columns = []
                for width in col_widths:
                    columns.append(line[start : start + width].strip())
                    start += width
                h.append(columns[0])
                k.append(columns[1])
                l.append(columns[2])
                m.append(columns[3] if max_order > 0 else 0)
                n.append(columns[4] if max_order > 0 else 0)
                p.append(columns[5] if max_order > 0 else 0)

        h = np.array(h).astype(int)
        k = np.array(k).astype(int)
        l = np.array(l).astype(int)

        m = np.array(m).astype(int)
        n = np.array(n).astype(int)
        p = np.array(p).astype(int)

        mod_HKL = np.column_stack([mod_vec_1, mod_vec_2, mod_vec_3])

        hkl = np.stack([h, k, l]) + np.einsum("ij,jm->im", mod_HKL, [m, n, p])

        s = np.linalg.norm(np.einsum("ij,jm->im", UB, hkl), axis=0)

        hkls = np.round(np.column_stack([*hkl, s]) * 1000, 1).astype(int)
        sort = np.lexsort(hkls.T).tolist()
        with open(filename, "w") as f:
            for i in sort[1:]:
                f.write(lines[i])
            f.write(lines[sort[0]])


def main():
    parser = argparse.ArgumentParser(description="Corrections for integration")

    parser.add_argument(
        "filename",
        type=str,
        help="Peaks Workspace",
    )

    parser.add_argument(
        "-f",
        "--formula",
        type=str,
        default="Yb3-Al5-O12",
        help="Chemical formula",
    )

    parser.add_argument(
        "-z",
        "--zparameter",
        type=float,
        default="8",
        help="Number of formula units",
    )

    parser.add_argument(
        "-g",
        "--pointgroup",
        type=str,
        default=None,
        help="Point group symmetry",
    )

    parser.add_argument(
        "-u",
        "--uvector",
        nargs="+",
        type=float,
        default=[0, 0, 1],
        help="Miller indices along beam",
    )

    parser.add_argument(
        "-v",
        "--vvector",
        nargs="+",
        type=float,
        default=[1, 0, 0],
        help="Miller indices in plane",
    )

    parser.add_argument(
        "-p",
        "--parameters",
        nargs="+",
        type=float,
        default=[0.1, 0.1, 0.1],
        help="Sample Parameters",
    )

    parser.add_argument(
        "-c", "--scale", type=float, default=None, help="Scale factor"
    )

    args = parser.parse_args()

    peaks = Peaks("peaks", args.filename, args.scale, args.pointgroup)
    peaks.load_peaks()

    mu = 0.0
    transmission = None
    crystal_size = None
    if (np.array(args.parameters) > 0).all():
        abs_correc = AbsorptionCorrection(
            "peaks",
            args.formula,
            args.zparameter,
            u_vector=args.uvector,
            v_vector=args.vvector,
            params=args.parameters,
            filename=args.filename,
        )
        mu = abs_correc.mu
        transmission = abs_correc.T
        crystal_size = args.parameters

    peaks.save_peaks(
        mu=mu, transmission=transmission, crystal_size=crystal_size
    )


if __name__ == "__main__":
    main()
