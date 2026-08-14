import garnet.reduction.search  # noqa: F401  registers FindUBFromConventionalCell

from mantid.api import AlgorithmManager

from mantid.simpleapi import (
    SelectCellWithForm,
    SelectCellOfType,
    ShowPossibleCells,
    TransformHKL,
    CalculatePeaksHKL,
    IndexPeaks,
    FindUBUsingFFT,
    FindUBUsingLatticeParameters,
    FindUBUsingIndexedPeaks,
    OptimizeLatticeForCellType,
    CalculateUMatrix,
    HasUB,
    LoadIsawUB,
    SaveIsawUB,
    CopySample,
    CreateEmptyTableWorkspace,
    CreateSingleValuedWorkspace,
    SetUB,
    FilterPeaks,
    DeleteWorkspace,
    mtd,
)

from mantid.geometry import PointGroupFactory, UnitCell

import json

from itertools import product, permutations

import numpy as np

from scipy.spatial.transform import Rotation

import scipy.spatial
import scipy.optimize
import scipy.linalg

lattice_group = {
    "Triclinic": "-1",
    "Monoclinic": "2/m",
    "Orthorhombic": "mmm",
    "Tetragonal": "4/mmm",
    "Rhombohedral": "-3m",
    "Hexagonal": "6/mmm",
    "Cubic": "m-3m",
}

centering_matrices = {
    "P": np.eye(3),
    "A": np.array([[2, 0, 0], [0, 1, 1], [0, 1, -1]]) / 2,
    "B": np.array([[1, 0, 1], [0, 2, 0], [1, 0, -1]]) / 2,
    "C": np.array([[1, 1, 0], [1, -1, 0], [0, 0, 2]]) / 2,
    "I": np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]]) / 2,
    "F": np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]) / 2,
    "R": np.array([[2, -1, -1], [1, 1, -2], [1, 1, 1]]) / 3,
}


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


class UBModel:
    def __init__(self, peaks):
        """
        Tools for working with peaks and UB.

        Parameters
        ----------
        peaks : str
            Table of peaks.

        """

        self.peaks = peaks

    def get_UB(self):
        """
        Current UB matrix.

        Returns
        -------
        UB : 2d-array
            UB-matrix.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            return ol.getUB().copy()

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()
            a, b, c = ol.a(), ol.b(), ol.c()
            alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

            self.a, self.b, self.c = a, b, c
            self.alpha, self.beta, self.gamma = alpha, beta, gamma

            return a, b, c, alpha, beta, gamma

    def get_lattice_parameter_uncertanties(self):
        """
        Get the errors in the lattice constants from the oriented lattice.

        Returns
        -------
        errors : list
            List of errors in lattice constants.
        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            params = (
                ol.errora(),
                ol.errorb(),
                ol.errorc(),
                ol.erroralpha(),
                ol.errorbeta(),
                ol.errorgamma(),
            )

            params = np.array(params)
            params[~np.isfinite(params)] = 0.0

            return params.round(8).tolist()

    def get_center_uncertainty(self, hkl, min_frac=0.01, max_frac=0.05):
        """
        Propagate lattice-parameter uncertainties to a peak's Q-space
        center uncertainty.

        Parameters
        ----------
        hkl : sequence of float
            Miller indices of the peak.
        min_frac : float, optional
            Floor on each lattice-parameter uncertainty as a fraction of
            its value. Default 0.01.
        max_frac : float, optional
            Cap on each lattice-parameter uncertainty as a fraction of
            its value. Default 0.05.

        Returns
        -------
        sigma_c : float
            Norm of the propagated Q-space center uncertainty (inverse
            angstroms), or 0.0 if lattice information is unavailable.
        """
        if not mtd.doesExist(self.peaks):
            return 0.0

        ol = mtd[self.peaks].sample().getOrientedLattice()
        U = ol.getU()

        params = np.array(self.get_lattice_parameters())
        errors = np.array(self.get_lattice_parameter_uncertanties())
        sigma_params = np.clip(
            errors, min_frac * np.abs(params), max_frac * np.abs(params)
        )

        hkl = np.asarray(hkl, dtype=float)

        def Q(p):
            return 2 * np.pi * U @ _B_matrix(*p) @ hkl

        jac = np.zeros((3, 6))
        for i in range(6):
            delta = max(1e-8, 1e-4 * abs(params[i]))
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += delta
            p_minus[i] -= delta
            jac[:, i] = (Q(p_plus) - Q(p_minus)) / (2 * delta)

        return float(np.linalg.norm(jac @ sigma_params))

    def get_max_d_spacing(self):
        """
        Obtain the maximum d-spacing from the oriented lattice.

        Returns
        -------
        d_max : float
            Maximum d-spacing.

        """

        if HasUB(Workspace=self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            return 1 / min([ol.astar(), ol.bstar(), ol.cstar()])

    def has_UB(self):
        """
        Check if peaks table has a UB determined.

        """

        return HasUB(Workspace=self.peaks)

    def save_UB(self, filename):
        """
        Save UB to file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.

        """

        SaveIsawUB(InputWorkspace=self.peaks, Filename=filename)

    def load_UB(self, filename, run_number=None):
        """
        Load UB from file.

        Parameters
        ----------
        filename : str
            Name of UB file with extension .mat.
        run_number : str, optional
            Run number to replace starred expression.

        """

        LoadIsawUB(
            InputWorkspace=self.peaks,
            Filename=filename.replace("*", str(run_number)),
        )

    def determine_UB_with_primitive_cell(self, min_d, max_d, tol=0.15):
        """
        Determine UB with primitive lattice using min/max lattice constant.

        Parameters
        ----------
        min_d : float
            Minimum lattice parameter in ansgroms.
        max_d : float
            Maximum lattice parameter in angstroms.
        tol : float, optional
            Indexing tolerance. The default is 0.15.

        """

        FindUBUsingFFT(
            PeaksWorkspace=self.peaks,
            MinD=min_d,
            MaxD=max_d,
            Tolerance=tol,
            Iterations=20,
            DegreesPerStep=0.3,
        )

    def determine_UB_with_lattice_parameters(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        tol=0.2,
    ):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        FindUBUsingLatticeParameters(
            PeaksWorkspace=self.peaks,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            Tolerance=tol,
            FixParameters=True,
            NumInitial=1000,
            Iterations=1,
        )

    def determine_UB_from_conventional_cell(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        centering="P",
        tol=0.2,
    ):
        """
        Determine UB with prior known lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.
        centering : str,
            Conventional cell reflection condition.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """
        alg = AlgorithmManager.create("FindUBFromConventionalCell")
        alg.initialize()
        alg.setProperty("PeaksWorkspace", self.peaks)
        alg.setProperty("a", a)
        alg.setProperty("b", b)
        alg.setProperty("c", c)
        alg.setProperty("alpha", alpha)
        alg.setProperty("beta", beta)
        alg.setProperty("gamma", gamma)
        alg.setProperty("Centering", centering)
        alg.setProperty("Tolerance", tol)
        alg.execute()

    def convert_conventional_to_primitive(
        self,
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        centering,
    ):
        uc = UnitCell(a, b, c, alpha, beta, gamma)

        G = uc.getG()

        P = self.centering_matrix(centering)

        Gp = P.T @ G @ P

        uc.recalculateFromGstar(np.linalg.inv(Gp))

        return uc.a(), uc.b(), uc.c(), uc.alpha(), uc.beta(), uc.gamma()

    def centering_matrix(self, centering):
        return centering_matrices[centering]

    def calculate_transform_extents(self, centering):
        P = self.centering_matrix(centering)

        return np.linalg.inv(P).T

    def transform_primitive_to_conventional(self, centering):
        self.transform_lattice(self.calculate_transform_extents(centering))

    def transform_conventional_to_primitive(self, centering):
        T = np.linalg.inv(self.calculate_transform_extents(centering))
        self.transform_lattice(T)

    def get_primitive_cell_length_range(self, centering):
        const = self.get_lattice_parameters()
        const = self.convert_conventional_to_primitive(*const, centering)

        d_min = 0.9 * np.min(const[:3])
        d_max = 1.1 * np.max(const[:3])

        return d_min, d_max

    def refine_UB_without_constraints(self, tol=0.1, sat_tol=None):
        """
        Refine UB with unconstrained lattice parameters.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol

        FindUBUsingIndexedPeaks(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
        )

    def refine_UB_with_constraints(self, cell, tol=0.1):
        """
        Refine UB with constraints corresponding to lattice system.

        +----------------+---------------+----------------------+
        | Lattice system | Lengths       | Angles               |
        +================+===============+======================+
        | Cubic          | :math:`a=b=c` | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Hexagonal      | :math:`a=b`   | :math:`α=β=90,γ=120` |
        +----------------+---------------+----------------------+
        | Rhombohedral   | :math:`a=b=c` | :math:`α=β=γ`        |
        +----------------+---------------+----------------------+
        | Tetragonal     | :math:`a=b`   | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Orthorhombic   | None          | :math:`α=β=γ=90`     |
        +----------------+---------------+----------------------+
        | Monoclinic     | None          | :math:`α=γ=90`       |
        +----------------+---------------+----------------------+
        | Triclinic      | None          | None                 |
        +----------------+---------------+----------------------+

        Parameters
        ----------
        cell : float
            Lattice system.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        OptimizeLatticeForCellType(
            PeaksWorkspace=self.peaks, CellType=cell, Apply=True, Tolerance=tol
        )

    def refine_U_only(self, a, b, c, alpha, beta, gamma):
        """
        Refine the U orientation only.

        Parameters
        ----------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        CalculateUMatrix(
            PeaksWorkspace=self.peaks,
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

    def select_cell(self, number, tol=0.1):
        """
        Transform to conventional cell using form number.

        Parameters
        ----------
        number : int
            Form number.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        SelectCellWithForm(
            PeaksWorkspace=self.peaks,
            FormNumber=number,
            Apply=True,
            Tolerance=tol,
        )

    def select_type(self, cell, centering, tol=0.1):
        """
        Transform to conventional cell using cell and centering type.

        Parameters
        ----------
        cell : str
            Cell type.
        centering : str
            Centering type.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        SelectCellOfType(
            PeaksWorkspace=self.peaks,
            CellType=cell if centering != "R" else "Rhombohedral",
            Centering=centering,
            Apply=True,
            Tolerance=tol,
        )

    def possible_conventional_cells(self, max_error=0.2, permutations=True):
        """
        List possible conventional cells.

        Parameters
        ----------
        max_error : float, optional
            Max scalar error to report form numbers. The default is 0.2.
        permutations : bool, optional
            Allow permutations of the lattice. The default is True.

        Returns
        -------
        vals : list
            List of form results.

        """

        result = ShowPossibleCells(
            PeaksWorkspace=self.peaks,
            MaxScalarError=max_error,
            AllowPermutations=permutations,
            BestOnly=False,
        )

        vals = [json.loads(cell) for cell in result.Cells]

        return vals

    def transform_lattice(self, transform, tol=0.1):
        """
        Apply a cell transformation to the lattice.

        Parameters
        ----------
        transform : 3x3 array-like
            Transform to apply to hkl values.
        tol : float, optional
            Indexing tolerance. The default is 0.1.

        """

        hkl_trans = ",".join(["{},{},{}".format(*row) for row in transform])

        TransformHKL(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=False,
        )

    def generate_lattice_transforms(self, cell):
        """
        Obtain possible transforms compatabile with a unit cell lattice.

        Parameters
        ----------
        cell : str
            Latttice system.

        Returns
        -------
        transforms : dict
            Transform dictionary with symmetry operation as key.

        """

        symbol = lattice_group[cell]

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transform = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            if np.linalg.det(T) > 0:
                name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
                transform[name] = T.tolist()

        return {key: transform[key] for key in sorted(transform.keys())}

    def index_peaks(
        self,
        tol=0.15,
        sat_tol=None,
        mod_vec_1=[0, 0, 0],
        mod_vec_2=[0, 0, 0],
        mod_vec_3=[0, 0, 0],
        max_order=0,
        cross_terms=False,
    ):
        """
        Index the peaks and calculate the lattice parameter uncertainties.

        Parameters
        ----------
        tol : float, optional
            Indexing tolerance. The default is 0.1.
        sat_tol : float, optional
            Satellite indexing tolerance. The default is None.
        mod_vec_1, mod_vec_2, mod_vec_3 : list, optional
            Modulation vectors. The default is [0,0,0].
        max_order : int, optional
            Maximum order greater than zero for satellites. The default is 0.
        cross_terms : bool, optional
            Include modulation cross terms. The default is False.

        Returns
        -------
        indexing : list
            Result of indexing including number indexed and errors.

        """

        tol_for_sat = sat_tol if sat_tol is not None else tol
        save_info = True if max_order > 0 else False

        indexing = IndexPeaks(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            ToleranceForSatellite=tol_for_sat,
            RoundHKLs=True,
            CommonUBForAll=True,
            ModVector1=mod_vec_1,
            ModVector2=mod_vec_2,
            ModVector3=mod_vec_3,
            MaxOrder=max_order,
            CrossTerms=cross_terms,
            SaveModulationInfo=save_info,
        )

        return indexing

    def calculate_hkl(self):
        """
        Calculate hkl values without rounding.

        """

        CalculatePeaksHKL(PeaksWorkspace=self.peaks, OverWrite=True)

    def copy_UB(self, workspace):
        """
        Copy UB to another workspace.

        Parameters
        ----------
        workspace : float
            Target workspace to copy the UB to.

        """

        CopySample(
            InputWorkspace=self.peaks,
            OutputWorkspace=workspace,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyShape=False,
        )

    def shortest_reciprocal_spacing(self, centering, max_index=2):
        qmin = np.inf
        hmin = None

        self.transform_conventional_to_primitive(centering)

        UB_prim = mtd[self.peaks].sample().getOrientedLattice().getUB().copy()

        for dhkl in product(range(-max_index, max_index + 1), repeat=3):
            dhkl = np.asarray(dhkl, dtype=float)

            if np.all(dhkl == 0):
                continue

            dq = 2 * np.pi * UB_prim @ dhkl
            q = np.linalg.norm(dq)

            if q < qmin:
                qmin = q
                hmin = dhkl.copy()

        self.transform_primitive_to_conventional(centering)

        return qmin, hmin


class Optimization:
    def __init__(self, peaks, tol=0.1):
        """
        Optimize lattice and orientation using nonlinear least squares.

        Parameters
        ----------
        peaks : str
            Name of peaks workspace to perform constrained UB optimization.
        tol : float
            Indexing tolerance for optimization.

        """

        self.peaks = peaks

        ub_inv = np.linalg.inv(self.get_UB()) / (2 * np.pi)

        Qs, hkls, Isig = [], [], []

        for pk in mtd[peaks]:
            hkl = np.array(pk.getHKL())
            Q = np.array(pk.getQSampleFrame())

            mod_Q = np.linalg.norm(Q)

            if mod_Q > 0:
                diff_hkl = np.abs(hkl - np.dot(ub_inv, Q))

                if (diff_hkl < tol).all():
                    intensity = pk.getIntensity()
                    sigma = pk.getSigmaIntensity()
                    intens_over_sig = intensity / sigma if sigma > 0 else 0.0
                    hkls.append(hkl)
                    Qs.append(Q)
                    Isig.append(intens_over_sig)

        self.Q = np.array(Qs)
        self.hkl = np.array(hkls)
        self.Isig = np.array(Isig)

        self.min_req = True if len(self.hkl) > 3 else False

    def get_UB(self):
        """
        Current UB matrux.

        Returns
        -------
        UB : 2d-array
            UB-matrix.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            return ol.getUB().copy()

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        if mtd.doesExist(self.peaks):
            ol = mtd[self.peaks].sample().getOrientedLattice()

            a, b, c = ol.a(), ol.b(), ol.c()
            alpha, beta, gamma = ol.alpha(), ol.beta(), ol.gamma()

            self.a, self.b, self.c = a, b, c
            self.alpha, self.beta, self.gamma = alpha, beta, gamma

            return a, b, c, alpha, beta, gamma

    def get_orientation_angles(self):
        """
        Current orientation angles.

        Returns
        -------
        phi : float
            Rotation axis azimuthal angle in radians.
        theta : float
            Rotation axis polar angle in radians.
        omega : float
            Rotation angle in radians.

        """

        if mtd.doesExist(self.peaks):
            U = mtd[self.peaks].sample().getOrientedLattice().getU()

            omega = np.arccos((np.trace(U) - 1) / 2)

            val, vec = np.linalg.eig(U)

            i_axis = np.argmin(np.abs(val - 1))
            if not np.isclose(val[i_axis], 1, atol=1e-3):
                raise ValueError(
                    "Orientation matrix U is not a proper rotation "
                    "(closest eigenvalue to +1 is {}); "
                    "the current UB is degenerate.".format(val[i_axis])
                )

            ux, uy, uz = vec[:, i_axis].real

            theta = np.arccos(uz)
            phi = np.arctan2(uy, ux)

            return phi, theta, omega

    def U_matrix(self, phi, theta, omega):
        u0 = np.cos(phi) * np.sin(theta)
        u1 = np.sin(phi) * np.sin(theta)
        u2 = np.cos(theta)

        w = omega * np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array(
            [
                [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
            ]
        )

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def fixed(self, x):
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        return (a, b, c, alpha, beta, gamma, *x)

    def cubic(self, x):
        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):
        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):
        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):
        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):
        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, hkl, Q, fun, W=np.eye(3), weights=None):
        """
        Optimization residual function.

        Parameters
        ----------
        x : list
            Parameters.
        hkl : list
            Miller indices.
        Q : list
            Q-sample vectors.
        fun : function
            Lattice constraint function.
        W: 3x3-array
            Weight matrix.
        weights : 1d-array, optional
            Per-peak scalar weights (e.g. I/σ). The default is None (uniform).

        Returns
        -------
        residual : list
            Least squares residuals.

        """

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        wr = (np.einsum("ij,lj->li", UB, hkl) * 2 * np.pi - Q) @ W.T

        if weights is not None:
            wr *= weights[:, None]

        return wr.flatten()

    def whiten_weight_matrix(self, Q):
        sigma = np.cov(Q.T)
        L = np.linalg.cholesky(sigma)
        W = np.linalg.inv(L)
        return W

    def optimize_lattice(self, cell, n_cycles=5, sigma_cut=3.0):
        """
        Refine the orientation and lattice parameters under constraints.

        Iterates least-squares refinement with I/σ weighting and σ-clipping
        outlier rejection. After each cycle peaks whose unweighted residual
        norm exceeds ``sigma_cut`` times the median are removed, and the next
        cycle starts from the previous solution.

        Parameters
        ----------
        cell : str
            Lattice centering to constrain parameters.
        n_cycles : int, optional
            Maximum number of sigma-clipping cycles. The default is 5.
        sigma_cut : float, optional
            Outlier rejection threshold in units of the median residual norm.
            The default is 3.0.

        """

        if mtd.doesExist(self.peaks) and self.min_req:
            a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

            phi, theta, omega = self.get_orientation_angles()

            fun_dict = {
                "Fixed": self.fixed,
                "Cubic": self.cubic,
                "Rhombohedral": self.rhombohedral,
                "Tetragonal": self.tetragonal,
                "Hexagonal": self.hexagonal,
                "Orthorhombic": self.orthorhombic,
                "Monoclinic": self.monoclinic,
                "Triclinic": self.triclinic,
            }

            x0_dict = {
                "Fixed": (),
                "Cubic": (a,),
                "Rhombohedral": (a, alpha),
                "Tetragonal": (a, c),
                "Hexagonal": (a, c),
                "Orthorhombic": (a, b, c),
                "Monoclinic": (a, b, c, beta),
                "Triclinic": (a, b, c, alpha, beta, gamma),
            }

            fun = fun_dict[cell]
            x0 = x0_dict[cell]

            W = self.whiten_weight_matrix(self.Q)

            x0 += (phi, theta, omega)

            hkl = self.hkl.copy()
            Q = self.Q.copy()
            Isig = self.Isig.copy()

            sol = None
            for _ in range(n_cycles):
                if len(hkl) <= len(x0):
                    break

                w_scale = (
                    np.median(Isig[Isig > 0]) if np.any(Isig > 0) else 1.0
                )
                weights = Isig / w_scale if w_scale > 0 else None

                args = (hkl, Q, fun, W, weights)
                sol = scipy.optimize.least_squares(
                    self.residual, x0=x0, args=args
                )

                raw_res = self.residual(sol.x, hkl, Q, fun, W).reshape(-1, 3)
                norms = np.sqrt((raw_res**2).sum(axis=1))

                cutoff = sigma_cut * np.median(norms)
                keep = norms <= cutoff

                if keep.all():
                    break

                hkl = hkl[keep]
                Q = Q[keep]
                Isig = Isig[keep]
                x0 = sol.x  # warm-start next cycle

            if sol is None:
                return

            a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

            B = self.B_matrix(a, b, c, alpha, beta, gamma)
            U = self.U_matrix(phi, theta, omega)

            UB = np.dot(U, B)

            J = sol.jac
            inv_cov = J.T.dot(J)

            cov = (
                np.linalg.inv(inv_cov)
                if np.linalg.det(inv_cov) > 0
                else np.zeros((len(sol.x), len(sol.x)))
            )

            chi2dof = np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
            cov *= chi2dof

            sig = np.sqrt(np.diagonal(cov))

            sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

            if np.isclose(a, sig_a):
                sig_a = 0
            if np.isclose(b, sig_b):
                sig_b = 0
            if np.isclose(c, sig_c):
                sig_c = 0

            if np.isclose(alpha, sig_alpha):
                sig_alpha = 0
            if np.isclose(beta, sig_beta):
                sig_beta = 0
            if np.isclose(gamma, sig_gamma):
                sig_gamma = 0

            ol = mtd[self.peaks].sample().getOrientedLattice()
            ol.setUB(UB)
            ol.setModUB(UB @ ol.getModHKL())
            ol.setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)

    def optimize_lattice_only(self, cell, n_cycles=5, sigma_cut=3.0):
        """
        Refine lattice parameters under cell constraints with U fixed.

        The crystal orientation (U matrix) is held constant; only the free
        lattice parameters allowed by ``cell`` are optimized.

        Parameters
        ----------
        cell : str
            Lattice system to constrain parameters.
        n_cycles : int, optional
            Maximum number of sigma-clipping cycles. The default is 5.
        sigma_cut : float, optional
            Outlier rejection threshold in units of the median residual norm.
            The default is 3.0.

        """

        if not (mtd.doesExist(self.peaks) and self.min_req):
            return

        a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

        phi, theta, omega = self.get_orientation_angles()

        fun_dict = {
            "Fixed": self.fixed,
            "Cubic": self.cubic,
            "Rhombohedral": self.rhombohedral,
            "Tetragonal": self.tetragonal,
            "Hexagonal": self.hexagonal,
            "Orthorhombic": self.orthorhombic,
            "Monoclinic": self.monoclinic,
            "Triclinic": self.triclinic,
        }

        x0_dict = {
            "Fixed": (),
            "Cubic": (a,),
            "Rhombohedral": (a, alpha),
            "Tetragonal": (a, c),
            "Hexagonal": (a, c),
            "Orthorhombic": (a, b, c),
            "Monoclinic": (a, b, c, beta),
            "Triclinic": (a, b, c, alpha, beta, gamma),
        }

        base_fun = fun_dict[cell]
        x0 = x0_dict[cell]

        if len(x0) == 0:
            return

        # Append fixed orientation so residual() receives the expected 9-tuple
        def fun(x):
            return base_fun((*x, phi, theta, omega))

        W = self.whiten_weight_matrix(self.Q)

        hkl = self.hkl.copy()
        Q = self.Q.copy()
        Isig = self.Isig.copy()

        sol = None
        for _ in range(n_cycles):
            if len(hkl) <= len(x0):
                break

            w_scale = np.median(Isig[Isig > 0]) if np.any(Isig > 0) else 1.0
            weights = Isig / w_scale if w_scale > 0 else None

            args = (hkl, Q, fun, W, weights)
            sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

            raw_res = self.residual(sol.x, hkl, Q, fun, W).reshape(-1, 3)
            norms = np.sqrt((raw_res**2).sum(axis=1))

            cutoff = sigma_cut * np.median(norms)
            keep = norms <= cutoff

            if keep.all():
                break

            hkl = hkl[keep]
            Q = Q[keep]
            Isig = Isig[keep]
            x0 = sol.x

        if sol is None:
            return

        a, b, c, alpha, beta, gamma, phi, theta, omega = fun(sol.x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        J = sol.jac
        inv_cov = J.T.dot(J)

        cov = (
            np.linalg.inv(inv_cov)
            if np.linalg.det(inv_cov) > 0
            else np.zeros((len(sol.x), len(sol.x)))
        )

        chi2dof = np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
        cov *= chi2dof

        sig = np.sqrt(np.diagonal(cov))

        sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

        if np.isclose(a, sig_a):
            sig_a = 0
        if np.isclose(b, sig_b):
            sig_b = 0
        if np.isclose(c, sig_c):
            sig_c = 0

        if np.isclose(alpha, sig_alpha):
            sig_alpha = 0
        if np.isclose(beta, sig_beta):
            sig_beta = 0
        if np.isclose(gamma, sig_gamma):
            sig_gamma = 0

        ol = mtd[self.peaks].sample().getOrientedLattice()
        ol.setUB(UB)
        ol.setModUB(UB @ ol.getModHKL())
        ol.setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)


def write_ub_info(info_file, run, min_d, opt, ub):
    """
    Write a lattice-refinement diagnostic text file.

    Parameters
    ----------
    info_file : str
        Output text file path.
    run : int
        Run number, for the report header.
    min_d : float
        Resolution (minimum d-spacing) used for peak prediction.
    opt : Optimization
        Lattice optimization (already `optimize_lattice()`-refined),
        for the count of peaks used.
    ub : UBModel
        UB model (already refined), for the lattice parameters and
        their uncertainties.

    """
    n_peaks = len(opt.hkl)

    a, b, c, alpha, beta, gamma = ub.get_lattice_parameters()
    (
        sig_a,
        sig_b,
        sig_c,
        sig_alpha,
        sig_beta,
        sig_gamma,
    ) = ub.get_lattice_parameter_uncertanties()

    with open(info_file, "w") as f:
        f.write("Run: {}\n".format(run))
        f.write("Peaks used in optimization: {}\n".format(n_peaks))
        f.write("Resolution (minimum d-spacing): {:.4f} Å\n".format(min_d))
        f.write("\nLattice parameters:\n")
        f.write("a = {:.4f} ± {:.4f} Å\n".format(a, sig_a))
        f.write("b = {:.4f} ± {:.4f} Å\n".format(b, sig_b))
        f.write("c = {:.4f} ± {:.4f} Å\n".format(c, sig_c))
        f.write("alpha = {:.4f} ± {:.4f} deg\n".format(alpha, sig_alpha))
        f.write("beta = {:.4f} ± {:.4f} deg\n".format(beta, sig_beta))
        f.write("gamma = {:.4f} ± {:.4f} deg\n".format(gamma, sig_gamma))


class RefineSingleCrystalGoniometer:
    def __init__(self, peaks, tol=0.12, cell="Triclinic", n_iter=1):
        self.peaks = peaks

        for iter in range(n_iter):
            self.table = self.peaks + "_#{}".format(iter)

            CreateEmptyTableWorkspace(OutputWorkspace=self.table)

            mtd[self.table].addColumn("float", "Requested Omega")
            mtd[self.table].addColumn("float", "Refined Omega")

            mtd[self.table].addColumn("float", "Requested Chi")
            mtd[self.table].addColumn("float", "Refined Chi")

            mtd[self.table].addColumn("float", "Requested Phi")
            mtd[self.table].addColumn("float", "Refined Phi")

            ol = mtd[self.peaks].sample().getOrientedLattice()

            self.U = ol.getU().copy()

            self.a = ol.a()
            self.b = ol.b()
            self.c = ol.c()
            self.alpha = ol.alpha()
            self.beta = ol.beta()
            self.gamma = ol.gamma()

            self.peak_dict = {}

            runs = np.unique(mtd[self.peaks].column("RunNumber")).tolist()

            IndexPeaks(
                PeaksWorkspace=self.peaks, Tolerance=tol, CommonUBForAll=False
            )

            for i, run in enumerate(runs):
                FilterPeaks(
                    InputWorkspace=self.peaks,
                    FilterVariable="RunNumber",
                    FilterValue=run,
                    Operator="=",
                    OutputWorkspace="_tmp",
                )

                Q = np.array(mtd["_tmp"].column("QLab"))
                hkl = np.array(mtd["_tmp"].column("IntHKL"))

                mask = hkl.any(axis=1)

                R = mtd["_tmp"].getPeak(0).getGoniometerMatrix().copy()

                omega, chi, phi = (
                    Rotation.from_matrix(R)
                    .as_euler("YZY", degrees=True)
                    .tolist()
                )

                self.peak_dict[run] = (omega, chi, phi), Q[mask], hkl[mask]

                DeleteWorkspace(Workspace="_tmp")

            self.optimize_lattice(cell)

    def calculate_goniometer(self, omega, chi, phi):
        return Rotation.from_euler(
            "YZY", [omega, chi, phi], degrees=True
        ).as_matrix()

    def get_orientation_angles(self):
        """
        Current orientation angles.

        Returns
        -------
        phi : float
            Rotation axis azimuthal angle in radians.
        theta : float
            Rotation axis polar angle in radians.
        omega : float
            Rotation angle in radians.

        """

        omega = np.arccos((np.trace(self.U) - 1) / 2)

        val, vec = np.linalg.eig(self.U)

        i_axis = np.argmin(np.abs(val - 1))
        if not np.isclose(val[i_axis], 1, atol=1e-3):
            raise ValueError(
                "Orientation matrix U is not a proper rotation "
                "(closest eigenvalue to +1 is {}); "
                "the current UB is degenerate.".format(val[i_axis])
            )

        ux, uy, uz = vec[:, i_axis].real

        theta = np.arccos(uz)
        phi = np.arctan2(uy, ux)

        return phi, theta, omega

    def get_lattice_parameters(self):
        """
        Current lattice parameters.

        Returns
        -------
        a, b, c : float
            Lattice constants in angstroms.
        alpha, beta, gamma : float
            Lattice angles in degrees.

        """

        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma

        return a, b, c, alpha, beta, gamma

    def U_matrix(self, phi, theta, omega):
        u0 = np.cos(phi) * np.sin(theta)
        u1 = np.sin(phi) * np.sin(theta)
        u2 = np.cos(theta)

        w = omega * np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(w).as_matrix()

        return U

    def B_matrix(self, a, b, c, alpha, beta, gamma):
        alpha, beta, gamma = np.deg2rad([alpha, beta, gamma])

        G = np.array(
            [
                [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
                [b * a * np.cos(gamma), b**2, b * c * np.cos(alpha)],
                [c * a * np.cos(beta), c * b * np.cos(alpha), c**2],
            ]
        )

        B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

        return B

    def fixed(self, x):
        a, b, c = self.a, self.b, self.c
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        return (a, b, c, alpha, beta, gamma, *x)

    def cubic(self, x):
        a, *params = x

        return (a, a, a, 90, 90, 90, *params)

    def rhombohedral(self, x):
        a, alpha, *params = x

        return (a, a, a, alpha, alpha, alpha, *params)

    def tetragonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 90, *params)

    def hexagonal(self, x):
        a, c, *params = x

        return (a, a, c, 90, 90, 120, *params)

    def orthorhombic(self, x):
        a, b, c, *params = x

        return (a, b, c, 90, 90, 90, *params)

    def monoclinic(self, x):
        a, b, c, beta, *params = x

        return (a, b, c, 90, beta, 90, *params)

    def triclinic(self, x):
        a, b, c, alpha, beta, gamma, *params = x

        return (a, b, c, alpha, beta, gamma, *params)

    def residual(self, x, peak_dict, func):
        """
        Optimization residual function.

        Parameters
        ----------
        x : list
            Parameters.
        peak_dict : dictionary
            Goniometer angles, Q-lab vectors, Miller indices.            .
        func : function
            Lattice constraint function.

        Returns
        -------
        residual : list
            Least squares residuals.

        """

        a, b, c, alpha, beta, gamma, phi, theta, omega, *params = func(x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        UB = np.dot(U, B)

        # ub_inv = np.linalg.inv(2 * np.pi * UB)

        params = np.array(params).reshape(-1, 3)

        diff = []

        for i, run in enumerate(peak_dict.keys()):
            (omega, chi, phi), Q, hkl = peak_dict[run]
            omega_off, chi_off, phi_off = params[i]
            R = self.calculate_goniometer(
                omega + omega_off, chi + chi_off, phi + phi_off
            )
            # hkl = np.einsum("ij,lj->li", ub_inv @ R.T, Q)
            # int_hkl = np.round(hkl)
            # diff += (hkl - int_hkl).flatten().tolist()
            diff += (
                (np.einsum("ij,lj->li", R @ UB, hkl) * 2 * np.pi - Q)
                .flatten()
                .tolist()
            )

        return diff + params.flatten().tolist()

    def optimize_lattice(self, cell):
        """
        Refine the orientation and lattice parameters under constraints.

        Parameters
        ----------
        cell : str
            Lattice centering to constrain paramters.

        """

        a, b, c, alpha, beta, gamma = self.get_lattice_parameters()

        phi, theta, omega = self.get_orientation_angles()

        fun_dict = {
            "Fixed": self.fixed,
            "Cubic": self.cubic,
            "Rhombohedral": self.rhombohedral,
            "Tetragonal": self.tetragonal,
            "Hexagonal": self.hexagonal,
            "Orthorhombic": self.orthorhombic,
            "Monoclinic": self.monoclinic,
            "Triclinic": self.triclinic,
        }

        x0_dict = {
            "Fixed": (),
            "Cubic": (a,),
            "Rhombohedral": (a, alpha),
            "Tetragonal": (a, c),
            "Hexagonal": (a, c),
            "Orthorhombic": (a, b, c),
            "Monoclinic": (a, b, c, beta),
            "Triclinic": (a, b, c, alpha, beta, gamma),
        }

        fun = fun_dict[cell]
        x0 = x0_dict[cell]

        n = 3 * len(self.peak_dict.keys())

        x0 += (phi, theta, omega) + (0,) * n
        args = (self.peak_dict, fun)

        sol = scipy.optimize.least_squares(self.residual, x0=x0, args=args)

        a, b, c, alpha, beta, gamma, phi, theta, omega, *params = fun(sol.x)

        B = self.B_matrix(a, b, c, alpha, beta, gamma)
        U = self.U_matrix(phi, theta, omega)

        params = np.array(params).reshape(-1, 3)

        peak_dict = {}
        for i, run in enumerate(self.peak_dict.keys()):
            (omega, chi, phi), Q, hkl = self.peak_dict[run]
            omega_off, chi_off, phi_off = params[i]
            omega_prime, chi_prime, phi_prime = (
                omega + omega_off,
                chi + chi_off,
                phi + phi_off,
            )
            mtd[self.table].addRow(
                [omega, omega_prime, chi, chi_prime, phi, phi_prime]
            )
            R = self.calculate_goniometer(omega_prime, chi_prime, phi_prime)
            peak_dict[run] = R

        for peak in mtd[self.peaks]:
            run = peak.getRunNumber()
            peak.setGoniometerMatrix(peak_dict[run])

        UB = np.dot(U, B)

        J = sol.jac
        cov = np.linalg.inv(J.T.dot(J))

        chi2dof = np.sum(sol.fun**2) / (sol.fun.size - sol.x.size)
        cov *= chi2dof

        sig = np.sqrt(np.diagonal(cov))

        sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma, *_ = fun(sig)

        if np.isclose(a, sig_a):
            sig_a = 0
        if np.isclose(b, sig_b):
            sig_b = 0
        if np.isclose(c, sig_c):
            sig_c = 0

        if np.isclose(alpha, sig_alpha):
            sig_alpha = 0
        if np.isclose(beta, sig_beta):
            sig_beta = 0
        if np.isclose(gamma, sig_gamma):
            sig_gamma = 0

        ol = mtd[self.peaks].sample().getOrientedLattice()
        ol.setUB(UB)
        ol.setModUB(UB @ ol.getModHKL())
        ol.setError(sig_a, sig_b, sig_c, sig_alpha, sig_beta, sig_gamma)


class Reorient:
    def __init__(
        self, peaks, UB_ref, crystal_system="Triclinic", lattice_system=None
    ):
        if lattice_system is None:
            lattice_system = crystal_system

        self.peaks = peaks

        ol = mtd[self.peaks].sample().getOrientedLattice()

        self.UB = ol.getUB().copy()
        self.UB_ref = UB_ref.copy()

        # Triclinic/Monoclinic/Orthorhombic have no two axes constrained
        # equal by symmetry, so a from-scratch UB determination has no way
        # to know which physical direction the reference calls a/b/c --
        # e.g. an orthorhombic cell can come back as (c, a, b). Resolve
        # that axis-labeling ambiguity first; for the other systems this
        # is already covered by the point group itself (its rotations
        # already permute the symmetry-equal axes), so it's a no-op.
        P = self.resolve_axis_ambiguity(crystal_system)

        transforms = self.cell_symmetry_matrices(
            crystal_system, lattice_system
        )

        self.minimize(transforms, P)

    def _lattice_parameters(self, UB):
        """
        a, b, c, alpha, beta, gamma for an arbitrary UB matrix, via a
        scratch workspace so this reuses Mantid's own UB<->lattice
        parameter convention instead of re-deriving the metric tensor.
        """

        name = "_reorient_lattice_params"

        CreateSingleValuedWorkspace(OutputWorkspace=name)
        SetUB(Workspace=name, UB=UB)

        ol = mtd[name].sample().getOrientedLattice()
        params = np.array(
            [ol.a(), ol.b(), ol.c(), ol.alpha(), ol.beta(), ol.gamma()]
        )

        DeleteWorkspace(Workspace=name)

        return params

    def resolve_axis_ambiguity(self, crystal_system):
        """
        Find the axis permutation (plus, if needed, a compensating single
        axis flip) of self.UB whose (a, b, c, alpha, beta, gamma) best
        matches self.UB_ref's, independent of crystal system (this is a
        general permutation-and-compare search, not specific cell-type
        logic). Only Triclinic/Monoclinic/Orthorhombic actually need it
        -- see __init__.

        A plain axis permutation is improper (det < 0) for an odd
        permutation (e.g. a single a<->b swap) -- left uncorrected, that
        would flip the resulting UB left-handed. Real UB determinations
        are always right-handed, so an odd permutation is only ever
        realistic paired with a compensating sign flip on one axis; which
        axis is tried explicitly (flipping changes 2 of the 3 angles, not
        just a sign, so the wrong choice would fail the angle match) and
        only kept if it actually restores det > 0.

        Returns
        -------
        P : ndarray, shape (3, 3)
            Proper (det > 0) axis-permutation/sign-flip matrix.
        """

        if crystal_system not in (
            "Triclinic",
            "Monoclinic",
            "Orthorhombic",
        ):
            return np.eye(3)

        ref_params = self._lattice_parameters(self.UB_ref)

        best_P, best_cost = np.eye(3), np.inf

        for perm in permutations(range(3)):
            perm_P = np.eye(3)[:, perm]

            if np.linalg.det(perm_P) > 0:
                candidates = [perm_P]
            else:
                candidates = [
                    np.diag(signs).astype(float) @ perm_P
                    for signs in ([-1, 1, 1], [1, -1, 1], [1, 1, -1])
                ]

            for P in candidates:
                if np.linalg.det(P) <= 0:
                    continue

                params = self._lattice_parameters(self.UB @ np.linalg.inv(P))

                length_cost = np.sum(
                    ((params[:3] - ref_params[:3]) / ref_params[:3]) ** 2
                )
                angle_cost = np.sum(
                    ((params[3:] - ref_params[3:]) / 180.0) ** 2
                )
                cost = length_cost + angle_cost

                if cost < best_cost:
                    best_cost, best_P = cost, P

        return best_P

    def cell_symmetry_matrices(self, crystal_system, lattice_system):
        if crystal_system == "Cubic":
            symbol = "m-3m"
        elif crystal_system == "Hexagonal":
            symbol = "6/mmm"
        elif crystal_system == "Tetragonal":
            symbol = "4/mmm"
        elif crystal_system == "Trigonal":
            if lattice_system == "Rhombohedral":
                symbol = "-3m r"
            elif lattice_system == "Hexagonal":
                symbol = "-3m"
        elif crystal_system == "Orthorhombic":
            symbol = "mmm"
        elif crystal_system == "Monoclinic":
            symbol = "2/m"
        elif crystal_system == "Triclinic":
            symbol = "-1"

        pg = PointGroupFactory.createPointGroup(symbol)

        coords = np.eye(3).astype(int)

        transforms = {}
        for symop in pg.getSymmetryOperations():
            T = np.column_stack([symop.transformHKL(vec) for vec in coords])
            name = "{}: ".format(symop.getOrder()) + symop.getIdentifier()
            transforms[name] = T

        return transforms

    def minimize(self, transforms, P, tol=0.12):
        """
        Search the crystal system's full point group (proper and
        improper) composed with the axis-permutation P, keeping only
        the combinations that stay a valid, proper (det > 0) transform,
        for whichever best matches UB_ref.

        P alone may be improper (an odd permutation, det = -1, e.g. a
        plain swap of two axes) -- composing it with the point group's
        *proper* rotations only would then always stay improper, since
        proper x proper = proper and proper x improper = improper. Point
        group elements are filtered by the combined transform's
        determinant here (not on their own beforehand) so the correct
        subset -- proper or improper -- is used in either case.
        """

        cost, T_best = np.inf, P.copy()

        for _, M in transforms.items():
            T = M @ P

            if np.linalg.det(T) <= 0:
                continue

            UBp = self.UB @ np.linalg.inv(T)
            fro = np.linalg.norm(UBp - self.UB_ref)
            if fro < cost:
                cost = fro
                T_best = T

        hkl_trans = ",".join(9 * ["{}"]).format(*T_best.flatten())

        TransformHKL(
            PeaksWorkspace=self.peaks,
            Tolerance=tol,
            HKLTransform=hkl_trans,
            FindError=False,
        )
