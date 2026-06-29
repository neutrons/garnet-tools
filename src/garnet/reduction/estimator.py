"""Poisson peak estimator with a fixed measured-background field.

This module is intended as a drop-in replacement for the experimental global
peak-shape estimator.  The main changes relative to the previous version are:

* The Poisson likelihood is evaluated on the raw sample counts ``d``.
* The measured background ``b`` enters the mean as a fixed field.
* A single additional flat sample-background offset ``B`` is fitted per peak.
* The per-peak amplitude/background fit is bounded with ``A >= 0, B >= 0``.
* The global outer optimization uses a cost-only objective by default, avoiding
  the inconsistent analytic gradient that caused convergence problems.
* A fixed high-SNR active set is selected before the global optimization.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.optimize
import scipy.stats

from garnet.plots.peaks import PeakEstimatePlot
from garnet.reduction.intensity import bin_extent
from garnet.reduction.intensity import revert_ellipsoid_parameters
from garnet.reduction.peaks import PeakModel

_CHI2_SCALE_3D = scipy.stats.chi2.ppf(0.997, df=3)  # ~= 14.16
_CHI2_SCALE_1D = scipy.stats.chi2.ppf(0.997, df=1)  # ~= 8.81
_R_SCALE_3D = np.sqrt(_CHI2_SCALE_3D)  # ~= 3.76
_R_SCALE_1D = np.sqrt(_CHI2_SCALE_1D)  # ~= 2.97


@dataclass
class OptimizerDiagnostics:
    """Small container for outer-optimization diagnostics."""

    success: bool = False
    message: str = ""
    nit: int = 0
    nfev: int = 0
    fun: float = np.nan
    grad_norm: float = np.nan
    n_peaks_total: int = 0
    n_peaks_active: int = 0
    n_bad_spd: int = 0
    n_bad_fit: int = 0


def _vech6_to_S(y):
    """Convert [S00, S11, S22, S12, S02, S01] to a symmetric matrix."""
    S = np.zeros((3, 3), dtype=float)
    S[0, 0] = y[0]
    S[1, 1] = y[1]
    S[2, 2] = y[2]
    S[1, 2] = S[2, 1] = y[3]
    S[0, 2] = S[2, 0] = y[4]
    S[0, 1] = S[1, 0] = y[5]
    return S


def _is_spd(S, min_eig=1e-12):
    """Return whether S is numerically positive definite and its min eig."""
    S = 0.5 * (S + S.T)
    try:
        vals = np.linalg.eigvalsh(S)
    except np.linalg.LinAlgError:
        return False, -np.inf
    return bool(np.all(vals > min_eig)), float(vals[0])


def _poisson_nll(d, mu, eps=1e-12):
    """Poisson negative log-likelihood, omitting log(d!)."""
    mu = np.maximum(mu, eps)
    return float(np.sum(mu - d * np.log(mu)))


def _fit_A_B_poisson(g, d, b, eps=1e-12, maxiter=50):
    """Fit bounded A and B for mu = A*g + B + b.

    Parameters
    ----------
    g : ndarray
        Gaussian template values on valid voxels.
    d : ndarray
        Raw sample counts on valid voxels.
    b : ndarray
        Fixed measured-background estimate on valid voxels.
    eps : float
        Numerical floor for the Poisson mean.
    maxiter : int
        Maximum inner L-BFGS-B iterations.

    Returns
    -------
    A, B, A_err, B_err, cost, ok : tuple
        Bounded MLE, Fisher errors, NLL cost, and success flag.
    """
    if g.size == 0 or d.size == 0 or b.size == 0:
        return 0.0, 0.0, np.inf, np.inf, np.inf, False

    if not np.any(g > 0.0):
        return 0.0, 0.0, np.inf, np.inf, np.inf, False

    net = d - b
    B0 = max(float(np.nanpercentile(net, 20.0)), eps)
    A0 = max(float(np.nanmax(net) - B0), eps)
    p0 = np.array([A0, B0], dtype=float)

    def fun(p):
        A, B = p
        mu = A * g + B + b
        return _poisson_nll(d, mu, eps=eps)

    def jac(p):
        A, B = p
        mu = np.maximum(A * g + B + b, eps)
        r = 1.0 - d / mu
        return np.array([np.sum(r * g), np.sum(r)], dtype=float)

    result = scipy.optimize.minimize(
        fun,
        p0,
        jac=jac,
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None)],
        options={"maxiter": maxiter, "ftol": 1e-11, "gtol": 1e-8},
    )

    A, B = result.x
    mu = np.maximum(A * g + B + b, eps)
    cost = _poisson_nll(d, mu, eps=eps)

    f_AA = np.sum(g * g / mu)
    f_AB = np.sum(g / mu)
    f_BB = np.sum(1.0 / mu)
    fisher = np.array([[f_AA, f_AB], [f_AB, f_BB]], dtype=float)

    try:
        cov = np.linalg.pinv(fisher)
        A_err = float(np.sqrt(max(cov[0, 0], eps)))
        B_err = float(np.sqrt(max(cov[1, 1], eps)))
    except np.linalg.LinAlgError:
        A_err = np.inf
        B_err = np.inf

    ok = bool(np.isfinite(cost) and np.isfinite(A) and np.isfinite(B))
    return float(A), float(B), A_err, B_err, cost, ok


def _initial_center(d, b, Q0, Q1, Q2, valid):
    """Return a robust initial centroid from clipped net counts."""
    w = np.where(valid, np.maximum(d - b, 0.0), 0.0)
    wsum = float(np.sum(w))

    if wsum <= 0.0:
        w = np.where(valid, d, 0.0)
        wsum = float(np.sum(w))

    if wsum <= 0.0:
        return np.zeros(3, dtype=float)

    return (
        np.array(
            [np.sum(w * Q0), np.sum(w * Q1), np.sum(w * Q2)],
            dtype=float,
        )
        / wsum
    )


def _clip_center_to_grid(c, x0, x1, x2):
    """Keep the centroid inside the binned local coordinate grid."""
    return np.array(
        [
            np.clip(c[0], np.nanmin(x0), np.nanmax(x0)),
            np.clip(c[1], np.nanmin(x1), np.nanmax(x1)),
            np.clip(c[2], np.nanmin(x2), np.nanmax(x2)),
        ],
        dtype=float,
    )


def predict_local_S(res, peak_index, projections):
    """
    Project the predicted S matrix (r^2) into the local frame.

    Parameters
    ----------
    res : ResolutionEllipsoid
        Resolution model with ``predict_lab_S``.
    peak_index : int
        Peak index.
    projections : list of 3 arrays
        Local projection axes [n, u, v] in the same frame as ``S_lab``.

    Returns
    -------
    S_local : (3, 3) array
        Predicted S matrix (r^2) in the local projection frame.
    """
    S_lab = res.predict_lab_S(peak_index)
    W = np.column_stack(projections)
    S_local = W.T @ S_lab @ W
    return 0.5 * (S_local + S_local.T)


def fit_peak_3d(
    d,
    b,
    x0,
    x1,
    x2,
    S_local,
    center_iter=3,
    center_damping=0.5,
    eps=1e-12,
):
    """
    Fit ``mu = A*g(x-c; S) + B + b`` for one 3D peak.

    The likelihood is evaluated on the raw sample counts ``d``.  The measured
    background ``b`` is treated as a fixed expected background field.  ``B`` is
    a single additional flat sample-background offset.

    Parameters
    ----------
    d : (n0, n1, n2) array
        Raw sample counts.
    b : (n0, n1, n2) array
        Fixed measured-background counts, already scaled to the sample run.
    x0, x1, x2 : 1-d arrays
        Bin-center coordinates along each local axis.
    S_local : (3, 3) array
        S matrix (r^2) in the local frame.
    center_iter : int
        Number of EM-like centroid refinement iterations.
    center_damping : float
        Damping factor for centroid updates.
    eps : float
        Numerical floor for the Poisson mean.

    Returns
    -------
    dict or None
        Fit information, or None when the fit is invalid.
    """
    d = np.asarray(d, dtype=float)
    b = np.asarray(b, dtype=float)
    S_local = 0.5 * (np.asarray(S_local, dtype=float) + S_local.T)

    valid = np.isfinite(d) & np.isfinite(b) & (d >= 0.0) & (b >= 0.0)
    if int(np.sum(valid)) < 6:
        return None

    ok, _ = _is_spd(S_local)
    if not ok:
        return None

    cov = S_local / _R_SCALE_3D**2
    ok, _ = _is_spd(cov)
    if not ok:
        return None

    try:
        Sigma_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return None

    Q0, Q1, Q2 = np.meshgrid(x0, x1, x2, indexing="ij")
    c = _initial_center(d, b, Q0, Q1, Q2, valid)
    c = _clip_center_to_grid(c, x0, x1, x2)

    A = B = A_err = B_err = cost = np.nan
    g = mu = dx = None

    n_iter = max(int(center_iter), 1)
    damping = float(np.clip(center_damping, 0.0, 1.0))

    for iteration in range(n_iter):
        dx = np.stack([Q0 - c[0], Q1 - c[1], Q2 - c[2]], axis=-1)
        quad = np.einsum("...i,ij,...j->...", dx, Sigma_inv, dx)
        g_full = np.where(valid, np.exp(-0.5 * quad), 0.0)

        gv = g_full[valid].ravel()
        dv = d[valid].ravel()
        bv = b[valid].ravel()

        A, B, A_err, B_err, cost, ok = _fit_A_B_poisson(
            gv,
            dv,
            bv,
            eps=eps,
        )
        if not ok:
            return None

        mu_v = np.maximum(A * gv + B + bv, eps)
        if iteration == n_iter - 1:
            break

        # EM-like expected signal counts per voxel.  This suppresses voxels
        # dominated by the measured background or by the fitted flat offset.
        signal_w = dv * (A * gv / mu_v)
        wsum = float(np.sum(signal_w))

        if wsum <= eps:
            break

        q0 = Q0[valid].ravel()
        q1 = Q1[valid].ravel()
        q2 = Q2[valid].ravel()
        c_new = (
            np.array(
                [
                    np.sum(signal_w * q0),
                    np.sum(signal_w * q1),
                    np.sum(signal_w * q2),
                ],
                dtype=float,
            )
            / wsum
        )

        c = (1.0 - damping) * c + damping * c_new
        c = _clip_center_to_grid(c, x0, x1, x2)

    dx_full = np.stack([Q0 - c[0], Q1 - c[1], Q2 - c[2]], axis=-1)
    quad = np.einsum("...i,ij,...j->...", dx_full, Sigma_inv, dx_full)
    g_full = np.where(valid, np.exp(-0.5 * quad), 0.0)

    gv = g_full[valid].ravel()
    dv = d[valid].ravel()
    bv = b[valid].ravel()
    dx_valid = dx_full[valid]

    A, B, A_err, B_err, cost, ok = _fit_A_B_poisson(gv, dv, bv, eps=eps)
    if not ok or A <= 0.0 or not np.isfinite(A_err) or A_err <= 0.0:
        return None

    mu_fit = np.maximum(A * gv + B + bv, eps)
    snr = A / A_err

    return {
        "A": A,
        "B": B,
        "A_err": A_err,
        "B_err": B_err,
        "c": c,
        "snr": snr,
        "cost": cost,
        "g": gv,
        "mu": mu_fit,
        "dx": dx_valid,
        "dv": dv,
        "bv": bv,
        "valid": valid,
    }


def _empirical_covariance(fit, W):
    """
    Empirical S matrix (r^2) in lab frame.

    Uses the expected signal counts ``d * A*g/mu`` as weights.  This is more
    consistent with the raw-count Poisson model than using ``A*g/mu`` alone.
    """
    responsibility = fit["A"] * fit["g"] / np.maximum(fit["mu"], 1e-12)
    weights = fit["dv"] * responsibility
    wsum = float(np.sum(weights))

    if wsum <= 0.0:
        return None

    dx = fit["dx"]
    S_local = (_R_SCALE_3D**2 / wsum) * (dx * weights[:, None]).T @ dx
    S_local = 0.5 * (S_local + S_local.T)

    S_lab = W @ S_local @ W.T
    return 0.5 * (S_lab + S_lab.T)


class PeakEstimator:
    """
    Nonlinear profile-likelihood peak integration.

    The global ellipsoid variance parameters are refined by minimizing the
    summed raw-count Poisson profile likelihood over a fixed active set of
    peaks.  Per-peak ``A``, ``B``, and ``c`` are refit inside the objective.
    """

    def __init__(
        self,
        snr_min=3.0,
        alpha=1e-8,
        active_snr_min=None,
        min_eig=1e-12,
        bad_fit_penalty=1e9,
        use_active_set=True,
        normalize_cost=True,
    ):
        self.snr_min = float(snr_min)
        self.alpha = float(alpha)
        self.active_snr_min = active_snr_min
        self.min_eig = float(min_eig)
        self.bad_fit_penalty = float(bad_fit_penalty)
        self.use_active_set = bool(use_active_set)
        self.normalize_cost = bool(normalize_cost)

        self._peak_data = {}
        self._fit = {}
        self._active_peaks = []
        self.moment_covs = {}
        self.diagnostics = OptimizerDiagnostics()
        self.optimization_result = None

    def collect_peaks(self, peaks_ws, data, md_ws, pc_ratio=1.0):
        """
        Bin all predicted peaks using the full Q-sample workspace.

        Parameters
        ----------
        peaks_ws : str
            Mantid peaks workspace name.
        data : DataModel
            Reduction data model.
        md_ws : str
            Full Q-sample workspace, for example ``"md"``.
        pc_ratio : float
            Scale factor ``pc_signal / pc_background`` for the background.
        """
        peak = PeakModel(peaks_ws)
        n_peak = peak.get_number_peaks()
        UB = peak.get_UB()
        I = np.eye(3)

        for i in range(n_peak):
            d_spacing = peak.get_d_spacing(i)
            Q_mod = 2.0 * np.pi / d_spacing
            hkl = peak.get_hkl(i)
            lamda = peak.get_wavelength(i)
            two_theta, az_phi = peak.get_angles(i)
            R = peak.get_goniometer_matrix(i)
            shape = peak.get_peak_shape(i)
            dQ = data.get_resolution_in_Q(lamda, two_theta)

            c0, c1, c2, r0, r1, r2, v0, v1, v2 = shape
            c0, c1, c2 = R @ [c0, c1, c2]
            v0, v1, v2 = (R @ np.column_stack([v0, v1, v2])).T
            shape = c0, c1, c2, r0, r1, r2, v0, v1, v2

            bin_params = R @ UB, hkl, lamda, I, two_theta, az_phi, shape, dQ
            bins, extents, projections, _, _ = bin_extent(*bin_params)
            result = data.bin_in_Q(md_ws, extents, bins, projections)

            if result is None:
                continue

            d, _, Q0, Q1, Q2 = result
            data.delete_workspace(md_ws + "_bin")

            b = np.zeros_like(d, dtype=float)
            if data.workspace_exists("bkg_md"):
                result = data.bin_in_Q("bkg_md", extents, bins, projections)

                if result is not None:
                    b, _, _, _, _ = result
                    data.delete_workspace("bkg_md_bin")

            b = np.asarray(b, dtype=float) * pc_ratio

            x0 = Q0[:, 0, 0] - Q_mod
            x1 = Q1[0, :, 0].copy()
            x2 = Q2[0, 0, :].copy()

            self._peak_data[i] = {
                "d": np.asarray(d, dtype=float),
                "b": b,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "Q_mod": Q_mod,
                "projections": projections,
                "W": np.column_stack(projections),
                "R": R,
                "hkl": hkl,
            }

    def estimate(self, peaks_ws, res):
        """
        Run the hierarchical fitting loop over all collected peaks.

        Parameters
        ----------
        peaks_ws : str
            Mantid peaks workspace name.
        res : ResolutionEllipsoid
            Already fit and applied resolution model.

        Returns
        -------
        results : dict
            ``{peak_index: (I, sigma_I, shape, info, hkl)}``.
        """
        peak_indices = list(self._peak_data.keys())
        design_matrices = res.build_design_matrices(peak_indices)
        x0 = np.asarray(res.model["variance_parameters"], dtype=float).copy()

        self._active_peaks = self._select_active_peaks(
            res,
            design_matrices,
            x0,
        )

        if not self.use_active_set:
            self._active_peaks = [
                i for i in peak_indices if i in design_matrices
            ]

        if len(self._active_peaks) == 0:
            self._active_peaks = [
                i for i in peak_indices if i in design_matrices
            ]

        result = scipy.optimize.minimize(
            self._outer_cost,
            x0,
            args=(design_matrices,),
            method="L-BFGS-B",
            jac=None,
            bounds=[(0.0, None)] * len(x0),
            options={
                "maxiter": 500,
                "maxls": 50,
                "ftol": 1e-10,
                "gtol": 1e-6,
                "eps": 1e-6,
            },
        )
        self.optimization_result = result

        x_opt = np.asarray(result.x, dtype=float)
        res.model["variance_parameters"] = x_opt
        res.apply()

        grad_norm = np.nan
        if getattr(result, "jac", None) is not None:
            grad_norm = float(np.linalg.norm(result.jac))

        self.diagnostics = OptimizerDiagnostics(
            success=bool(result.success),
            message=str(result.message),
            nit=int(getattr(result, "nit", 0)),
            nfev=int(getattr(result, "nfev", 0)),
            fun=float(result.fun),
            grad_norm=grad_norm,
            n_peaks_total=len(self._peak_data),
            n_peaks_active=len(self._active_peaks),
        )

        self._final_fit_pass(peaks_ws, res)
        return self._assemble_results(peaks_ws, res)

    def _local_S_from_design(self, design_matrix, x, W):
        """Evaluate local S from one design matrix and parameter vector."""
        y = design_matrix @ x
        S_lab = _vech6_to_S(y)
        S_local = W.T @ S_lab @ W
        return 0.5 * (S_local + S_local.T)

    def _select_active_peaks(self, res, design_matrices, x0):
        """Choose a fixed initial active set for the global refinement."""
        snr_cut = self.snr_min
        if self.active_snr_min is not None:
            snr_cut = float(self.active_snr_min)

        active = []
        for i, pd in self._peak_data.items():
            if i not in design_matrices:
                continue

            S_local = self._local_S_from_design(
                design_matrices[i],
                x0,
                pd["W"],
            )
            ok, _ = _is_spd(S_local, min_eig=self.min_eig)
            if not ok:
                S_local = predict_local_S(res, i, pd["projections"])

            fit = fit_peak_3d(
                pd["d"],
                pd["b"],
                pd["x0"],
                pd["x1"],
                pd["x2"],
                S_local,
            )
            if fit is not None and fit["snr"] >= snr_cut:
                active.append(i)

        return active

    def _outer_cost(self, x, design_matrices):
        """Cost-only global Poisson profile objective."""
        total_cost = 0.0
        n_terms = 0
        n_bad_spd = 0
        n_bad_fit = 0

        for i in self._active_peaks:
            if i not in design_matrices:
                continue

            pd = self._peak_data[i]
            S_local = self._local_S_from_design(
                design_matrices[i],
                x,
                pd["W"],
            )
            ok, min_eig = _is_spd(S_local, min_eig=self.min_eig)
            if not ok:
                total_cost += self.bad_fit_penalty * (1.0 + abs(min_eig))
                n_bad_spd += 1
                continue

            fit = fit_peak_3d(
                pd["d"],
                pd["b"],
                pd["x0"],
                pd["x1"],
                pd["x2"],
                S_local,
            )
            if fit is None:
                total_cost += self.bad_fit_penalty
                n_bad_fit += 1
                continue

            total_cost += fit["cost"]
            n_terms += int(np.sum(fit["valid"]))

        if self.normalize_cost and n_terms > 0:
            total_cost /= float(n_terms)

        total_cost += 0.5 * self.alpha * float(np.dot(x, x))

        self.diagnostics.n_bad_spd = n_bad_spd
        self.diagnostics.n_bad_fit = n_bad_fit

        return float(total_cost)

    def _final_fit_pass(self, peaks_ws, res):
        """Fit every collected peak with the optimized resolution model."""
        peak = PeakModel(peaks_ws)
        self._fit = {}

        for i, pd in self._peak_data.items():
            S_local = predict_local_S(res, i, pd["projections"])
            fit = fit_peak_3d(
                pd["d"],
                pd["b"],
                pd["x0"],
                pd["x1"],
                pd["x2"],
                S_local,
            )
            if fit is None:
                continue
            self._fit[i] = {**fit, "S_local": S_local}

        for i, fit in self._fit.items():
            c = fit["c"]
            if not np.all(np.isfinite(c)):
                continue

            W = self._peak_data[i]["W"]
            dQ = W @ c
            R = self._peak_data[i]["R"]
            Q = R @ peak.get_sample_Q(i)
            Q = R.T @ (Q + dQ)
            peak.set_peak_center(i, *Q)

    def _assemble_results(self, peaks_ws, res):
        """Assemble integrated intensities and empirical moment covariances."""
        results = {}

        for i, fit in self._fit.items():
            A = fit["A"]
            A_err = fit["A_err"]
            S_local = fit["S_local"]
            pd = self._peak_data[i]

            cov = S_local / _R_SCALE_3D**2
            det_cov = np.linalg.det(cov)
            scale = (2.0 * np.pi) ** 1.5 * np.sqrt(max(det_cov, 0.0))

            I = A * scale
            sigma_I = A_err * scale

            c = fit["c"]
            r, V = res._ellipsoid_from_S(S_local)
            local_params = (
                c[0],
                c[1],
                c[2],
                r[0],
                r[1],
                r[2],
                V[:, 0],
                V[:, 1],
                V[:, 2],
            )
            shape = revert_ellipsoid_parameters(
                local_params, pd["projections"]
            )

            info = {
                "bkg": fit["B"],
                "bkg_err": fit["B_err"],
                "intens": I,
                "intens_err": sigma_I,
                "snr": fit["snr"],
                "poisson_cost": fit["cost"],
            }
            results[i] = I, sigma_I, shape, info, pd["hkl"]

        self.moment_covs = {}
        for i, fit in self._fit.items():
            if fit["snr"] < self.snr_min:
                continue

            S = _empirical_covariance(fit, self._peak_data[i]["W"])
            if S is None:
                continue

            R = self._peak_data[i]["R"]
            S_sample = R.T @ S @ R
            self.moment_covs[i] = 0.5 * (S_sample + S_sample.T)

        return results

    def _extract_normalized_profiles(self):
        """
        Collect normalized 1D marginal profiles for high-SNR peaks.

        Returns
        -------
        zs, yns, ens : each a list of 3 ndarrays
            Per-axis z coordinates, normalized counts, and uncertainties.
        """
        zs = [[], [], []]
        yns = [[], [], []]
        ens = [[], [], []]
        sum_axes_per_k = [(1, 2), (0, 2), (0, 1)]
        eps = 1e-10

        for i, fit in self._fit.items():
            if fit["snr"] < self.snr_min:
                continue

            c = fit["c"]
            S_local = fit["S_local"]
            cov_local = S_local / _R_SCALE_3D**2
            sigmas = np.sqrt(np.maximum(np.diag(cov_local), 0.0))

            pd = self._peak_data[i]
            d = pd["d"]
            b = pd["b"]
            xs = [pd["x0"], pd["x1"], pd["x2"]]

            for k in range(3):
                if sigmas[k] <= 0.0:
                    continue

                axes = sum_axes_per_k[k]
                dk = np.nansum(d, axis=axes)
                bk = np.nansum(b, axis=axes)
                n_collapsed = int(np.prod([d.shape[axis] for axis in axes]))
                flat = fit["B"] * n_collapsed

                net = dk - bk - flat
                ek = np.sqrt(np.maximum(dk + bk + flat, 0.0))
                z = (xs[k] - c[k]) / sigmas[k]

                valid = np.isfinite(net) & np.isfinite(ek) & (ek > 0.0)
                if int(np.sum(valid)) < 3:
                    continue

                g = np.exp(-0.5 * z**2)
                w = np.where(valid, 1.0 / (ek**2 + eps), 0.0)
                sw = float(np.sum(w))
                if sw <= 0.0:
                    continue

                net_bar = float(np.sum(w * net) / sw)
                g_bar = float(np.sum(w * g) / sw)
                num = np.sum(w * (g - g_bar) * (net - net_bar))
                den = np.sum(w * (g - g_bar) ** 2)

                if den <= 0.0:
                    continue

                A1d = num / den
                B1d = net_bar - A1d * g_bar
                if A1d <= 0.0:
                    continue

                N = int(np.sum(valid))
                resid = net - (A1d * g + B1d)
                resid_var = np.sum(w * resid**2) / max(N - 2, 1)
                A1d_err = np.sqrt(max(resid_var / den, 0.0))
                B1d_err = np.sqrt(
                    max(resid_var * (1.0 / sw + g_bar**2 / den), 0.0)
                )

                y_hat = (net - B1d) / A1d
                e_base = np.sqrt(ek**2 + B1d_err**2)
                e_hat = np.sqrt(
                    (e_base / A1d) ** 2 + (y_hat * A1d_err / A1d) ** 2
                )

                zs[k].extend(z[valid].tolist())
                yns[k].extend(y_hat[valid].tolist())
                ens[k].extend(e_hat[valid].tolist())

        return (
            [np.asarray(zs[k], dtype=float) for k in range(3)],
            [np.asarray(yns[k], dtype=float) for k in range(3)],
            [np.asarray(ens[k], dtype=float) for k in range(3)],
        )

    def plot_estimate(self, filename):
        """
        Save a 3-panel plot of normalized 1D peak profiles in z-space.

        Parameters
        ----------
        filename : str
            Output path, for example ``"run_est.png"``.
        """
        zs, yns, ens = self._extract_normalized_profiles()
        plot = PeakEstimatePlot(zs, yns, ens)
        plot.save_plot(filename)
        plot.close()
