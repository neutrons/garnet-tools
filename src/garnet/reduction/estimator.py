import numpy as np

from garnet.reduction.peaks import PeakModel
from garnet.reduction.intensity import (
    bin_axes,
    bin_extent,
    revert_ellipsoid_parameters,
)


def predict_local_cov(res, peak_index, projections):
    """
    Project the ResolutionEllipsoid predicted covariance into the local S-W frame.

    Parameters
    ----------
    res : ResolutionEllipsoid
    peak_index : int
    projections : list of 3 arrays
        [n, u, v] unit vectors in sample frame.

    Returns
    -------
    cov_local : (3, 3) array
        Predicted covariance in the local projection frame.
    """

    cov_s = res.predict_sample_cov(peak_index)
    W = np.column_stack(projections)
    cov_local = W.T @ cov_s @ W
    return 0.5 * (cov_local + cov_local.T)


def fit_peak_3d(d, x0, x1, x2, cov_local):
    """
    Fit f(x) = A * exp(-½ xᵀ Σ⁻¹ x) + B with Σ fixed from the global model.

    Background B comes from the 15th percentile of the 3D box.
    Centroid c is found by a squared-residual weighted mean.
    Amplitude A is found by weighted linear regression against the Gaussian template.

    Parameters
    ----------
    d : (n0, n1, n2) array
        Raw counts.
    x0, x1, x2 : 1-d arrays
        Bin-centre coordinates along each local axis (x0 already centred
        by subtracting |Q|; x1 and x2 centred at 0).
    cov_local : (3, 3) array
        Covariance matrix in the local frame (Σ, fixed).

    Returns
    -------
    A, B, A_err, B_err : float
    c : (3,) array  centroid shift in local frame
    chi2_nu : float
    snr : float
    """

    valid = np.isfinite(d) & (d >= 0)
    N = int(valid.sum())

    zero = (0.0, 0.0, 0.0, 0.0, np.zeros(3), 0.0)

    if N < 6:
        return zero

    B = float(np.nanpercentile(d[valid], 15))
    B = max(B, 0.0)

    Q0, Q1, Q2 = np.meshgrid(x0, x1, x2, indexing="ij")

    # centroid: weight by squared excess above background
    w = np.where(valid, np.maximum(d - B, 0.0) ** 2, 0.0)
    wsum = float(w.sum())

    if wsum > 0:
        c = np.array(
            [
                float(np.sum(w * Q0)) / wsum,
                float(np.sum(w * Q1)) / wsum,
                float(np.sum(w * Q2)) / wsum,
            ]
        )
    else:
        c = np.zeros(3)

    # Gaussian template centered at c with fixed Σ
    dx = np.stack([Q0 - c[0], Q1 - c[1], Q2 - c[2]], axis=-1)

    try:
        Sigma_inv = np.linalg.inv(cov_local)
    except np.linalg.LinAlgError:
        return zero

    quad = np.einsum("...i,ij,...j->...", dx, Sigma_inv, dx)
    g = np.where(valid, np.exp(-0.5 * quad), 0.0)

    # clip negatives from background subtraction; deviance requires d >= 0
    dv = np.where(valid, np.maximum(d, 0.0), 0.0)
    gv = g[valid]
    dv_flat = dv[valid]

    # Poisson MLE via IRLS: weights w_i = 1/mu_i converge to the deviance MLE.
    # Initialise with unweighted WLS.
    eps = 1e-6
    A, B_fit = B, B
    for _ in range(10):
        mu = np.maximum(A * gv + B_fit, eps)
        w = 1.0 / mu
        sw = float(w.sum())
        if sw <= 0.0:
            break
        d_bar = float((w * dv_flat).sum()) / sw
        g_bar = float((w * gv).sum()) / sw
        num = float((w * (gv - g_bar) * (dv_flat - d_bar)).sum())
        den = float((w * (gv - g_bar) ** 2).sum())
        if den <= 0.0:
            break
        A_new = max(num / den, 0.0)
        B_new = max(d_bar - A_new * g_bar, 0.0)
        if abs(A_new - A) < 1e-8 * (A + eps) and abs(B_new - B_fit) < 1e-8 * (
            B_fit + eps
        ):
            A, B_fit = A_new, B_new
            break
        A, B_fit = A_new, B_new

    if A <= 0.0:
        return 0.0, B_fit, 0.0, 0.0, c, 0.0

    # Fisher-information standard errors for the Poisson MLE
    mu_fit = np.maximum(A * gv + B_fit, eps)
    fisher_A = float(np.sum(gv**2 / mu_fit))
    fisher_B = float(np.sum(1.0 / mu_fit))
    A_err = 1.0 / np.sqrt(max(fisher_A, eps))
    B_err = 1.0 / np.sqrt(max(fisher_B, eps))

    snr = A / A_err if A_err > 0.0 else 0.0

    return A, B_fit, A_err, B_err, c, snr


def moment_covariance(d, x0, x1, x2, c, W):
    """
    Empirical covariance in sample frame from weighted second moments.

    Parameters
    ----------
    d : (n0, n1, n2) array
    x0, x1, x2 : 1-d coordinate arrays in local frame
    c : (3,) centroid in local frame
    W : (3, 3) matrix whose columns are the local projection vectors in sample frame

    Returns
    -------
    cov_sample : (3, 3) array or None
    """

    B = float(np.nanpercentile(d[np.isfinite(d)], 15))
    w = np.maximum(d - max(B, 0.0), 0.0)
    wsum = float(w.sum())

    if wsum <= 0.0:
        return None

    Q0, Q1, Q2 = np.meshgrid(x0, x1, x2, indexing="ij")

    dx = np.stack(
        [
            (Q0 - c[0]).ravel(),
            (Q1 - c[1]).ravel(),
            (Q2 - c[2]).ravel(),
        ]
    )

    w_flat = w.ravel()
    cov_local = (dx * w_flat) @ dx.T / wsum
    cov_local = 0.5 * (cov_local + cov_local.T)

    cov_sample = W @ cov_local @ W.T
    return 0.5 * (cov_sample + cov_sample.T)


class PeakEstimator:
    """
    Hierarchical constrained peak integration.

    The global covariance shape comes from ResolutionEllipsoid. Per-peak free
    parameters are amplitude (A), background (B), and centroid shift (3-D).
    Strong clean peaks feed their empirical second-moment covariance back into
    ResolutionEllipsoid so the global model improves across iterations.

    Usage
    -----
    estimator = PeakEstimator()
    estimator.collect_peaks(peaks_ws, data, r_cut, "md")
    estimator.estimate(peaks_ws, res)
    # peaks workspace now has refined centroids; res has improved model
    # then run the normal per-bank integrate_peaks loop
    """

    def __init__(self, n_iter=3, snr_min=3.0):
        self.n_iter = n_iter
        self.snr_min = snr_min

        self._peak_data = {}
        self._fit = {}

    def collect_peaks(
        self,
        peaks_ws,
        data,
        r_cut,
        md_ws,
        bkg_md=None,
        pc_ratio=1.0,
        R_bkg=None,
    ):
        """
        Bin all predicted peaks using the full Q-sample workspace.
        Called once before estimate().

        Parameters
        ----------
        peaks_ws : str
        data : DataModel
        r_cut : float  (used only as fallback for shape initialisation)
        md_ws : str  full Q-sample workspace (e.g. "md")
        bkg_md : str or None
            Background Q-lab workspace (no Lorentz correction).  When given,
            the binned background is scaled by pc_ratio and subtracted from
            each peak box before fitting.
        pc_ratio : float
            Scale factor = pc_signal / pc_background (proton charge ratio).
        R_bkg : (3, 3) array or None
            Goniometer matrix for the background run.  For isotropic powder
            backgrounds the signal R is reused, so this is only needed when
            the background has a preferred orientation.
        """

        peak = PeakModel(peaks_ws)
        n_peak = peak.get_number_peaks()
        UB = peak.get_UB()

        for i in range(n_peak):
            d_spacing = peak.get_d_spacing(i)
            Q_center = 2.0 * np.pi / d_spacing
            hkl = peak.get_hkl(i)
            lamda = peak.get_wavelength(i)
            two_theta, az_phi = peak.get_angles(i)
            R = peak.get_goniometer_matrix(i)
            shape = peak.get_peak_shape(i)
            dQ = data.get_resolution_in_Q(lamda, two_theta)

            bin_params = UB, hkl, lamda, R, two_theta, az_phi, shape, dQ
            bins, extents, projections, transform, conversion = bin_extent(
                *bin_params
            )

            result = data.bin_in_Q(md_ws, extents, bins, projections)

            if result is None:
                continue

            d_arr, _, Q0, Q1, Q2 = result

            data.delete_workspace(md_ws + "_bin")

            if bkg_md is not None:
                # For powder backgrounds, R_bkg does not matter (isotropic), so
                # we reuse the signal projections.  Pass R_bkg only when the
                # background has a preferred orientation.
                if R_bkg is not None:
                    bkg_bin_params = (
                        UB,
                        hkl,
                        lamda,
                        R_bkg,
                        two_theta,
                        az_phi,
                        shape,
                        dQ,
                    )
                    _, bkg_extents, bkg_projections, _, _ = bin_extent(
                        *bkg_bin_params
                    )
                else:
                    bkg_extents, bkg_projections = extents, projections

                result_bkg = data.bin_in_Q(
                    bkg_md, bkg_extents, bins, bkg_projections
                )

                if result_bkg is not None:
                    d_bkg, _, _, _, _ = result_bkg
                    data.delete_workspace(bkg_md + "_bin")
                    d_arr = d_arr - pc_ratio * d_bkg

            # store axes only to keep memory usage linear in n_peaks
            x0 = Q0[:, 0, 0] - Q_center  # centred along n axis
            x1 = Q1[0, :, 0]  # centred at 0 along u
            x2 = Q2[0, 0, :]  # centred at 0 along v

            self._peak_data[i] = {
                "d": d_arr,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "Q_center": Q_center,
                "projections": projections,
                "W": np.column_stack(projections),
                "hkl": hkl,
            }

    def estimate(self, peaks_ws, res):
        """
        Run the hierarchical fitting loop over all collected peaks.

        Parameters
        ----------
        peaks_ws : str
        res : ResolutionEllipsoid  already fit and applied

        Returns
        -------
        dict : {peak_index: (I, sigma_I, shape, info, hkl)}
        """

        peak = PeakModel(peaks_ws)

        for iteration in range(self.n_iter):
            for i, pd in self._peak_data.items():
                cov_local = predict_local_cov(res, i, pd["projections"])

                A, B, A_err, B_err, c, snr = fit_peak_3d(
                    pd["d"],
                    pd["x0"],
                    pd["x1"],
                    pd["x2"],
                    cov_local,
                )

                self._fit[i] = {
                    "A": A,
                    "B": B,
                    "A_err": A_err,
                    "B_err": B_err,
                    "c": c,
                    "snr": snr,
                    "cov_local": cov_local,
                }

            for i, fit in self._fit.items():
                c = fit["c"]
                if not np.all(np.isfinite(c)) or fit["A"] <= 0.0:
                    continue

                W = self._peak_data[i]["W"]
                dQ_sample = W @ c

                Q_s = np.array(peak.get_sample_Q(i))
                peak.set_peak_center(i, *(Q_s + dQ_sample))

            if iteration < self.n_iter - 1:
                for i, fit in self._fit.items():
                    if fit["snr"] < self.snr_min:
                        continue

                    pd = self._peak_data[i]
                    cov_s = moment_covariance(
                        pd["d"],
                        pd["x0"],
                        pd["x1"],
                        pd["x2"],
                        fit["c"],
                        pd["W"],
                    )
                    if cov_s is not None:
                        self._write_peak_shape(peaks_ws, i, cov_s, res)

                res.fit()
                res.apply()

        self.moment_covs = {}
        for i, fit in self._fit.items():
            if fit["snr"] < self.snr_min:
                continue
            pd = self._peak_data[i]
            cov_s = moment_covariance(
                pd["d"], pd["x0"], pd["x1"], pd["x2"], fit["c"], pd["W"]
            )
            if cov_s is not None:
                self.moment_covs[i] = cov_s

        results = {}

        for i, fit in self._fit.items():
            A = fit["A"]
            A_err = fit["A_err"]
            cov_local = fit["cov_local"]
            pd = self._peak_data[i]

            det_cov = np.linalg.det(cov_local)
            scale = (2.0 * np.pi) ** 1.5 * np.sqrt(max(det_cov, 0.0))

            I = A * scale
            sigma_I = A_err * scale

            c = fit["c"]
            radii, V = res._ellipsoid_from_covariance(cov_local)
            local_params = (
                c[0],
                c[1],
                c[2],
                radii[0],
                radii[1],
                radii[2],
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
                "I_ell": I,
                "s_ell": sigma_I,
                "snr": fit["snr"],
            }

            results[i] = I, sigma_I, shape, info, pd["hkl"]

        return results

    def _write_peak_shape(self, peaks_ws, peak_index, cov_sample, res):
        """Write empirical sample-frame covariance as peak shape."""

        pk = PeakModel(peaks_ws)
        radii, V_sample = res._ellipsoid_from_covariance(cov_sample)

        Q_s = np.array(pk.get_sample_Q(peak_index))

        pk.set_peak_shape(
            peak_index,
            *Q_s,
            radii[0],
            radii[1],
            radii[2],
            list(V_sample[:, 0]),
            list(V_sample[:, 1]),
            list(V_sample[:, 2]),
        )
