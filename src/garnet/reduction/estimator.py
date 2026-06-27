import numpy as np

from garnet.plots.peaks import PeakEstimatePlot
from garnet.reduction.peaks import PeakModel
from garnet.reduction.intensity import (
    bin_extent,
    revert_ellipsoid_parameters,
)


def predict_local_cov(res, peak_index, projections):
    """
    Project the predicted covariance into the local frame.

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

    cov_lab = res.predict_lab_cov(peak_index)
    W = np.column_stack(projections)
    cov_local = W.T @ cov_lab @ W
    return 0.5 * (cov_local + cov_local.T)


def fit_peak_3d(d, b, x0, x1, x2, cov_local):
    """
    Fit f(x) = A * exp(-½ xᵀ Σ⁻¹ x) + B with Σ fixed from the global model.

    Background B comes from the 15th percentile of the 3D box.
    Centroid c is found by a squared-residual weighted mean.
    Amplitude A is found by weighted linear regression against the Gaussian template.

    Parameters
    ----------
    d : (n0, n1, n2) array
        Signal counts.
    b : (n0, n1, n2) array
        Background counts.
    x0, x1, x2 : 1-d arrays
        Bin-center coordinates along each local axis (x0 already centred
        by subtracting |Q|; x1 and x2 centred at 0).
    cov_local : (3, 3) array
        Covariance matrix in the local frame.

    Returns
    -------
    A, B, A_err, B_err : float
    c : (3,) array  centroid shift in local frame
    chi2_nu : float
    snr : float
    """

    s = d - b

    valid = np.isfinite(s) & (s >= 0)
    N = int(valid.sum())

    zero = (0.0, 0.0, 0.0, 0.0, np.zeros(3), 0.0)

    if N < 6:
        return zero

    B = np.nanpercentile(s[valid], 15)
    B = max(B, 0.0)

    Q0, Q1, Q2 = np.meshgrid(x0, x1, x2, indexing="ij")

    # centroid: weight by squared excess above background
    w = np.where(valid, np.maximum(s - B, 0.0) ** 2, 0.0)
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

    dv = np.where(valid, np.maximum(s, 0.0), 0.0)
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


def moment_covariance(d, b, x0, x1, x2, c, W):
    """
    Empirical covariance in local frame from weighted second moments.

    Parameters
    ----------
    d : (n0, n1, n2) array
    b : (n0, n1, n2) array
    x0, x1, x2 : 1-d coordinate arrays in local frame
    c : (3,) centroid in local frame
    W : (3, 3) matrix whose columns are the local projection vectors in local frame

    Returns
    -------
    cov : (3, 3) array or None
    """

    s = d - b

    B = np.nanpercentile(d[np.isfinite(s)], 15)
    sig = np.maximum(s - max(B, 0.0), 0.0)
    noise = np.sqrt(np.maximum(d + b, 1.0))
    w = np.where(np.isfinite(d), sig / noise, 0.0)
    wsum = w.sum()

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

    cov = W @ cov_local @ W.T
    return 0.5 * (cov + cov.T)


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
    estimator.collect_peaks(peaks_ws, data, "md")
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
        md_ws,
        pc_ratio=1.0,
    ):
        """
        Bin all predicted peaks using the full Q-sample workspace.
        Called once before estimate().

        Parameters
        ----------
        peaks_ws : str
        data : DataModel
        md_ws : str  full Q-sample workspace (e.g. "md")
        pc_ratio : float
            Scale factor = pc_signal / pc_background (proton charge ratio).
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

            bin_params = UB, hkl, lamda, I, two_theta, az_phi, shape, dQ
            bins, extents, projections, transform, conversion = bin_extent(
                *bin_params
            )

            result = data.bin_in_Q(md_ws, extents, bins, projections)

            d, _, Q0, Q1, Q2 = result

            data.delete_workspace(md_ws + "_bin")

            b = np.zeros_like(d)
            if data.workspace_exists("bkg_md"):
                result = data.bin_in_Q("bkg_md", extents, bins, projections)

                b, _, Q0, Q1, Q2 = result

                data.delete_workspace("bkg_md_bin")

            b *= pc_ratio

            x0 = Q0[:, 0, 0] - Q_mod
            x1 = Q1[0, :, 0]
            x2 = Q2[0, 0, :]

            self._peak_data[i] = {
                "d": d,
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
                    pd["b"],
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
                dQ = W @ c

                R = self._peak_data[i]["R"]

                Q = R @ peak.get_sample_Q(i)
                Q = R.T @ (Q + dQ)
                peak.set_peak_center(i, *Q)

            if iteration < self.n_iter - 1:
                for i, fit in self._fit.items():
                    if fit["snr"] < self.snr_min:
                        continue

                    pd = self._peak_data[i]
                    cov = moment_covariance(
                        pd["d"],
                        pd["b"],
                        pd["x0"],
                        pd["x1"],
                        pd["x2"],
                        fit["c"],
                        pd["W"],
                    )

                    if cov is not None:
                        R = self._peak_data[i]["R"]

                        cov = R.T @ cov @ R

                        self._write_peak_shape(peaks_ws, i, cov, res)

                res.fit()
                res.apply()

        self.moment_covs = {}
        for i, fit in self._fit.items():
            if fit["snr"] < self.snr_min:
                continue
            pd = self._peak_data[i]
            cov = moment_covariance(
                pd["d"],
                pd["b"],
                pd["x0"],
                pd["x1"],
                pd["x2"],
                fit["c"],
                pd["W"],
            )
            if cov is not None:
                self.moment_covs[i] = cov

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

    def _extract_normalized_profiles(self):
        """
        For all high-SNR peaks collect 1D marginal profiles normalized to
        (y - B) / A vs z = (x - mu) / sigma.

        Returns
        -------
        zs, yns, ens : each a list of 3 ndarrays (one per axis)
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
            cov_local = fit["cov_local"]
            sigmas = np.sqrt(np.maximum(np.diag(cov_local), 0.0))

            pd = self._peak_data[i]
            d = pd["d"]
            b = pd.get("b", np.zeros_like(d))
            xs = [pd["x0"], pd["x1"], pd["x2"]]

            for k in range(3):
                if sigmas[k] <= 0:
                    continue

                dk = np.nansum(d, axis=sum_axes_per_k[k])
                bk = np.nansum(b, axis=sum_axes_per_k[k])
                ek = np.sqrt(np.maximum(dk + bk, 0.0))
                net = dk - bk
                z = (xs[k] - c[k]) / sigmas[k]
                g = np.exp(-0.5 * z**2)

                valid = np.isfinite(net) & np.isfinite(ek) & (ek > 0)
                if valid.sum() < 3:
                    continue

                w = np.where(valid, 1.0 / (ek**2 + eps), 0.0)
                sw = w.sum()
                if sw <= 0:
                    continue

                net_bar = (w * net).sum() / sw
                g_bar = (w * g).sum() / sw
                num = (w * (g - g_bar) * (net - net_bar)).sum()
                den = (w * (g - g_bar) ** 2).sum()

                if den <= 0:
                    continue

                A1d = float(num / den)
                B1d = float(net_bar - A1d * g_bar)

                if A1d <= 0:
                    continue

                N = int(valid.sum())
                resid = net - (A1d * g + B1d)
                resid_var = float((w * resid**2).sum()) / max(N - 2, 1)
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
            [np.array(zs[k]) for k in range(3)],
            [np.array(yns[k]) for k in range(3)],
            [np.array(ens[k]) for k in range(3)],
        )

    def plot_estimate(self, filename):
        """
        Save a 3-panel plot of the normalized 1D peak profiles in z-space.

        Each panel shows (y - B) / A vs z = (x - mu) / sigma accumulated
        over all high-SNR peaks, which should collapse onto exp(-z^2/2).

        Parameters
        ----------
        filename : str
            Output path (e.g. "run_est.png").
        """

        zs, yns, ens = self._extract_normalized_profiles()
        plot = PeakEstimatePlot(zs, yns, ens)
        plot.save_plot(filename)
        plot.close()
