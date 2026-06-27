import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from mantid.simpleapi import mtd
from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid

from scipy.optimize import nnls

# r0,r1,r2 stored in the peak workspace are 99.7% containment radii,
# matching the convention in ellipsoid.py (S_matrix = U diag(r²) U.T,
# mah²≤1 ↔ 99.7% ellipsoid).
_CHI2_SCALE_3D = scipy.stats.chi2.ppf(0.997, df=3)  # ≈ 14.16
_R_SCALE = np.sqrt(_CHI2_SCALE_3D)  # ≈ 3.76


class ResolutionEllipsoid:
    def __init__(
        self,
        peaks_ws,
        r_cut=np.inf,
        sig_noise_cut=5.0,
        min_peaks=10,
        scale_bounds=(0.5, 2.0),
        mosaic="isotropic",
    ):
        self.peaks_ws = peaks_ws
        self.r_cut = r_cut
        self.sig_noise_cut = sig_noise_cut
        self.min_peaks = min_peaks
        self.scale_bounds = scale_bounds
        self.mosaic = mosaic
        self.model = None
        self.prior_cov_sigma = None
        self.prior_center_sigma = None
        self.lamda_min = np.inf
        self.lamda_max = 0
        self.two_theta_min = np.inf
        self.two_theta_max = 0

    def _transverse_directions(self, qhat):
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(qhat, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        m1 = np.cross(qhat, ref)
        m1 /= np.linalg.norm(m1)
        m2 = np.cross(qhat, m1)
        m2 /= np.linalg.norm(m2)
        return m1, m2

    def _normalize_columns(self, V):
        V = np.asarray(V, dtype=float)
        n = np.linalg.norm(V, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return V / n

    def _vech6(self, M):
        return np.array(
            [M[0, 0], M[1, 1], M[2, 2], M[1, 2], M[0, 2], M[0, 1]], dtype=float
        )

    def _outer6(self, a):
        M = np.outer(a, a)
        return self._vech6(M)

    def _get_peak_offset(self, ws, no):
        shape = ws.getPeak(no).getPeakShape()

        UB = ws.sample().getOrientedLattice().getUB()

        hkl = ws.getPeak(no).getHKL()
        R = ws.getPeak(no).getGoniometerMatrix()

        Qobs = ws.getPeak(no).getQLabFrame()
        Qcalc = 2 * np.pi * R @ UB @ hkl

        d0, d1, d2 = Qobs - Qcalc

        if shape.shapeName() == "ellipsoid":
            try:
                d = eval(shape.toJSON())
            except:
                d = None

            if d is not None:
                if "translation0" in d.keys():
                    d0 += d["translation0"]
                    d1 += d["translation1"]
                    d2 += d["translation2"]

        return np.array([d0, d1, d2])

    def _get_peak_shape(self, ws, no, r_cut):
        shape = ws.getPeak(no).getPeakShape()

        radii = np.full(3, r_cut, dtype=float)
        v0, v1, v2 = np.eye(3).tolist()

        if shape.shapeName() == "ellipsoid":
            try:
                d = eval(shape.toJSON())
            except:
                d = None

            if d is not None:
                v0 = [float(x) for x in d["direction0"].split()]
                v1 = [float(x) for x in d["direction1"].split()]
                v2 = [float(x) for x in d["direction2"].split()]

                radii = np.array(
                    [
                        float(d["radius0"]),
                        float(d["radius1"]),
                        float(d["radius2"]),
                    ]
                )

        V = np.column_stack([v0, v1, v2])
        V = self._normalize_columns(V)

        return radii, V

    def _set_peak_shape(self, ws, no, radii, V_sample):
        V_sample = self._normalize_columns(V_sample)
        v0, v1, v2 = V_sample.T

        shape = PeakShapeEllipsoid(
            [V3D(*v0), V3D(*v1), V3D(*v2)],
            list(radii),
            list(radii),
            list(radii),
        )
        ws.getPeak(no).setPeakShape(shape)

    def _covariance_from_ellipsoid(self, radii, V):
        # radii are 99.7% containment radii; divide by _R_SCALE to get sigma
        sigma = np.asarray(radii, dtype=float) / _R_SCALE
        return V @ np.diag(sigma**2) @ V.T

    def _ellipsoid_from_covariance(self, cov):
        cov = 0.5 * (cov + cov.T)

        W, V = np.linalg.eigh(cov)

        # convert 1-sigma eigenvalues to 99.7% containment radii
        radii = np.sqrt(np.maximum(W, 0.0)) * _R_SCALE
        V = self._normalize_columns(V)

        if np.linalg.det(V) < 0:
            V[:, -1] = -V[:, -1]

        return radii, V

    def _peak_params(self, peak):
        two_theta = peak.getScattering()
        phi = peak.getAzimuthal()

        lamda = peak.getWavelength()

        kf_x = np.sin(two_theta) * np.cos(phi)
        kf_y = np.sin(two_theta) * np.sin(phi)
        kf_z = np.cos(two_theta)

        nu = np.arcsin(kf_y)
        gamma = np.arctan2(kf_x, kf_z)

        return gamma, nu, lamda

    def _Q_magnitude(self, two_theta, lamda):
        return (4.0 * np.pi / lamda) * np.sin(0.5 * two_theta)

    def _tof_path_length(self, lamda, tof):
        return 0.003956034 * tof / lamda

    def _stoica_wilkinson_transform_from_peak(self, peak):
        two_theta = peak.getScattering()
        phi = peak.getAzimuthal()

        ki_hat = np.array([0.0, 0.0, 1.0])

        kf_hat = np.array(
            [
                np.sin(two_theta) * np.cos(phi),
                np.sin(two_theta) * np.sin(phi),
                np.cos(two_theta),
            ]
        )

        n = kf_hat - ki_hat
        n /= np.linalg.norm(n)

        u = kf_hat + ki_hat
        u /= np.linalg.norm(u)

        v = np.cross(n, u)
        v /= np.linalg.norm(v)

        return np.vstack([n, u, v])

    def _sample_axes_to_lab(self, R, V_sample):
        return R @ V_sample

    def _lab_axes_to_sample(self, R, V_lab):
        return R.T @ V_lab

    def _model_design_lab(self, peak):
        """
        Approximation to the resolution function.

        J. B. Forsyth, Single Crystal Pulsed Neutron Diffraction, in
        Chemical Crystallography with Pulsed Neutrons and Synchroton X-Rays,
        edited by M. A. Carrondo and G. A. Jeffrey (Springer Netherlands,
        Dordrecht, 1988), pp. 117–135.

        A. D. Stoica, On the resolution of slow-neutron spectrometers. II. The
        resolution function for time-of-flight diffractometry, Acta Cryst A 31,
        193 (1975).

        Parameters
        ----------
        peak : peak object
            Single crystal peak.

        Returns
        -------
        A : ndarray
            Design matrix.

        """
        two_theta = peak.getScattering()
        phi = peak.getAzimuthal()
        lamda = peak.getWavelength()

        k = 2.0 * np.pi / lamda

        s = np.sin(two_theta)
        c = np.cos(two_theta)
        cp = np.cos(phi)
        sp = np.sin(phi)

        alpha_i = np.array([1.0, 0.0, 0.0])
        beta_i = np.array([0.0, 1.0, 0.0])

        ki = np.array([0.0, 0.0, 1.0])
        kf = np.array([s * cp, s * sp, c])

        alpha_f = np.array([c * cp, c * sp, -s])
        beta_f = np.array([-sp, cp, 0.0])

        q_lambda = kf - ki

        Q_vec = k * q_lambda
        Q2 = np.dot(Q_vec, Q_vec)

        cols = [
            k**2 * self._outer6(alpha_i),  # sigma_alpha_i^2
            k**2 * self._outer6(beta_i),  # sigma_beta_i^2
            k**2 * self._outer6(alpha_f),  # sigma_alpha_f^2
            k**2 * self._outer6(beta_f),  # sigma_beta_f^2
            k**2 * self._outer6(q_lambda),  # sigma_dl_mod^2 (σ_λ/λ)
        ]

        if self.mosaic == "isotropic":
            mosaic_iso = Q2 * np.eye(3) - np.outer(Q_vec, Q_vec)
            cols.append(self._vech6(mosaic_iso))

        elif self.mosaic == "diagonal":
            R = peak.getGoniometerMatrix()
            for j in range(3):
                cols.append(self._outer6(np.cross(Q_vec, R[:, j])))

        else:  # "full"
            R = peak.getGoniometerMatrix()
            s2 = np.sqrt(0.5)
            mosaic_dirs = [
                R[:, 0],
                R[:, 1],
                R[:, 2],
                (R[:, 0] + R[:, 1]) * s2,
                (R[:, 0] + R[:, 2]) * s2,
                (R[:, 1] + R[:, 2]) * s2,
            ]
            for v in mosaic_dirs:
                cols.append(self._outer6(np.cross(Q_vec, v)))

        return np.column_stack(cols)

    def _predict_cov_lab(self, peak):
        A = self._model_design_lab(peak)
        y = A @ self.model["variance_parameters"]
        cov = np.array(
            [[y[0], y[5], y[4]], [y[5], y[1], y[3]], [y[4], y[3], y[2]]]
        )
        return 0.5 * (cov + cov.T)

    def robust_nnls(self, A, y, max_iter=20, c=1.345, eps=1e-12):
        """
        Robust NNLS using Huber-style iterative reweighting.
        A x ~= y, x >= 0
        """

        finite = np.all(np.isfinite(A), axis=1) & np.isfinite(y)
        A = A[finite]
        y = y[finite]

        # initial unweighted fit
        x, _ = nnls(A, y)

        for _ in range(max_iter):
            y_fit = A @ x

            # compare widths rather than variances
            r = np.sqrt(np.maximum(y_fit, eps)) - np.sqrt(np.maximum(y, eps))

            # robust scale estimate
            mad = np.median(np.abs(r - np.median(r)))
            scale = 1.4826 * mad + eps

            z = r / scale

            # Huber weights
            w = np.ones_like(z)
            mask = np.abs(z) > c
            w[mask] = c / np.abs(z[mask])

            # weighted NNLS
            sw = np.sqrt(w)
            x_new, _ = nnls(A * sw[:, None], y * sw)

            if np.linalg.norm(x_new - x) < 1e-10 * (np.linalg.norm(x) + eps):
                x = x_new
                break

            x = x_new

        residual_norm = np.linalg.norm(A @ x - y)
        return x, residual_norm, w

    def fit(self):
        ws = mtd[self.peaks_ws]

        A_blocks = []
        y_blocks = []
        used = []

        for i, peak in enumerate(ws):
            sig_noise = peak.getIntensityOverSigma()
            if not np.isfinite(sig_noise) or sig_noise < self.sig_noise_cut:
                continue

            two_theta = peak.getScattering()
            lamda = peak.getWavelength()

            if lamda < self.lamda_min:
                self.lamda_min = lamda
            if lamda > self.lamda_max:
                self.lamda_max = lamda

            if two_theta < self.two_theta_min:
                self.two_theta_min = two_theta
            if two_theta > self.two_theta_max:
                self.two_theta_max = two_theta

            R = peak.getGoniometerMatrix()

            radii_s, V_s = self._get_peak_shape(ws, i, self.r_cut)

            if not np.all(np.isfinite(radii_s)) or np.any(radii_s <= 0):
                continue

            V_lab = self._sample_axes_to_lab(R, V_s)
            V_lab = self._normalize_columns(V_lab)

            cov_lab_obs = self._covariance_from_ellipsoid(radii_s, V_lab)
            cov_lab_obs = 0.5 * (cov_lab_obs + cov_lab_obs.T)

            y_p = self._vech6(cov_lab_obs)
            A_p = self._model_design_lab(peak)

            Q = self._Q_magnitude(two_theta, lamda)

            if not (np.isfinite(Q) and Q > 0):
                continue

            w = sig_noise / Q**2

            if not (np.all(np.isfinite(y_p)) and np.all(np.isfinite(A_p))):
                continue

            A_blocks.append(w * A_p)
            y_blocks.append(w * y_p)
            used.append(i)

        if not A_blocks:
            return

        A = np.vstack(A_blocks)
        y = np.concatenate(y_blocks)

        x, residual_norm, robust_weights = self.robust_nnls(A, y)

        sq = lambda v: np.sqrt(max(v, 0.0))
        base = {
            "sigma_alpha_i": sq(x[0]),
            "sigma_beta_i": sq(x[1]),
            "sigma_alpha_f": sq(x[2]),
            "sigma_beta_f": sq(x[3]),
            "sigma_dl_mod": sq(x[4]),
        }
        if self.mosaic == "isotropic":
            base["sigma_mosaic"] = sq(x[5])
        elif self.mosaic == "diagonal":
            base["sigma_mosaic_0"] = sq(x[5])
            base["sigma_mosaic_1"] = sq(x[6])
            base["sigma_mosaic_2"] = sq(x[7])
        else:  # full
            labels = ["00", "11", "22", "01", "02", "12"]
            for k, lab in enumerate(labels):
                base[f"sigma_mosaic_{lab}"] = sq(x[5 + k])

        self.model = {
            **base,
            "variance_parameters": x,
            "residual_norm": residual_norm,
            "used_peaks": used,
        }
        return self.model

    def apply(self):
        if self.model is None:
            raise RuntimeError("Call fit() before apply().")

        lo, hi = self.scale_bounds
        ws = mtd[self.peaks_ws]

        for i, peak in enumerate(ws):
            R = peak.getGoniometerMatrix()

            cov_lab = self._predict_cov_lab(peak)
            radii, V_lab = self._ellipsoid_from_covariance(cov_lab)

            V_sample = self._lab_axes_to_sample(R, V_lab)
            V_sample = self._normalize_columns(V_sample)

            self._set_peak_shape(ws, i, radii, V_sample)

    def predict_lab_cov(self, peak_index):
        ws = mtd[self.peaks_ws]
        peak = ws.getPeak(peak_index)
        cov_lab = self._predict_cov_lab(peak)
        return 0.5 * (cov_lab + cov_lab.T)

    def predict_sample_cov(self, peak_index):
        cov_lab = self.predict_lab_cov(peak_index)
        ws = mtd[self.peaks_ws]
        peak = ws.getPeak(peak_index)
        R = peak.getGoniometerMatrix()
        cov_sam = R.T @ cov_lab @ R
        return 0.5 * (cov_sam + cov_sam.T)

    def estimate_prior_sigmas(self, moment_covs=None):
        """
        Estimate prior width for covariance and centroid parameters.

        Parameters
        ----------
        moment_covs : dict or None
            {peak_index: (3,3) sample-frame empirical covariance} from
            PeakEstimator.moment_covs.  When provided these true data-driven
            observations are used instead of the peak shapes in the workspace
            (which are set by res.apply() and are trivially 1:1 with the model).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before estimate_prior_sigmas().")

        ws = mtd[self.peaks_ws]
        cov_obs_vecs = []
        offset_vecs = []

        cov_indices = (
            list(moment_covs.keys())
            if moment_covs is not None
            else self.model["used_peaks"]
        )

        moment_cov_obs = {}  # peak_index -> C_w (S-W frame, un-normalised)

        for i in cov_indices:
            peak = ws.getPeak(i)
            R = np.array(peak.getGoniometerMatrix()).reshape(3, 3)
            Q_mag = np.linalg.norm(np.array(peak.getQLabFrame()))
            if Q_mag == 0:
                continue

            T = self._stoica_wilkinson_transform_from_peak(peak)

            if moment_covs is not None:
                cov_lab = R @ moment_covs[i] @ R.T
            else:
                radii, V_s = self._get_peak_shape(ws, i, self.r_cut)
                V_lab = self._sample_axes_to_lab(R, V_s)
                cov_lab = self._covariance_from_ellipsoid(radii, V_lab)

            cov_lab = 0.5 * (cov_lab + cov_lab.T)
            C_w = T @ cov_lab @ T.T
            moment_cov_obs[i] = C_w

            cov_obs_vecs.append(
                np.array(
                    [
                        C_w[0, 0],
                        C_w[1, 1],
                        C_w[2, 2],
                        C_w[1, 2],
                        C_w[0, 2],
                        C_w[0, 1],
                    ]
                )
                / Q_mag**2
            )

        self._moment_cov_obs = moment_cov_obs

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)
            R = peak.getGoniometerMatrix()
            Q_mag = np.linalg.norm(np.array(peak.getQLabFrame()))
            if Q_mag == 0:
                continue
            T = self._stoica_wilkinson_transform_from_peak(peak)
            offset_s = self._get_peak_offset(ws, i)
            offset_vecs.append(T @ (R @ offset_s) / Q_mag)

        cov_obs_vecs = np.array(cov_obs_vecs)  # (n_peaks, 6)
        offset_vecs = np.array(offset_vecs)  # (n_peaks, 3)

        self.prior_cov_sigma = np.sqrt(np.mean(cov_obs_vecs**2, axis=0))
        self.prior_center_sigma = np.sqrt(np.mean(offset_vecs**2, axis=0))

        self._offset_vecs = offset_vecs

        return self.prior_center_sigma, self.prior_cov_sigma

    def diagnostics(self):
        if self.model is None:
            raise RuntimeError("Call fit() before diagnostics().")

        ws = mtd[self.peaks_ws]
        rows = []

        moment_cov_obs = getattr(self, "_moment_cov_obs", {})

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)

            R = peak.getGoniometerMatrix()

            cov_lab_pred = self._predict_cov_lab(peak)

            T = self._stoica_wilkinson_transform_from_peak(peak)

            if i in moment_cov_obs:
                cov_w_obs = moment_cov_obs[i]
            else:
                radii, V_sample = self._get_peak_shape(ws, i, self.r_cut)
                V_lab = self._sample_axes_to_lab(R, V_sample)
                cov_lab_obs = self._covariance_from_ellipsoid(radii, V_lab)
                cov_lab_obs = 0.5 * (cov_lab_obs + cov_lab_obs.T)
                cov_w_obs = T @ cov_lab_obs @ T.T

            cov_w_pred = T @ cov_lab_pred @ T.T

            # Project translation offset along predicted ellipsoid axes
            offset_s = self._get_peak_offset(ws, i)
            offset_lab = R @ offset_s
            radii_pred, V_lab_pred = self._ellipsoid_from_covariance(
                cov_lab_pred
            )
            projs = [
                float(np.dot(offset_lab, V_lab_pred[:, k])) for k in range(3)
            ]
            norms = [projs[k] / max(radii_pred[k], 1e-12) for k in range(3)]

            two_theta = peak.getScattering()
            phi = peak.getAzimuthal()

            kx_hat = np.sin(two_theta) * np.cos(phi)
            ky_hat = np.sin(two_theta) * np.sin(phi)
            kz_hat = np.cos(two_theta)

            gamma = np.arctan2(kx_hat, kz_hat)
            nu = np.arcsin(ky_hat)

            rows.append(
                {
                    "i": i,
                    "gamma": gamma,
                    "nu": nu,
                    "lambda": peak.getWavelength(),
                    "signal_noise": peak.getIntensityOverSigma(),
                    "Q": peak.getQSampleFrame().norm(),
                    "obs_x0": cov_w_obs[0, 0],
                    "obs_x1": cov_w_obs[1, 1],
                    "obs_x2": cov_w_obs[2, 2],
                    "pred_x0": cov_w_pred[0, 0],
                    "pred_x1": cov_w_pred[1, 1],
                    "pred_x2": cov_w_pred[2, 2],
                    "offset_x0": projs[0],
                    "offset_x1": projs[1],
                    "offset_x2": projs[2],
                    "offset_norm_x0": norms[0],
                    "offset_norm_x1": norms[1],
                    "offset_norm_x2": norms[2],
                }
            )

        return rows

    def plot_diagnostics(self, filename):
        rows = self.diagnostics()

        gamma = np.array([r["gamma"] for r in rows])
        nu = np.array([r["nu"] for r in rows])
        lamda = np.array([r["lambda"] for r in rows])
        Q = np.array([r["Q"] for r in rows])
        signal_noise = np.array([r["signal_noise"] for r in rows])

        s = 1 if Q.size > 1000 else 10

        obs = {
            k: np.array([r[f"obs_{k}"] for r in rows])
            for k in ["x0", "x1", "x2"]
        }
        pred = {
            k: np.array([r[f"pred_{k}"] for r in rows])
            for k in ["x0", "x1", "x2"]
        }
        offset = {
            k: np.array([r[f"offset_{k}"] for r in rows])
            for k in ["x0", "x1", "x2"]
        }

        hi = np.sqrt(
            max(
                max(obs[k].max() for k in obs),
                max(pred[k].max() for k in pred),
            )
        )

        max_res = max((abs(obs[k] - pred[k]) / Q).max() for k in obs) * 100

        max_trans = max(abs(offset[k]).max() for k in offset)

        names = {
            "x0": "{|Q|}",
            "x1": "{\\Delta{Q}_1}",
            "x2": "{\\Delta{Q}_2}",
        }

        fig, axes = plt.subplots(
            2, 6, figsize=(18, 7), constrained_layout=True
        )

        sc_lambda = sc_sn = sc_resid = None

        for k, lab in enumerate(["x0", "x1", "x2"]):
            c0 = k
            c1 = k + 3
            name = names[lab]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            # [0, c0] obs vs calc – lambda color
            ax = axes[0, c0]
            sc_lambda = ax.scatter(
                sig_pred,
                sig_obs,
                c=lamda,
                s=s,
                cmap="viridis",
                marker=".",
                rasterized=True,
            )
            ax.plot([0, hi], [0, hi], "k--", lw=1)
            ax.set_xlabel("$r(\\mathrm{calc})$ [$\\AA^{-1}$]")
            if k == 0:
                ax.set_ylabel("$r(\\mathrm{obs})$ [$\\AA^{-1}$]")
            else:
                ax.tick_params(labelleft=False)
            ax.set_title(f"${name}$-axis")
            ax.set_aspect("equal", adjustable="box")
            ax.minorticks_on()

            # [0, c1] residual vs Q – S/N color
            ax = axes[0, c1]
            resid = (sig_obs - sig_pred) / Q * 100
            sc_sn = ax.scatter(
                Q,
                resid,
                c=signal_noise,
                s=s,
                cmap="plasma",
                norm="log",
                marker=".",
                rasterized=True,
            )
            ax.axhline(0, color="k", lw=1, linestyle="--")
            ax.set_ylim(-max_res, max_res)
            ax.tick_params(labelbottom=False)
            if k == 0:
                ax.set_ylabel("$[r_{\\rm obs}-r_{\\rm calc}]/|Q|$ [%]")
            else:
                ax.tick_params(labelleft=False)
            ax.set_title(f"${name}$-axis")
            ax.minorticks_on()

            # [1, c0] gamma/nu map – relative residual % color
            ax = axes[1, c0]
            resid_map = (
                np.abs(sig_obs - sig_pred) / np.maximum(sig_pred, 1e-12) * 100
            )
            sc_resid = ax.scatter(
                np.rad2deg(gamma),
                np.rad2deg(nu),
                c=resid_map,
                s=s,
                cmap="binary",
                norm="linear",
                marker=".",
                rasterized=True,
            )
            ax.set_xlabel("$\\gamma$ [$^\\circ$]")
            if k == 0:
                ax.set_ylabel("$\\nu$ [$^\\circ$]")
            else:
                ax.tick_params(labelleft=False)
            ax.set_aspect("equal", adjustable="box")
            ax.minorticks_on()

            # [1, c1] center offset vs Q – S/N color
            ax = axes[1, c1]
            resid = offset[lab] / Q * 100
            sc_sn = ax.scatter(
                Q,
                resid,
                c=signal_noise,
                s=s,
                cmap="plasma",
                norm="log",
                marker=".",
                rasterized=True,
            )
            ax.axhline(0, color="k", lw=1, linestyle="--")
            ax.set_ylim(-max_trans, max_trans)
            ax.set_xlabel("$|Q|$ [$\\AA^{-1}$]")
            if k == 0:
                ax.set_ylabel("$\Delta{c}/|Q|$ [%]")
            else:
                ax.tick_params(labelleft=False)
            ax.minorticks_on()

        cb = fig.colorbar(
            sc_lambda, ax=list(axes[0, [0, 1, 2]]), label="$\\lambda$ [$\\AA$]"
        )
        cb.ax.minorticks_on()
        cb = fig.colorbar(
            sc_sn,
            ax=axes[:, [3, 4, 5]].ravel().tolist(),
            label="$I/\\sigma$",
        )
        cb.ax.minorticks_on()
        cb = fig.colorbar(
            sc_resid,
            ax=list(axes[1, [2]]),
            label="$|r_{\\rm obs}/r_{\\rm calc}-1|$ [%]",
        )
        cb.ax.minorticks_on()

        fig.savefig(filename, bbox_inches="tight")
