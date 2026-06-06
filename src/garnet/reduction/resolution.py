import matplotlib.pyplot as plt
import numpy as np

from mantid.simpleapi import mtd
from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid

from scipy.optimize import nnls


class ResolutionEllipsoid:
    def __init__(
        self,
        peaks_ws,
        r_cut=np.inf,
        sig_noise_cut=5.0,
        min_peaks=10,
        scale_bounds=(0.5, 2.0),
    ):
        self.peaks_ws = peaks_ws
        self.r_cut = r_cut
        self.sig_noise_cut = sig_noise_cut
        self.min_peaks = min_peaks
        self.scale_bounds = scale_bounds
        self.model = None
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
        return V @ np.diag(np.asarray(radii, dtype=float) ** 2) @ V.T

    def _ellipsoid_from_covariance(self, cov):
        cov = 0.5 * (cov + cov.T)

        W, V = np.linalg.eigh(cov)

        radii = np.sqrt(W)
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

        if Q2 > 0:
            qhat = Q_vec / np.sqrt(Q2)
            m1, m2 = self._transverse_directions(qhat)
        else:
            m1 = np.array([1.0, 0.0, 0.0])
            m2 = np.array([0.0, 1.0, 0.0])

        A = np.column_stack(
            [
                k**2 * self._outer6(alpha_i),  # sigma_alpha_i^2
                k**2 * self._outer6(beta_i),  # sigma_beta_i^2
                k**2 * self._outer6(alpha_f),  # sigma_alpha_f^2
                k**2 * self._outer6(beta_f),  # sigma_beta_f^2
                (k / lamda) ** 2
                * self._outer6(q_lambda),  # sigma_dl_timing^2 (σ_λ = const)
                k**2
                * self._outer6(q_lambda),  # sigma_dl_mod^2 (σ_λ/λ = const)
                Q2 * self._outer6(m1),  # sigma_mosaic_1^2
                Q2 * self._outer6(m2),  # sigma_mosaic_2^2
            ]
        )

        return A

    def _predict_cov_lab(self, peak):
        A = self._model_design_lab(peak)

        x = np.array(
            [
                self.model["sigma_alpha_i"] ** 2,
                self.model["sigma_beta_i"] ** 2,
                self.model["sigma_alpha_f"] ** 2,
                self.model["sigma_beta_f"] ** 2,
                self.model["sigma_dl_timing"] ** 2,
                self.model["sigma_dl_mod"] ** 2,
                self.model["sigma_mosaic_1"] ** 2,
                self.model["sigma_mosaic_2"] ** 2,
            ]
        )

        y = A @ x

        cov = np.array(
            [
                [y[0], y[5], y[4]],
                [y[5], y[1], y[3]],
                [y[4], y[3], y[2]],
            ]
        )

        return 0.5 * (cov + cov.T)

    def robust_nnls(self, A, y, max_iter=20, c=1.345, eps=1e-12):
        """
        Robust NNLS using Huber-style iterative reweighting.
        A x ~= y, x >= 0
        """

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
            if sig_noise < self.sig_noise_cut:
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

            V_lab = self._sample_axes_to_lab(R, V_s)
            V_lab = self._normalize_columns(V_lab)

            cov_lab_obs = self._covariance_from_ellipsoid(radii_s, V_lab)
            cov_lab_obs = 0.5 * (cov_lab_obs + cov_lab_obs.T)

            y_p = self._vech6(cov_lab_obs)
            A_p = self._model_design_lab(peak)

            Q = self._Q_magnitude(two_theta, lamda)

            w = sig_noise / Q**2

            A_blocks.append(w * A_p)
            y_blocks.append(w * y_p)
            used.append(i)

        A = np.vstack(A_blocks)
        y = np.concatenate(y_blocks)

        x, residual_norm, robust_weights = self.robust_nnls(A, y)

        self.model = {
            "sigma_alpha_i": np.sqrt(max(x[0], 0.0)),
            "sigma_beta_i": np.sqrt(max(x[1], 0.0)),
            "sigma_alpha_f": np.sqrt(max(x[2], 0.0)),
            "sigma_beta_f": np.sqrt(max(x[3], 0.0)),
            "sigma_dl_timing": np.sqrt(max(x[4], 0.0)),
            "sigma_dl_mod": np.sqrt(max(x[5], 0.0)),
            "sigma_mosaic_1": np.sqrt(max(x[6], 0.0)),
            "sigma_mosaic_2": np.sqrt(max(x[7], 0.0)),
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

            # radii = np.clip(radii, lo * self.r_cut, hi * self.r_cut)

            V_sample = self._lab_axes_to_sample(R, V_lab)
            V_sample = self._normalize_columns(V_sample)

            self._set_peak_shape(ws, i, radii, V_sample)

    def diagnostics(self):
        if self.model is None:
            raise RuntimeError("Call fit() before diagnostics().")

        ws = mtd[self.peaks_ws]
        rows = []

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)

            R = peak.getGoniometerMatrix()

            radii, V_sample = self._get_peak_shape(ws, i, self.r_cut)
            V_lab = self._sample_axes_to_lab(R, V_sample)

            cov_lab_obs = self._covariance_from_ellipsoid(radii, V_lab)
            cov_lab_obs = 0.5 * (cov_lab_obs + cov_lab_obs.T)

            cov_lab_pred = self._predict_cov_lab(peak)

            T = self._stoica_wilkinson_transform_from_peak(peak)

            cov_w_obs = T @ cov_lab_obs @ T.T
            cov_w_pred = T @ cov_lab_pred @ T.T

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

        obs = {
            "x0": np.array([r["obs_x0"] for r in rows]),
            "x1": np.array([r["obs_x1"] for r in rows]),
            "x2": np.array([r["obs_x2"] for r in rows]),
        }

        pred = {
            "x0": np.array([r["pred_x0"] for r in rows]),
            "x1": np.array([r["pred_x1"] for r in rows]),
            "x2": np.array([r["pred_x2"] for r in rows]),
        }

        hi_obs = max([obs[key].max() for key in obs.keys()])
        hi_pred = max([pred[key].max() for key in pred.keys()])

        hi = np.sqrt(max([hi_obs, hi_pred]))

        names = {
            "x0": "{|Q|}",
            "x1": "{\Delta{Q}_1}",
            "x2": "{\Delta{Q}_2}",
        }

        fig, axes = plt.subplots(
            3,
            3,
            figsize=(10, 10),
            constrained_layout=True,
            sharex="row",
            sharey="row",
        )

        for j, lab in enumerate(["x0", "x1", "x2"]):
            ax = axes[0, j]

            name = names[lab]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            sc = ax.scatter(
                sig_pred,
                sig_obs,
                c=lamda,
                cmap="viridis",
                marker=".",
                rasterized=True,
            )

            ax.plot([0, hi], [0, hi], "k--", lw=1)

            ax.set_xlabel("$r(\\mathrm{{calc}})$ [$\AA^{{-1}}$]")
            if j == 0:
                ax.set_ylabel("$r(\\mathrm{{obs}})$ [$\AA^{{-1}}$]")
            ax.set_title(f"${name}$-axis")
            ax.set_aspect("equal", adjustable="box")
            ax.minorticks_on()

        cb = fig.colorbar(sc, ax=axes[0, :], label="$\\lambda$ [$\\AA$]")
        cb.ax.minorticks_on()

        for j, lab in enumerate(["x0", "x1", "x2"]):
            ax = axes[1, j]

            name = names[lab]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            resid = (sig_obs - sig_pred) / Q * 100

            sc = ax.scatter(
                Q,
                resid,
                c=signal_noise,
                cmap="plasma",
                norm="log",
                marker=".",
                rasterized=True,
            )

            ax.axhline(0, color="k", lw=1, linestyle="--")
            ax.set_xlabel("$|Q|$ [$\AA^{{-1}}$]")
            if j == 0:
                ax.set_ylabel(
                    "$[r(\\mathrm{{obs}})-r(\\mathrm{{calc}})]/|Q|$ [%]"
                )
            ax.minorticks_on()

        cb = fig.colorbar(sc, ax=axes[1, :], label="$I/\\sigma$")
        cb.ax.minorticks_on()

        for j, lab in enumerate(["x0", "x1", "x2"]):
            ax = axes[2, j]

            name = names[lab]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            resid = np.abs(sig_obs - sig_pred) / sig_pred * 100

            sc = ax.scatter(
                np.rad2deg(gamma),
                np.rad2deg(nu),
                c=resid,
                cmap="binary",
                norm="linear",
                marker=".",
                rasterized=True,
            )

            ax.set_xlabel("$\\gamma$ [$^\\circ$]")
            if j == 0:
                ax.set_ylabel("$\\nu$ [$^\\circ$]")
            ax.minorticks_on()
            ax.set_aspect("equal", adjustable="box")

        cb = fig.colorbar(
            sc,
            ax=axes[2, :],
            label="$|r(\\mathrm{{obs}})-r(\\mathrm{{calc}})|/r(\\mathrm{{calc}})$ [%]",
        )
        cb.ax.minorticks_on()

        fig.savefig(filename, bbox_inches="tight")
