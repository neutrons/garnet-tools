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
        r_cut=1.0,
        sig_noise_cut=20.0,
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

    def _normalize_columns(self, V):
        V = np.asarray(V, dtype=float)
        n = np.linalg.norm(V, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return V / n

    def _vech_diag(self, M):
        return np.array([M[0, 0], M[1, 1], M[2, 2]], dtype=float)

    def _get_peak_shape(self, ws, no, r_cut):
        shape = ws.getPeak(no).getPeakShape()

        if shape.shapeName() == "ellipsoid":
            d = eval(shape.toJSON())

            v0 = [float(x) for x in d["direction0"].split()]
            v1 = [float(x) for x in d["direction1"].split()]
            v2 = [float(x) for x in d["direction2"].split()]

            radii = np.array(
                [
                    min(float(d["radius0"]), r_cut),
                    min(float(d["radius1"]), r_cut),
                    min(float(d["radius2"]), r_cut),
                ]
            )
        else:
            radii = np.full(3, r_cut, dtype=float)
            v0, v1, v2 = np.eye(3).tolist()

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
        order = np.argsort(W)[::-1]

        W = W[order]
        V = V[:, order]

        radii = np.sqrt(np.maximum(W, 0.0))
        V = self._normalize_columns(V)

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

    def _wilkinson_transform_from_peak(self, peak):
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

        x = kf_hat
        z = np.cross(ki_hat, kf_hat)
        z /= np.linalg.norm(z)

        y = np.cross(z, x)
        y /= np.linalg.norm(y)

        return np.vstack([x, y, z])

    def _sample_axes_to_lab(self, R, V_sample):
        return R @ V_sample

    def _lab_axes_to_sample(self, R, V_lab):
        return R.T @ V_lab

    def _model_design_lab_wilkinson(self, peak):
        two_theta = peak.getScattering()
        lamda = peak.getWavelength()

        theta = 0.5 * two_theta
        H = (4.0 * np.pi / lamda) * np.sin(theta)

        s = np.sin(theta)
        ct = 1.0 / np.tan(theta)

        A = np.zeros((3, 5), dtype=float)

        # parameters:
        # x[0] = (sigma_t / T)^2 = (sigma_lambda / lambda)^2
        # x[1] = sigma_alpha_i^2
        # x[2] = sigma_alpha_f^2
        # x[3] = sigma_beta_i^2 + sigma_beta_f^2, or one combined beta term
        # x[4] = eta_s^2

        # Hx
        A[0, 0] = H**2
        A[0, 1] = H**2 * ct**2 / 4.0
        A[0, 2] = H**2 * ct**2 / 4.0

        # Hy
        A[1, 1] = H**2 / 4.0
        A[1, 2] = H**2 / 4.0
        A[1, 4] = H**2

        # Hz
        A[2, 3] = H**2 / (4.0 * s**2)
        A[2, 4] = H**2

        return A

    def _predict_cov_lab(self, peak):
        A = self._model_design_lab_wilkinson(peak)

        x = np.array(
            [
                self.model["sigma_dlambda_over_lambda"] ** 2,
                self.model["sigma_alpha_i"] ** 2,
                self.model["sigma_alpha_f"] ** 2,
                self.model["sigma_beta_combined"] ** 2,
                self.model["eta_s"] ** 2,
            ]
        )

        sig2 = A @ x
        sig2 = np.maximum(sig2, 0.0)

        cov_w = np.diag(sig2)

        T = self._wilkinson_transform_from_peak(peak)
        cov_lab = T.T @ cov_w @ T

        return 0.5 * (cov_lab + cov_lab.T)

    def fit(self):
        ws = mtd[self.peaks_ws]

        A_blocks = []
        y_blocks = []
        used = []

        for i, peak in enumerate(ws):
            if peak.getIntensityOverSigma() < self.sig_noise_cut:
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

            T = self._wilkinson_transform_from_peak(peak)

            cov_w_obs = T @ cov_lab_obs @ T.T
            y_p = self._vech_diag(cov_w_obs)
            A_p = self._model_design_lab_wilkinson(peak)

            A_blocks.append(A_p)
            y_blocks.append(y_p)
            used.append(i)

        A = np.vstack(A_blocks)
        y = np.concatenate(y_blocks)

        x, residual_norm = nnls(A, y)

        self.model = {
            "sigma_dlambda_over_lambda": np.sqrt(max(x[0], 0.0)),
            "sigma_alpha_i": np.sqrt(max(x[1], 0.0)),
            "sigma_alpha_f": np.sqrt(max(x[2], 0.0)),
            "sigma_beta_combined": np.sqrt(max(x[3], 0.0)),
            "eta_s": np.sqrt(max(x[4], 0.0)),
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
            radii_lab, V_lab = self._ellipsoid_from_covariance(cov_lab)

            radii_lab = np.clip(radii_lab, lo * self.r_cut, hi * self.r_cut)

            V_sample = self._lab_axes_to_sample(R, V_lab)
            V_sample = self._normalize_columns(V_sample)

            self._set_peak_shape(ws, i, radii_lab, V_sample)

    def diagnostics(self):
        if self.model is None:
            raise RuntimeError("Call fit() before diagnostics().")

        ws = mtd[self.peaks_ws]
        rows = []

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)

            R = peak.getGoniometerMatrix()

            radii_s, V_s = self._get_peak_shape(ws, i, self.r_cut)
            V_lab = self._sample_axes_to_lab(R, V_s)

            cov_lab_obs = self._covariance_from_ellipsoid(radii_s, V_lab)
            cov_lab_obs = 0.5 * (cov_lab_obs + cov_lab_obs.T)

            cov_lab_pred = self._predict_cov_lab(peak)

            T = self._wilkinson_transform_from_peak(peak)

            cov_w_obs = T @ cov_lab_obs @ T.T
            cov_w_pred = T @ cov_lab_pred @ T.T

            rows.append(
                {
                    "i": i,
                    "two_theta": peak.getScattering(),
                    "lambda": peak.getWavelength(),
                    "signal_noise": peak.getIntensityOverSigma(),
                    "obs_x": cov_w_obs[0, 0],
                    "obs_y": cov_w_obs[1, 1],
                    "obs_z": cov_w_obs[2, 2],
                    "pred_x": cov_w_pred[0, 0],
                    "pred_y": cov_w_pred[1, 1],
                    "pred_z": cov_w_pred[2, 2],
                }
            )

        return rows

    def plot_diagnostics(self, filename):
        rows = self.diagnostics()

        two_theta = np.array([r["two_theta"] for r in rows])
        lamda = np.array([r["lambda"] for r in rows])

        obs = {
            "x": np.array([r["obs_x"] for r in rows]),
            "y": np.array([r["obs_y"] for r in rows]),
            "z": np.array([r["obs_z"] for r in rows]),
        }

        pred = {
            "x": np.array([r["pred_x"] for r in rows]),
            "y": np.array([r["pred_y"] for r in rows]),
            "z": np.array([r["pred_z"] for r in rows]),
        }

        fig, axes = plt.subplots(
            2, 3, figsize=(13, 7), constrained_layout=True
        )

        for j, lab in enumerate(["x", "y", "z"]):
            ax = axes[0, j]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            sc = ax.scatter(sig_obs, sig_pred, c=lamda, s=25)

            lo = min(sig_obs.min(), sig_pred.min())
            hi = max(sig_obs.max(), sig_pred.max())

            ax.plot([lo, hi], [lo, hi], "k--", lw=1)

            ax.set_xlabel(f"observed $\\sigma_{lab}$ [$\\AA^{{-1}}$]")
            ax.set_ylabel(f"predicted $\\sigma_{lab}$ [$\\AA^{{-1}}$]")
            ax.set_title(f"${lab}$-axis width")
            ax.set_aspect("equal", adjustable="box")
            ax.minorticks_on()
            cb = fig.colorbar(sc, ax=ax, label="$\\lambda$ [$\\AA$]")
            cb.ax.minorticks_on()

        for j, lab in enumerate(["x", "y", "z"]):
            ax = axes[1, j]

            sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
            sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

            resid = sig_pred - sig_obs

            sc = ax.scatter(
                np.rad2deg(two_theta),
                resid,
                c=lamda,
                s=25,
            )

            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel("$2\\theta$ [deg]")
            ax.set_ylabel(f"$\\Delta\\sigma_{lab}$ [$\\AA^{{-1}}$]")
            ax.set_title(f"${lab}$-axis width")
            ax.minorticks_on()
            cb = fig.colorbar(sc, ax=ax, label="$\\lambda$ [$\\AA$]")
            cb.ax.minorticks_on()

        fig.savefig(filename, bbox_inches="tight")
