import time
import numpy as np

import scipy.spatial.transform
import scipy.special
import scipy.stats
import scipy.signal
import scipy.ndimage

from lmfit import Minimizer, Parameters, fit_report


class PeakEllipsoid:
    def __init__(self):
        self.params = Parameters()

        self.lamda_center = 1.0
        self.lamda_cov = 1.0

        self.mode_weights_1d = 1.0
        self.mode_weights_2d = 1.0
        self.mode_weights_3d = 1.0

        self.prior_center_sigma = 1.0
        self.prior_cov = np.eye(3)
        self.prior_cov_sigma = 1.0

    def vech6(self, M):
        return np.array(
            [M[0, 0], M[1, 1], M[2, 2], M[1, 2], M[0, 2], M[0, 1]],
            dtype=float,
        )

    def update_constraints(self, x0, x1, x2, dx):
        dx0 = x0[:, 0, 0][1] - x0[:, 0, 0][0]
        dx1 = x1[0, :, 0][1] - x1[0, :, 0][0]
        dx2 = x2[0, 0, :][1] - x2[0, 0, :][0]

        r0_max = (x0[:, 0, 0][-1] - x0[:, 0, 0][0]) / 2
        r1_max = (x1[0, :, 0][-1] - x1[0, :, 0][0]) / 2
        r2_max = (x2[0, 0, :][-1] - x2[0, 0, :][0]) / 2

        c0, c1, c2 = 0, 0, 0

        c0_min, c1_min, c2_min = (
            c0 - r0_max,
            c1 - r1_max,
            c2 - r2_max,
        )
        c0_max, c1_max, c2_max = (
            c0 + r0_max,
            c1 + r1_max,
            c2 + r2_max,
        )

        r_max = 0.5 * np.max([r0_max, r1_max, r2_max])

        dx = 2 * np.min([dx0, dx1, dx2])

        self.params.add("c0", value=c0, min=c0_min, max=c0_max)
        self.params.add("c1", value=c1, min=c1_min, max=c1_max)
        self.params.add("c2", value=c2, min=c2_min, max=c2_max)

        self.params.add("r0", value=r_max, min=dx, max=2 * r0_max)
        self.params.add("r1", value=r_max, min=dx, max=2 * r1_max)
        self.params.add("r2", value=r_max, min=dx, max=2 * r2_max)

        self.params.add("u0", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("u1", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("u2", value=0.0, min=-np.pi, max=np.pi)

        self.combine_params = None

    def copy_combine(self):
        self.combine_params = self.params.copy()

    def update_estimate(self, shape):
        c0, c1, c2, r0, r1, r2, v0, v1, v2 = shape

        U = np.column_stack([v0, v1, v2])
        if np.linalg.det(U) < 0:
            U[:, 2] *= -1

        u = scipy.spatial.transform.Rotation.from_matrix(U).as_rotvec()

        u0, u1, u2 = u

        self.params["c0"].set(value=0)
        self.params["c1"].set(value=0)
        self.params["c2"].set(value=0)

        for name, value in zip(["r0", "r1", "r2"], [r0, r1, r2]):
            p = self.params[name]
            if p.min < value < p.max:
                p.set(value=value)
            else:
                p.set(value=2 * p.min)

        self.params["u0"].set(value=u0)
        self.params["u1"].set(value=u1)
        self.params["u2"].set(value=u2)

        self.params["u0"].set(min=-np.pi, max=np.pi)
        self.params["u1"].set(min=-np.pi, max=np.pi)
        self.params["u2"].set(min=-np.pi, max=np.pi)

        self.estimated_fit = (
            np.array([0.0, 0.0, 0.0]),
            self.S_matrix(r0, r1, r2, u0, u1, u2),
        )

        self.prior_center_sigma = np.mean([r0, r1, r2]) / 3
        self.prior_cov_sigma = np.mean([r0**2, r1**2, r2**2]) / 3

        self.prior_cov = self.S_matrix(r0, r1, r2, u0, u1, u2)

    def prior_residual(self, params):
        terms = []

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        c = np.array([c0, c1, c2])

        terms += (
            np.sqrt(self.lamda_center) * c / self.prior_center_sigma
        ).tolist()

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        S = self.S_matrix(r0, r1, r2, u0, u1, u2)
        dS = self.vech6(S - self.prior_cov)
        terms += (np.sqrt(self.lamda_cov) * dS / self.prior_cov_sigma).tolist()

        return np.asarray(terms, dtype=float)

    def prior_jacobian(self, params, S, dr, du):
        names = ["c0", "c1", "c2", "S00", "S11", "S22", "S12", "S02", "S01"]
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), len(names)), dtype=float)

        for col, name in enumerate(["c0", "c1", "c2"]):
            jac[params_list.index(name), col] = (
                np.sqrt(self.lamda_center) / self.prior_center_sigma
            )

        d_inv_S = {
            "r0": dr[0],
            "r1": dr[1],
            "r2": dr[2],
            "u0": du[0],
            "u1": du[1],
            "u2": du[2],
        }

        cov_scale = np.sqrt(self.lamda_cov) / self.prior_cov_sigma
        for name in ["r0", "r1", "r2", "u0", "u1", "u2"]:
            dS = -S @ d_inv_S[name] @ S
            jac[params_list.index(name), 3:] = cov_scale * self.vech6(dS)

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]
        return jac[ind]

    def S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):
        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def det_S(self, r0, r1, r2, u0, u1, u2):
        S = self.S_matrix(r0, r1, r2, u0, u1, u2)
        return np.linalg.det(S)

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):
        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S

    def set_resolution_sigma(self, prior_center_sigma, prior_cov_sigma):
        self.prior_center_sigma = prior_center_sigma
        self.prior_cov_sigma = prior_cov_sigma

    def data_norm(self, d, n, v, rel_err=30):
        mask = (n > 0) & np.isfinite(n)

        d[~mask] = np.nan
        n[~mask] = np.nan
        v[~mask] = np.nan

        y_int = d / n
        e_int = np.sqrt(v + np.nanpercentile(v, rel_err)) / n

        return y_int, e_int

    def profile_project(self, x0, x1, x2, d, n, w, mode="3d"):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d0 = np.asarray(d, dtype=float).copy()
        n0 = np.asarray(n, dtype=float).copy()
        w0 = np.asarray(w, dtype=float).copy()

        valid = np.isfinite(d0) & np.isfinite(n0) & (n0 > 0)
        d0[~valid] = np.nan
        n0[~valid] = np.nan
        w0[~valid] = np.nan

        if mode == "1d_0":
            d_int = np.nansum(d0, axis=(1, 2))
            n_int = np.nanmean(n0 / dx1 / dx2, axis=(1, 2))
            v_int = np.nansum(np.clip(d0, 0, None), axis=(1, 2))
            w_int = np.nanmean(w0, axis=(1, 2))
        elif mode == "1d_1":
            d_int = np.nansum(d0, axis=(0, 2))
            n_int = np.nanmean(n0 / dx0 / dx2, axis=(0, 2))
            v_int = np.nansum(np.clip(d0, 0, None), axis=(0, 2))
            w_int = np.nanmean(w0, axis=(0, 2))
        elif mode == "1d_2":
            d_int = np.nansum(d0, axis=(0, 1))
            n_int = np.nanmean(n0 / dx0 / dx1, axis=(0, 1))
            v_int = np.nansum(np.clip(d0, 0, None), axis=(0, 1))
            w_int = np.nanmean(w0, axis=(0, 1))
        elif mode == "2d_0":
            d_int = np.nansum(d0, axis=0)
            n_int = np.nanmean(n0 / dx0, axis=0)
            v_int = np.nansum(np.clip(d0, 0, None), axis=0)
            w_int = np.nanmean(w0, axis=0)
        elif mode == "2d_1":
            d_int = np.nansum(d0, axis=1)
            n_int = np.nanmean(n0 / dx1, axis=1)
            v_int = np.nansum(np.clip(d0, 0, None), axis=1)
            w_int = np.nanmean(w0, axis=1)
        elif mode == "2d_2":
            d_int = np.nansum(d0, axis=2)
            n_int = np.nanmean(n0 / dx2, axis=2)
            v_int = np.nansum(np.clip(d0, 0, None), axis=2)
            w_int = np.nanmean(w0, axis=2)
        elif mode == "3d":
            d_int = d0.copy()
            n_int = n0.copy()
            v_int = np.clip(d0, 0, None).copy()
            w_int = w0.copy()
        else:
            raise ValueError("Unknown mode {}".format(mode))

        bad = ~np.isfinite(n_int) | (n_int <= 0)
        d_int[bad] = np.nan
        n_int[bad] = np.nan
        v_int[bad] = np.nan
        w_int[bad] = np.nan

        return d_int, n_int, v_int, w_int

    def normalize(self, x0, x1, x2, d, n, w, mode="3d"):
        d_int, n_int, v_int, w_int = self.profile_project(
            x0, x1, x2, d, n, w, mode=mode
        )

        y_int, e_int = self.data_norm(d_int, n_int, v_int)

        return y_int, e_int / w_int

    def ellipsoid_covariance(self, inv_S, mode="3d", perc=99.7):
        if mode == "3d":
            scale = scipy.stats.chi2.ppf(perc / 100, df=3)
            inv_var = inv_S * scale
        elif mode == "2d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[1:, 1:] * scale
        elif mode == "2d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[0::2, 0::2] * scale
        elif mode == "2d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=2)
            inv_var = inv_S[:2, :2] * scale
        elif mode == "1d_0":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[0, 0] * scale
        elif mode == "1d_1":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[1, 1] * scale
        elif mode == "1d_2":
            scale = scipy.stats.chi2.ppf(perc / 100, df=1)
            inv_var = inv_S[2, 2] * scale

        return inv_var

    def coerce_weight(self, target, weight=None):
        target = np.asarray(target)

        if weight is None:
            return np.ones_like(target, dtype=float)
        if np.isscalar(weight):
            return np.full_like(target, weight, dtype=float)

        return np.asarray(weight, dtype=float)

    def coerce_weights(self, targets, weights=None):
        if weights is None:
            weights = [None] * len(targets)

        return [
            self.coerce_weight(target, weight)
            for target, weight in zip(targets, weights)
        ]

    def uniform_mode_weights(self, ys, es, val=1.0):
        n_valid = sum(
            np.count_nonzero(np.isfinite(y) & np.isfinite(e) & (e > 0))
            for y, e in zip(ys, es)
        )

        scale = np.sqrt(val) / np.sqrt(max(n_valid, 1))

        return [np.full_like(e, scale, dtype=float) for e in es]

    def chi_2_fit(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
            m = 11
            k = 3
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
            m = 7
            k = 2
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
            m = 7
            k = 2
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
            m = 7
            k = 2
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
            m = 4
            k = 1
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
            m = 4
            k = 1
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2
            m = 4
            k = 1

        mask = (d2 <= 2 ** (2 / k)) & np.isfinite(e) & (e > 0)

        n = np.sum(mask)

        dof = n - m

        if dof <= 0:
            return np.inf
        else:
            return np.nansum(((y_fit[mask] - y[mask]) / e[mask]) ** 2) / dof

    def estimate_intensity(self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[1:, 1:], dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[0::2, 0::2], dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_S[:2, :2], dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_S[0, 0] * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_S[1, 1] * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_S[2, 2] * dx**2

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if mode == "3d":
            dx = np.prod([dx0, dx1, dx2])
            # k = 3
        elif mode == "2d_0":
            dx = np.prod([dx1, dx2])
            # k = 2
        elif mode == "2d_1":
            dx = np.prod([dx0, dx2])
            # k = 2
        elif mode == "2d_2":
            dx = np.prod([dx0, dx1])
            # k = 2
        elif mode == "1d_0":
            dx = dx0
            # k = 1
        elif mode == "1d_1":
            dx = dx1
            # k = 1
        elif mode == "1d_2":
            dx = dx2
            # k = 1

        pk = (d2 <= 1**2) & np.isfinite(y) & (e > 0)
        # bkg = (d2 > 1**2) & (d2 < 2 ** (2 / k)) & (e > 0)

        b = self.params["B{}".format(mode)].value
        b_err = self.params["B{}".format(mode)].stderr

        if b_err is None:
            b_err = b

        I = np.nansum(y[pk] - b) * dx
        sig = np.sqrt(np.nansum(e[pk] ** 2 + b_err**2)) * dx

        return I, sig

    def gaussian(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2

        return np.exp(-0.5 * d2)

    def inv_S_deriv_r(self, r0, r1, r2, u0, u1, u2):
        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2 / r0**3, 0, 0]) @ U.T
        dinv_S1 = U @ np.diag([0, -2 / r1**3, 0]) @ U.T
        dinv_S2 = U @ np.diag([0, 0, -2 / r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_u(self, r0, r1, r2, u0, u1, u2):
        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        U = self.U_matrix(u0, u1, u2)
        dU0, dU1, dU2 = self.U_deriv_u(u0, u1, u2)

        dinv_S0 = dU0 @ V @ U.T + U @ V @ dU0.T
        dinv_S1 = dU1 @ V @ U.T + U @ V @ dU1.T
        dinv_S2 = dU2 @ V @ U.T + U @ V @ dU2.T

        return dinv_S0, dinv_S1, dinv_S2

    def U_deriv_u(self, u0, u1, u2, delta=1e-6):
        dU0 = self.U_matrix(u0 + delta, u1, u2) - self.U_matrix(
            u0 - delta, u1, u2
        )
        dU1 = self.U_matrix(u0, u1 + delta, u2) - self.U_matrix(
            u0, u1 - delta, u2
        )
        dU2 = self.U_matrix(u0, u1, u2 + delta) - self.U_matrix(
            u0, u1, u2 - delta
        )

        return 0.5 * dU0 / delta, 0.5 * dU1 / delta, 0.5 * dU2 / delta

    def gaussian_integral(self, inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        return np.sqrt((2 * np.pi) ** k * det)

    def gaussian_integral_jac_S(self, inv_S, d_inv_S, mode="3d"):
        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            k = 3
            det = 1 / np.linalg.det(inv_var)
        elif "2d" in mode:
            k = 2
            det = 1 / np.linalg.det(inv_var)
        elif "1d" in mode:
            k = 1
            det = 1 / inv_var

        g = np.sqrt((2 * np.pi) ** k * det)

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
        elif mode == "2d_0":
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g0 = g1 * 0
        elif mode == "2d_1":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g2 = np.einsum("ij,ji->", inv_var, d_inv_var[2])
            g1 = g2 * 0
        elif mode == "2d_2":
            g0 = np.einsum("ij,ji->", inv_var, d_inv_var[0])
            g1 = np.einsum("ij,ji->", inv_var, d_inv_var[1])
            g2 = g0 * 0
        elif mode == "1d_0":
            g0 = d_inv_var[0] * inv_var
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            g1 = d_inv_var[1] * inv_var
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            g2 = d_inv_var[2] * inv_var
            g0 = g1 = g2 * 0

        return 0.5 * g * np.array([g0, g1, g2])

    def gaussian_jac_c(self, x0, x1, x2, c, inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g0 = np.zeros_like(g1)
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g2 = np.einsum("ij,j...->i...", inv_var, dx)
            g1 = np.zeros_like(g0)
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1 = np.einsum("ij,j...->i...", inv_var, dx)
            g2 = np.zeros_like(g0)
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = inv_var * dx
            g1 = g2 = np.zeros_like(g0)
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = inv_var * dx
            g2 = g0 = np.zeros_like(g1)
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = inv_var * dx
            g0 = g1 = np.zeros_like(g2)

        g = np.exp(-0.5 * d2)

        return g * np.array([g0, g1, g2])

    def gaussian_jac_S(self, x0, x1, x2, c, inv_S, d_inv_S, mode="3d"):
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            dx = [dx0, dx1, dx2]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g0 = g1 * 0
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g2 = np.einsum("i...,ij,j...->...", dx, d_inv_var[2], dx)
            g1 = g2 * 0
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0 = np.einsum("i...,ij,j...->...", dx, d_inv_var[0], dx)
            g1 = np.einsum("i...,ij,j...->...", dx, d_inv_var[1], dx)
            g2 = g0 * 0
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
            d2 = inv_var * dx**2
            g0 = d_inv_var[0] * dx**2
            g1 = g2 = g0 * 0
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
            d2 = inv_var * dx**2
            g1 = d_inv_var[1] * dx**2
            g2 = g0 = g1 * 0
        elif mode == "1d_2":
            dx = dx2[0, 0, :]
            d2 = inv_var * dx**2
            g2 = d_inv_var[2] * dx**2
            g0 = g1 = g2 * 0

        g = np.exp(-0.5 * d2)

        return -0.5 * g * np.array([g0, g1, g2])

    def counts_to_intensity_uncertainty(self, d, n):
        d = np.asarray(d, dtype=float)
        n = np.asarray(n, dtype=float)
        y = np.divide(
            d, n, out=np.full_like(d, np.nan, dtype=float), where=n > 0
        )
        e = np.divide(
            np.sqrt(np.clip(d, 0, None) + 1.0),
            n,
            out=np.full_like(d, np.nan, dtype=float),
            where=n > 0,
        )
        return y, e

    def poisson_deviance_residual_factor(self, d, mu, w=None, eps=1e-12):
        d = np.asarray(d, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if w is None:
            w = np.ones_like(d, dtype=float)
        else:
            w = np.asarray(w, dtype=float)

        valid = (
            np.isfinite(d)
            & np.isfinite(mu)
            & np.isfinite(w)
            & (d >= 0)
            & (mu > eps)
        )

        r = np.full_like(mu, np.nan, dtype=float)
        dr_dmu = np.full_like(mu, np.nan, dtype=float)

        if not np.any(valid):
            return r, dr_dmu, valid

        dv = d[valid]
        muv = np.clip(mu[valid], eps, None)

        term = muv - dv
        positive = dv > 0
        term[positive] += dv[positive] * np.log(dv[positive] / muv[positive])
        term = np.maximum(term, 0.0)

        rv = np.sign(muv - dv) * np.sqrt(2.0 * term)

        near = np.isclose(muv, dv, rtol=1e-8, atol=1e-10) | (
            np.abs(rv) < 1e-10
        )
        fv = np.empty_like(muv)
        fv[near] = 1.0 / np.sqrt(muv[near])
        far = ~near
        fv[far] = (1.0 - dv[far] / muv[far]) / rv[far]

        rv *= w[valid]
        fv *= w[valid]

        r[valid] = rv
        dr_dmu[valid] = fv

        return r, dr_dmu, valid

    def mode_model_counts(self, params, x0, x1, x2, d, n, w, c, inv_S, mode):
        A = params["A" + mode].value
        B = params["B" + mode].value
        g = self.gaussian(x0, x1, x2, c, inv_S, mode)
        mu = n * (A * g + B)
        return mu, g, A, B

    def residual_mode_poisson(
        self, params, x0, x1, x2, d, n, w, c, inv_S, mode
    ):
        w = self.coerce_weight(d, w)
        mu, g, A, B = self.mode_model_counts(
            params, x0, x1, x2, d, n, w, c, inv_S, mode
        )
        res, _, mask = self.poisson_deviance_residual_factor(d, mu, w)
        return res.flatten()[mask.flatten()]

    def jacobian_mode_poisson(
        self,
        params,
        x0,
        x1,
        x2,
        d,
        n,
        w,
        c,
        inv_S,
        dr,
        du,
        mode,
    ):
        params_list = [name for name, par in params.items()]
        n_params = len(params_list)

        w = self.coerce_weight(d, w)
        mu, g, A, B = self.mode_model_counts(
            params, x0, x1, x2, d, n, w, c, inv_S, mode
        )
        _, factor, mask = self.poisson_deviance_residual_factor(d, mu, w)

        n_pix = np.asarray(d).size
        jac = np.zeros((n_params, n_pix), dtype=float)

        dA = factor * n * g
        dB = factor * n

        yc_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode=mode)
        dc0, dc1, dc2 = factor * n * A * yc_gauss

        yr_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode=mode)
        dr0, dr1, dr2 = factor * n * A * yr_gauss

        yu_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode=mode)
        du0, du1, du2 = factor * n * A * yu_gauss

        names_values = {
            "A" + mode: dA,
            "B" + mode: dB,
            "c0": dc0,
            "c1": dc1,
            "c2": dc2,
            "r0": dr0,
            "r1": dr1,
            "r2": dr2,
            "u0": du0,
            "u1": du1,
            "u2": du2,
        }

        for name, value in names_values.items():
            if name in params_list:
                jac[params_list.index(name), :] = np.asarray(value).flatten()

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]
        return jac[ind][:, mask.flatten()]

    def residual_1d(
        self, params, x0, x1, x2, ds, ns, ws=None, c=None, inv_S=None
    ):
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)

        return np.concatenate(
            [
                self.residual_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, "1d_0"
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, "1d_1"
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, "1d_2"
                ),
            ]
        )

    def jacobian_1d(
        self,
        params,
        x0,
        x1,
        x2,
        ds,
        ns,
        ws=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)

        return np.column_stack(
            [
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, dr, du, "1d_0"
                ),
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, dr, du, "1d_1"
                ),
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, dr, du, "1d_2"
                ),
            ]
        )

    def residual_2d(
        self, params, x0, x1, x2, ds, ns, ws=None, c=None, inv_S=None
    ):
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)

        return np.concatenate(
            [
                self.residual_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, "2d_0"
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, "2d_1"
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, "2d_2"
                ),
            ]
        )

    def jacobian_2d(
        self,
        params,
        x0,
        x1,
        x2,
        ds,
        ns,
        ws=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)

        return np.column_stack(
            [
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, dr, du, "2d_0"
                ),
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, dr, du, "2d_1"
                ),
                self.jacobian_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, dr, du, "2d_2"
                ),
            ]
        )

    def residual_3d(
        self, params, x0, x1, x2, d, n, w=None, c=None, inv_S=None
    ):
        w = self.coerce_weight(d, w)
        return self.residual_mode_poisson(
            params, x0, x1, x2, d, n, w, c, inv_S, "3d"
        )

    def jacobian_3d(
        self,
        params,
        x0,
        x1,
        x2,
        d,
        n,
        w=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        w = self.coerce_weight(d, w)
        return self.jacobian_mode_poisson(
            params, x0, x1, x2, d, n, w, c, inv_S, dr, du, "3d"
        )

    def residual(self, params, args_1d, args_2d, args_3d):
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value

        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        cost_1d = self.residual_1d(params, *args_1d, c, inv_S)
        cost_2d = self.residual_2d(params, *args_2d, c, inv_S)
        cost_3d = self.residual_3d(params, *args_3d, c, inv_S)
        cost_prior = self.prior_residual(params)

        cost = np.concatenate([cost_1d, cost_2d, cost_3d, cost_prior])
        cost = np.nan_to_num(cost, nan=0.0, posinf=1e16, neginf=-1e16)

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value

        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        dr = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)
        S = np.linalg.inv(inv_S)

        jac_1d = self.jacobian_1d(params, *args_1d, c, inv_S, dr, du)
        jac_2d = self.jacobian_2d(params, *args_2d, c, inv_S, dr, du)
        jac_3d = self.jacobian_3d(params, *args_3d, c, inv_S, dr, du)
        jac_prior = self.prior_jacobian(params, S, dr, du)

        jac = np.column_stack([jac_1d, jac_2d, jac_3d, jac_prior])
        jac = np.nan_to_num(jac, nan=0.0, posinf=1e16, neginf=-1e16)

        return jac.T

    def collect_mode_fit_metrics(self, x0, x1, x2, c, inv_S, mode_data):
        args = x0, x1, x2, c, inv_S

        metrics = {}

        for mode, (y, e) in mode_data.items():
            A = self.params["A" + mode].value
            B = self.params["B" + mode].value

            y_fit = A * self.gaussian(*args, mode) + B

            if mode == "3d":
                valid = np.isfinite(y) & (e >= 0)
            else:
                valid = np.isfinite(y) & (e > 0)

            y_fit[~valid] = np.nan

            fit_triplet = (y_fit, y, e)

            chi2 = self.chi_2_fit(x0, x1, x2, c, inv_S, *fit_triplet, mode)
            I0, s0 = self.estimate_intensity(
                x0, x1, x2, c, inv_S, *fit_triplet, mode
            )

            metrics[mode] = [I0, s0, chi2, fit_triplet]

        return metrics

    def extract_result(self, args_1d, args_2d, args_3d, xmod):
        x0, x1, x2, d1d, n1d, w1d = args_1d
        x0, x1, x2, d2d, n2d, w2d = args_2d
        x0, x1, x2, d3d, n3d, w3d = args_3d

        d1d_0, d1d_1, d1d_2 = d1d
        d2d_0, d2d_1, d2d_2 = d2d
        n1d_0, n1d_1, n1d_2 = n1d
        n2d_0, n2d_1, n2d_2 = n2d

        y1d_0, e1d_0 = self.counts_to_intensity_uncertainty(d1d_0, n1d_0)
        y1d_1, e1d_1 = self.counts_to_intensity_uncertainty(d1d_1, n1d_1)
        y1d_2, e1d_2 = self.counts_to_intensity_uncertainty(d1d_2, n1d_2)

        y2d_0, e2d_0 = self.counts_to_intensity_uncertainty(d2d_0, n2d_0)
        y2d_1, e2d_1 = self.counts_to_intensity_uncertainty(d2d_1, n2d_1)
        y2d_2, e2d_2 = self.counts_to_intensity_uncertainty(d2d_2, n2d_2)

        y3d, e3d = self.counts_to_intensity_uncertainty(d3d, n3d)

        self.redchi2 = []
        self.intensity = []
        self.sigma = []

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c0_err = self.params["c0"].stderr
        c1_err = self.params["c1"].stderr
        c2_err = self.params["c2"].stderr

        if c0_err is None:
            c0_err = c0
        if c1_err is None:
            c1_err = c1
        if c2_err is None:
            c2_err = c2

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c_err = c0_err, c1_err, c2_err

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        mode_1d = {
            "1d_0": (y1d_0, e1d_0),
            "1d_1": (y1d_1, e1d_1),
            "1d_2": (y1d_2, e1d_2),
        }
        metrics_1d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_1d
        )

        modes = ["1d_0", "1d_1", "1d_2"]

        y1 = [metrics_1d[mode][3] for mode in modes]
        self.redchi2.append([metrics_1d[mode][2] for mode in modes])
        self.intensity.append([metrics_1d[mode][0] for mode in modes])
        self.sigma.append([metrics_1d[mode][1] for mode in modes])

        mode_2d = {
            "2d_0": (y2d_0, e2d_0),
            "2d_1": (y2d_1, e2d_1),
            "2d_2": (y2d_2, e2d_2),
        }
        metrics_2d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_2d
        )

        modes = ["2d_0", "2d_1", "2d_2"]

        y2 = [metrics_2d[mode][3] for mode in modes]
        self.redchi2.append([metrics_2d[mode][2] for mode in modes])
        self.intensity.append([metrics_2d[mode][0] for mode in modes])
        self.sigma.append([metrics_2d[mode][1] for mode in modes])

        mode_3d = {"3d": (y3d, e3d)}
        metrics_3d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_3d
        )

        y3 = metrics_3d["3d"][3]
        self.redchi2.append(metrics_3d["3d"][2])
        self.intensity.append(metrics_3d["3d"][0])
        self.sigma.append(metrics_3d["3d"][1])

        self.fit_metrics = {
            **{
                mode: {
                    "I0": values[0],
                    "s0": values[1],
                    "chi2": values[2],
                }
                for mode, values in metrics_1d.items()
            },
            **{
                mode: {
                    "I0": values[0],
                    "s0": values[1],
                    "chi2": values[2],
                }
                for mode, values in metrics_2d.items()
            },
            "3d": {
                "I0": metrics_3d["3d"][0],
                "s0": metrics_3d["3d"][1],
                "chi2": metrics_3d["3d"][2],
            },
        }

        if not np.linalg.det(inv_S) > 0:
            print("Improper optimal covariance")
            return None

        S = np.linalg.inv(inv_S)

        V, W = np.linalg.eigh(S)

        c0, c1, c2 = c

        c0 += xmod
        c = c0, c1, c2

        self.estimated_fit[0][0] += xmod

        r0, r1, r2 = np.sqrt(V)

        v0, v1, v2 = W.T

        fitting = (x0 + xmod, x1, x2, *y3)

        self.peak_pos = c, c_err

        self.best_fit = c, S, *fitting

        self.best_prof = (
            (x0[:, 0, 0] + xmod, *y1[0]),
            (x1[0, :, 0], *y1[1]),
            (x2[0, 0, :], *y1[2]),
        )

        self.best_proj = (
            (x1[0, :, :], x2[0, :, :], *y2[0]),
            (x0[:, 0, :] + xmod, x2[:, 0, :], *y2[1]),
            (x0[:, :, 0] + xmod, x1[:, :, 0], *y2[2]),
        )

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def extract_amplitude_background(self):
        A0 = self.params["A1d_0"]
        A1 = self.params["A1d_1"]
        A2 = self.params["A1d_2"]

        B0 = self.params["B1d_0"]
        B1 = self.params["B1d_1"]
        B2 = self.params["B1d_2"]

        return np.array([A0, A1, A2]), np.array([B0, B1, B2])

    def estimate_center_weighted(
        self, d, n, kernel, frac=0.9, min_weight=1e-12
    ):
        d = np.asarray(d, dtype=float)
        n = np.asarray(n, dtype=float)
        k = np.asarray(kernel, dtype=float)

        valid = np.isfinite(d) & np.isfinite(n) & (n > 0)

        d0 = np.where(valid, d, 0.0)
        n0 = np.where(valid, n, 0.0)

        ybar = np.sum(d0) / np.sum(n0)

        r = d0 - ybar * n0

        num = scipy.signal.correlate(r, k, mode="same", method="direct")
        den = scipy.signal.correlate(n0, k**2, mode="same", method="direct")

        c = np.divide(
            num,
            np.sqrt(den),
            out=np.full_like(num, np.nan),
            where=den > min_weight,
        )

        i0 = len(d) // 2

        i = np.nanargmax(c)
        if i < len(d) // 8 or i > 7 * len(d) // 8:
            return i0

        return i

    def quick_gaussian(self, x0, x1, x2, d, n, mode="3d"):
        mask = n > 0

        if mask.sum() <= 3:
            return None

        w = np.ones_like(n)

        d_int, n_int, *_ = self.profile_project(x0, x1, x2, d, n, w, mode=mode)

        y = d / n
        y[~mask] = np.nan

        B = np.nanpercentile(y, 5)
        A = np.nanpercentile(y, 95) - B

        if A <= 0 or not np.isfinite(A):
            A = 1
        if B <= 0 or not np.isfinite(B):
            B = 1

        self.params.add("A" + mode, value=A, min=0, max=np.inf)
        self.params.add("B" + mode, value=B, min=0, max=np.inf)

        return A > B

    def estimate_envelope(
        self,
        x0,
        x1,
        x2,
        d_int,
        n_int,
        wgt,
        report_fit=False,
    ):
        d = d_int.copy()
        n = n_int.copy()

        if self.combine_params is not None:
            self.params = self.combine_params.copy()

        if (np.array(d.shape) < 3).any():
            return None

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        d1d_0, n1d_0, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="1d_0"
        )
        d1d_1, n1d_1, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="1d_1"
        )
        d1d_2, n1d_2, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="1d_2"
        )

        est1d_0 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_0")
        est1d_1 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_1")
        est1d_2 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_2")

        if est1d_0 is None or est1d_1 is None or est1d_2 is None:
            return None

        d1d = [d1d_0, d1d_1, d1d_2]
        n1d = [n1d_0, n1d_1, n1d_2]
        w1d = self.uniform_mode_weights(d1d, n1d, self.mode_weights_1d)

        args_1d = [x0, x1, x2, d1d, n1d, w1d]

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        d2d_0, n2d_0, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="2d_0"
        )
        d2d_1, n2d_1, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="2d_1"
        )
        d2d_2, n2d_2, _, _ = self.profile_project(
            x0, x1, x2, d, n, wgt, mode="2d_2"
        )

        est2d_0 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_0")
        est2d_1 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_1")
        est2d_2 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_2")

        if est2d_0 is None or est2d_1 is None or est2d_2 is None:
            return None

        d2d = [d2d_0, d2d_1, d2d_2]
        n2d = [n2d_0, n2d_1, n2d_2]
        w2d = self.uniform_mode_weights(d2d, n2d, self.mode_weights_2d)

        args_2d = [x0, x1, x2, d2d, n2d, w2d]

        d3d, n3d, _, _ = self.profile_project(x0, x1, x2, d, n, wgt, mode="3d")
        w3d = self.uniform_mode_weights([d3d], [n3d], self.mode_weights_3d)[0]

        est3d = self.quick_gaussian(x0, x1, x2, d, n, mode="3d")

        if est3d is None:
            return None

        args_3d = [x0, x1, x2, d3d, n3d, w3d]

        protocol = [False] * 9
        n_iter = 30

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        self.update_adaptive_prior(args_1d, args_2d, args_3d)

        protocol = [True] * 3 + [False] * 3 + [False] * 3
        n_iter = 30

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        protocol = [False] * 3 + [True] * 3 + [False] * 3
        n_iter = 30

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        amplitude, background = self.estimate_peak_strength()

        if np.all(amplitude > background):
            protocol = [False] * 3 + [True] * 3 + [True] * 3
            n_iter = 30

            self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        return args_1d, args_2d, args_3d

    def estimate_peak_strength(self):
        amplitude, background = self.extract_amplitude_background()

        amplitude = np.array([param for param in amplitude], dtype=float)
        background = np.array([param for param in background], dtype=float)

        return amplitude, background

    def update_adaptive_prior(self, args_1d, args_2d, args_3d):
        x0, x1, x2, d1d, n1d, w1d = args_1d
        x0, x1, x2, d2d, n2d, w2d = args_2d
        x0, x1, x2, d3d, n3d, w3d = args_3d

        d1d_0, d1d_1, d1d_2 = d1d
        d2d_0, d2d_1, d2d_2 = d2d
        n1d_0, n1d_1, n1d_2 = n1d
        n2d_0, n2d_1, n2d_2 = n2d

        y1d_0, e1d_0 = self.counts_to_intensity_uncertainty(d1d_0, n1d_0)
        y1d_1, e1d_1 = self.counts_to_intensity_uncertainty(d1d_1, n1d_1)
        y1d_2, e1d_2 = self.counts_to_intensity_uncertainty(d1d_2, n1d_2)

        y2d_0, e2d_0 = self.counts_to_intensity_uncertainty(d2d_0, n2d_0)
        y2d_1, e2d_1 = self.counts_to_intensity_uncertainty(d2d_1, n2d_1)
        y2d_2, e2d_2 = self.counts_to_intensity_uncertainty(d2d_2, n2d_2)

        y3d, e3d = self.counts_to_intensity_uncertainty(d3d, n3d)

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        r0 = self.params["r0"].value
        r1 = self.params["r1"].value
        r2 = self.params["r2"].value

        u0 = self.params["u0"].value
        u1 = self.params["u1"].value
        u2 = self.params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        mode_1d = {
            "1d_0": (y1d_0, e1d_0),
            "1d_1": (y1d_1, e1d_1),
            "1d_2": (y1d_2, e1d_2),
        }
        metrics_1d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_1d
        )

        mode_2d = {
            "2d_0": (y2d_0, e2d_0),
            "2d_1": (y2d_1, e2d_1),
            "2d_2": (y2d_2, e2d_2),
        }
        metrics_2d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_2d
        )

        mode_3d = {"3d": (y3d, e3d)}
        metrics_3d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_3d
        )

        I_1d_0, sig_1d_0, *_ = metrics_1d["1d_0"]
        I_1d_1, sig_1d_1, *_ = metrics_1d["1d_1"]
        I_1d_2, sig_1d_2, *_ = metrics_1d["1d_2"]

        I_2d_0, sig_2d_0, *_ = metrics_2d["2d_0"]
        I_2d_1, sig_2d_1, *_ = metrics_2d["2d_1"]
        I_2d_2, sig_2d_2, *_ = metrics_2d["2d_2"]

        I_3d, sig_3d, *_ = metrics_3d["3d"]

        weights = (
            [self.mode_weights_1d] * 3
            + [self.mode_weights_2d] * 3
            + [self.mode_weights_3d]
        )

        signal_to_noise = [
            I_1d_0 / sig_1d_0,
            I_1d_1 / sig_1d_1,
            I_1d_2 / sig_1d_2,
            I_2d_0 / sig_2d_0,
            I_2d_1 / sig_2d_1,
            I_2d_2 / sig_2d_2,
            I_3d / sig_3d,
        ]

        sn = self.safe_sn(signal_to_noise)

        w = np.asarray(weights, dtype=float)

        sn_eff = np.exp(np.sum(w * np.log(sn)) / np.sum(w))

        scale = self.prior_strength_from_sn(sn_eff)

        self.lamda_cov = scale
        self.lamda_center = scale

    def safe_sn(self, sn, floor=1.0, ceiling=1e4):
        sn = np.nan_to_num(sn, nan=floor, posinf=ceiling, neginf=floor)
        return np.clip(sn, floor, ceiling)

    def prior_strength_from_sn(
        self,
        sn_eff,
        sn0=10.0,
        power=2.0,
        lam_min=1.0,
        lam_max=10.0,
    ):
        sn_eff = self.safe_sn(sn_eff)

        return lam_min + (lam_max - lam_min) / (1.0 + (sn_eff / sn0) ** power)

    def _isig_3d(self, params, args_3d):
        x0, x1, x2, d3d, n3d, _ = args_3d
        y3d, e3d = self.counts_to_intensity_uncertainty(d3d, n3d)

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value
        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        dx = np.prod(self.voxels(x0, x1, x2))

        c0v, c1v, c2v = c
        dx_vec = [x0 - c0v, x1 - c1v, x2 - c2v]
        d2 = np.einsum("i...,ij,j...->...", dx_vec, inv_S, dx_vec)
        pk = (d2 <= 1) & np.isfinite(y3d) & (e3d > 0)

        b = params["B3d"].value
        b_err = params["B3d"].stderr or params["B3d"].value

        I = np.nansum(y3d[pk] - b) * dx
        sig = np.sqrt(np.nansum(e3d[pk] ** 2 + b_err**2)) * dx

        return I / sig if sig > 0 else -np.inf

    def sweep(
        self, args_1d, args_2d, args_3d, protocol, n_iter=50, report_fit=False
    ):
        self.params["c0"].set(vary=protocol[0])
        self.params["c1"].set(vary=protocol[1])
        self.params["c2"].set(vary=protocol[2])

        self.params["r0"].set(vary=protocol[3])
        self.params["r1"].set(vary=protocol[4])
        self.params["r2"].set(vary=protocol[5])

        self.params["u0"].set(vary=protocol[6])
        self.params["u1"].set(vary=protocol[7])
        self.params["u2"].set(vary=protocol[8])

        out = Minimizer(
            self.residual,
            self.params,
            fcn_args=(args_1d, args_2d, args_3d),
            nan_policy="omit",
        )

        ssr_before = np.nansum(
            self.residual(self.params, args_1d, args_2d, args_3d) ** 2
        )
        isig_before = self._isig_3d(self.params, args_3d)

        result = out.minimize(
            method="least_squares",
            jac=self.jacobian,
            max_nfev=n_iter,
        )

        if report_fit:
            print(fit_report(result))

        ssr_after = np.nansum(
            self.residual(result.params, args_1d, args_2d, args_3d) ** 2
        )
        isig_after = self._isig_3d(result.params, args_3d)

        if ssr_after < ssr_before and isig_after >= isig_before:
            self.params = result.params

    def calculate_intensity(self, A, H, r0, r1, r2, u0, u1, u2, mode="3d"):
        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)
        g = self.gaussian_integral(inv_S, mode)

        return A * g

    def voxels(self, x0, x1, x2):
        return (
            x0[1, 0, 0] - x0[0, 0, 0],
            x1[0, 1, 0] - x1[0, 0, 0],
            x2[0, 0, 1] - x2[0, 0, 0],
        )

    def voxel_volume(self, x0, x1, x2):
        return np.prod(self.voxels(x0, x1, x2))

    def fit(
        self,
        x0_prof,
        x1_proj,
        x2_proj,
        d,
        n,
        dx,
        xmod,
        voxel_weights,
    ):
        fit_start = time.perf_counter()

        def log_fit_time(outcome):
            elapsed = time.perf_counter() - fit_start
            print("fit [{}] {:.3f}s".format(outcome, elapsed))

        x0 = x0_prof - xmod
        x1 = x1_proj.copy()
        x2 = x2_proj.copy()

        d_val = d.copy()
        n_val = n.copy()

        y = d_val / n_val
        e = np.sqrt(np.clip(d_val, 0, None) + 1) / n_val

        if (np.array(y.shape) <= 3).any():
            log_fit_time("small-shape")
            return None

        y_max = np.nanmax(y)

        det_mask = n_val > 0

        if y_max <= 0 or np.sum(det_mask) < 11:
            log_fit_time("insufficient-data")
            return None

        coords = np.argwhere(det_mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0) + 1

        y = y[i0:j0, i1:j1, i2:j2].copy()
        e = e[i0:j0, i1:j1, i2:j2].copy()

        d_val = d_val[i0:j0, i1:j1, i2:j2].copy()
        n_val = n_val[i0:j0, i1:j1, i2:j2].copy()

        if (np.array(y.shape) <= 3).any():
            log_fit_time("small-cropped-shape")
            return None

        x0 = x0[i0:j0, i1:j1, i2:j2].copy()
        x1 = x1[i0:j0, i1:j1, i2:j2].copy()
        x2 = x2[i0:j0, i1:j1, i2:j2].copy()

        voxel_weights = voxel_weights[i0:j0, i1:j1, i2:j2].copy()

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        if not np.nansum(y) > 0:
            print("Invalid data")
            log_fit_time("invalid-data")
            return None

        weights = None
        try:
            weights = self.estimate_envelope(
                x0, x1, x2, d_val, n_val, voxel_weights
            )
        except Exception as e:
            print("Exception estimating envelope: {}".format(e))
            log_fit_time("estimate-error")
            return None

        if weights is None:
            print("Invalid weight estimate")
            log_fit_time("invalid-weights")
            return None

        self.copy_combine()

        log_fit_time("ok")
        return weights

    def peak_roi(self, x0, x1, x2, c, S, scale, p=0.997):
        c0, c1, c2 = c

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        x = np.array([x0 - c0, x1 - c1, x2 - c2])

        S_inv = np.linalg.inv(S)

        ellipsoid = np.einsum("ij,jklm,iklm->klm", S_inv, x, x)

        mask = ellipsoid <= 1

        structure = np.ones((3, 3, 3), dtype=bool)

        pk = scipy.ndimage.binary_dilation(mask, structure=structure)

        # bound = np.zeros_like(pk, dtype=bool)
        # bound[0, :, :] = True
        # bound[-1, :, :] = True
        # bound[:, 0, :] = True
        # bound[:, -1, :] = True
        # bound[:, :, 0] = True
        # bound[:, :, -1] = True

        # clip = pk & (~bound)
        # pk = clip.copy()

        mask = scipy.ndimage.binary_dilation(pk, structure=structure)

        n_pk = np.sum(pk)

        for i in range(3):
            bkg = mask & (~pk)

            n_bkg = np.sum(bkg)

            if n_bkg < 2 * n_pk:
                mask = scipy.ndimage.binary_dilation(mask, structure=structure)

        scale = scipy.stats.chi2.ppf(p, df=3)

        inv_var = scale * S_inv

        d2 = np.einsum("i...,ij,j...->...", x, inv_var, x)

        det = 1 / np.linalg.det(inv_var)

        y = np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** 3 * det)

        return pk, bkg, mask, y

    def extract_raw_intensity(self, counts, pk, bkg):
        d = counts.copy()
        d[np.isinf(d)] = np.nan

        d_pk = d[pk].copy()
        d_bkg = d[bkg].copy()

        pk_intens = np.nansum(d_pk)
        pk_err = np.sqrt(pk_intens)

        bkg_intens = np.nansum(d_bkg)
        bkg_err = np.sqrt(bkg_intens)

        vol_pk = pk.sum()
        vol_bkg = bkg.sum()

        ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

        intens = pk_intens - ratio * bkg_intens
        sig = np.sqrt(pk_err**2 + ratio**2 * bkg_err**2)

        if not sig > 0:
            sig = float("inf")

        return intens, sig

    def extract_intensity(self, d, n, pk, bkg, kernel):
        core = pk & (n > 0)
        shell = bkg & (n > 0)

        # d_min, d_max = 0, np.inf
        # if np.sum(shell) > 0:
        #     d_med = np.nanmedian(d[shell])
        #     d_mad = np.nanmedian(np.abs(d[shell] - d_med))
        #     k = 1 / scipy.stats.norm.ppf(0.75)
        #     d_min, d_max = d_med - k * d_mad, d_med + k * d_mad

        shell = bkg & (n >= 0)  # & (d >= d_min) & (d <= d_max)

        d_pk = d[core].copy()
        d_bkg = d[shell].copy()

        n_pk = n[pk].copy()
        n_bkg = n[bkg].copy()

        k_pk = kernel[pk]
        k_bkg = kernel[bkg]

        bkg_cnts = np.nansum(d_bkg)
        bkg_norm = np.nansum(n_bkg * k_bkg)

        if bkg_cnts == 0.0:
            bkg_cnts = float("nan")
        if bkg_norm == 0.0:
            bkg_norm = float("nan")

        b = bkg_cnts / bkg_norm
        b_err = np.sqrt(bkg_cnts) / bkg_norm

        if not np.isfinite(b):
            b = 0
        if not np.isfinite(b_err):
            b_err = 0

        pk_cnts = np.nansum(d_pk)
        pk_norm = np.nansum(n_pk * k_pk)

        if pk_cnts == 0.0:
            pk_cnts = float("nan")
        if pk_cnts == 0.0:
            pk_norm = float("nan")

        vol_pk = float(core.sum())
        vol_bkg = float(shell.sum())

        vol = k_pk.sum()

        ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

        pk_intens = pk_cnts
        bkg_intens = bkg_cnts

        pk_err = np.sqrt(pk_cnts)
        bkg_err = np.sqrt(bkg_cnts)

        raw_intens = pk_intens - ratio * bkg_intens
        raw_sig = np.sqrt(pk_err**2 + ratio**2 * bkg_err**2)

        intens = vol * raw_intens / pk_norm
        sig = vol * raw_sig / pk_norm

        if not sig > 0:
            sig = float("-inf")

        data_norm = pk_cnts, pk_norm, bkg_cnts, bkg_norm, ratio

        return intens, sig, b, b_err, vol, *data_norm

    def matched_filter(self, d, n, v, pk, bkg, kernel, rel_error=0.15):
        mask = (pk | bkg) & (n > 0)
        p = kernel.copy()
        p[~mask] = np.nan

        b = np.nanquantile(d[mask], rel_error)

        e = np.sqrt(v + b)
        e[np.isclose(d, 0)] = 1

        q = n * p
        w = 1 / e**2

        Sqq = np.nansum(w * q**2)
        Sqn = np.nansum(w * q * n)
        Snn = np.nansum(w * n**2)

        Tq = np.nansum(w * q * d)
        Tn = np.nansum(w * n * d)

        den = Sqq * Snn - Sqn**2

        I = (Snn * Tq - Sqn * Tn) / den
        b = (Sqq * Tn - Sqn * Tq) / den
        sig = np.sqrt(Snn / den)

        A = I * np.nanmax(p)

        return I, sig, A, b

    def fitted_profile(self, x0, x1, x2, d, n, c, S, p=0.997, eta=0.5):
        scale = np.sqrt(scipy.stats.chi2.ppf(p, df=1))

        c0, c1, c2 = c

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        C = S.copy()

        sigma = np.sqrt(C[0, 0]) / scale

        weights = np.ones_like(n)

        d_int, n_int, v_int, _ = self.profile_project(
            x0, x1, x2, d, n, weights, mode="1d_0"
        )

        y = d_int / n_int
        e = np.sqrt(np.clip(d_int, 0, None) + 1) / n_int

        x = x0[:, 0, 0] - c0

        for i in range(3):
            norm = np.sqrt(2 * np.pi) * sigma

            kernel = np.exp(-0.5 * (x / sigma) ** 2) / norm

            pk = np.abs(x) < scale * sigma
            bkg = np.abs(x) >= scale * sigma

            I, I_err, A, b = self.matched_filter(
                d_int, n_int, v_int, pk, bkg, kernel
            )

            w = np.clip(y - b, 0, None)

            wgt = np.nansum(w)

            if wgt > 0:
                sigma_hat = np.clip(np.nansum(w * x**2) / wgt, dx0, sigma)
                sigma = (1 - eta) * sigma + eta * sigma_hat

        y_fit = I * kernel + b

        return I, I_err, b, x, y_fit, y, e

    def integrate(self, x0, x1, x2, d, n, c, S):
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d3x = self.voxel_volume(x0, x1, x2)

        d[np.isinf(d)] = np.nan
        n[np.isinf(n)] = np.nan

        pk, bkg, mask, kernel = self.peak_roi(x0, x1, x2, c, S, 1)

        result = self.extract_intensity(d, n, pk, bkg, kernel)

        intens, sig, b, b_err, N, *data_norm = result

        pk_data, pk_norm, bkg_data, bkg_norm, ratio = data_norm

        intens *= d3x
        sig *= d3x

        self.intensity.append(intens)
        self.sigma.append(sig)

        self.weights = (x0[pk], x1[pk], x2[pk]), d[pk].copy()

        self.info = [d3x, b, b_err]

        y = d / n
        e = np.sqrt(np.clip(d, 0, None) + 1) / n

        intens_raw, sig_raw = self.extract_raw_intensity(d, pk, bkg)

        self.info += [intens_raw, sig_raw]

        self.info += [N, pk_data, pk_norm, bkg_data, bkg_norm, ratio]

        if not np.isfinite(sig):
            sig = float("inf")

        xye = (x0, x1, x2), (dx0, dx1, dx2), y, e

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        self.peak_background_mask = x0, x1, x2, pk, bkg

        result = self.fitted_profile(x0, x1, x2, d, n, c, S)

        I, I_err, b, x, y_fit, y, e = result

        self.integral = x, y_fit, y, e

        result = self.matched_filter(d, n, d, pk, bkg, kernel)

        I_filt, sig_filt, A_filt, b_filt = result

        chi2_3d = self.redchi2[-1]

        if np.isfinite(chi2_3d) and chi2_3d > 1:
            sig_filt = sig_filt * np.sqrt(chi2_3d)
            result = I_filt, sig_filt, A_filt, b_filt

        self.filter = result

        self.intensity.append(I)
        self.sigma.append(I_err)

        return intens, sig if I_filt > sig_filt else intens
