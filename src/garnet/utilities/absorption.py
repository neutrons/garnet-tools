import numpy as np

from scipy.spatial.transform import Rotation


class AbsorptionEllipsoid:
    """
    Analytical absorption correction for an ellipsoidal sample shape.

    Evaluates the absorption factor and absorption-weighted path
    length (Tbar) for every peak at once via a fixed, pre-sampled
    Monte Carlo point cloud inside a unit ellipsoid, transformed
    analytically per orientation .
    """

    def __init__(self, N=100, seed=42, beta=3):
        self.prepare_absorption_table(N=N, seed=seed, beta=beta)

    def prepare_absorption_table(self, N=100, seed=42, beta=3):
        rng = np.random.default_rng(seed)

        v = rng.normal(size=(N, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        u = rng.random(N).astype(np.float32)
        r = u ** (1.0 / beta)
        p = v * r[:, None]

        w = (3.0 / beta) * (r ** (3.0 - beta))

        self.sample_points = p.astype(np.float32)
        self.sample_weights = w.astype(np.float32)
        self.N_mc = N

    def ellipsoid_parameters(self, coeffs):
        alpha, beta, gamma, vol, ratio_b, ratio_c = coeffs

        a = np.cbrt(6 * vol / (np.pi * ratio_b * ratio_c))

        thickness = a
        width = a * ratio_b
        height = a * ratio_c

        return alpha, beta, gamma, thickness, width, height

    def calculate_ellipsoid_surface(self, coeffs):
        params = self.ellipsoid_parameters(coeffs)
        alpha, beta, gamma, thickness, width, height = params

        R = Rotation.from_euler(
            "ZYX", [gamma, beta, alpha], degrees=True
        ).as_matrix()

        D = np.diag([1 / width**2, 1 / height**2, 1 / thickness**2]) * 4

        return D, R, R @ D @ R.T

    def _exit_lengths_for_directions(self, Q, yQ, cquad, dirs):
        a = np.einsum("mi,ij,mj->m", dirs, Q, dirs)

        b = 2 * (yQ @ dirs.T)

        disc = b * b - 4 * (cquad[:, None] * a[None, :])
        disc = np.maximum(disc, 0.0)

        eps = 1e-12
        a_safe = np.where(np.abs(a) < eps, np.sign(a) * eps + eps, a)

        t = (-b + np.sqrt(disc)) / (2 * a_safe[None, :])
        t = np.maximum(t, 0.0)
        return t

    def _absorption_factors(self, mu, Q, yQ, cquad, n_in_rev, n_out_rev):
        P = n_in_rev.shape[0]

        dirs = np.vstack([n_in_rev, n_out_rev])
        t = self._exit_lengths_for_directions(Q, yQ, cquad, dirs)

        t_total = t[:, :P] + t[:, P:]

        w = np.exp(-t_total * mu[None, :])
        ws = self.sample_weights[:, None]

        ws_sum = np.sum(ws, axis=0)
        A = np.sum(ws * w, axis=0) / ws_sum

        denom = np.sum(ws * w, axis=0)
        Tbar = np.sum(ws * w * t_total, axis=0) / np.clip(denom, 1e-30, None)

        return A, Tbar

    def correction(self, coeffs, mu, ri_hat, sf_hat):
        """
        Absorption factor and Tbar for every peak, given the ellipsoid..

        Returns
        -------
        T : ndarray
            Absorption factor per peak, clipped away from zero.
        Tbar : ndarray
            Absorption-weighted path length per peak.
        """
        D, R, Q = self.calculate_ellipsoid_surface(coeffs)

        S = (R @ np.diag(1 / np.sqrt(np.diag(D)))).astype(np.float32)
        y = self.sample_points @ S.T
        yQ = y @ Q
        cquad = np.einsum("ij,ij->i", yQ, y) - 1.0

        A, Tbar = self._absorption_factors(
            np.asarray(mu, dtype=np.float32),
            Q,
            yQ,
            cquad,
            np.asarray(ri_hat),
            np.asarray(sf_hat),
        )

        T = np.clip(A, 1e-8, None)
        return T, Tbar
