import time
import numpy as np

import scipy.optimize
import scipy.spatial.transform
import scipy.stats
import scipy.signal
import scipy.ndimage
import scipy.special

from lmfit import Minimizer, Parameters, fit_report

_R_SCALE_3D = np.sqrt(scipy.stats.chi2.ppf(0.997, df=3))

# Orthonormal basis spanning the traceless subspace of log-radius space
# (each column sums to zero and is orthogonal to [1, 1, 1] and to the
# other column), used to separate an overall log-size change from the
# two independent log-aspect-ratio ("shape") changes.
SHAPE_BASIS = np.array(
    [
        [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(6.0)],
        [0.0, -2.0 / np.sqrt(6.0)],
    ]
)

# Parameters that determine the modeled Gaussian's center/shape/orientation
# (and therefore its support), as opposed to the per-mode amplitude/background
# parameters A<mode>/B<mode>, which the missing-support penalty is defined to
# be independent of.
_MISSING_SUPPORT_GEOMETRY_PARAMS = (
    "c0",
    "c1",
    "c2",
    "log_size",
    "shape_1",
    "shape_2",
    "domega_x",
    "domega_y",
    "domega_z",
)


class PeakEllipsoid:
    def __init__(self):
        """
        Initialize an empty fitter with default parameters, weights, and prior state.

        The prior state is later overwritten by `update_estimate`.
        """
        self.params = Parameters()

        self.mode_weights_1d = 1.0
        self.mode_weights_2d = 1.0
        self.mode_weights_3d = 1.0

        self.prior_center_sigma = 1.0
        self.prior_cov = np.eye(3)
        self.prior_cov_sigma = 0.3
        self.prior_distortion_sigma = 0.1
        self.prior_rot_sigma = 0.1

        # Missing-support penalty (see missing_support_penalty): tolerated
        # fraction of modeled peak support in zero-normalization voxels,
        # softplus transition width, and penalty strength.
        self.missing_tolerance = 0.03
        self.missing_smoothing = 0.01
        self.missing_sigma = 0.05

        self._prior_radii = np.ones(3)
        self._prior_inv_sqrt = np.eye(3)

        self._prior_r = np.ones(3)
        self._prior_U = np.eye(3)

    def vech6(self, M):
        """
        Half-vectorize a symmetric 3x3 matrix into its 6 unique entries.

        Parameters
        ----------
        M : ndarray of shape (3, 3)
            Symmetric matrix to vectorize.

        Returns
        -------
        v : ndarray of shape (6,)
            The entries ``[M00, M11, M22, M12, M02, M01]``.
        """
        return np.array(
            [M[0, 0], M[1, 1], M[2, 2], M[1, 2], M[0, 2], M[0, 1]],
            dtype=float,
        )

    def update_constraints(self, x0, x1, x2, dx):
        """
        (Re)initialize `self.params` bounds and starting values from the data extent.

        Centers start at zero, bounded to half the coordinate extent per
        axis; size/shape start at zero; orientation offsets start at zero,
        bounded to [-pi, pi]. Also resets `self.combine_params` to None.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        dx : float
            Unused (voxel spacing is recomputed from `x0, x1, x2`).
        """
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

        self.params.add("c0", value=c0, min=c0_min, max=c0_max)
        self.params.add("c1", value=c1, min=c1_min, max=c1_max)
        self.params.add("c2", value=c2, min=c2_min, max=c2_max)

        self.params.add("log_size", value=0.0)
        self.params.add("shape_1", value=0.0)
        self.params.add("shape_2", value=0.0)

        self.params.add("domega_x", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("domega_y", value=0.0, min=-np.pi, max=np.pi)
        self.params.add("domega_z", value=0.0, min=-np.pi, max=np.pi)

        self.combine_params = None

    def copy_combine(self):
        """
        Snapshot `self.params` into `self.combine_params` for `estimate_envelope` to resume from.
        """
        self.combine_params = self.params.copy()

    def update_estimate(self, shape):
        """
        Seed `self.params` and the resolution-model prior from an initial ellipsoid estimate.

        Centers reset to the origin; `log_size`/`shape_*`/`domega_*` reset to
        zero so the fit starts out reproducing this reference ellipsoid.
        Also (re)initializes the SNR-adaptive prior state and prior widths,
        and sets `self.estimated_fit`.

        Parameters
        ----------
        shape : tuple
            ``(c0, c1, c2, r0, r1, r2, v0, v1, v2)``: (unused) center,
            principal radii, and orthonormal principal axis vectors
            (each shape (3,)).
        """
        c0, c1, c2, r0, r1, r2, v0, v1, v2 = shape

        U = np.column_stack([v0, v1, v2])
        if np.linalg.det(U) < 0:
            U[:, 2] *= -1

        self.params["c0"].set(value=0)
        self.params["c1"].set(value=0)
        self.params["c2"].set(value=0)

        self.params["log_size"].set(value=0.0)
        self.params["shape_1"].set(value=0.0)
        self.params["shape_2"].set(value=0.0)

        self.params["domega_x"].set(value=0.0)
        self.params["domega_y"].set(value=0.0)
        self.params["domega_z"].set(value=0.0)

        prior_cov = U @ np.diag([r0**2, r1**2, r2**2]) @ U.T

        self.estimated_fit = (np.array([0.0, 0.0, 0.0]), prior_cov)

        self.prior_cov = prior_cov

        r_sq, U0 = np.linalg.eigh(self.prior_cov)
        self._prior_radii = np.sqrt(np.maximum(r_sq, 1e-12))
        self._prior_inv_sqrt = U0 @ np.diag(1.0 / self._prior_radii) @ U0.T

        # Reference (predicted) radii/orientation that log_size, shape_1,
        # shape_2, and domega_x, domega_y, domega_z are defined relative to.
        self._prior_r = np.array([r0, r1, r2], dtype=float)
        self._prior_U = U

        self.prior_center_sigma = 1.0
        self.prior_cov_sigma = 0.3
        self.prior_distortion_sigma = 0.1
        self.prior_rot_sigma = 0.1

    def shape_from_params(self, params):
        """
        Reconstruct absolute radii and orientation rotation vector from the size/shape/orientation parameters.

        Converts `log_size, shape_1, shape_2` to radii via `SHAPE_BASIS`
        and `self._prior_r`, and `domega_*` to an absolute rotation vector
        via ``R = self._prior_U @ Exp(domega)``.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `log_size, shape_1, shape_2,
            domega_x, domega_y, domega_z`.

        Returns
        -------
        r0, r1, r2 : float
            Ellipsoid principal radii along the rotated axes.
        u0, u1, u2 : float
            Absolute orientation rotation vector components.
        """
        log_size = params["log_size"].value
        shape = np.array([params["shape_1"].value, params["shape_2"].value])
        domega = np.array(
            [
                params["domega_x"].value,
                params["domega_y"].value,
                params["domega_z"].value,
            ]
        )

        log_radius_change = log_size + SHAPE_BASIS @ shape
        r0, r1, r2 = self._prior_r * np.exp(log_radius_change)

        delta_R = scipy.spatial.transform.Rotation.from_rotvec(
            domega
        ).as_matrix()
        R = self._prior_U @ delta_R
        u0, u1, u2 = scipy.spatial.transform.Rotation.from_matrix(
            R
        ).as_rotvec()

        return r0, r1, r2, u0, u1, u2

    def prior_residual(self, params):
        """
        Whitened regularization residuals pulling the fit toward the resolution-model prior.

        Stacks center (Mahalanobis distance from the origin), size,
        shape, and orientation residuals, each scaled by its current
        SNR-adaptive prior sigma.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `c0, c1, c2, log_size, shape_1,
            shape_2, domega_x, domega_y, domega_z`.

        Returns
        -------
        terms : ndarray of shape (9,)
            Concatenated residuals: 3 center + 1 size + 2 shape + 3
            orientation.
        """
        terms = []

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value
        c = np.array([c0, c1, c2])

        # Center: P_center = || Sigma0^{-1/2} (c - c0) / sigma_c ||^2
        u_mu = _R_SCALE_3D * self._prior_inv_sqrt @ c
        terms += (u_mu / self.prior_center_sigma).tolist()

        # Size residual: P_size = (log_size / sigma_log_s)^2
        terms.append(params["log_size"].value / self.prior_cov_sigma)

        # Shape residuals: P_shape = sum (shape_i / sigma_log_d)^2
        terms.append(params["shape_1"].value / self.prior_distortion_sigma)
        terms.append(params["shape_2"].value / self.prior_distortion_sigma)

        # Orientation residual: P_rotation = || domega / sigma_rot ||^2
        terms.append(params["domega_x"].value / self.prior_rot_sigma)
        terms.append(params["domega_y"].value / self.prior_rot_sigma)
        terms.append(params["domega_z"].value / self.prior_rot_sigma)

        return np.asarray(terms, dtype=float)

    def prior_jacobian(self, params):
        """
        Analytic Jacobian of `prior_residual` with respect to the varying fit parameters.

        Column ordering matches `prior_residual` (3 center + 1 size + 2
        shape + 3 orientation); only the center block mixes across
        parameters, the rest are diagonal.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `c0, c1, c2, log_size, shape_1,
            shape_2, domega_x, domega_y, domega_z`.

        Returns
        -------
        jac : ndarray of shape (n_vary, 9)
            Jacobian rows restricted to `vary=True` parameters.
        """
        # Residuals: [r_c0, r_c1, r_c2, r_size, r_sh1, r_sh2, r_om_x, r_om_y, r_om_z]
        # 3 center + 1 size + 2 shape + 3 orientation = 9 columns
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), 9), dtype=float)

        # Center: ∂r_c_i/∂c_j = R * (S₀^{-1/2})_{i,j} / σ_c
        center_scale = _R_SCALE_3D / self.prior_center_sigma
        for j, cname in enumerate(["c0", "c1", "c2"]):
            idx = params_list.index(cname)
            jac[idx, :3] = center_scale * self._prior_inv_sqrt[:, j]

        jac[params_list.index("log_size"), 3] = 1.0 / self.prior_cov_sigma

        shape_scale = 1.0 / self.prior_distortion_sigma
        jac[params_list.index("shape_1"), 4] = shape_scale
        jac[params_list.index("shape_2"), 5] = shape_scale

        rot_scale = 1.0 / self.prior_rot_sigma
        jac[params_list.index("domega_x"), 6] = rot_scale
        jac[params_list.index("domega_y"), 7] = rot_scale
        jac[params_list.index("domega_z"), 8] = rot_scale

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]
        return jac[ind]

    def S_matrix(self, r0, r1, r2, u0, u1, u2):
        """
        Ellipsoid covariance-like matrix S = U diag(r0^2, r1^2, r2^2) U^T.

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.

        Returns
        -------
        S : ndarray of shape (3, 3)
            Ellipsoid covariance-like matrix.
        """
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):
        """
        Inverse ellipsoid covariance-like matrix inv_S = U diag(1/r0^2, 1/r1^2, 1/r2^2) U^T.

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.

        Returns
        -------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        """
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):
        """
        Ellipsoid orientation rotation matrix from an axis-angle rotation vector.

        Parameters
        ----------
        u0, u1, u2 : float
            Axis-angle rotation vector components.

        Returns
        -------
        U : ndarray of shape (3, 3)
            Proper orthogonal rotation matrix.
        """
        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def det_S(self, r0, r1, r2, u0, u1, u2):
        """
        Determinant of the ellipsoid covariance-like matrix S, i.e. (r0*r1*r2)**2.

        Rotation-invariant, so computed directly from the radii.

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Unused (det(S) does not depend on orientation).

        Returns
        -------
        det : float
            Determinant of S.
        """
        return (r0 * r1 * r2) ** 2

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):
        """
        Package the peak center and inverse ellipsoid covariance from scalar parameters.

        Parameters
        ----------
        c0, c1, c2 : float
            Peak center coordinates.
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.

        Returns
        -------
        c : ndarray of shape (3,)
            Peak center [c0, c1, c2].
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        """
        c = np.array([c0, c1, c2])

        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)

        return c, inv_S

    def data_norm(self, d, n, v, rel_err=30):
        """
        Normalize raw counts by monitor counts and estimate their uncertainty.

        Voxels with non-positive or non-finite normalization are masked
        to NaN in-place in `d`, `n`, and `v` before normalizing.

        Parameters
        ----------
        d : ndarray
            Raw observed event counts. Modified in place.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
            Modified in place.
        v : ndarray
            Variance proxy (e.g. clipped counts), same shape as `d`.
            Modified in place.
        rel_err : float, optional
            Percentile of `v` used as a variance floor. Default 30.

        Returns
        -------
        y_int : ndarray
            Normalized intensity d / n.
        e_int : ndarray
            Estimated uncertainty sqrt(v + percentile(v, rel_err)) / n.
        """
        mask = (n > 0) & np.isfinite(n)

        d[~mask] = np.nan
        n[~mask] = np.nan
        v[~mask] = np.nan

        y_int = d / n
        e_int = np.sqrt(v + np.nanpercentile(v, rel_err)) / n

        return y_int, e_int

    def profile_project(self, x0, x1, x2, d, n, w, mode="3d"):
        """
        Project raw counts, normalization, variance proxy, and weight onto a given mode.

        `"1d_i"` sums/averages over the two axes other than `i`; `"2d_i"`
        sums/averages over axis `i` only; `"3d"` returns full-array copies.
        Normalization is rate-averaged (rescaled by the integrated-out
        voxel size) rather than summed. Invalid input voxels are excluded
        before projecting; projected voxels with invalid resulting
        normalization are masked to NaN in all outputs.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : ndarray
            Per-voxel fit weight, same shape as `d`.
        mode : str, optional
            One of "1d_0", "1d_1", "1d_2", "2d_0", "2d_1", "2d_2", "3d".
            Default "3d".

        Returns
        -------
        d_int : ndarray
            Projected (summed) raw counts.
        n_int : ndarray
            Projected (rate-averaged) normalization counts.
        v_int : ndarray
            Projected (summed) variance proxy from clip(d, 0, None).
        w_int : ndarray
            Projected (averaged) fit weight.

        Raises
        ------
        ValueError
            If `mode` is not a recognized mode string.
        """
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
        """
        Project raw data onto a mode and normalize it into a weighted intensity/uncertainty.

        Combines `profile_project` and `data_norm`, dividing the
        resulting uncertainty by the projected fit weight.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : ndarray
            Per-voxel fit weight, same shape as `d`.
        mode : str, optional
            Projection mode passed to `profile_project`. Default "3d".

        Returns
        -------
        y_int : ndarray
            Normalized intensity for the given mode.
        e_int : ndarray
            Weight-scaled uncertainty for the given mode.
        """
        d_int, n_int, v_int, w_int = self.profile_project(
            x0, x1, x2, d, n, w, mode=mode
        )

        y_int, e_int = self.data_norm(d_int, n_int, v_int)

        return y_int, e_int / w_int

    def ellipsoid_covariance(self, inv_S, mode="3d", perc=99.7):
        """
        Rescale (a projection of) inv_S to an arbitrary confidence-contour percentile.

        `inv_S` is on the 99.7%-contour convention; this converts it (or
        the marginal/slice selected by `mode`) to the inverse-variance
        for percentile `perc` of the corresponding chi-squared
        distribution.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".
        perc : float, optional
            Target confidence-contour percentile (0-100). Default 99.7.

        Returns
        -------
        inv_var : ndarray or float
            Rescaled inverse-variance: (3, 3) or (2, 2) matrix for "3d"/
            "2d_*" modes, scalar for "1d_*" modes.
        """
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
        """
        Normalize a weight argument into an array matching the target's shape.

        Parameters
        ----------
        target : array_like
            Array whose shape the returned weight array should match.
        weight : None, scalar, or array_like, optional
            None gives all-ones; a scalar is broadcast; array-like is
            coerced to a float ndarray as-is.

        Returns
        -------
        weight : ndarray
            Float array of weights matching `target`'s shape.
        """
        target = np.asarray(target)

        if weight is None:
            return np.ones_like(target, dtype=float)
        if np.isscalar(weight):
            return np.full_like(target, weight, dtype=float)

        return np.asarray(weight, dtype=float)

    def coerce_weights(self, targets, weights=None):
        """
        Apply `coerce_weight` element-wise across parallel lists of targets and weights.

        Parameters
        ----------
        targets : sequence of array_like
            Arrays whose shapes the returned weights should match.
        weights : None or sequence, optional
            If None, each target gets an all-ones weight; otherwise a
            same-length sequence, each element passed to `coerce_weight`.

        Returns
        -------
        weights : list of ndarray
            One coerced weight array per target.
        """
        if weights is None:
            weights = [None] * len(targets)

        return [
            self.coerce_weight(target, weight)
            for target, weight in zip(targets, weights)
        ]

    def uniform_mode_weights(self, ys, es, val=1.0):
        """
        Build uniform per-mode weight arrays scaled by the total valid point count.

        Each array is filled with sqrt(val) / sqrt(n_valid), where
        n_valid is the total finite, positive-uncertainty point count
        across all ys/es pairs, giving each mode a total weight
        contribution proportional to `val` regardless of point count.

        Parameters
        ----------
        ys : sequence of ndarray
            Per-mode intensity arrays (used only via `es`, for counting).
        es : sequence of ndarray
            Per-mode uncertainty arrays, one per element of `ys`.
        val : float, optional
            Target weight scale for this group of modes. Default 1.0.

        Returns
        -------
        weights : list of ndarray
            One uniform weight array per element of `es`.
        """
        n_valid = sum(
            np.count_nonzero(np.isfinite(y) & np.isfinite(e) & (e > 0))
            for y, e in zip(ys, es)
        )

        scale = np.sqrt(val) / np.sqrt(max(n_valid, 1))

        return [np.full_like(e, scale, dtype=float) for e in es]

    def poisson_deviance_fit(
        self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d"
    ):
        """
        Reduced Poisson deviance goodness-of-fit statistic within 1-sigma of the peak center.

        Restricts to voxels with Mahalanobis distance d2 <= 2**(2/k),
        approximates raw counts/normalization from (y, e) via the Poisson
        error model, and returns the summed deviance
        2*(mu - d + d*log(d/mu)) divided by degrees of freedom.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        y_fit : ndarray
            Fitted (model) normalized intensity, same shape as `y`.
        y : ndarray
            Observed normalized intensity for this mode.
        e : ndarray
            Uncertainty on `y`, same shape as `y`.
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        reduced_deviance : float
            Summed Poisson deviance over degrees of freedom, or inf if
            degrees of freedom are not positive.
        """
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

        mask = (d2 <= 2 ** (2 / k)) & np.isfinite(e) & (e > 0) & np.isfinite(y)

        n_pts = np.sum(mask)

        dof = n_pts - m

        if dof <= 0:
            return np.inf

        ym = y[mask]
        em = e[mask]
        yfm = y_fit[mask]

        eps = 1e-10

        # Approximate normalization from Poisson error model:
        # e = sqrt(d+1)/n  →  n ≈ y/e^2 when d>0,  n ≈ 1/e when d=0
        n_approx = np.where(
            ym > 0,
            ym / np.clip(em**2, eps, None),
            1.0 / np.clip(em, eps, None),
        )

        # Approximate raw counts and model counts
        d_approx = np.clip(n_approx * ym, 0.0, None)
        mu = np.clip(n_approx * yfm, eps, None)

        # Poisson deviance: 2*(mu - d + d*log(d/mu)); reduces to 2*mu when d=0
        dev = 2.0 * np.where(
            d_approx > 0,
            mu - d_approx + d_approx * np.log(d_approx / mu),
            mu,
        )

        return float(np.nansum(dev)) / dof

    def estimate_intensity(
        self, x0, x1, x2, c, inv_S, y_fit, y, e, mode="3d", bkg_offset=None
    ):
        """
        Background-subtracted box-sum intensity and uncertainty within the 1-sigma peak region (I_1d/I_2d/I_3d).

        Sums the background-subtracted normalized intensity over voxels
        with Mahalanobis distance <= 1, scaled by voxel volume. Distinct
        from the `I_ell` and `I_prof` estimators.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        y_fit : ndarray
            Unused (kept to match the `poisson_deviance_fit` signature).
        y : ndarray
            Observed normalized intensity for this mode.
        e : ndarray
            Uncertainty on `y`, same shape as `y`.
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".
        bkg_offset : ndarray, optional
            Additional per-voxel background-rate offset to subtract
            alongside the fitted `B<mode>` level.

        Returns
        -------
        I : float
            Background-subtracted, volume-scaled summed intensity.
        sig : float
            Propagated uncertainty on `I`.
        """
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

        if bkg_offset is not None:
            I = np.nansum(y[pk] - b - bkg_offset[pk]) * dx
        else:
            I = np.nansum(y[pk] - b) * dx
        sig = np.sqrt(np.nansum(e[pk] ** 2 + b_err**2)) * dx

        return I, sig

    def gaussian(self, x0, x1, x2, c, inv_S, mode="3d"):
        """
        Unnormalized (peak-value-1) Gaussian profile for a given mode.

        `inv_S` is rescaled from the 99.7%-contour convention to
        1-sigma, so the result is 1 at the center and exp(-0.5) at
        Mahalanobis distance 1.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        g : ndarray
            Unnormalized Gaussian profile on the mode's coordinate grid.
        """
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
        """
        Analytic derivatives of `inv_S` with respect to each radius r0, r1, r2.

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.

        Returns
        -------
        dinv_S0, dinv_S1, dinv_S2 : ndarray of shape (3, 3)
            d(inv_S)/d(r0), d(r1), d(r2).
        """
        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2 / r0**3, 0, 0]) @ U.T
        dinv_S1 = U @ np.diag([0, -2 / r1**3, 0]) @ U.T
        dinv_S2 = U @ np.diag([0, 0, -2 / r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_size_shape(self, r0, r1, r2, u0, u1, u2):
        """
        Analytic derivatives of `inv_S` with respect to log_size, shape_1, shape_2.

        Chain-rules `inv_S_deriv_r` through
        d(r_i)/d(log_size) = r_i and d(r_i)/d(shape_k) = r_i * SHAPE_BASIS[i, k].

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.

        Returns
        -------
        dinv_S_size, dinv_S_shape1, dinv_S_shape2 : ndarray of shape (3, 3)
            d(inv_S)/d(log_size), d(shape_1), d(shape_2).
        """
        d_inv_S_r = self.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
        r = np.array([r0, r1, r2])

        dinv_S_size = sum(r[i] * d_inv_S_r[i] for i in range(3))
        dinv_S_shape1 = sum(
            r[i] * SHAPE_BASIS[i, 0] * d_inv_S_r[i] for i in range(3)
        )
        dinv_S_shape2 = sum(
            r[i] * SHAPE_BASIS[i, 1] * d_inv_S_r[i] for i in range(3)
        )

        return dinv_S_size, dinv_S_shape1, dinv_S_shape2

    def inv_S_deriv_domega(self, r0, r1, r2, domega, delta=1e-6):
        """
        Finite-difference derivatives of `inv_S` with respect to domega_x, domega_y, domega_z.

        Central-differences inv_S = R @ diag(1/r0^2, 1/r1^2, 1/r2^2) @ R.T
        with R(domega) = self._prior_U @ Exp(domega).

        Parameters
        ----------
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        domega : sequence of float
            Relative-rotation vector (domega_x, domega_y, domega_z).
        delta : float, optional
            Finite-difference step size. Default 1e-6.

        Returns
        -------
        dinv_S_x, dinv_S_y, dinv_S_z : ndarray of shape (3, 3)
            d(inv_S)/d(domega_x), d(domega_y), d(domega_z).
        """
        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])
        domega = np.asarray(domega, dtype=float)

        def inv_S_at(dw):
            R = (
                self._prior_U
                @ scipy.spatial.transform.Rotation.from_rotvec(dw).as_matrix()
            )
            return R @ V @ R.T

        derivs = []
        for k in range(3):
            step = np.zeros(3)
            step[k] = delta
            derivs.append(
                (inv_S_at(domega + step) - inv_S_at(domega - step))
                / (2 * delta)
            )

        return tuple(derivs)

    def gaussian_integral(self, inv_S, mode="3d"):
        """
        Normalization integral sqrt((2*pi)**k * det) of the unit-peak Gaussian for a given mode.

        `k` is the mode's dimensionality and `det` the determinant of the
        1-sigma covariance from `inv_S`. Multiplying by amplitude `A`
        gives the total fitted intensity for that mode.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        integral : float
            Normalization integral of the unit-peak Gaussian.
        """
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
        """
        Derivative of `gaussian_integral` with respect to a shape parameter group.

        Uses Jacobi's formula d(det(M))/dx = det(M) * trace(M^-1 dM/dx)
        on the 1-sigma covariance implied by `inv_S`.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        d_inv_S : sequence of ndarray
            Three derivatives of `inv_S` (e.g. from `inv_S_deriv_r`,
            `inv_S_deriv_size_shape`, or `inv_S_deriv_domega`), each
            shape (3, 3).
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        dg : ndarray of shape (3,)
            Derivative of `gaussian_integral` w.r.t. each parameter in
            `d_inv_S`.
        """
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

        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        # d(det(S))/dx = -det(S) * trace(S @ d(inv_var)/dx), where S = inv_var^-1
        # (Jacobi's formula applied to det(inv_var) = 1/det(S)). All three
        # shape parameters are contracted against the mode's own
        # submatrix (not just the "own-axis" ones): under a rotated
        # orientation the submatrix genuinely depends on all three radii,
        # and a rotation about one axis perturbs the *other* two axes'
        # submatrix, so no axis can be assumed zero a priori.
        if mode == "3d" or "2d" in mode:
            S_mode = np.linalg.inv(inv_var)
            g0, g1, g2 = (
                -np.einsum("ij,ji->", S_mode, dv) for dv in d_inv_var
            )
        else:
            S_mode = 1.0 / inv_var
            g0, g1, g2 = (-dv * S_mode for dv in d_inv_var)

        return 0.5 * g * np.array([g0, g1, g2])

    def gaussian_jac_c(self, x0, x1, x2, c, inv_S, mode="3d"):
        """
        Derivative of the unnormalized `gaussian` profile with respect to the center c0, c1, c2.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        dg : ndarray
            Stacked [dg/dc0, dg/dc1, dg/dc2] on the mode's coordinate
            grid; entries for axes unused by `mode` are zero.
        """
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
        """
        Derivative of the unnormalized `gaussian` profile with respect to a shape parameter group.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        d_inv_S : sequence of ndarray
            Three derivatives of `inv_S` (e.g. from `inv_S_deriv_r`,
            `inv_S_deriv_size_shape`, or `inv_S_deriv_domega`), each
            shape (3, 3).
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        dg : ndarray
            Stacked derivatives of the Gaussian profile w.r.t. each
            parameter in `d_inv_S`, on the mode's coordinate grid.
        """
        c0, c1, c2 = c

        dx0, dx1, dx2 = x0 - c0, x1 - c1, x2 - c2

        inv_var = self.ellipsoid_covariance(inv_S, mode)
        d_inv_var = [self.ellipsoid_covariance(val, mode) for val in d_inv_S]

        if mode == "3d":
            dx = [dx0, dx1, dx2]
        elif mode == "2d_0":
            dx = [dx1[0, :, :], dx2[0, :, :]]
        elif mode == "2d_1":
            dx = [dx0[:, 0, :], dx2[:, 0, :]]
        elif mode == "2d_2":
            dx = [dx0[:, :, 0], dx1[:, :, 0]]
        elif mode == "1d_0":
            dx = dx0[:, 0, 0]
        elif mode == "1d_1":
            dx = dx1[0, :, 0]
        elif mode == "1d_2":
            dx = dx2[0, 0, :]

        # Each of the three shape parameters is contracted against the
        # mode's own submatrix/dx (not just the "own-axis" ones): under a
        # rotated orientation the submatrix genuinely depends on all three
        # radii, and a rotation about one axis perturbs the *other* two
        # axes' submatrix, so no axis can be assumed zero a priori.
        if mode == "3d" or "2d" in mode:
            d2 = np.einsum("i...,ij,j...->...", dx, inv_var, dx)
            g0, g1, g2 = (
                np.einsum("i...,ij,j...->...", dx, dv, dx) for dv in d_inv_var
            )
        else:
            d2 = inv_var * dx**2
            g0, g1, g2 = (dv * dx**2 for dv in d_inv_var)

        g = np.exp(-0.5 * d2)

        return -0.5 * g * np.array([g0, g1, g2])

    def counts_to_intensity_uncertainty(self, d, n):
        """
        Convert raw counts and normalization into normalized intensity and its uncertainty.

        Uncertainty uses the sqrt(d + 1) Poisson approximation (avoids
        zero uncertainty at d == 0).

        Parameters
        ----------
        d : ndarray
            Raw observed event counts.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.

        Returns
        -------
        y : ndarray
            Normalized intensity d / n, NaN where n <= 0.
        e : ndarray
            Uncertainty sqrt(clip(d, 0, None) + 1) / n, NaN where n <= 0.
        """
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
        """
        Signed Poisson deviance residual r = sign(mu-d)*sqrt(2*(mu-d+d*log(d/mu))) and dr/dmu.

        Used to chain-rule the Jacobian of a residual-based least-squares
        fit against a Poisson likelihood (`jacobian_mode_poisson`). Near
        mu == d, evaluated via the limiting expression to avoid 0/0. For
        mu < 0 (unphysical mid-optimization), evaluated at a small
        positive floor and linearly extrapolated to the true mu, giving a
        large correctly-signed residual with a bounded Jacobian. Weights
        apply multiplicatively to both `r` and `dr_dmu`.

        Parameters
        ----------
        d : ndarray
            Observed (approximate) raw counts.
        mu : ndarray
            Model (expected) counts, same shape as `d`.
        w : ndarray, optional
            Per-voxel weight. Default 1 everywhere.
        eps : float, optional
            Positive floor for clipping `mu`. Default 1e-12.

        Returns
        -------
        r : ndarray
            Signed weighted deviance residual, NaN where invalid.
        dr_dmu : ndarray
            Weighted derivative of `r` w.r.t. `mu`, NaN where invalid.
        valid : ndarray of bool
            Mask where `d`, `mu`, `w` are finite and d >= 0.
        """
        d = np.asarray(d, dtype=float)
        mu = np.asarray(mu, dtype=float)
        if w is None:
            w = np.ones_like(d, dtype=float)
        else:
            w = np.asarray(w, dtype=float)

        valid = np.isfinite(d) & np.isfinite(mu) & np.isfinite(w) & (d >= 0)

        r = np.full_like(mu, np.nan, dtype=float)
        dr_dmu = np.full_like(mu, np.nan, dtype=float)

        if not np.any(valid):
            return r, dr_dmu, valid

        dv = d[valid]
        mu_true = mu[valid]
        muv = np.clip(mu_true, eps, None)

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

        # For mu < 0: the eps-clip gives near-zero r and a gradient that pushes mu
        # further negative.  Instead, evaluate the deviance at a small positive floor
        # and extrapolate linearly to the true (negative) mu.  This gives a large
        # negative residual — correct gradient direction — with a bounded Jacobian.
        neg = mu_true < 0
        if np.any(neg):
            floor = 0.01
            dv_neg = dv[neg]
            muv_fl = np.full(neg.sum(), floor)
            term_fl = muv_fl - dv_neg
            pos_fl = dv_neg > 0
            term_fl[pos_fl] += dv_neg[pos_fl] * np.log(
                dv_neg[pos_fl] / muv_fl[pos_fl]
            )
            term_fl = np.maximum(term_fl, 0.0)
            rv_fl = np.sign(muv_fl - dv_neg) * np.sqrt(2.0 * term_fl)
            near_fl = np.isclose(muv_fl, dv_neg, rtol=1e-8, atol=1e-10) | (
                np.abs(rv_fl) < 1e-10
            )
            fv_fl = np.where(
                near_fl,
                1.0 / np.sqrt(muv_fl),
                (1.0 - dv_neg / muv_fl) / rv_fl,
            )
            rv[neg] = rv_fl + (mu_true[neg] - floor) * fv_fl
            fv[neg] = fv_fl

        rv *= w[valid]
        fv *= w[valid]

        r[valid] = rv
        dr_dmu[valid] = fv

        return r, dr_dmu, valid

    def mode_model_counts(
        self, params, x0, x1, x2, d, n, w, c, inv_S, mode, bkg=None
    ):
        """
        Poisson model counts mu = n*(A*gaussian + B) [+ bkg] for one mode.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `A<mode>` and `B<mode>`.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Unused (kept for a uniform call signature).
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : ndarray
            Unused (kept for a uniform call signature).
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
        bkg : ndarray, optional
            Additional fixed background counts to add to the model.

        Returns
        -------
        mu : ndarray
            Model (expected) counts.
        g : ndarray
            Unnormalized Gaussian profile used in the model.
        A, B : float
            Current `A<mode>` and `B<mode>` parameter values.
        """
        A = params["A" + mode].value
        B = params["B" + mode].value
        g = self.gaussian(x0, x1, x2, c, inv_S, mode)
        mu = n * (A * g + B)
        if bkg is not None:
            mu = mu + bkg
        return mu, g, A, B

    def residual_mode_poisson(
        self, params, x0, x1, x2, d, n, w, c, inv_S, mode, bkg=None
    ):
        """
        Flattened Poisson-deviance residual vector for one mode, for use in least-squares fitting.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `A<mode>` and `B<mode>`.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts for this mode.
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : None, scalar, or ndarray
            Per-voxel weight; coerced via `coerce_weight`.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
        bkg : ndarray, optional
            Additional fixed background counts added to the model.

        Returns
        -------
        res : ndarray
            Flattened signed Poisson deviance residuals for valid voxels.
        """
        w = self.coerce_weight(d, w)
        mu, g, A, B = self.mode_model_counts(
            params, x0, x1, x2, d, n, w, c, inv_S, mode, bkg
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
        bkg=None,
    ):
        """
        Analytic Jacobian of `residual_mode_poisson` with respect to the varying fit parameters.

        Chain-rules the derivative factor from
        `poisson_deviance_residual_factor` through
        mu = n*(A*gaussian + B) [+ bkg], using the Gaussian derivatives
        w.r.t. center (`gaussian_jac_c`) and shape (`gaussian_jac_S`).

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `A<mode>`, `B<mode>`, `c0, c1,
            c2, log_size, shape_1, shape_2, domega_x, domega_y, domega_z`.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts for this mode.
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : None, scalar, or ndarray
            Per-voxel weight; coerced via `coerce_weight`.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray
            Derivatives of `inv_S` w.r.t. log_size, shape_1, shape_2
            (from `inv_S_deriv_size_shape`).
        du : tuple of ndarray
            Derivatives of `inv_S` w.r.t. domega_x, domega_y, domega_z
            (from `inv_S_deriv_domega`).
        mode : str
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
        bkg : ndarray, optional
            Additional fixed background counts added to the model.

        Returns
        -------
        jac : ndarray of shape (n_vary, n_valid)
            Jacobian restricted to varying parameters and valid voxels.
        """
        params_list = [name for name, par in params.items()]
        n_params = len(params_list)

        w = self.coerce_weight(d, w)
        mu, g, A, B = self.mode_model_counts(
            params, x0, x1, x2, d, n, w, c, inv_S, mode, bkg
        )
        _, factor, mask = self.poisson_deviance_residual_factor(d, mu, w)

        n_pix = np.asarray(d).size
        jac = np.zeros((n_params, n_pix), dtype=float)

        dA = factor * n * g
        dB = factor * n

        yc_gauss = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode=mode)
        dc0, dc1, dc2 = factor * n * A * yc_gauss

        yr_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, dr, mode=mode)
        d_size, d_shape1, d_shape2 = factor * n * A * yr_gauss

        yu_gauss = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode=mode)
        d_omega_x, d_omega_y, d_omega_z = factor * n * A * yu_gauss

        names_values = {
            "A" + mode: dA,
            "B" + mode: dB,
            "c0": dc0,
            "c1": dc1,
            "c2": dc2,
            "log_size": d_size,
            "shape_1": d_shape1,
            "shape_2": d_shape2,
            "domega_x": d_omega_x,
            "domega_y": d_omega_y,
            "domega_z": d_omega_z,
        }

        for name, value in names_values.items():
            if name in params_list:
                jac[params_list.index(name), :] = np.asarray(value).flatten()

        ind = [i for i, (name, par) in enumerate(params.items()) if par.vary]
        return jac[ind][:, mask.flatten()]

    def residual_1d(
        self,
        params,
        x0,
        x1,
        x2,
        ds,
        ns,
        ws=None,
        bkgs=None,
        c=None,
        inv_S=None,
    ):
        """
        Concatenated Poisson-deviance residuals across all three 1D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        ds : sequence of ndarray
            Raw observed counts for modes ("1d_0", "1d_1", "1d_2").
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights; coerced via `coerce_weights`.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.

        Returns
        -------
        res : ndarray
            Concatenated `residual_mode_poisson` outputs for "1d_0",
            "1d_1", "1d_2".
        """
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)
        b0, b1, b2 = bkgs if bkgs is not None else (None, None, None)

        return np.concatenate(
            [
                self.residual_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, "1d_0", b0
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, "1d_1", b1
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, "1d_2", b2
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
        bkgs=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        """
        Column-stacked Poisson-deviance Jacobians across all three 1D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        ds : sequence of ndarray
            Raw observed counts for modes ("1d_0", "1d_1", "1d_2").
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights; coerced via `coerce_weights`.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. log_size, shape_1, shape_2.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. domega_x, domega_y, domega_z.

        Returns
        -------
        jac : ndarray
            Column-stacked `jacobian_mode_poisson` outputs for "1d_0",
            "1d_1", "1d_2".
        """
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)
        b0, b1, b2 = bkgs if bkgs is not None else (None, None, None)

        return np.column_stack(
            [
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d0,
                    n0,
                    w0,
                    c,
                    inv_S,
                    dr,
                    du,
                    "1d_0",
                    b0,
                ),
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d1,
                    n1,
                    w1,
                    c,
                    inv_S,
                    dr,
                    du,
                    "1d_1",
                    b1,
                ),
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d2,
                    n2,
                    w2,
                    c,
                    inv_S,
                    dr,
                    du,
                    "1d_2",
                    b2,
                ),
            ]
        )

    def residual_2d(
        self,
        params,
        x0,
        x1,
        x2,
        ds,
        ns,
        ws=None,
        bkgs=None,
        c=None,
        inv_S=None,
    ):
        """
        Concatenated Poisson-deviance residuals across all three 2D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        ds : sequence of ndarray
            Raw observed counts for modes ("2d_0", "2d_1", "2d_2").
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights; coerced via `coerce_weights`.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.

        Returns
        -------
        res : ndarray
            Concatenated `residual_mode_poisson` outputs for "2d_0",
            "2d_1", "2d_2".
        """
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)
        b0, b1, b2 = bkgs if bkgs is not None else (None, None, None)

        return np.concatenate(
            [
                self.residual_mode_poisson(
                    params, x0, x1, x2, d0, n0, w0, c, inv_S, "2d_0", b0
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d1, n1, w1, c, inv_S, "2d_1", b1
                ),
                self.residual_mode_poisson(
                    params, x0, x1, x2, d2, n2, w2, c, inv_S, "2d_2", b2
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
        bkgs=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        """
        Column-stacked Poisson-deviance Jacobians across all three 2D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        ds : sequence of ndarray
            Raw observed counts for modes ("2d_0", "2d_1", "2d_2").
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights; coerced via `coerce_weights`.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. log_size, shape_1, shape_2.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. domega_x, domega_y, domega_z.

        Returns
        -------
        jac : ndarray
            Column-stacked `jacobian_mode_poisson` outputs for "2d_0",
            "2d_1", "2d_2".
        """
        d0, d1, d2 = ds
        n0, n1, n2 = ns
        w0, w1, w2 = self.coerce_weights(ds, ws)
        b0, b1, b2 = bkgs if bkgs is not None else (None, None, None)

        return np.column_stack(
            [
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d0,
                    n0,
                    w0,
                    c,
                    inv_S,
                    dr,
                    du,
                    "2d_0",
                    b0,
                ),
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d1,
                    n1,
                    w1,
                    c,
                    inv_S,
                    dr,
                    du,
                    "2d_1",
                    b1,
                ),
                self.jacobian_mode_poisson(
                    params,
                    x0,
                    x1,
                    x2,
                    d2,
                    n2,
                    w2,
                    c,
                    inv_S,
                    dr,
                    du,
                    "2d_2",
                    b2,
                ),
            ]
        )

    def residual_3d(
        self, params, x0, x1, x2, d, n, w=None, bkg=None, c=None, inv_S=None
    ):
        """
        Poisson-deviance residual vector for the full 3D volume.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : None, scalar, or ndarray, optional
            Per-voxel weight; coerced via `coerce_weight`.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.

        Returns
        -------
        res : ndarray
            Flattened signed Poisson deviance residuals for valid voxels.
        """
        w = self.coerce_weight(d, w)
        return self.residual_mode_poisson(
            params, x0, x1, x2, d, n, w, c, inv_S, "3d", bkg
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
        bkg=None,
        c=None,
        inv_S=None,
        dr=None,
        du=None,
    ):
        """
        Poisson-deviance Jacobian for the full 3D volume.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : None, scalar, or ndarray, optional
            Per-voxel weight; coerced via `coerce_weight`.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
        c : sequence of float, optional
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. log_size, shape_1, shape_2.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` w.r.t. domega_x, domega_y, domega_z.

        Returns
        -------
        jac : ndarray of shape (n_vary, n_valid)
            Jacobian restricted to varying parameters and valid voxels.
        """
        w = self.coerce_weight(d, w)
        return self.jacobian_mode_poisson(
            params, x0, x1, x2, d, n, w, c, inv_S, dr, du, "3d", bkg
        )

    def residual(self, params, args_1d, args_2d, args_3d):
        """
        Full stacked residual vector combining all modes and the prior.

        Concatenates the 1D, 2D, 3D Poisson-deviance residuals with the
        prior residual and a scalar missing-support penalty, sanitizing
        NaN/inf so `scipy.optimize.least_squares` sees a well-behaved
        cost vector. This is the residual function passed to
        `lmfit.Minimizer` in `sweep`.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        args_1d : sequence
            Positional arguments for `residual_1d`: (x0, x1, x2, ds, ns,
            ws, bkgs).
        args_2d : sequence
            Positional arguments for `residual_2d` (same layout).
        args_3d : sequence
            Positional arguments for `residual_3d`: (x0, x1, x2, d, n, w,
            bkg).

        Returns
        -------
        cost : ndarray
            Concatenated residual vector (1D + 2D + 3D + prior terms).
        """
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0, r1, r2, u0, u1, u2 = self.shape_from_params(params)

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        cost_1d = self.residual_1d(params, *args_1d, c, inv_S)
        cost_2d = self.residual_2d(params, *args_2d, c, inv_S)
        cost_3d = self.residual_3d(params, *args_3d, c, inv_S)
        cost_prior = self.prior_residual(params)
        cost_missing = self.missing_support_penalty(
            c, inv_S, args_3d[0], args_3d[1], args_3d[2], args_3d[4]
        )

        cost = np.concatenate(
            [cost_1d, cost_2d, cost_3d, cost_prior, [cost_missing]]
        )
        cost = np.nan_to_num(cost, nan=0.0, posinf=1e16, neginf=-1e16)

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):
        """
        Full stacked Jacobian matching the residual ordering of `residual`.

        Column-stacks the 1D, 2D, 3D Poisson-deviance Jacobians with the
        prior Jacobian and the missing-support penalty's Jacobian
        column, sanitizes NaN/inf, and transposes to the
        (n_residuals, n_vary) convention expected by
        `scipy.optimize.least_squares`. Passed to `lmfit.Minimizer.minimize`
        in `sweep`.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        args_1d : sequence
            Positional arguments for `jacobian_1d`: (x0, x1, x2, ds, ns,
            ws, bkgs).
        args_2d : sequence
            Positional arguments for `jacobian_2d` (same layout).
        args_3d : sequence
            Positional arguments for `jacobian_3d`: (x0, x1, x2, d, n, w,
            bkg).

        Returns
        -------
        jac : ndarray of shape (n_residuals, n_vary)
            Full Jacobian matrix, transposed to match `residual`.
        """
        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value

        r0, r1, r2, u0, u1, u2 = self.shape_from_params(params)

        domega = np.array(
            [
                params["domega_x"].value,
                params["domega_y"].value,
                params["domega_z"].value,
            ]
        )

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        dr = self.inv_S_deriv_size_shape(r0, r1, r2, u0, u1, u2)
        du = self.inv_S_deriv_domega(r0, r1, r2, domega)

        jac_1d = self.jacobian_1d(params, *args_1d, c, inv_S, dr, du)
        jac_2d = self.jacobian_2d(params, *args_2d, c, inv_S, dr, du)
        jac_3d = self.jacobian_3d(params, *args_3d, c, inv_S, dr, du)
        jac_prior = self.prior_jacobian(params)
        jac_missing = self.missing_support_penalty_jacobian(
            params,
            args_3d[0],
            args_3d[1],
            args_3d[2],
            args_3d[4],
            c,
            inv_S,
            dr,
            du,
        )

        jac = np.column_stack([jac_1d, jac_2d, jac_3d, jac_prior, jac_missing])
        jac = np.nan_to_num(jac, nan=0.0, posinf=1e16, neginf=-1e16)

        return jac.T

    def collect_mode_fit_metrics(
        self, x0, x1, x2, c, inv_S, mode_data, bkg_offset=None
    ):
        """
        Fitted profile, chi-squared, and box-sum intensity for each requested mode.

        For each mode, evaluates A*gaussian + B (plus any `bkg_offset`),
        masks invalid points, and computes the reduced Poisson deviance
        (`poisson_deviance_fit`) and box-sum intensity/uncertainty
        (`estimate_intensity`).

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode_data : dict
            Mode string -> (y, e) observed normalized intensity and
            uncertainty.
        bkg_offset : dict, optional
            Mode string -> additional per-voxel background-rate offset;
            modes absent get no offset.

        Returns
        -------
        metrics : dict
            Mode string -> [I0, s0, chi2, fit_triplet], where I0/s0 are
            the box-sum intensity/uncertainty, chi2 the reduced Poisson
            deviance, and fit_triplet = (y_fit, y, e).
        """
        args = x0, x1, x2, c, inv_S

        metrics = {}

        for mode, (y, e) in mode_data.items():
            A = self.params["A" + mode].value
            B = self.params["B" + mode].value

            y_fit = A * self.gaussian(*args, mode) + B

            if bkg_offset is not None and mode in bkg_offset:
                y_fit = y_fit + bkg_offset[mode]

            if mode == "3d":
                valid = np.isfinite(y) & (e >= 0)
            else:
                valid = np.isfinite(y) & (e > 0)

            y_fit[~valid] = np.nan

            fit_triplet = (y_fit, y, e)

            chi2 = self.poisson_deviance_fit(
                x0, x1, x2, c, inv_S, *fit_triplet, mode
            )
            I0, s0 = self.estimate_intensity(
                x0,
                x1,
                x2,
                c,
                inv_S,
                *fit_triplet,
                mode,
                bkg_offset=(
                    bkg_offset[mode]
                    if bkg_offset is not None and mode in bkg_offset
                    else None
                ),
            )

            metrics[mode] = [I0, s0, chi2, fit_triplet]

        return metrics

    def missing_support_fraction(self, x0, x1, x2, c, inv_S, n, mode="3d"):
        """
        Fraction of the peak's total Gaussian mass not captured by valid voxels.

        The denominator is the analytic whole-space integral
        (`gaussian_integral`), not a box-grid sum, so support extending
        past the fitting box counts as missing too, same as an internal
        zero-normalization gap. Diagnostic only: doesn't affect the fit.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        n : ndarray
            Normalization counts on the same grid; non-finite/non-positive
            entries mark unmeasured voxels.
        mode : str, optional
            Mode accepted by `gaussian`. Default "3d".

        Returns
        -------
        missing_fraction : float
            missing_support / total_support, or nan if total_support <= 0.
        missing_support : float
            total_support - valid_support, clipped at 0.
        total_support : float
            Analytic whole-space Gaussian integral (box-independent).
        """
        profile = self.gaussian(x0, x1, x2, c, inv_S, mode=mode)

        voxel_volume = np.prod(self.voxels(x0, x1, x2))

        weighted_profile = profile * voxel_volume

        valid_mask = np.isfinite(n) & (n > 0)

        valid_support = float(np.sum(weighted_profile[valid_mask]))
        total_support = float(self.gaussian_integral(inv_S, mode=mode))

        if not np.isfinite(total_support) or total_support <= 0:
            return (
                float("nan"),
                max(total_support - valid_support, 0.0),
                total_support,
            )

        missing_support = max(total_support - valid_support, 0.0)
        missing_fraction = missing_support / total_support

        return missing_fraction, missing_support, total_support

    def missing_support_penalty(self, c, inv_S, x0, x1, x2, n, mode="3d"):
        """
        Smooth softplus penalty on the missing-support fraction (`missing_support_fraction`).

        Negligible below `self.missing_tolerance`; rises smoothly above
        it with transition width `self.missing_smoothing` and strength
        `self.missing_sigma`, using a softplus for differentiability at
        the threshold.

        Parameters
        ----------
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        n : ndarray
            Normalization counts on the same grid as x0, x1, x2.
        mode : str, optional
            Mode accepted by `gaussian`. Default "3d".

        Returns
        -------
        penalty : float
            Scalar softplus residual, or 0.0 if the missing fraction is
            non-finite.
        """
        missing_fraction, _, _ = self.missing_support_fraction(
            x0, x1, x2, c, inv_S, n, mode=mode
        )

        if not np.isfinite(missing_fraction):
            return 0.0

        z = (
            missing_fraction - self.missing_tolerance
        ) / self.missing_smoothing

        return (
            self.missing_smoothing * np.logaddexp(0.0, z) / self.missing_sigma
        )

    def missing_support_penalty_jacobian(
        self, params, x0, x1, x2, n, c, inv_S, dr, du, mode="3d"
    ):
        """
        Analytic Jacobian row of `missing_support_penalty` with respect to the varying fit parameters.

        Chain-rules the softplus derivative through the quotient-rule
        derivative of the missing-support fraction f = W_missing / W
        (W the whole-space integral, W_missing the clipped shortfall).
        dW/dtheta comes from `gaussian_integral_jac_S` (0 for center
        parameters); the per-voxel profile derivative reuses
        `gaussian_jac_c`/`gaussian_jac_S`. A<mode>/B<mode> entries are
        left at zero since they don't affect profile geometry.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        n : ndarray
            Normalization counts on the same grid as x0, x1, x2.
        c : sequence of float
            Peak center (c0, c1, c2).
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray
            Derivatives of `inv_S` w.r.t. log_size, shape_1, shape_2.
        du : tuple of ndarray
            Derivatives of `inv_S` w.r.t. domega_x, domega_y, domega_z.
        mode : str, optional
            Mode accepted by `gaussian`. Default "3d".

        Returns
        -------
        jac : ndarray of shape (n_vary, 1)
            Jacobian column restricted to `vary=True` parameters.
        """
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), 1), dtype=float)

        profile = self.gaussian(x0, x1, x2, c, inv_S, mode=mode)
        voxel_volume = np.prod(self.voxels(x0, x1, x2))
        weighted_profile = profile * voxel_volume

        valid_mask = np.isfinite(n) & (n > 0)
        valid_support = np.sum(weighted_profile[valid_mask])
        total_support = self.gaussian_integral(inv_S, mode=mode)

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]

        if not np.isfinite(total_support) or total_support <= 0:
            return jac[ind]

        # missing_support = max(total_support - valid_support, 0); its
        # derivative is 0 in the clipped (over-covered) regime, exactly
        # like a ReLU.
        active = (total_support - valid_support) > 0.0
        missing_support = max(total_support - valid_support, 0.0)
        missing_fraction = missing_support / total_support

        z = (
            missing_fraction - self.missing_tolerance
        ) / self.missing_smoothing
        dpenalty_df = scipy.special.expit(z) / self.missing_sigma

        dg_c = self.gaussian_jac_c(x0, x1, x2, c, inv_S, mode=mode)
        dg_size_shape = self.gaussian_jac_S(
            x0, x1, x2, c, inv_S, dr, mode=mode
        )
        dg_omega = self.gaussian_jac_S(x0, x1, x2, c, inv_S, du, mode=mode)

        # total_support is the analytic whole-space integral: translation
        # invariant (0 for c0,c1,c2), and its size/shape/orientation
        # derivatives come from gaussian_integral_jac_S rather than a
        # box-grid sum.
        dW_total_size_shape = self.gaussian_integral_jac_S(
            inv_S, dr, mode=mode
        )
        dW_total_omega = self.gaussian_integral_jac_S(inv_S, du, mode=mode)

        dprofile_dW_total_by_name = {
            "c0": (dg_c[0], 0.0),
            "c1": (dg_c[1], 0.0),
            "c2": (dg_c[2], 0.0),
            "log_size": (dg_size_shape[0], dW_total_size_shape[0]),
            "shape_1": (dg_size_shape[1], dW_total_size_shape[1]),
            "shape_2": (dg_size_shape[2], dW_total_size_shape[2]),
            "domega_x": (dg_omega[0], dW_total_omega[0]),
            "domega_y": (dg_omega[1], dW_total_omega[1]),
            "domega_z": (dg_omega[2], dW_total_omega[2]),
        }

        for name in _MISSING_SUPPORT_GEOMETRY_PARAMS:
            if name not in params or not params[name].vary:
                continue

            dprofile, dW_total = dprofile_dW_total_by_name[name]
            dW_valid = np.sum(dprofile[valid_mask]) * voxel_volume
            dmissing_dtheta = (dW_total - dW_valid) if active else 0.0

            df_dtheta = (
                total_support * dmissing_dtheta - missing_support * dW_total
            ) / total_support**2

            idx = params_list.index(name)
            jac[idx, 0] = dpenalty_df * df_dtheta

        return jac[ind]

    def extract_result(self, args_1d, args_2d, args_3d, xmod):
        """
        Finalize the fit: compute per-mode metrics and populate the post-fit result attributes.

        Converts raw counts to normalized intensity/uncertainty for every
        mode and calls `collect_mode_fit_metrics` to populate
        `self.reddev`, `self.intensity`, `self.sigma`, `self.fit_metrics`,
        and the missing-support attributes. Also derives the final
        ellipsoid (center, radii, axes) from S = inv(inv_S), shifts the
        center back by `xmod`, and populates `self.peak_pos`,
        `self.best_fit`, `self.best_prof`, `self.best_bkg_prof`, and
        `self.best_proj`.

        Parameters
        ----------
        args_1d : sequence
            (x0, x1, x2, d1d, n1d, w1d, bkg_1d, ...) where d1d/n1d are
            3-tuples of per-axis 1D-projected counts/normalization and
            bkg_1d is None or a matching 3-tuple of background counts.
        args_2d : sequence
            Analogous to `args_1d` for the three 2D projections.
        args_3d : sequence
            Analogous to `args_1d` for the full 3D volume.
        xmod : float
            Offset subtracted from axis-0 before fitting (e.g. to wrap a
            periodic axis); added back to the reported center.

        Returns
        -------
        c0, c1, c2 : float
            Fitted peak center (c0 with `xmod` added back).
        r0, r1, r2 : float
            Principal radii (1-sigma) of the fitted ellipsoid.
        v0, v1, v2 : ndarray of shape (3,)
            Principal axis vectors of the fitted ellipsoid.

            Returns None instead of this tuple if the optimized inverse
            covariance is not positive definite.
        """
        x0, x1, x2, d1d, n1d, w1d, bkg_1d, *_ = args_1d
        x0, x1, x2, d2d, n2d, w2d, bkg_2d, *_ = args_2d
        x0, x1, x2, d3d, n3d, w3d, bkg_3d, *_ = args_3d

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

        self.reddev = []
        self.intensity = []
        self.sigma = []

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        c0_err = self.params["c0"].stderr
        c1_err = self.params["c1"].stderr
        c2_err = self.params["c2"].stderr

        # Convert whitened σ_μ back to approximate physical scale per axis
        _c_err_scale = (
            self._prior_radii * self.prior_center_sigma / _R_SCALE_3D
        )
        if c0_err is None:
            c0_err = abs(c0) if c0 != 0 else float(_c_err_scale[0])
        if c1_err is None:
            c1_err = abs(c1) if c1 != 0 else float(_c_err_scale[1])
        if c2_err is None:
            c2_err = abs(c2) if c2 != 0 else float(_c_err_scale[2])

        r0, r1, r2, u0, u1, u2 = self.shape_from_params(self.params)

        c_err = c0_err, c1_err, c2_err

        c, inv_S = self.centroid_inverse_covariance(
            c0, c1, c2, r0, r1, r2, u0, u1, u2
        )

        (
            self.missing_fraction,
            self.missing_support,
            self.total_support,
        ) = self.missing_support_fraction(x0, x1, x2, c, inv_S, n3d)

        def _bkg_norm(bkg_arr, n_arr):
            return np.where(
                np.isfinite(n_arr) & (n_arr > 0),
                bkg_arr / n_arr,
                0.0,
            )

        bkg_offset_1d = None
        if bkg_1d is not None:
            b1d_0, b1d_1, b1d_2 = bkg_1d
            bkg_offset_1d = {
                "1d_0": _bkg_norm(b1d_0, n1d_0),
                "1d_1": _bkg_norm(b1d_1, n1d_1),
                "1d_2": _bkg_norm(b1d_2, n1d_2),
            }

        bkg_offset_2d = None
        if bkg_2d is not None:
            b2d_0, b2d_1, b2d_2 = bkg_2d
            bkg_offset_2d = {
                "2d_0": _bkg_norm(b2d_0, n2d_0),
                "2d_1": _bkg_norm(b2d_1, n2d_1),
                "2d_2": _bkg_norm(b2d_2, n2d_2),
            }

        bkg_offset_3d = None
        if bkg_3d is not None:
            bkg_offset_3d = {"3d": _bkg_norm(bkg_3d, n3d)}

        mode_1d = {
            "1d_0": (y1d_0, e1d_0),
            "1d_1": (y1d_1, e1d_1),
            "1d_2": (y1d_2, e1d_2),
        }
        metrics_1d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_1d, bkg_offset_1d
        )

        modes = ["1d_0", "1d_1", "1d_2"]

        y1 = [metrics_1d[mode][3] for mode in modes]
        self.reddev.append([metrics_1d[mode][2] for mode in modes])
        self.intensity.append([metrics_1d[mode][0] for mode in modes])
        self.sigma.append([metrics_1d[mode][1] for mode in modes])

        mode_2d = {
            "2d_0": (y2d_0, e2d_0),
            "2d_1": (y2d_1, e2d_1),
            "2d_2": (y2d_2, e2d_2),
        }
        metrics_2d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_2d, bkg_offset_2d
        )

        modes = ["2d_0", "2d_1", "2d_2"]

        y2 = [metrics_2d[mode][3] for mode in modes]
        self.reddev.append([metrics_2d[mode][2] for mode in modes])
        self.intensity.append([metrics_2d[mode][0] for mode in modes])
        self.sigma.append([metrics_2d[mode][1] for mode in modes])

        mode_3d = {"3d": (y3d, e3d)}
        metrics_3d = self.collect_mode_fit_metrics(
            x0, x1, x2, c, inv_S, mode_3d, bkg_offset_3d
        )

        y3 = metrics_3d["3d"][3]
        self.reddev.append(metrics_3d["3d"][2])
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

        self.best_bkg_prof = None
        scale = getattr(self, "_bkg_scale", None)
        if bkg_1d is not None and scale is not None:
            x_coords = (x0[:, 0, 0] + xmod, x1[0, :, 0], x2[0, 0, :])
            bkg_prof = []
            for x_k, b1d_k, n1d_k in zip(x_coords, bkg_1d, n1d):
                b_raw_proj = b1d_k * scale
                valid = (n1d_k > 0) & np.isfinite(n1d_k) & np.isfinite(b1d_k)
                y_bkg = np.where(valid, b1d_k / n1d_k, np.nan)
                e_bkg = np.where(
                    valid,
                    np.sqrt(np.clip(b_raw_proj, 0, None) + 1)
                    / (scale * n1d_k),
                    np.nan,
                )
                bkg_prof.append((x_k, y_bkg, e_bkg))
            self.best_bkg_prof = bkg_prof

        self.best_proj = (
            (x1[0, :, :], x2[0, :, :], *y2[0]),
            (x0[:, 0, :] + xmod, x2[:, 0, :], *y2[1]),
            (x0[:, :, 0] + xmod, x1[:, :, 0], *y2[2]),
        )

        return c0, c1, c2, r0, r1, r2, v0, v1, v2

    def extract_amplitude_background(self):
        """
        Collect the fitted 1D-mode amplitude and background parameters.

        Returns
        -------
        amplitudes : ndarray of shape (3,)
            A1d_0, A1d_1, A1d_2 `lmfit.Parameter` objects.
        backgrounds : ndarray of shape (3,)
            B1d_0, B1d_1, B1d_2 `lmfit.Parameter` objects.
        """
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
        """
        Estimate a 1D peak-center index via matched-filter correlation with a kernel.

        Cross-correlates the background-subtracted counts with `kernel`
        (normalized by sqrt(correlate(n^2, kernel^2))) and returns the
        index of maximum correlation, falling back to the array midpoint
        if that maximum falls in the outer eighths (an unreliable edge
        estimate).

        Parameters
        ----------
        d : ndarray
            1D raw observed event counts.
        n : ndarray
            1D normalization (monitor/solid-angle) counts, same shape as
            `d`.
        kernel : ndarray
            1D correlation kernel (e.g. a Gaussian template).
        frac : float, optional
            Unused, reserved for future use. Default 0.9.
        min_weight : float, optional
            Minimum correlated normalization-squared value to accept a
            correlation value. Default 1e-12.

        Returns
        -------
        index : int
            Estimated integer index of the peak center along `d`.
        """
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

    def _mode_mah2(self, x0, x1, x2, mode):
        """
        Squared Mahalanobis distance from the estimated-fit ellipsoid, in the projected subspace.

        Uses `self.estimated_fit` (fixed reference set by
        `update_estimate`), not the current fit parameters. Output shape
        matches `profile_project`'s for the same mode.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        mode : str
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".

        Returns
        -------
        mah2 : ndarray
            Squared Mahalanobis distance to the estimated-fit center,
            shape matching the mode's projected grid. inf where the
            relevant covariance sub-block is singular/non-positive.
        """
        c_est, S_est = self.estimated_fit

        if mode == "3d":
            dx = np.stack(
                [x0 - c_est[0], x1 - c_est[1], x2 - c_est[2]], axis=-1
            )
            try:
                return np.einsum(
                    "...i,ij,...j->...", dx, np.linalg.inv(S_est), dx
                )
            except np.linalg.LinAlgError:
                return np.full(x0.shape, np.inf)

        elif mode == "1d_0":
            xk = x0[:, 0, 0]
            skk = S_est[0, 0]
            return (
                (xk - c_est[0]) ** 2 / skk
                if skk > 0
                else np.full(xk.shape, np.inf)
            )

        elif mode == "1d_1":
            xk = x1[0, :, 0]
            skk = S_est[1, 1]
            return (
                (xk - c_est[1]) ** 2 / skk
                if skk > 0
                else np.full(xk.shape, np.inf)
            )

        elif mode == "1d_2":
            xk = x2[0, 0, :]
            skk = S_est[2, 2]
            return (
                (xk - c_est[2]) ** 2 / skk
                if skk > 0
                else np.full(xk.shape, np.inf)
            )

        elif mode == "2d_0":
            xa, xb = x1[0, :, :], x2[0, :, :]
            ij = [1, 2]
            dxs = np.stack([xa - c_est[1], xb - c_est[2]], axis=-1)
        elif mode == "2d_1":
            xa, xb = x0[:, 0, :], x2[:, 0, :]
            ij = [0, 2]
            dxs = np.stack([xa - c_est[0], xb - c_est[2]], axis=-1)
        elif mode == "2d_2":
            xa, xb = x0[:, :, 0], x1[:, :, 0]
            ij = [0, 1]
            dxs = np.stack([xa - c_est[0], xb - c_est[1]], axis=-1)
        else:
            return np.full(x0.shape, np.inf)

        S_sub = S_est[np.ix_(ij, ij)]
        try:
            return np.einsum(
                "...i,ij,...j->...", dxs, np.linalg.inv(S_sub), dxs
            )
        except np.linalg.LinAlgError:
            return np.full(xa.shape, np.inf)

    def quick_gaussian(self, x0, x1, x2, d, n, mode="3d", b=None, m=None):
        """
        Quick amplitude/background estimate for a mode, seeding `self.params`.

        Projects the data onto `mode`, optionally subtracts a rescaled
        measured background (`b`, `m`), estimates background `B` from
        the shell beyond Mahalanobis distance 2 and amplitude `A` from
        the 95th percentile within distance 1 above `B`, then adds
        non-negative-bounded `A<mode>`/`B<mode>` parameters.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".
        b : ndarray, optional
            Measured background counts, same shape as `d`.
        m : ndarray, optional
            Measured background monitor counts, same shape as `d`, used
            with `n` to rescale `b` into signal-count units.

        Returns
        -------
        stronger : bool or None
            True if estimated amplitude exceeds background. None if
            fewer than 4 valid voxels (no estimate made).
        """
        w = np.ones_like(n)

        d_proj, n_proj, _, _ = self.profile_project(
            x0, x1, x2, d, n, w, mode=mode
        )

        valid = (n_proj > 0) & np.isfinite(d_proj) & np.isfinite(n_proj)
        if valid.sum() <= 3:
            return None

        y = np.where(valid, d_proj / n_proj, np.nan)

        if b is not None and m is not None:
            scale = np.nanmedian(np.where(n > 0, m / n, np.nan))
            if np.isfinite(scale) and scale > 0:
                bkg_nc = self.project_background(
                    np.where(np.isfinite(b), b / scale, 0.0), mode
                )
                bkg_rate_proj = np.where(n_proj > 0, bkg_nc / n_proj, 0.0)
                bkg_valid = valid & np.isfinite(bkg_rate_proj)
                y_net = np.where(bkg_valid, y - bkg_rate_proj, y)
            else:
                y_net = y
        else:
            y_net = y

        mah2 = self._mode_mah2(x0, x1, x2, mode)

        peak_roi = valid & (mah2 <= 1.0)
        bkg_shell = valid & (mah2 >= 4.0)

        if bkg_shell.sum() >= 3:
            B = float(np.nanmedian(y_net[bkg_shell]))
        else:
            B = float(np.nanpercentile(y_net[valid], 5))

        src = peak_roi if peak_roi.sum() >= 3 else valid
        A = float(np.nanpercentile(y_net[src], 95)) - B

        if not np.isfinite(A):
            A = 1
        if not np.isfinite(B):
            B = 0

        B = max(B, 0.0)
        A = max(A, 0.0)

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
        b=None,
        m=None,
    ):
        """
        Build all mode projections, seed initial amplitudes, and iteratively fit the ellipsoid.

        Restores `self.combine_params` if available, projects the data
        onto all seven modes, quick-estimates each mode's
        amplitude/background (`quick_gaussian`), converts a measured
        background into per-mode background-count arrays, then runs an
        annealed multiresolution schedule of `sweep` calls: geometry
        parameters are unfrozen in stages (amplitude/background only,
        then + center, + size, + shape, + orientation), each interleaved
        with `update_adaptive_prior`, while the 1D/2D auxiliary residual
        weights (`alpha_1d`/`alpha_2d`, relative to `self.mode_weights_1d`/
        `self.mode_weights_2d`) are wound down stage by stage. A final
        `sweep` at a small residual `alpha_1d`/`alpha_2d` (0.01x) leaves
        the reported geometry and covariance effectively determined by
        the 3D likelihood and priors alone.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d_int : ndarray
            Raw observed event counts, 3D array.
        n_int : ndarray
            Normalization (monitor/solid-angle) counts, same shape as
            `d_int`.
        wgt : ndarray
            Per-voxel fit weight, same shape as `d_int`.
        report_fit : bool, optional
            Print an `lmfit` fit report after each `sweep` call. Default
            False.
        b : ndarray, optional
            Measured background counts, same shape as `d_int`.
        m : ndarray, optional
            Measured background monitor counts, same shape as `d_int`.

        Returns
        -------
        args_1d : list
            [x0, x1, x2, d1d, n1d, w1d, bkg_1d] for the 1D-mode
            residual/Jacobian calls.
        args_2d : list
            Analogous list for the 2D-mode calls.
        args_3d : list
            Analogous list for the 3D-mode calls.

            None instead of this tuple if the data shape is too small
            (any axis < 3) or any mode's `quick_gaussian` estimate fails.
        """
        d = d_int.copy()
        n = n_int.copy()

        if self.combine_params is not None:
            self.params = self.combine_params.copy()

        if (np.array(d.shape) < 3).any():
            return None

        c0 = self.params["c0"].value
        c1 = self.params["c1"].value
        c2 = self.params["c2"].value

        r0, r1, r2, u0, u1, u2 = self.shape_from_params(self.params)

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

        est1d_0 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_0", b=b, m=m)
        est1d_1 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_1", b=b, m=m)
        est1d_2 = self.quick_gaussian(x0, x1, x2, d, n, mode="1d_2", b=b, m=m)

        if est1d_0 is None or est1d_1 is None or est1d_2 is None:
            return None

        d1d = [d1d_0, d1d_1, d1d_2]
        n1d = [n1d_0, n1d_1, n1d_2]
        w1d = self.uniform_mode_weights(d1d, n1d, self.mode_weights_1d)

        # scale = median(m/n) ≈ 1/k; b/scale converts background to signal units
        bkg_counts_3d = None
        if b is not None and m is not None:
            scale = np.nanmedian(np.where(n > 0, m / n, np.nan))
            if np.isfinite(scale) and scale > 0:
                self._bkg_scale = scale
                bkg_counts_3d = np.where(np.isfinite(b), b / scale, 0.0)
            else:
                self._bkg_scale = None

        bkg_1d = None
        if bkg_counts_3d is not None:
            bkg_1d = [
                self.project_background(bkg_counts_3d, "1d_0"),
                self.project_background(bkg_counts_3d, "1d_1"),
                self.project_background(bkg_counts_3d, "1d_2"),
            ]

        args_1d = [x0, x1, x2, d1d, n1d, w1d, bkg_1d]

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

        est2d_0 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_0", b=b, m=m)
        est2d_1 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_1", b=b, m=m)
        est2d_2 = self.quick_gaussian(x0, x1, x2, d, n, mode="2d_2", b=b, m=m)

        if est2d_0 is None or est2d_1 is None or est2d_2 is None:
            return None

        d2d = [d2d_0, d2d_1, d2d_2]
        n2d = [n2d_0, n2d_1, n2d_2]
        w2d = self.uniform_mode_weights(d2d, n2d, self.mode_weights_2d)

        bkg_2d = None
        if bkg_counts_3d is not None:
            bkg_2d = [
                self.project_background(bkg_counts_3d, "2d_0"),
                self.project_background(bkg_counts_3d, "2d_1"),
                self.project_background(bkg_counts_3d, "2d_2"),
            ]

        args_2d = [x0, x1, x2, d2d, n2d, w2d, bkg_2d]

        d3d, n3d, _, _ = self.profile_project(x0, x1, x2, d, n, wgt, mode="3d")
        w3d = self.uniform_mode_weights([d3d], [n3d], self.mode_weights_3d)[0]

        est3d = self.quick_gaussian(x0, x1, x2, d, n, mode="3d", b=b, m=m)

        if est3d is None:
            return None

        bkg_3d = (
            self.project_background(bkg_counts_3d, "3d")
            if bkg_counts_3d is not None
            else None
        )

        args_3d = [x0, x1, x2, d3d, n3d, w3d, bkg_3d]

        n_iter = 30

        # Annealed multiresolution schedule: the 1D/2D auxiliary views
        # guide the early geometry stages (weighted by alpha_1d/alpha_2d,
        # relative to self.mode_weights_1d/2d), then their influence is
        # wound down so the final stage is (nearly) a 3D-only MAP fit.
        stages = [
            ([False] * 9, 1.0, 1.0),
            ([True] * 3 + [False] * 6, 1.0, 0.25),
            ([True] * 4 + [False] * 5, 1.0, 0.5),
            ([True] * 6 + [False] * 3, 0.5, 1.0),
            ([True] * 9, 0.2, 1.0),
        ]

        for i, (protocol, alpha_1d, alpha_2d) in enumerate(stages):
            args_1d[5] = self.uniform_mode_weights(
                d1d, n1d, alpha_1d * self.mode_weights_1d
            )
            args_2d[5] = self.uniform_mode_weights(
                d2d, n2d, alpha_2d * self.mode_weights_2d
            )

            self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

            if i < len(stages) - 1:
                self.update_adaptive_prior(args_1d, args_2d, args_3d)

        # Final stage: alpha_1d/alpha_2d are wound down to a small residual
        # weight (not exactly zero) so A1d_*/B1d_*/A2d_*/B2d_* keep a
        # well-conditioned Jacobian column instead of going fully
        # unconstrained; the reported geometry and covariance are
        # effectively determined by the 3D likelihood and priors alone.
        args_1d[5] = self.uniform_mode_weights(
            d1d, n1d, 0.01 * self.mode_weights_1d
        )
        args_2d[5] = self.uniform_mode_weights(
            d2d, n2d, 0.01 * self.mode_weights_2d
        )

        self.sweep(args_1d, args_2d, args_3d, [True] * 9, n_iter, report_fit)

        return args_1d, args_2d, args_3d

    def estimate_peak_strength(self):
        """
        Fitted 1D-mode amplitude and background values as float arrays.

        Returns
        -------
        amplitude : ndarray of shape (3,)
            Fitted ``A1d_0, A1d_1, A1d_2`` values.
        background : ndarray of shape (3,)
            Fitted ``B1d_0, B1d_1, B1d_2`` values.
        """
        amplitude, background = self.extract_amplitude_background()

        amplitude = np.array([param for param in amplitude], dtype=float)
        background = np.array([param for param in background], dtype=float)

        return amplitude, background

    def update_adaptive_prior(self, args_1d, args_2d, args_3d):
        """
        Recompute the SNR-adaptive prior widths and update which parameter groups vary.

        Estimates the current 3D I/sigma (`_isig_3d`), converts it to
        prior widths via `prior_widths_from_snr`, and stores them. A
        width of 0 freezes the corresponding group (center, size/shape,
        or orientation) at its reference value (vary=False); otherwise
        it's allowed to vary.

        Parameters
        ----------
        args_1d, args_2d : sequence
            Unused; kept for a uniform call signature.
        args_3d : sequence
            3D-mode residual arguments passed to `_isig_3d` to estimate
            the current SNR.
        """
        snr = self._isig_3d(self.params, args_3d)

        (
            sigma_c,
            sigma_log_s,
            sigma_log_d,
            sigma_rot,
        ) = self.prior_widths_from_snr(snr)

        self.prior_center_sigma = sigma_c
        self.prior_cov_sigma = sigma_log_s
        self.prior_distortion_sigma = sigma_log_d
        self.prior_rot_sigma = sigma_rot

        center_vary = sigma_c > 0.0
        radius_vary = sigma_log_s > 0.0 or sigma_log_d > 0.0
        orient_vary = sigma_rot > 0.0

        for name in ("c0", "c1", "c2"):
            self.params[name].set(vary=center_vary)
            if not center_vary:
                self.params[name].set(value=0.0)

        for name in ("log_size", "shape_1", "shape_2"):
            self.params[name].set(vary=radius_vary)
            if not radius_vary:
                self.params[name].set(value=0.0)

        for name in ("domega_x", "domega_y", "domega_z"):
            self.params[name].set(vary=orient_vary)
            if not orient_vary:
                self.params[name].set(value=0.0)

    def prior_widths_from_snr(self, snr):
        """
        Map an SNR estimate to SNR-adaptive prior widths via `smooth_saturating`.

        Lower SNR yields tighter widths (pulling toward the
        resolution-model prediction); higher SNR relaxes them.

        Parameters
        ----------
        snr : float
            Estimated I/sigma signal-to-noise ratio, e.g. from `_isig_3d`.

        Returns
        -------
        sigma_c : float
            Center prior width, in Mahalanobis units.
        sigma_log_s : float
            Prior width for the mean log-radius scale.
        sigma_log_d : float
            Prior width for the anisotropic log-radius distortion.
        sigma_rot : float
            Prior width for the relative orientation rotation vector, in
            radians.
        """

        x = snr if snr > 1 else 1

        sigma_c = self.smooth_saturating(x, 5.0, 100, 20)
        sigma_log_s = self.smooth_saturating(x, 1.0, 50, 15)
        sigma_log_d = self.smooth_saturating(x, 0.05, 2.0, 10)
        sigma_rot = self.smooth_saturating(x, 0.05, 2.0, 10)

        return sigma_c, sigma_log_s, sigma_log_d, sigma_rot

    def smooth_saturating(
        self,
        x,
        f_min=0.0,
        f_max=1.0,
        x0=2.0,
        p=0.5,
        n=1.0,
        clip_below=True,
    ):
        """
        Smooth function with a minimum at x=1 that monotonically approaches f_max.

        Parameters
        ----------
        x : float or array_like
            Input coordinate, intended for x >= 1.
        f_min, f_max : float, optional
            Value at x=1 and asymptotic value as x -> infinity.
        x0 : float, optional
            Location (> 1) reaching fraction `p` of the way from f_min
            to f_max. Default 2.0.
        p : float, optional
            Fraction of the max reached at x0, in (0, 1). Default 0.5.
        n : float, optional
            Shape parameter (1 = exponential rise, 2 = smoother
            minimum). Default 1.0.
        clip_below : bool, optional
            Clip x < 1 to x = 1. Default True.

        Returns
        -------
        y : float or ndarray
            Smooth saturating function value.

        Raises
        ------
        ValueError
            If x0 <= 1, p not in (0, 1), or n <= 0.
        """
        if x0 <= 1:
            raise ValueError("x0 must be greater than 1.")
        if not (0 < p < 1):
            raise ValueError("p must satisfy 0 < p < 1.")
        if n <= 0:
            raise ValueError("n must be positive.")

        x = np.asarray(x, dtype=float)

        if clip_below:
            x_eff = np.maximum(x, 1.0)
        else:
            x_eff = x

        t = (x_eff - 1.0) / (x0 - 1.0)
        k = -np.log(1.0 - p)

        y = f_min + (f_max - f_min) * (1.0 - np.exp(-k * t**n))

        return y

    def _isig_3d(self, params, args_3d, scales=(0.5, 1.0, 2.0)):
        """
        Quick 3D I/sigma estimate, tried at a few radii scales, keeping the best.

        For each scale, voxels within Mahalanobis distance 1 (radii
        scaled by `scale`) are peak and voxels between 1 and 2**(1/3)
        are background, rescaled by the peak/background voxel-count
        ratio before subtraction. Trying multiple scales avoids
        underestimating SNR when the current radii don't yet match the
        true peak.

        Parameters
        ----------
        params : lmfit.Parameters
            Fit parameters; must contain `c0, c1, c2, log_size, shape_1,
            shape_2, domega_x, domega_y, domega_z`.
        args_3d : sequence
            (x0, x1, x2, d3d, n3d, w3d, bkg_3d) or shorter without
            bkg_3d; bkg_3d, if present, is subtracted from d3d first.
        scales : tuple of float, optional
            Radii scale factors to try. Default (0.5, 1.0, 2.0).

        Returns
        -------
        best_isig : float
            Best I/sigma across `scales`, or -inf if none valid.
        """
        x0, x1, x2, d3d, n3d, *rest = args_3d
        bkg_3d = rest[1] if len(rest) > 1 else None

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value
        r0, r1, r2, u0, u1, u2 = self.shape_from_params(params)

        dx_vec = [x0 - c0, x1 - c1, x2 - c2]

        d = d3d.copy()
        if bkg_3d is not None:
            d = d - bkg_3d

        best_isig = -np.inf

        for scale in scales:
            inv_S = self.inv_S_matrix(
                scale * r0, scale * r1, scale * r2, u0, u1, u2
            )

            d2 = np.einsum("i...,ij,j...->...", dx_vec, inv_S, dx_vec)

            pk = (d2 <= 1) & (n3d > 0)
            bkg = (d2 >= 1) & (d2 < np.cbrt(2)) & (n3d > 0)

            b = np.nansum(d[bkg])

            p = np.nansum(pk)
            q = np.nansum(bkg)

            vol_ratio = p / q if q > 0 else 0

            I = np.nansum(d[pk]) - vol_ratio * b
            sig = np.sqrt(np.nansum(d[pk]) + vol_ratio**2 * b)

            isig = I / sig if sig > 0 else -np.inf

            best_isig = max(best_isig, isig)

        return best_isig

    def sweep(
        self,
        args_1d,
        args_2d,
        args_3d,
        protocol=None,
        n_iter=50,
        report_fit=False,
    ):
        """
        Run one `lmfit`/Levenberg-Marquardt minimization pass, accepted only if SNR improves.

        Optionally sets which parameters vary via `protocol`, minimizes
        `self.residual`/`self.jacobian` with `lmfit.Minimizer`'s
        "leastsq" method, and writes the result back to `self.params`
        only if post-fit 3D I/sigma (`_isig_3d` at scales=[1]) is not
        worse than before.

        Parameters
        ----------
        args_1d : sequence
            Positional arguments for `residual_1d`/`jacobian_1d`.
        args_2d : sequence
            Positional arguments for `residual_2d`/`jacobian_2d`.
        args_3d : sequence
            Positional arguments for `residual_3d`/`jacobian_3d`.
        protocol : sequence of bool, optional
            9-element `vary` flags for c0, c1, c2, log_size, shape_1,
            shape_2, domega_x, domega_y, domega_z. If None, current
            `vary` settings are left unchanged.
        n_iter : int, optional
            Maximum function evaluations for the solver. Default 50.
        report_fit : bool, optional
            Print an `lmfit` fit report after minimizing. Default False.
        """
        if protocol is not None:
            self.params["c0"].set(vary=protocol[0])
            self.params["c1"].set(vary=protocol[1])
            self.params["c2"].set(vary=protocol[2])

            self.params["log_size"].set(vary=protocol[3])
            self.params["shape_1"].set(vary=protocol[4])
            self.params["shape_2"].set(vary=protocol[5])

            self.params["domega_x"].set(vary=protocol[6])
            self.params["domega_y"].set(vary=protocol[7])
            self.params["domega_z"].set(vary=protocol[8])

        out = Minimizer(
            self.residual,
            self.params,
            fcn_args=(args_1d, args_2d, args_3d),
            nan_policy="omit",
        )

        isig_before = self._isig_3d(self.params, args_3d, scales=[1])

        result = out.minimize(
            method="leastsq",
            Dfun=self.jacobian,
            col_deriv=0,
            max_nfev=n_iter,
        )

        if report_fit:
            print(fit_report(result))

        isig_after = self._isig_3d(result.params, args_3d, scales=[1])

        if isig_after >= isig_before:
            self.params = result.params

    def calculate_intensity(self, A, H, r0, r1, r2, u0, u1, u2, mode="3d"):
        """
        Total fitted intensity for a mode: A * gaussian_integral(inv_S, mode).

        Parameters
        ----------
        A : float
            Fitted Gaussian amplitude for this mode.
        H : float
            Unused (reserved).
        r0, r1, r2 : float
            Principal radii along the rotated axes.
        u0, u1, u2 : float
            Orientation rotation vector components.
        mode : str, optional
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".
            Default "3d".

        Returns
        -------
        intensity : float
            Total integrated intensity A * gaussian_integral(...).
        """
        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)
        g = self.gaussian_integral(inv_S, mode)

        return A * g

    def voxels(self, x0, x1, x2):
        """
        Voxel spacing along each axis, from adjacent-element differences of the meshgrid.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.

        Returns
        -------
        dx0, dx1, dx2 : float
            Voxel spacing along each axis.
        """
        return (
            x0[1, 0, 0] - x0[0, 0, 0],
            x1[0, 1, 0] - x1[0, 0, 0],
            x2[0, 0, 1] - x2[0, 0, 0],
        )

    def voxel_volume(self, x0, x1, x2):
        """
        Volume of a single voxel, the product of the per-axis voxel spacings.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.

        Returns
        -------
        volume : float
            Product of the three per-axis voxel spacings from `voxels`.
        """
        return np.prod(self.voxels(x0, x1, x2))

    def project_background(self, bkg, mode):
        """
        Sum a 3D background-count array onto a given mode's projection.

        Parameters
        ----------
        bkg : ndarray
            3D background counts array. Non-finite entries treated as
            zero.
        mode : str
            One of "3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2".

        Returns
        -------
        bkg_proj : ndarray
            `bkg` summed over axes integrated out by `mode` (a copy for
            "3d").
        """
        bkg = np.where(np.isfinite(bkg), bkg, 0.0)
        if mode == "1d_0":
            return np.sum(bkg, axis=(1, 2))
        elif mode == "1d_1":
            return np.sum(bkg, axis=(0, 2))
        elif mode == "1d_2":
            return np.sum(bkg, axis=(0, 1))
        elif mode == "2d_0":
            return np.sum(bkg, axis=0)
        elif mode == "2d_1":
            return np.sum(bkg, axis=1)
        elif mode == "2d_2":
            return np.sum(bkg, axis=2)
        elif mode == "3d":
            return bkg.copy()

    def subtract_profile(self, d, n, perc=30):
        """
        Subtract a per-axis-0-slice background percentile from the raw counts.

        For each index along axis 0, estimates a background level as the
        `perc` percentile of the normalized intensity over axes 1 and 2,
        then subtracts it (rescaled by `n`) from the raw counts.

        Parameters
        ----------
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        perc : float, optional
            Percentile (0-100) used as the background estimate per
            axis-0 slice. Default 30.

        Returns
        -------
        d_sub : ndarray
            Background-subtracted raw counts, same shape as `d`.
        n : ndarray
            Unmodified normalization counts (for a convenient
            ``d, n = subtract_profile(d, n)`` pattern).
        """
        y = np.where(np.isfinite(n) & (n > 0), d / n, np.nan)
        b = np.nanpercentile(y, perc, axis=(1, 2))
        c = b[:, np.newaxis, np.newaxis] * n
        return d - c, n

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
        b=None,
        m=None,
    ):
        """
        Top-level fit entry point: crop data to the valid detector region and fit the ellipsoid.

        Converts counts to normalized intensity, rejects too-small or
        all-invalid input, crops arrays to the bounding box of voxels
        with positive normalization, then calls `estimate_envelope` to
        perform the iterative fit, snapshotting the result via
        `copy_combine` on success. Logs elapsed time and outcome for
        each early-exit path or success.

        Parameters
        ----------
        x0_prof : ndarray
            3D meshgrid coordinates along axis 0, original (not
            peak-centered) frame.
        x1_proj, x2_proj : ndarray
            3D meshgrid coordinates along axes 1 and 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        dx : object
            Unused (voxel spacing is recomputed via `voxels`).
        xmod : float
            Offset subtracted from `x0_prof` for the fit's internal
            axis-0 coordinate; stored as `self._fit_xmod`.
        voxel_weights : ndarray
            Per-voxel fit weight, same shape as `d`.
        b : ndarray, optional
            Measured background counts, same shape as `d`.
        m : ndarray, optional
            Measured background monitor counts, same shape as `d`.

        Returns
        -------
        weights : tuple or None
            The (args_1d, args_2d, args_3d) tuple from `estimate_envelope`
            on success; None on any early-exit condition (too small,
            insufficient valid data, all-invalid crop, an exception in
            `estimate_envelope`, or `estimate_envelope` returning None).
        """
        fit_start = time.perf_counter()

        def log_fit_time(outcome):
            elapsed = time.perf_counter() - fit_start
            print("fit [{}] {:.3f}s".format(outcome, elapsed))

        self._fit_xmod = xmod

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

        # d_val, n_val = self.subtract_profile(d_val, n_val)

        coords = np.argwhere(det_mask)

        i0, i1, i2 = coords.min(axis=0)
        j0, j1, j2 = coords.max(axis=0) + 1

        self._fit_crop = (i0, j0, i1, j1, i2, j2)

        y = y[i0:j0, i1:j1, i2:j2].copy()
        e = e[i0:j0, i1:j1, i2:j2].copy()

        d_val = d_val[i0:j0, i1:j1, i2:j2].copy()
        n_val = n_val[i0:j0, i1:j1, i2:j2].copy()

        if b is not None:
            b = b[i0:j0, i1:j1, i2:j2].copy()
        if m is not None:
            m = m[i0:j0, i1:j1, i2:j2].copy()

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
                x0, x1, x2, d_val, n_val, voxel_weights, b=b, m=m
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
        """
        Build peak/background/combined masks and a p-confidence Gaussian kernel around an ellipsoid.

        The peak mask is the 1-sigma ellipsoid dilated by one voxel; the
        combined mask grows by further dilations (up to 3 attempts)
        until the background shell has at least twice as many voxels as
        the peak. The kernel `y` is a Gaussian at the `p`-confidence
        inverse covariance, independent of the mask-growing logic.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        c : sequence of float
            Peak center (c0, c1, c2).
        S : ndarray of shape (3, 3)
            Ellipsoid covariance-like matrix.
        scale : float
            Unused (recomputed locally from `p` before use).
        p : float, optional
            Confidence-contour probability (0-1) for kernel `y`. Default
            0.997.

        Returns
        -------
        pk : ndarray of bool
            Peak region mask (1-sigma ellipsoid, dilated by one voxel).
        bkg : ndarray of bool
            Background shell mask (mask minus pk).
        mask : ndarray of bool
            Combined peak-plus-background mask after dilation growth.
        y : ndarray
            Gaussian kernel at the p-confidence-contour inverse
            covariance, normalized to integrate to 1.
        """
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
        """
        Simple background-subtracted box-sum intensity from raw counts, with Poisson error.

        Subtracts the background counts, scaled by the peak/background
        voxel-count ratio, from the peak-region raw counts.

        Parameters
        ----------
        counts : ndarray
            Raw observed event counts, 3D array. Infinite values treated
            as NaN.
        pk : ndarray of bool
            Peak region mask, same shape as `counts`.
        bkg : ndarray of bool
            Background region mask, same shape as `counts`.

        Returns
        -------
        intens : float
            Background-subtracted summed raw intensity.
        sig : float
            Propagated Poisson uncertainty; inf if variance is
            non-positive.
        """
        d = counts.copy()
        d[np.isinf(d)] = np.nan

        d_pk = d[pk].copy()
        d_bkg = d[bkg].copy()

        pk_intens = np.nansum(d_pk)
        pk_err = np.sqrt(max(pk_intens, 0.0))

        bkg_intens = np.nansum(d_bkg)
        bkg_err = np.sqrt(max(bkg_intens, 0.0))

        vol_pk = pk.sum()
        vol_bkg = bkg.sum()

        ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

        intens = pk_intens - ratio * bkg_intens
        sig = np.sqrt(pk_err**2 + ratio**2 * bkg_err**2)

        if not sig > 0:
            sig = float("inf")

        return intens, sig

    def extract_intensity(
        self,
        d,
        n,
        pk,
        bkg,
        vol_frac=1.0,
        bkg_meas=None,
    ):
        """
        Normalization-aware background-subtracted box-sum intensity (the I_ell estimator).

        Sums raw counts and normalization over the peak and
        background-shell regions (positive-normalization voxels only).
        If `bkg_meas` is supplied it's used directly (assumed exact)
        instead of the shell; otherwise shell counts are scaled by the
        peak/shell voxel-count ratio, as in `extract_raw_intensity` but
        additionally normalized by monitor counts. Final
        intensity/uncertainty are rescaled by the peak voxel count and
        by `vol_frac` to correct for ellipsoid volume clipped by the
        detector edge.

        Parameters
        ----------
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        pk : ndarray of bool
            Peak region mask, same shape as `d`.
        bkg : ndarray of bool
            Background shell region mask, same shape as `d`.
        vol_frac : float, optional
            Fraction (0-1] of the theoretical ellipsoid volume covered
            by valid voxels; divides `intens`/`sig` to correct edge
            clipping. Default 1.0.
        bkg_meas : ndarray, optional
            Measured/fitted background counts, same shape as `d`, used
            in place of the shell if provided (assumed zero variance).

        Returns
        -------
        intens : float
            Background-subtracted, volume- and edge-corrected intensity.
        sig : float
            Propagated uncertainty; -inf if variance is non-positive.
        b, b_err : float
            Estimated background rate (per monitor count) and its
            uncertainty.
        vol : float
            Number of voxels in the peak region.
        vol_frac : float
            The `vol_frac` argument, echoed back.
        vol_pk : int
            Same as `vol`, as an integer.
        pk_cnts, pk_norm : float
            Summed raw counts and normalization in the peak region (NaN
            if zero).
        bkg_cnts, bkg_norm : float
            Summed background counts/normalization used (from `bkg_meas`
            or the shell).
        ratio : float
            Peak/shell voxel-count ratio used to scale shell counts (1.0
            when `bkg_meas` is used).
        """
        core = pk & (n > 0)
        shell = bkg & (n > 0)

        d_pk = d[core].copy()
        d_bkg = d[shell].copy()

        n_pk = n[core].copy()
        n_bkg = n[shell].copy()

        pk_cnts = np.nansum(d_pk)
        pk_norm = np.nansum(n_pk)

        if pk_cnts == 0.0:
            pk_cnts = float("nan")
        if pk_norm == 0.0:
            pk_norm = float("nan")

        vol_pk = float(core.sum())
        vol_bkg = float(shell.sum())

        vol = vol_pk

        if bkg_meas is not None:
            bkg_cnts = np.nansum(bkg_meas[core])
            bkg_norm = pk_norm

            # bkg_meas is a fitted/smoothed background model, assumed exact
            bkg_variance = 0.0

            b = (
                bkg_cnts / bkg_norm
                if np.isfinite(bkg_norm) and bkg_norm > 0
                else 0.0
            )
            b_err = (
                np.sqrt(bkg_variance) / bkg_norm
                if np.isfinite(bkg_norm) and bkg_norm > 0
                else 0.0
            )

            raw_intens = pk_cnts - bkg_cnts
            raw_sig = np.sqrt(np.abs(pk_cnts) + bkg_variance)

            ratio = 1.0
        else:
            bkg_cnts = np.nansum(d_bkg)
            bkg_norm = np.nansum(n_bkg)

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

            ratio = vol_pk / vol_bkg if vol_bkg > 0 else 0

            raw_intens = pk_cnts - ratio * bkg_cnts
            raw_sig = np.sqrt(np.abs(pk_cnts) + ratio**2 * np.abs(bkg_cnts))

        intens = vol * raw_intens / pk_norm
        sig = vol * raw_sig / pk_norm

        if vol_frac > 0:
            intens /= vol_frac
            sig /= vol_frac

        if not sig > 0:
            sig = float("-inf")

        data_norm = pk_cnts, pk_norm, bkg_cnts, bkg_norm, ratio

        return intens, sig, b, b_err, vol, vol_frac, int(vol_pk), *data_norm

    def matched_filter(self, d, n, kernel, mask, bkg_meas=None):
        """
        Matched-filter Poisson maximum-likelihood fit of an intensity and background level.

        Fits mu = n*(I*kernel + b) + bkg_meas to `d` by maximizing the
        Poisson log-likelihood over I and b via L-BFGS-B, seeded from
        percentile-based estimates. Uncertainty on `I` comes from the
        observed Fisher information at the optimum.

        Parameters
        ----------
        d : ndarray
            Raw observed event counts, restricted to `mask` internally.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        kernel : ndarray
            Matched-filter template (e.g. a normalized Gaussian profile),
            same shape as `d`.
        mask : ndarray of bool
            Voxels to include in the fit.
        bkg_meas : ndarray, optional
            Additional fixed measured background counts, same shape as
            `d`, added to the model.

        Returns
        -------
        I : float
            Fitted matched-filter intensity scale.
        I_err : float
            Uncertainty on `I`; inf if the Fisher information is
            non-positive.
        A : float
            Peak amplitude: `I` scaled by the kernel's max within `mask`.
        b : float
            Fitted background rate.

            Returns (0.0, inf, 0.0, 0.0) if `mask` selects no voxels.
        """
        valid = mask & (n > 0) & np.isfinite(d)

        if not valid.any():
            return 0.0, np.inf, 0.0, 0.0

        dv = d[valid].astype(float)
        nv = n[valid].astype(float)
        pv = np.asarray(kernel, dtype=float)[valid]
        bv = (
            bkg_meas[valid].astype(float)
            if bkg_meas is not None
            else np.zeros_like(dv)
        )

        yv = np.where(nv > 0, dv / nv, 0.0)
        b0 = float(np.clip(np.nanpercentile(yv, 5), 0.0, None))
        A0 = float(np.clip(np.nanpercentile(yv, 95) - b0, 0.0, None))
        p_max = (
            float(np.nanmax(pv)) if pv.size > 0 and np.nanmax(pv) > 0 else 1.0
        )
        I0 = A0 / p_max

        def neg_ll(params):
            I, b = params
            mu = np.clip(nv * (I * pv + b) + bv, 1e-10, None)
            return float(np.sum(mu - dv * np.log(mu)))

        res = scipy.optimize.minimize(
            neg_ll,
            [I0, b0],
            method="L-BFGS-B",
            bounds=[(0.0, None), (0.0, None)],
        )

        I, b = res.x

        mu_fit = np.clip(nv * (I * pv + b) + bv, 1e-10, None)
        fisher_I = float(np.sum((nv * pv) ** 2 / mu_fit))
        I_err = 1.0 / np.sqrt(fisher_I) if fisher_I > 0 else np.inf

        karr = np.asarray(kernel, dtype=float)
        A = (
            I * float(np.nanmax(karr[mask & np.isfinite(karr)]))
            if (mask & np.isfinite(karr)).any()
            else 0.0
        )

        return I, I_err, A, b

    def fitted_profile(
        self, x0, x1, x2, d, n, c, S, p=0.997, eta=0.5, bkg_meas=None
    ):
        """
        Iteratively matched-filter fit a 1D Gaussian profile along axis 0 (the I_prof estimator).

        Alternates `matched_filter` fits of a Gaussian kernel with
        re-estimating sigma from the weighted second moment of the net
        profile, damped by `eta` and floored at the voxel size. Initial
        sigma is sqrt(S[0,0]) rescaled from the 99.7%-contour convention
        using the 1D chi-squared quantile at `p`.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        c : sequence of float
            Peak center (c0, c1, c2).
        S : ndarray of shape (3, 3)
            Ellipsoid covariance-like matrix (99.7%-contour convention).
        p : float, optional
            Confidence-contour probability (0-1) for the initial sigma
            and the peak/background split. Default 0.997.
        eta : float, optional
            Damping factor (0-1) for updating sigma between iterations.
            Default 0.5.
        bkg_meas : ndarray, optional
            Measured/fitted background counts, 3D array, same shape as
            `d`, summed onto the axis-0 profile.

        Returns
        -------
        I, I_err : float
            Fitted matched-filter intensity and its uncertainty.
        b : float
            Fitted background rate.
        x : ndarray
            Axis-0 coordinate relative to the center.
        y_fit : ndarray
            Fitted profile I*kernel + b (plus background offset if
            `bkg_meas` given).
        y, e : ndarray
            Observed normalized intensity profile and its uncertainty.
        """
        scale = np.sqrt(scipy.stats.chi2.ppf(p, df=1))

        c0, c1, c2 = c

        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        C = S.copy()

        sigma = np.sqrt(C[0, 0]) / scale

        weights = np.ones_like(n)

        d_int, n_int, _, _ = self.profile_project(
            x0, x1, x2, d, n, weights, mode="1d_0"
        )

        bkg_1d = None
        bkg_norm = None
        if bkg_meas is not None:
            bkg_1d = np.nansum(bkg_meas, axis=(1, 2))
            bkg_norm = np.where(n_int > 0, bkg_1d / n_int, 0.0)

        y = d_int / n_int
        e = np.sqrt(np.clip(d_int, 0, None) + 1) / n_int

        x = x0[:, 0, 0] - c0

        for i in range(3):
            norm = np.sqrt(2 * np.pi) * sigma

            kernel = np.exp(-0.5 * (x / sigma) ** 2) / norm

            pk = np.abs(x) < scale * sigma
            bkg_mask = np.abs(x) >= scale * sigma

            I, I_err, A, b = self.matched_filter(
                d_int, n_int, kernel, pk | bkg_mask, bkg_1d
            )

            y_net = y - b if bkg_norm is None else y - b - bkg_norm
            w = np.clip(y_net, 0, None)

            wgt = np.nansum(w)

            if wgt > 0:
                sigma_hat = np.clip(np.nansum(w * x**2) / wgt, dx0, sigma)
                sigma = (1 - eta) * sigma + eta * sigma_hat

        y_fit = I * kernel + b
        if bkg_norm is not None:
            y_fit = y_fit + bkg_norm

        return I, I_err, b, x, y_fit, y, e

    def background_profile(self, x0, x1, x2, d, n, b, m):
        """
        Projected 1D background intensity and Poisson uncertainty for each axis.

        Crops inputs to the fit region (`self._fit_crop`) so coordinates
        match `self.best_prof`, rescales the background from monitor to
        signal units by the median ratio m/n, then projects onto each
        1D axis normalized by projected normalization counts.

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates (uncropped).
        d : ndarray
            Raw observed event counts, 3D array (uncropped).
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`
            (uncropped).
        b : ndarray or None
            Measured background counts, same shape as `d` (uncropped).
        m : ndarray or None
            Measured background monitor counts, same shape as `d`
            (uncropped).

        Returns
        -------
        result : list of tuple or None
            3 tuples (x, y_bkg, e_bkg), one per axis (1d_0, 1d_1, 1d_2).
            None if `b` or `m` is None, or the median monitor scale
            factor is not finite and positive.
        """
        if b is None or m is None:
            return None

        # crop to the fit region so x coordinates align with best_prof
        if hasattr(self, "_fit_crop"):
            i0, j0, i1, j1, i2, j2 = self._fit_crop
            x0 = x0[i0:j0, i1:j1, i2:j2]
            x1 = x1[i0:j0, i1:j1, i2:j2]
            x2 = x2[i0:j0, i1:j1, i2:j2]
            d = d[i0:j0, i1:j1, i2:j2]
            n = n[i0:j0, i1:j1, i2:j2]
            b = b[i0:j0, i1:j1, i2:j2]
            m = m[i0:j0, i1:j1, i2:j2]

        scale = np.nanmedian(np.where(n > 0, m / n, np.nan))
        if not (np.isfinite(scale) and scale > 0):
            return None

        w = np.ones_like(n)
        coords = (x0[:, 0, 0], x1[0, :, 0], x2[0, 0, :])
        result = []

        for mode, x_coord in zip(("1d_0", "1d_1", "1d_2"), coords):
            _, n_proj, _, _ = self.profile_project(
                x0, x1, x2, d, n, w, mode=mode
            )
            b_proj = self.project_background(b, mode)

            valid = (n_proj > 0) & np.isfinite(n_proj) & np.isfinite(b_proj)
            denom = np.where(valid, scale * n_proj, np.nan)

            y_bkg = np.where(valid, b_proj / denom, np.nan)
            e_bkg = np.where(
                valid, np.sqrt(np.clip(b_proj, 0, None) + 1) / denom, np.nan
            )

            result.append((x_coord, y_bkg, e_bkg))

        return result

    def integrate(self, x0, x1, x2, d, n, c, S, b=None, m=None):
        """
        Final post-fit intensity integration combining box-sum, ellipsoid, and profile estimators.

        Builds peak/background masks for both the fitted and
        estimated-fit ellipsoids (the latter giving an edge-clipping
        volume-fraction correction, applied to the earlier 3D box-sum
        intensity), then computes and stores the normalization-aware
        box-sum estimator (`extract_intensity`, I_ell), the 1D
        profile-fit estimator (`fitted_profile`, I_prof), a
        chi-squared-inflated matched-filter estimator (`matched_filter`),
        and a raw box-sum estimator for diagnostics. Populates
        `self.intensity`, `self.sigma`, `self.weights`,
        `self.data_norm_fit`, `self.peak_background_mask`,
        `self.integral`, `self.filter`, and `self.info` (all intensity
        estimators and their uncertainties, plus background/volume
        diagnostics).

        Parameters
        ----------
        x0, x1, x2 : ndarray
            3D meshgrid coordinates.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        c : sequence of float
            Fitted peak center (c0, c1, c2).
        S : ndarray of shape (3, 3)
            Fitted ellipsoid covariance-like matrix.
        b : ndarray, optional
            Measured background counts, same shape as `d`.
        m : ndarray, optional
            Measured background monitor counts, same shape as `d`.

        Returns
        -------
        intens : float
            Final normalization-aware box-sum intensity (I_ell), scaled
            by voxel volume.
        sig : float
            Uncertainty on `intens`; inf if not finite.
        """
        dx0, dx1, dx2 = self.voxels(x0, x1, x2)

        d3x = self.voxel_volume(x0, x1, x2)

        d = d.copy()
        n = n.copy()

        d[np.isinf(d)] = np.nan
        n[np.isinf(n)] = np.nan

        bkg_meas = None
        if b is not None and m is not None:
            valid_bkg = np.isfinite(b) & np.isfinite(m) & (m > 0)
            bkg_meas = np.where(valid_bkg, n * b / m, 0.0)

        pk, bkg, mask, kernel = self.peak_roi(x0, x1, x2, c, S, 1)

        c_prior, S_prior = self.estimated_fit

        pk_prior, _, _, kernel_prior = self.peak_roi(
            x0, x1, x2, c_prior, S_prior, 1
        )

        core_prior = pk_prior & (n > 0)

        vol_frac = min(kernel_prior[core_prior].sum() * d3x / 0.997, 1.0)

        # I_3d (from the Gaussian profile fit) only sums voxels actually
        # present on the detector, same edge-clipping issue extract_intensity
        # corrects for below; apply the same correction here for consistency.
        if vol_frac > 0:
            self.intensity[2] /= vol_frac
            self.sigma[2] /= vol_frac

        result = self.extract_intensity(d, n, pk, bkg, vol_frac, bkg_meas)

        intens, sig, b, b_err, N, vol_frac, n_vox, *data_norm = result

        pk_data, pk_norm, bkg_data, bkg_norm, ratio = data_norm

        intens *= d3x
        sig *= d3x

        self.intensity.append(intens)
        self.sigma.append(sig)

        self.weights = (x0[pk], x1[pk], x2[pk]), d[pk].copy()

        y = np.divide(d, n, out=np.full_like(d, np.nan), where=n > 0)
        e = np.divide(
            np.sqrt(np.clip(d, 0, None) + 1),
            n,
            out=np.full_like(d, np.nan),
            where=n > 0,
        )

        intens_raw, sig_raw = self.extract_raw_intensity(d, pk, bkg)

        if not np.isfinite(sig):
            sig = float("inf")

        xye = (x0, x1, x2), (dx0, dx1, dx2), y, e

        params = (intens, sig, b, b_err)

        self.data_norm_fit = xye, params

        self.peak_background_mask = x0, x1, x2, pk, bkg

        result = self.fitted_profile(x0, x1, x2, d, n, c, S, bkg_meas=bkg_meas)

        I, I_err, b_prof, x_prof, y_fit, y_prof, e_prof = result

        self.integral = x_prof, y_fit, y_prof, e_prof

        result = self.matched_filter(d, n, kernel, pk | bkg, bkg_meas)

        I_filt, sig_filt, A_filt, b_filt = result

        chi2_3d = self.reddev[-1]

        if np.isfinite(chi2_3d) and chi2_3d > 1:
            sig_filt = sig_filt * np.sqrt(chi2_3d)
            result = I_filt, sig_filt, A_filt, b_filt

        self.filter = result

        self.intensity.append(I)
        self.sigma.append(I_err)

        I1d = self.intensity[0]
        s1d = self.sigma[0]
        I2d = self.intensity[1]
        s2d = self.sigma[1]

        self.info = {
            "d3x": d3x,
            "bkg": b,
            "bkg_err": b_err,
            "vol_frac": vol_frac,
            "n_vox": n_vox,
            "intens_raw": intens_raw,
            "sig_raw": sig_raw,
            "pk_data": pk_data,
            "pk_norm": pk_norm,
            "bkg_data": bkg_data,
            "bkg_norm": bkg_norm,
            "ratio": ratio,
            "I_1d_0": I1d[0],
            "s_1d_0": s1d[0],
            "I_1d_1": I1d[1],
            "s_1d_1": s1d[1],
            "I_1d_2": I1d[2],
            "s_1d_2": s1d[2],
            "I_2d_0": I2d[0],
            "s_2d_0": s2d[0],
            "I_2d_1": I2d[1],
            "s_2d_1": s2d[1],
            "I_2d_2": I2d[2],
            "s_2d_2": s2d[2],
            "I_3d": self.intensity[2],
            "s_3d": self.sigma[2],
            "I_ell": self.intensity[3],
            "s_ell": self.sigma[3],
            "I_prof": self.intensity[4],
            "s_prof": self.sigma[4],
        }

        return intens, sig
