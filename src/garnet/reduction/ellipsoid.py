import time
import numpy as np

import scipy.optimize
import scipy.spatial.transform
import scipy.stats
import scipy.signal
import scipy.ndimage

from lmfit import Minimizer, Parameters, fit_report

_R_SCALE_1D = np.sqrt(scipy.stats.chi2.ppf(0.997, df=1))
_R_SCALE_2D = np.sqrt(scipy.stats.chi2.ppf(0.997, df=2))
_R_SCALE_3D = np.sqrt(scipy.stats.chi2.ppf(0.997, df=3))

# r0,r1,r2 (and S) are calibrated to the full-3D 99.7% confidence contour
# (_R_SCALE_3D, df=3). A mode's own sub-block of S needs its determinant
# rescaled to that mode's own df=d 99.7%-contour convention.
_MODE_NDIM = {
    "1d_0": 1,
    "1d_1": 1,
    "1d_2": 1,
    "2d_0": 2,
    "2d_1": 2,
    "2d_2": 2,
    "3d": 3,
}
_MODE_R_SCALE = {
    "1d_0": _R_SCALE_1D,
    "1d_1": _R_SCALE_1D,
    "1d_2": _R_SCALE_1D,
    "2d_0": _R_SCALE_2D,
    "2d_1": _R_SCALE_2D,
    "2d_2": _R_SCALE_2D,
    "3d": _R_SCALE_3D,
}


class PeakEllipsoid:
    def __init__(self):
        """
        Initialize a PeakEllipsoid fitter with default fit parameters.

        Sets up an empty `lmfit.Parameters` container, uniform per-mode
        fit weights, and default (unit) resolution-model prior state
        (prior covariance, prior radii/orientation, and prior widths)
        that is later overwritten by `update_estimate`.
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

        self._prior_radii = np.ones(3)
        self._prior_inv_sqrt = np.eye(3)

        self._prior_r = np.ones(3)
        self._prior_u = np.zeros(3)
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

        Centers are initialized to zero and bounded to within half the
        coordinate-array extent along each axis; radii are initialized to
        half the largest axis extent and bounded between twice the
        smallest voxel spacing and twice the per-axis half-extent;
        rotation-vector components are initialized to zero and bounded to
        ``[-pi, pi]``. Also resets `self.combine_params` to ``None``.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        dx : float
            Unused placeholder for a voxel-spacing argument (the voxel
            spacing actually used is recomputed from ``x0, x1, x2``).
        """
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
        """
        Snapshot the current fit parameters into `self.combine_params`.

        Populates `self.combine_params` with a copy of `self.params`, so
        that a subsequent `estimate_envelope` call can resume from this
        state instead of the constraint defaults.
        """
        self.combine_params = self.params.copy()

    def update_estimate(self, shape):
        """
        Seed `self.params` and the resolution-model prior from an initial ellipsoid estimate.

        Centers are reset to the origin (since coordinates are peak-centered),
        radii are set to the given values (clipped to twice the lower bound
        if outside the parameter bounds), and the orientation is derived
        from the given axis vectors (flipping the third axis if needed to
        keep `U` a proper rotation). Also (re)initializes the SNR-adaptive
        prior state (`self.prior_cov`, `self._prior_radii`,
        `self._prior_inv_sqrt`, `self._prior_r`, `self._prior_u`,
        `self._prior_U`) and resets the prior widths
        (`self.prior_center_sigma`, `self.prior_cov_sigma`,
        `self.prior_distortion_sigma`, `self.prior_rot_sigma`) to their
        default starting values. Also sets `self.estimated_fit`.

        Parameters
        ----------
        shape : tuple
            ``(c0, c1, c2, r0, r1, r2, v0, v1, v2)`` where ``c0, c1, c2``
            are the (unused) center coordinates, ``r0, r1, r2`` are the
            ellipsoid principal radii, and ``v0, v1, v2`` are the
            corresponding orthonormal principal axis vectors (each
            ndarray of shape (3,)).
        """
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

        self.prior_cov = self.S_matrix(r0, r1, r2, u0, u1, u2)

        r_sq, U0 = np.linalg.eigh(self.prior_cov)
        self._prior_radii = np.sqrt(np.maximum(r_sq, 1e-12))
        self._prior_inv_sqrt = U0 @ np.diag(1.0 / self._prior_radii) @ U0.T

        # Predicted radii/orientation, kept axis-matched to r0,r1,r2/u0,u1,u2
        # for the per-axis log-radius and relative-rotation prior terms.
        self._prior_r = np.array([r0, r1, r2], dtype=float)
        self._prior_u = np.array([u0, u1, u2], dtype=float)
        self._prior_U = self.U_matrix(u0, u1, u2)

        self.prior_center_sigma = 1.0
        self.prior_cov_sigma = 0.3
        self.prior_distortion_sigma = 0.1
        self.prior_rot_sigma = 0.1

    def omega_deriv_u(self, u0, u1, u2, delta=1e-6):
        """
        Finite-difference derivative of the prior-relative rotation vector.

        Computes d(omega)/d(u_k) for k=0,1,2 by central differences, where
        omega is the rotation vector of ``U0^T @ U(u0,u1,u2)`` (the
        orientation relative to the predicted resolution-model
        orientation `self._prior_U`).

        Parameters
        ----------
        u0 : float
            First component of the current rotation vector.
        u1 : float
            Second component of the current rotation vector.
        u2 : float
            Third component of the current rotation vector.
        delta : float, optional
            Finite-difference step size. Default is 1e-6.

        Returns
        -------
        domega0 : ndarray of shape (3,)
            d(omega)/d(u0).
        domega1 : ndarray of shape (3,)
            d(omega)/d(u1).
        domega2 : ndarray of shape (3,)
            d(omega)/d(u2).
        """

        def omega_at(uu0, uu1, uu2):
            U = self.U_matrix(uu0, uu1, uu2)
            R_rel = self._prior_U.T @ U
            return scipy.spatial.transform.Rotation.from_matrix(
                R_rel
            ).as_rotvec()

        domega0 = omega_at(u0 + delta, u1, u2) - omega_at(u0 - delta, u1, u2)
        domega1 = omega_at(u0, u1 + delta, u2) - omega_at(u0, u1 - delta, u2)
        domega2 = omega_at(u0, u1, u2 + delta) - omega_at(u0, u1, u2 - delta)

        return (
            0.5 * domega0 / delta,
            0.5 * domega1 / delta,
            0.5 * domega2 / delta,
        )

    def prior_residual(self, params):
        """
        Regularization residual terms pulling the fit toward the resolution-model prior.

        Builds the stacked whitened residual vector for four prior
        components: center (Mahalanobis distance of the center from the
        origin under the prior covariance), radius scale (mean log-radius
        deviation from the predicted radii), radius distortion (per-axis
        log-radius deviation after removing the mean), and orientation
        (rotation-vector distance from the predicted orientation). Each
        component is divided by its current SNR-adaptive prior sigma
        (`self.prior_center_sigma`, `self.prior_cov_sigma`,
        `self.prior_distortion_sigma`, `self.prior_rot_sigma`).

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``c0, c1, c2, r0, r1,
            r2, u0, u1, u2``.

        Returns
        -------
        terms : ndarray of shape (10,)
            Concatenated residuals: 3 center + 1 scale + 3 distortion +
            3 orientation components.
        """
        terms = []

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value
        c = np.array([c0, c1, c2])

        # Center: P_center = || Sigma0^{-1/2} (c - c0) / sigma_c ||^2
        u_mu = _R_SCALE_3D * self._prior_inv_sqrt @ c
        terms += (u_mu / self.prior_center_sigma).tolist()

        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        # Radius decomposition: per-axis log-radius vs. predicted radii
        delta = np.log(np.array([r0, r1, r2])) - np.log(self._prior_r)
        alpha = np.mean(delta)
        d = delta - alpha

        # Scale residual: P_scale = (alpha / sigma_log_s)^2
        terms.append(alpha / self.prior_cov_sigma)

        # Distortion residuals: P_distortion = sum (d_i / sigma_log_d)^2
        terms += (d / self.prior_distortion_sigma).tolist()

        # Orientation: relative rotation vector from U0 to U
        U = self.U_matrix(u0, u1, u2)
        R_rel = self._prior_U.T @ U
        omega = scipy.spatial.transform.Rotation.from_matrix(R_rel).as_rotvec()

        # Orientation residual: P_rotation = || omega / sigma_rot ||^2
        terms += (omega / self.prior_rot_sigma).tolist()

        return np.asarray(terms, dtype=float)

    def prior_jacobian(self, params):
        """
        Analytic Jacobian of `prior_residual` with respect to the varying fit parameters.

        Residual ordering matches `prior_residual`: 3 center + 1 scale +
        3 distortion + 3 orientation = 10 columns. Center derivatives are
        exact (linear in `c`); radius-decomposition derivatives are exact
        (analytic d(log r)/dr); orientation derivatives use the
        finite-difference helper `omega_deriv_u`.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``c0, c1, c2, r0, r1,
            r2, u0, u1, u2``.

        Returns
        -------
        jac : ndarray of shape (n_vary, 10)
            Jacobian rows restricted to parameters with ``vary=True``, one
            row per varying parameter and one column per prior residual
            term.
        """
        # Residuals: [r_c0, r_c1, r_c2, r_alpha, r_d0, r_d1, r_d2, r_om0, r_om1, r_om2]
        # 3 center + 1 scale + 3 distortion + 3 orientation = 10 columns
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), 10), dtype=float)

        # Center: ∂r_c_i/∂c_j = R * (S₀^{-1/2})_{i,j} / σ_c
        center_scale = _R_SCALE_3D / self.prior_center_sigma
        for j, cname in enumerate(["c0", "c1", "c2"]):
            idx = params_list.index(cname)
            jac[idx, :3] = center_scale * self._prior_inv_sqrt[:, j]

        # Radius decomposition: d(alpha)/d(r_j) = (1/3)/r_j,
        # d(d_i)/d(r_j) = (delta_ij - 1/3)/r_j
        r = np.array(
            [params["r0"].value, params["r1"].value, params["r2"].value]
        )
        eta_scale = 1.0 / self.prior_cov_sigma
        cov_d = 1.0 / self.prior_distortion_sigma
        centering = np.eye(3) - np.full((3, 3), 1.0 / 3.0)

        for j, rname in enumerate(["r0", "r1", "r2"]):
            idx = params_list.index(rname)
            dlogr = 1.0 / r[j]
            jac[idx, 3] = eta_scale * dlogr / 3.0
            jac[idx, 4:7] = cov_d * centering[:, j] * dlogr

        # Orientation: finite-difference d(omega)/d(u_k)
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value
        domega = self.omega_deriv_u(u0, u1, u2)
        rot_scale = 1.0 / self.prior_rot_sigma

        for k, uname in enumerate(["u0", "u1", "u2"]):
            idx = params_list.index(uname)
            jac[idx, 7:10] = rot_scale * domega[k]

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]
        return jac[ind]

    #: Every mode with its own A<mode>/B<mode> amplitude/background pair,
    #: hence its own volume penalty term.
    _PENALTY_MODES = (
        "1d_0",
        "1d_1",
        "1d_2",
        "2d_0",
        "2d_1",
        "2d_2",
        "3d",
    )

    def _mode_S_det(self, r0, r1, r2, u0, u1, u2, mode):
        """
        Determinant of the covariance sub-block S for a mode, on that mode's own df=d contour scale.

        The sub-block is the plain sub-block of
        ``S = U diag(r0^2, r1^2, r2^2) U^T`` selected the same way
        `ellipsoid_covariance` selects `inv_S` sub-blocks per mode (no
        chi-squared scale factor applied at this stage). Since `S` is
        generally not diagonal in the lab frame, 1D/2D sub-block
        determinants (and hence this quantity) do depend on orientation,
        unlike the full 3x3 determinant. For ``mode="3d"``, `S` is
        diagonal with entries ``r0^2, r1^2, r2^2`` in its own (rotated)
        local frame, and the determinant is rotation-invariant, so
        ``det(S) = (r0*r1*r2)**2`` is computed directly without building
        `U` or the full matrix product.

        `r0, r1, r2` (and therefore `S`) are calibrated to the full-3D
        99.7% confidence contour (`_R_SCALE_3D`, chi-squared with
        ``df=3``). A mode of dimensionality ``d = _MODE_NDIM[mode]`` (1,
        2, or 3) has its own ``df=d`` chi-squared contour convention, so
        the raw sub-block determinant is rescaled by
        ``(_MODE_R_SCALE[mode] / _R_SCALE_3D) ** (2*d)`` to convert it
        from the df=3 scale to the mode's own df=d scale. This factor is
        exactly 1 for ``mode="3d"`` (already on the df=3 scale) and
        greater than 1 for 1D/2D modes (a df=d contour at the same
        confidence level is narrower than the corresponding df=3 slice).

        Parameters
        ----------
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.

        Returns
        -------
        det_S : float
            df=d-rescaled determinant of the mode's covariance sub-block
            (a plain scalar value for 1D modes, where the "sub-block" is
            a single diagonal entry of `S`).
        """

        if mode == "3d":
            det_S = (r0 * r1 * r2) ** 2
        else:
            S = self.S_matrix(r0, r1, r2, u0, u1, u2)

            if mode == "2d_0":
                S_mode = S[1:, 1:]
            elif mode == "2d_1":
                S_mode = S[0::2, 0::2]
            elif mode == "2d_2":
                S_mode = S[:2, :2]
            elif mode == "1d_0":
                S_mode = S[0, 0]
            elif mode == "1d_1":
                S_mode = S[1, 1]
            elif mode == "1d_2":
                S_mode = S[2, 2]
            else:
                raise ValueError("Unknown mode {}".format(mode))

            det_S = np.linalg.det(S_mode) if np.ndim(S_mode) == 2 else S_mode

        d = _MODE_NDIM[mode]
        rescale = (_MODE_R_SCALE[mode] / _R_SCALE_3D) ** (2 * d)

        return det_S * rescale

    def volume_penalty_residual(self, params, mode, eps=1e-12):
        """
        Constant penalty term minimizing sqrt(B) / (pi^(d/4) * A * det(S)^(1/4)) for one mode.

        A, B are the given mode's amplitude/background, and S is that
        mode's own covariance sub-block (see `_mode_S_det`) — for the
        ``"3d"`` mode, ``det(S)^(1/4)`` reduces to ``sqrt(r0*r1*r2)``. The
        ``pi^(d/4)`` prefactor (``d`` = mode dimensionality) matches the
        ``(2*pi)^(d/2)`` volume element of the Gaussian integral
        normalization used by `gaussian_integral`, up to the factor of 2.
        Unlike the SNR-adaptive prior widths, this term is always active
        at a fixed strength, discouraging fits with weak amplitude and/or
        a small volume relative to the background level.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``A<mode>, B<mode>, r0,
            r1, r2, u0, u1, u2``.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
        eps : float, optional
            Small positive floor used to avoid division by zero in `A`
            and the volume term. Default is 1e-12.

        Returns
        -------
        penalty : ndarray of shape (1,)
            The single-element penalty residual
            ``sqrt(B) / (pi**(d/4) * A * det(S)^(1/4))`` for this mode.
        """

        A = max(params["A" + mode].value, eps)
        B = max(params["B" + mode].value, 0.0)
        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

        d = _MODE_NDIM[mode]
        det_S = self._mode_S_det(r0, r1, r2, u0, u1, u2, mode)
        vol = max(det_S, eps) ** 0.25

        penalty = np.sqrt(B) / (np.pi ** (d / 4.0) * A * vol)

        return np.array([penalty])

    def volume_penalty_jacobian(self, params, mode, eps=1e-12, delta=1e-6):
        """
        Jacobian of `volume_penalty_residual` for one mode w.r.t. varying fit parameters.

        The A/B derivatives are analytic; the r0,r1,r2,u0,u1,u2
        derivatives are central finite differences of ``det(S)^(1/4)``,
        since 1D/2D sub-block determinants (unlike the full 3x3
        determinant) don't have a simple closed form in terms of the
        radii alone.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``A<mode>, B<mode>, r0,
            r1, r2, u0, u1, u2``.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
        eps : float, optional
            Small positive floor matching `volume_penalty_residual`.
            Default is 1e-12.
        delta : float, optional
            Central finite-difference step size for the shape
            derivatives. Default is 1e-6.

        Returns
        -------
        jac : ndarray of shape (n_vary, 1)
            Jacobian column restricted to parameters with ``vary=True``,
            one row per varying parameter.
        """
        params_list = [name for name, _ in params.items()]
        jac = np.zeros((len(params_list), 1), dtype=float)

        A = max(params["A" + mode].value, eps)
        B = max(params["B" + mode].value, 0.0)
        r = [params["r0"].value, params["r1"].value, params["r2"].value]
        u = [params["u0"].value, params["u1"].value, params["u2"].value]

        d = _MODE_NDIM[mode]
        pi_factor = np.pi ** (d / 4.0)

        def vol(rr, uu):
            det_S = self._mode_S_det(*rr, *uu, mode)
            return max(det_S, eps) ** 0.25

        vol0 = vol(r, u)
        penalty = np.sqrt(B) / (pi_factor * A * vol0)

        aname, bname = "A" + mode, "B" + mode
        if aname in params_list:
            jac[params_list.index(aname), 0] = -penalty / A
        if bname in params_list and B > 0:
            jac[params_list.index(bname), 0] = penalty / (2.0 * B)

        shape_names = ["r0", "r1", "r2", "u0", "u1", "u2"]
        for i, name in enumerate(shape_names):
            if name not in params_list:
                continue
            r_p, u_p = list(r), list(u)
            r_m, u_m = list(r), list(u)
            if i < 3:
                r_p[i] += delta
                r_m[i] -= delta
            else:
                u_p[i - 3] += delta
                u_m[i - 3] -= delta
            dvol = (vol(r_p, u_p) - vol(r_m, u_m)) / (2.0 * delta)
            jac[params_list.index(name), 0] = -penalty / vol0 * dvol

        ind = [i for i, (_, par) in enumerate(params.items()) if par.vary]
        return jac[ind]

    def S_matrix(self, r0, r1, r2, u0, u1, u2):
        """
        Build the ellipsoid covariance-like matrix S = U diag(r0^2, r1^2, r2^2) U^T.

        Parameters
        ----------
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.

        Returns
        -------
        S : ndarray of shape (3, 3)
            The ellipsoid covariance-like matrix.
        """
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([r0**2, r1**2, r2**2])

        S = np.dot(np.dot(U, V), U.T)

        return S

    def inv_S_matrix(self, r0, r1, r2, u0, u1, u2):
        """
        Build the inverse ellipsoid covariance-like matrix inv_S = U diag(1/r0^2, 1/r1^2, 1/r2^2) U^T.

        Parameters
        ----------
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.

        Returns
        -------
        inv_S : ndarray of shape (3, 3)
            The inverse ellipsoid covariance-like matrix.
        """
        U = self.U_matrix(u0, u1, u2)

        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        inv_S = np.dot(np.dot(U, V), U.T)

        return inv_S

    def U_matrix(self, u0, u1, u2):
        """
        Build the ellipsoid orientation rotation matrix from a rotation vector.

        Parameters
        ----------
        u0 : float
            First component of the axis-angle rotation vector.
        u1 : float
            Second component of the axis-angle rotation vector.
        u2 : float
            Third component of the axis-angle rotation vector.

        Returns
        -------
        U : ndarray of shape (3, 3)
            Proper orthogonal rotation matrix, per
            ``scipy.spatial.transform.Rotation.from_rotvec``.
        """
        u = np.array([u0, u1, u2])

        U = scipy.spatial.transform.Rotation.from_rotvec(u).as_matrix()

        return U

    def det_S(self, r0, r1, r2, u0, u1, u2):
        """
        Determinant of the ellipsoid covariance-like matrix S.

        `S` is diagonal with entries ``r0^2, r1^2, r2^2`` in its own
        (rotated) local frame, and the determinant is rotation-invariant,
        so this is computed directly from the radii without building `U`
        or the full matrix product.

        Parameters
        ----------
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            Unused (`det(S)` does not depend on orientation).
        u1 : float
            Unused (`det(S)` does not depend on orientation).
        u2 : float
            Unused (`det(S)` does not depend on orientation).

        Returns
        -------
        det : float
            ``det(S)``, equal to ``(r0*r1*r2)**2``.
        """
        return (r0 * r1 * r2) ** 2

    def centroid_inverse_covariance(self, c0, c1, c2, r0, r1, r2, u0, u1, u2):
        """
        Package the peak center and inverse ellipsoid covariance from scalar parameters.

        Parameters
        ----------
        c0 : float
            Peak center coordinate along axis 0.
        c1 : float
            Peak center coordinate along axis 1.
        c2 : float
            Peak center coordinate along axis 2.
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.

        Returns
        -------
        c : ndarray of shape (3,)
            The peak center ``[c0, c1, c2]``.
        inv_S : ndarray of shape (3, 3)
            The inverse ellipsoid covariance-like matrix.
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
            Raw observed event counts. Modified in place (masked to NaN
            where invalid).
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as
            `d`. Modified in place (masked to NaN where invalid).
        v : ndarray
            Variance proxy (e.g. clipped counts), same shape as `d`.
            Modified in place (masked to NaN where invalid).
        rel_err : float, optional
            Percentile of `v` added inside the square root when
            estimating the uncertainty, used as a variance floor.
            Default is 30.

        Returns
        -------
        y_int : ndarray
            Normalized intensity ``d / n``.
        e_int : ndarray
            Estimated uncertainty
            ``sqrt(v + percentile(v, rel_err)) / n``.
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

        For `"1d_i"` modes, sums/averages over the two axes other than
        `i`; for `"2d_i"` modes, sums/averages over axis `i` only; for
        `"3d"`, returns copies of the full arrays. Normalization counts
        are averaged (not summed) and rescaled by the voxel size of the
        integrated-out axes so that `n_int` remains a rate (as needed for
        `d_int / n_int` to be a mean intensity). Voxels with invalid
        input (non-finite or non-positive `n`) are excluded before
        projecting, and projected voxels with invalid resulting
        normalization are masked to NaN in all four outputs.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array matching `x0`'s shape.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : ndarray
            Per-voxel fit weight, same shape as `d`.
        mode : str, optional
            Projection mode: one of ``"1d_0", "1d_1", "1d_2", "2d_0",
            "2d_1", "2d_2", "3d"``. Default is ``"3d"``.

        Returns
        -------
        d_int : ndarray
            Projected (summed) raw counts.
        n_int : ndarray
            Projected (rate-averaged) normalization counts.
        v_int : ndarray
            Projected (summed) variance proxy from ``clip(d, 0, None)``.
        w_int : ndarray
            Projected (averaged) fit weight.

        Raises
        ------
        ValueError
            If `mode` is not one of the recognized mode strings.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : ndarray
            Per-voxel fit weight, same shape as `d`.
        mode : str, optional
            Projection mode passed to `profile_project`. Default is
            ``"3d"``.

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
        Rescale (a projection of) the inverse covariance to a given confidence-contour level.

        `r0, r1, r2` are on the 99.7%-contour convention (see
        `_R_SCALE_3D`), so `inv_S` itself corresponds to the
        `perc=99.7` contour in 3D; this method converts `inv_S` (or a
        lower-dimensional marginal/slice of it, depending on `mode`) to
        the inverse-variance matrix for an arbitrary percentile `perc`
        of the corresponding chi-squared distribution.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``, selecting which
            marginal/slice of `inv_S` to use. Default is ``"3d"``.
        perc : float, optional
            Target confidence-contour percentile (0-100). Default is
            99.7.

        Returns
        -------
        inv_var : ndarray or float
            For `"3d"` and `"2d_*"` modes, the rescaled inverse-variance
            matrix (ndarray of shape (3, 3) or (2, 2) respectively); for
            `"1d_*"` modes, the rescaled scalar inverse variance (float).
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
            If ``None``, a weight of 1.0 everywhere is returned. If a
            scalar, it is broadcast to `target`'s shape. If array-like,
            it is coerced to a float ndarray as-is. Default is ``None``.

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
            Sequence of arrays whose shapes the returned weights should
            match.
        weights : None or sequence, optional
            If ``None``, each target gets a weight of ``None`` (resolved
            to all-ones by `coerce_weight`). Otherwise a sequence of the
            same length as `targets`, each element passed to
            `coerce_weight`. Default is ``None``.

        Returns
        -------
        weights : list of ndarray
            One coerced weight array per target, in the same order as
            `targets`.
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

        Each returned weight array is filled with the constant
        ``sqrt(val) / sqrt(n_valid)``, where `n_valid` is the total
        number of finite, positive-uncertainty points summed across all
        `ys`/`es` pairs. This gives each mode a total weight contribution
        proportional to `val`, independent of how many valid points it
        has.

        Parameters
        ----------
        ys : sequence of ndarray
            Per-mode intensity arrays (only used, via `es`, to count
            valid points).
        es : sequence of ndarray
            Per-mode uncertainty arrays, one per element of `ys`.
        val : float, optional
            Overall target weight scale for this group of modes. Default
            is 1.0.

        Returns
        -------
        weights : list of ndarray
            One uniform weight array per element of `es`, matching its
            shape.
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

        Restricts to voxels within a Mahalanobis distance
        ``d2 <= 2**(2/k)`` of the center (`k` = dimensionality of `mode`),
        approximates the underlying raw counts and normalization from the
        normalized intensity/uncertainty (`y`, `e`) via the Poisson error
        model, then computes the reduced Poisson deviance
        ``2*(mu - d + d*log(d/mu))`` between the approximated observed
        counts and the model counts, divided by the degrees of freedom.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        y_fit : ndarray
            Fitted (model) normalized intensity, same shape as `y`.
        y : ndarray
            Observed normalized intensity for this mode.
        e : ndarray
            Uncertainty on `y`, same shape as `y`.
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        reduced_deviance : float
            The Poisson deviance summed over included voxels, divided by
            the degrees of freedom, or ``inf`` if the degrees of freedom
            are not positive.
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
        Background-subtracted box-sum intensity and uncertainty within the 1-sigma peak region.

        Sums the background-subtracted normalized intensity over voxels
        within Mahalanobis distance 1 of the center (``d2 <= 1``),
        scaled by the voxel volume of the mode's non-integrated axes.
        The background level and its uncertainty are taken from the
        fitted ``B<mode>`` parameter. This is the `I_1d`, `I_2d`, or
        `I_3d` box-sum estimator (depending on `mode`), distinct from
        `I_ell` and `I_prof`.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        y_fit : ndarray
            Fitted (model) normalized intensity (unused except to match
            the calling signature shared with `poisson_deviance_fit`).
        y : ndarray
            Observed normalized intensity for this mode.
        e : ndarray
            Uncertainty on `y`, same shape as `y`.
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.
        bkg_offset : ndarray, optional
            Additional per-voxel background-rate offset (e.g. from a
            measured background monitor) to subtract along with the
            fitted background level `B<mode>`. Default is ``None``.

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
        Evaluate the unnormalized (peak-value-1) Gaussian profile for a given mode.

        The inverse covariance used is `inv_S` rescaled from the
        99.7%-contour convention to a unit-variance (1-sigma) convention
        via `ellipsoid_covariance`, so that the returned Gaussian has
        value 1 at the center and falls to ``exp(-0.5)`` at Mahalanobis
        distance 1.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        g : ndarray
            Unnormalized Gaussian profile evaluated on the mode's
            coordinate grid, with the same shape as the corresponding
            projected axis/axes of `x0`, `x1`, `x2`.
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
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.

        Returns
        -------
        dinv_S0 : ndarray of shape (3, 3)
            d(inv_S)/d(r0).
        dinv_S1 : ndarray of shape (3, 3)
            d(inv_S)/d(r1).
        dinv_S2 : ndarray of shape (3, 3)
            d(inv_S)/d(r2).
        """
        U = self.U_matrix(u0, u1, u2)

        dinv_S0 = U @ np.diag([-2 / r0**3, 0, 0]) @ U.T
        dinv_S1 = U @ np.diag([0, -2 / r1**3, 0]) @ U.T
        dinv_S2 = U @ np.diag([0, 0, -2 / r2**3]) @ U.T

        return dinv_S0, dinv_S1, dinv_S2

    def inv_S_deriv_u(self, r0, r1, r2, u0, u1, u2):
        """
        Analytic derivatives of `inv_S` with respect to each rotation-vector component u0, u1, u2.

        Uses the finite-difference orientation derivatives from
        `U_deriv_u` combined with the product rule for
        ``inv_S = U @ V @ U.T``.

        Parameters
        ----------
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.

        Returns
        -------
        dinv_S0 : ndarray of shape (3, 3)
            d(inv_S)/d(u0).
        dinv_S1 : ndarray of shape (3, 3)
            d(inv_S)/d(u1).
        dinv_S2 : ndarray of shape (3, 3)
            d(inv_S)/d(u2).
        """
        V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

        U = self.U_matrix(u0, u1, u2)
        dU0, dU1, dU2 = self.U_deriv_u(u0, u1, u2)

        dinv_S0 = dU0 @ V @ U.T + U @ V @ dU0.T
        dinv_S1 = dU1 @ V @ U.T + U @ V @ dU1.T
        dinv_S2 = dU2 @ V @ U.T + U @ V @ dU2.T

        return dinv_S0, dinv_S1, dinv_S2

    def U_deriv_u(self, u0, u1, u2, delta=1e-6):
        """
        Finite-difference derivatives of the rotation matrix U with respect to u0, u1, u2.

        Parameters
        ----------
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.
        delta : float, optional
            Finite-difference step size. Default is 1e-6.

        Returns
        -------
        dU0 : ndarray of shape (3, 3)
            d(U)/d(u0), by central difference.
        dU1 : ndarray of shape (3, 3)
            d(U)/d(u1), by central difference.
        dU2 : ndarray of shape (3, 3)
            d(U)/d(u2), by central difference.
        """
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
        """
        Normalization integral of the unit-peak Gaussian for a given mode.

        Computes ``sqrt((2*pi)**k * det)``, where `k` is the
        dimensionality of `mode` and `det` is the determinant (or scalar
        value, for 1D modes) of the 1-sigma covariance matrix
        corresponding to `inv_S`. Multiplying this by the peak amplitude
        `A` gives the volume under the normalized Gaussian, i.e. the
        total fitted intensity for that mode.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

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
        Derivative of `gaussian_integral` with respect to the shape (r0, r1, r2 or u0, u1, u2) parameters.

        Uses the Jacobi's-formula identity
        ``d(det(M))/dx = det(M) * trace(M^-1 dM/dx)`` applied to the
        1-sigma covariance implied by `inv_S`, restricted per `mode` to
        the relevant marginal/slice.

        Parameters
        ----------
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        d_inv_S : sequence of ndarray
            Three derivatives of `inv_S`, e.g. from `inv_S_deriv_r` or
            `inv_S_deriv_u`, each of shape (3, 3).
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        dg : ndarray of shape (3,)
            Derivative of `gaussian_integral` with respect to each of the
            three shape parameters corresponding to `d_inv_S`; entries
            for axes not used by `mode` are zero.
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
        """
        Derivative of the unnormalized `gaussian` profile with respect to the center c0, c1, c2.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        dg : ndarray
            Stacked derivatives ``[dg/dc0, dg/dc1, dg/dc2]`` evaluated on
            the mode's coordinate grid; entries for axes not used by
            `mode` are zero, with shape matching the corresponding
            projected axis/axes of `x0`, `x1`, `x2`.
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

        Given the derivatives of `inv_S` with respect to a group of
        three shape parameters (radii via `inv_S_deriv_r`, or
        orientation via `inv_S_deriv_u`), returns the corresponding
        derivatives of the Gaussian profile itself.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix (99.7%-contour
            convention).
        d_inv_S : sequence of ndarray
            Three derivatives of `inv_S` with respect to the shape
            parameter group (e.g. from `inv_S_deriv_r` or
            `inv_S_deriv_u`), each of shape (3, 3).
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        dg : ndarray
            Stacked derivatives of the Gaussian profile with respect to
            each of the three shape parameters corresponding to
            `d_inv_S`, evaluated on the mode's coordinate grid; entries
            for axes not used by `mode` are zero.
        """
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
        """
        Convert raw counts and normalization into normalized intensity and its uncertainty.

        Uncertainty uses the ``sqrt(d + 1)`` Poisson approximation
        (avoiding zero uncertainty when ``d == 0``).

        Parameters
        ----------
        d : ndarray
            Raw observed event counts.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.

        Returns
        -------
        y : ndarray
            Normalized intensity ``d / n``, NaN where ``n <= 0``.
        e : ndarray
            Uncertainty ``sqrt(clip(d, 0, None) + 1) / n``, NaN where
            ``n <= 0``.
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
        Signed Poisson deviance residual and its derivative factor with respect to mu.

        Computes the signed square-root deviance residual
        ``r = sign(mu - d) * sqrt(2*(mu - d + d*log(d/mu)))`` (0 when
        ``d == 0``) for each valid voxel, together with the factor
        ``dr/dmu`` needed to chain-rule the Jacobian of a residual-based
        least-squares fit against a Poisson likelihood
        (`jacobian_mode_poisson`). Near `mu == d`, the residual and its
        derivative are evaluated via the limiting expression to avoid a
        0/0 division. For `mu < 0` (unphysical during optimization), the
        deviance is evaluated at a small positive floor and linearly
        extrapolated to the true (negative) `mu`, giving a large, correctly
        signed residual with a bounded Jacobian instead of diverging.
        Weights are applied multiplicatively to both `r` and `dr/dmu`.

        Parameters
        ----------
        d : ndarray
            Observed (approximate) raw counts.
        mu : ndarray
            Model (expected) counts, same shape as `d`.
        w : ndarray, optional
            Per-voxel weight applied to the residual and its derivative.
            If ``None``, weights of 1 are used. Default is ``None``.
        eps : float, optional
            Small positive floor used when clipping `mu` for the
            non-negative branch. Default is 1e-12.

        Returns
        -------
        r : ndarray
            Signed weighted deviance residual, same shape as `mu`, NaN
            where invalid.
        dr_dmu : ndarray
            Weighted derivative of `r` with respect to `mu`, same shape
            as `mu`, NaN where invalid.
        valid : ndarray of bool
            Mask of voxels where `d`, `mu`, and `w` are finite and
            ``d >= 0``.
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
        Compute the Poisson model counts mu = n*(A*gaussian + B) [+ bkg] for one mode.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``A<mode>`` and
            ``B<mode>``.
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts for this mode (unused in the
            computation but kept for a uniform call signature).
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : ndarray
            Per-voxel fit weight for this mode (unused in the
            computation but kept for a uniform call signature).
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
        bkg : ndarray, optional
            Additional fixed background counts to add to the model.
            Default is ``None``.

        Returns
        -------
        mu : ndarray
            Model (expected) counts.
        g : ndarray
            Unnormalized Gaussian profile used in the model.
        A : float
            Current value of the ``A<mode>`` amplitude parameter.
        B : float
            Current value of the ``B<mode>`` background parameter.
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
            Current fit parameters; must contain ``A<mode>`` and
            ``B<mode>``.
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts for this mode.
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : None, scalar, or ndarray
            Per-voxel weight for this mode; coerced via `coerce_weight`.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
            Default is ``None``.

        Returns
        -------
        res : ndarray
            Flattened array of signed Poisson deviance residuals for
            voxels with valid data.
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

        Chain-rules the deviance-residual derivative factor from
        `poisson_deviance_residual_factor` through the model
        ``mu = n*(A*gaussian + B) [+ bkg]``, using the precomputed
        Gaussian derivatives with respect to center (`gaussian_jac_c`)
        and shape (`gaussian_jac_S`, applied once for `dr` and once for
        `du`).

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``A<mode>``, ``B<mode>``,
            ``c0, c1, c2, r0, r1, r2, u0, u1, u2``.
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts for this mode.
        n : ndarray
            Normalization (monitor/solid-angle) counts for this mode.
        w : None, scalar, or ndarray
            Per-voxel weight for this mode; coerced via `coerce_weight`.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        dr : tuple of ndarray
            Derivatives of `inv_S` with respect to ``r0, r1, r2`` (from
            `inv_S_deriv_r`).
        du : tuple of ndarray
            Derivatives of `inv_S` with respect to ``u0, u1, u2`` (from
            `inv_S_deriv_u`).
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
            Default is ``None``.

        Returns
        -------
        jac : ndarray of shape (n_vary, n_valid)
            Jacobian rows restricted to varying parameters and columns
            restricted to voxels with valid data.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        ds : sequence of ndarray
            Raw observed counts for modes ``("1d_0", "1d_1", "1d_2")``,
            in that order.
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights, one per element of `ds`; coerced via
            `coerce_weights`. Default is ``None``.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts, one per element
            of `ds`. If ``None``, no extra background is added. Default
            is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.

        Returns
        -------
        res : ndarray
            Concatenation of `residual_mode_poisson` outputs for modes
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
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
        Concatenated (column-stacked) Poisson-deviance Jacobians across all three 1D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        ds : sequence of ndarray
            Raw observed counts for modes ``("1d_0", "1d_1", "1d_2")``,
            in that order.
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights, one per element of `ds`; coerced via
            `coerce_weights`. Default is ``None``.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts, one per element
            of `ds`. Default is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``r0, r1, r2``.
            Default is ``None``.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``u0, u1, u2``.
            Default is ``None``.

        Returns
        -------
        jac : ndarray
            Column-stack of `jacobian_mode_poisson` outputs for modes
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        ds : sequence of ndarray
            Raw observed counts for modes ``("2d_0", "2d_1", "2d_2")``,
            in that order.
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights, one per element of `ds`; coerced via
            `coerce_weights`. Default is ``None``.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts, one per element
            of `ds`. Default is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.

        Returns
        -------
        res : ndarray
            Concatenation of `residual_mode_poisson` outputs for modes
            ``"2d_0"``, ``"2d_1"``, ``"2d_2"``.
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
        Concatenated (column-stacked) Poisson-deviance Jacobians across all three 2D projections.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        ds : sequence of ndarray
            Raw observed counts for modes ``("2d_0", "2d_1", "2d_2")``,
            in that order.
        ns : sequence of ndarray
            Normalization counts, one per element of `ds`.
        ws : sequence, optional
            Per-mode weights, one per element of `ds`; coerced via
            `coerce_weights`. Default is ``None``.
        bkgs : sequence of ndarray, optional
            Per-mode additional fixed background counts, one per element
            of `ds`. Default is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``r0, r1, r2``.
            Default is ``None``.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``u0, u1, u2``.
            Default is ``None``.

        Returns
        -------
        jac : ndarray
            Column-stack of `jacobian_mode_poisson` outputs for modes
            ``"2d_0"``, ``"2d_1"``, ``"2d_2"``.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : None, scalar, or ndarray, optional
            Per-voxel weight; coerced via `coerce_weight`. Default is
            ``None``.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
            Default is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.

        Returns
        -------
        res : ndarray
            Flattened array of signed Poisson deviance residuals for
            voxels with valid data.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        w : None, scalar, or ndarray, optional
            Per-voxel weight; coerced via `coerce_weight`. Default is
            ``None``.
        bkg : ndarray, optional
            Additional fixed background counts added to the model.
            Default is ``None``.
        c : sequence of float, optional
            Peak center ``(c0, c1, c2)``. Default is ``None``.
        inv_S : ndarray of shape (3, 3), optional
            Inverse ellipsoid covariance-like matrix. Default is
            ``None``.
        dr : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``r0, r1, r2``.
            Default is ``None``.
        du : tuple of ndarray, optional
            Derivatives of `inv_S` with respect to ``u0, u1, u2``.
            Default is ``None``.

        Returns
        -------
        jac : ndarray of shape (n_vary, n_valid)
            Jacobian rows restricted to varying parameters and columns
            restricted to voxels with valid data.
        """
        w = self.coerce_weight(d, w)
        return self.jacobian_mode_poisson(
            params, x0, x1, x2, d, n, w, c, inv_S, dr, du, "3d", bkg
        )

    def residual(self, params, args_1d, args_2d, args_3d):
        """
        Full stacked residual vector combining all modes and the prior.

        Computes the center and inverse covariance from `params`, then
        concatenates the 1D, 2D, and 3D Poisson-deviance residuals with
        the prior residual (`prior_residual`). Non-finite entries are
        replaced with 0 or a large finite value so that
        `scipy.optimize.least_squares` sees a well-behaved cost vector.
        This is the residual function passed to `lmfit.Minimizer` in
        `sweep`.

        Note: the volume penalty (`volume_penalty_residual`) is currently
        disabled here — see the comment in the method body.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        args_1d : sequence
            Positional arguments (after `params, x0, x1, x2`) for
            `residual_1d`, i.e. ``(x0, x1, x2, ds, ns, ws, bkgs)``.
        args_2d : sequence
            Positional arguments for `residual_2d`, i.e.
            ``(x0, x1, x2, ds, ns, ws, bkgs)``.
        args_3d : sequence
            Positional arguments for `residual_3d`, i.e.
            ``(x0, x1, x2, d, n, w, bkg)``.

        Returns
        -------
        cost : ndarray
            Concatenated residual vector (1D + 2D + 3D + prior terms),
            with NaN/inf sanitized.
        """
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
        # Volume penalty (volume_penalty_residual) is disabled: as written
        # it rewards *larger* volumes, backwards from the intended
        # anti-collapse regularizer. Paused pending a redesign that
        # rewards smaller volumes with larger intensity above background.

        cost = np.concatenate([cost_1d, cost_2d, cost_3d, cost_prior])
        cost = np.nan_to_num(cost, nan=0.0, posinf=1e16, neginf=-1e16)

        return cost

    def jacobian(self, params, args_1d, args_2d, args_3d):
        """
        Full stacked Jacobian matching the residual ordering of `residual`.

        Computes the center, inverse covariance, and shape derivatives
        (`inv_S_deriv_r`, `inv_S_deriv_u`) from `params`, then
        column-stacks the 1D, 2D, and 3D Poisson-deviance Jacobians with
        the prior Jacobian (`prior_jacobian`), and transposes to the
        ``(n_residuals, n_vary)`` convention expected by
        `scipy.optimize.least_squares`. This is the Jacobian function
        passed to `lmfit.Minimizer.minimize` in `sweep`.

        Note: the volume penalty Jacobian (`volume_penalty_jacobian`) is
        currently disabled here — see the comment in the method body.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters.
        args_1d : sequence
            Positional arguments for `jacobian_1d`, i.e.
            ``(x0, x1, x2, ds, ns, ws, bkgs)``.
        args_2d : sequence
            Positional arguments for `jacobian_2d`, i.e.
            ``(x0, x1, x2, ds, ns, ws, bkgs)``.
        args_3d : sequence
            Positional arguments for `jacobian_3d`, i.e.
            ``(x0, x1, x2, d, n, w, bkg)``.

        Returns
        -------
        jac : ndarray of shape (n_residuals, n_vary)
            Full Jacobian matrix, with NaN/inf sanitized, transposed to
            match the residual vector produced by `residual`.
        """
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

        jac_1d = self.jacobian_1d(params, *args_1d, c, inv_S, dr, du)
        jac_2d = self.jacobian_2d(params, *args_2d, c, inv_S, dr, du)
        jac_3d = self.jacobian_3d(params, *args_3d, c, inv_S, dr, du)
        jac_prior = self.prior_jacobian(params)
        # Volume penalty (volume_penalty_jacobian) is disabled: see the
        # matching comment in residual().

        jac = np.column_stack([jac_1d, jac_2d, jac_3d, jac_prior])
        jac = np.nan_to_num(jac, nan=0.0, posinf=1e16, neginf=-1e16)

        return jac.T

    def collect_mode_fit_metrics(
        self, x0, x1, x2, c, inv_S, mode_data, bkg_offset=None
    ):
        """
        Evaluate the fitted profile, chi-squared, and box-sum intensity for each requested mode.

        For each mode, evaluates the fitted model ``A*gaussian + B``
        (plus any `bkg_offset`), masks invalid points, then computes the
        reduced Poisson deviance (`poisson_deviance_fit`) and the
        background-subtracted box-sum intensity/uncertainty
        (`estimate_intensity`).

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        inv_S : ndarray of shape (3, 3)
            Inverse ellipsoid covariance-like matrix.
        mode_data : dict
            Mapping from mode string to a ``(y, e)`` tuple of observed
            normalized intensity and uncertainty for that mode.
        bkg_offset : dict, optional
            Mapping from mode string to an additional per-voxel
            background-rate offset array for that mode. Modes not
            present in this dict get no offset. Default is ``None``.

        Returns
        -------
        metrics : dict
            Mapping from mode string to a list
            ``[I0, s0, chi2, fit_triplet]``, where `I0`/`s0` are the
            box-sum intensity/uncertainty, `chi2` is the reduced Poisson
            deviance, and `fit_triplet` is ``(y_fit, y, e)``.
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

    def extract_result(self, args_1d, args_2d, args_3d, xmod):
        """
        Finalize the fit: compute per-mode metrics and populate the post-fit result attributes.

        Converts raw counts to normalized intensity/uncertainty for every
        1D/2D/3D mode, computes background offsets from `bkg_1d`/`bkg_2d`/
        `bkg_3d` if present, and calls `collect_mode_fit_metrics` for each
        mode group to populate `self.reddev`, `self.intensity`,
        `self.sigma`, and `self.fit_metrics`. Also derives the final
        ellipsoid shape (center, radii, principal axes) by eigen-decomposing
        ``S = inv(inv_S)``, shifts the center back by `xmod` (the modulo
        offset the caller applied before fitting), and populates
        `self.peak_pos`, `self.best_fit`, `self.best_prof`,
        `self.best_bkg_prof`, and `self.best_proj` for downstream plotting
        and intensity extraction.

        Parameters
        ----------
        args_1d : sequence
            ``(x0, x1, x2, d1d, n1d, w1d, bkg_1d, ...)`` where `d1d`,
            `n1d` are 3-tuples of per-axis 1D-projected raw counts and
            normalization, and `bkg_1d` is either ``None`` or a 3-tuple
            of per-axis 1D-projected background counts.
        args_2d : sequence
            ``(x0, x1, x2, d2d, n2d, w2d, bkg_2d, ...)`` analogous to
            `args_1d` but for the three 2D projections.
        args_3d : sequence
            ``(x0, x1, x2, d3d, n3d, w3d, bkg_3d, ...)`` analogous to
            `args_1d` but for the full 3D volume.
        xmod : float
            Offset that was subtracted from the axis-0 coordinate before
            fitting (e.g. to wrap a periodic axis); added back to the
            reported center and coordinates.

        Returns
        -------
        c0 : float
            Fitted peak center along axis 0 (with `xmod` added back).
        c1 : float
            Fitted peak center along axis 1.
        c2 : float
            Fitted peak center along axis 2.
        r0 : float
            First principal radius (1-sigma) of the fitted ellipsoid.
        r1 : float
            Second principal radius (1-sigma) of the fitted ellipsoid.
        r2 : float
            Third principal radius (1-sigma) of the fitted ellipsoid.
        v0 : ndarray of shape (3,)
            First principal axis vector of the fitted ellipsoid.
        v1 : ndarray of shape (3,)
            Second principal axis vector of the fitted ellipsoid.
        v2 : ndarray of shape (3,)
            Third principal axis vector of the fitted ellipsoid.

            If the optimized inverse covariance is not positive definite
            (an improper fit), ``None`` is returned instead of this
            tuple.
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
            The ``A1d_0, A1d_1, A1d_2`` `lmfit.Parameter` objects, one per
            axis.
        backgrounds : ndarray of shape (3,)
            The ``B1d_0, B1d_1, B1d_2`` `lmfit.Parameter` objects, one per
            axis.
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

        Subtracts the mean-normalized background level, cross-correlates
        the residual counts with `kernel` (normalized by
        ``sqrt(correlate(n^2, kernel^2))``), and returns the index of the
        maximum correlation. Falls back to the array midpoint if the
        located maximum lies in the outer eighths of the array (treated
        as an unreliable/edge estimate).

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
            Unused tuning parameter reserved for future use. Default is
            0.9.
        min_weight : float, optional
            Minimum correlated normalization-squared value required to
            accept a correlation value (otherwise it is set to NaN).
            Default is 1e-12.

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

        Uses `self.estimated_fit` (the ``(center, S)`` pair set by
        `update_estimate`) rather than the current fit parameters, so
        this gives a fixed reference region independent of in-progress
        parameter changes. The output shape matches `profile_project`'s
        output shape for the same mode: for `"1d_k"` modes, shape
        ``(nk,)`` along the single projected axis; for `"2d_k"` modes,
        shape ``(ni, nj)`` in the plane perpendicular to axis `k`; for
        `"3d"`, the full 3D shape.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.

        Returns
        -------
        mah2 : ndarray
            Squared Mahalanobis distance to the estimated-fit center,
            using the estimated-fit covariance `S_est`, with shape
            matching the mode's projected coordinate grid. Returns an
            array of ``inf`` where `S_est` (or its relevant sub-block) is
            singular or non-positive, and for unrecognized `mode`
            strings.
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

        Projects the data onto `mode`, optionally subtracts a measured
        background (`b`, `m`) rescaled to signal units, then estimates
        the background level `B` from the shell region beyond
        Mahalanobis distance 2 (`_mode_mah2` >= 4) and the amplitude `A`
        from the 95th percentile within the 1-sigma core region
        (`_mode_mah2` <= 1) above `B`. Adds ``A<mode>`` and ``B<mode>``
        parameters (both bounded to be non-negative) to `self.params`.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.
        b : ndarray, optional
            Measured background counts, same shape as `d`. Default is
            ``None``.
        m : ndarray, optional
            Measured background monitor/normalization counts, same shape
            as `d`, used with `n` to rescale `b` into signal-count units.
            Default is ``None``.

        Returns
        -------
        stronger : bool or None
            ``True`` if the estimated amplitude exceeds the estimated
            background, else ``False``. Returns ``None`` if fewer than 4
            valid projected voxels are available (no estimate made, and
            `self.params` is left unmodified).
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

        Restores `self.combine_params` as the starting parameters if
        available, projects the data onto all seven modes, quick-estimates
        each mode's amplitude/background via `quick_gaussian`, converts a
        measured background (`b`, `m`) into projected background-count
        arrays for each mode, then runs an initial unconstrained-amplitude
        `sweep` followed by `n_loop` rounds alternating
        `update_adaptive_prior` with two `sweep` calls (one varying
        center only, one varying radii and orientation).

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d_int : ndarray
            Raw observed event counts, 3D array.
        n_int : ndarray
            Normalization (monitor/solid-angle) counts, same shape as
            `d_int`.
        wgt : ndarray
            Per-voxel fit weight, same shape as `d_int`.
        report_fit : bool, optional
            If ``True``, print an `lmfit` fit report after each `sweep`
            call. Default is ``False``.
        b : ndarray, optional
            Measured background counts, same shape as `d_int`. Default
            is ``None``.
        m : ndarray, optional
            Measured background monitor/normalization counts, same shape
            as `d_int`. Default is ``None``.

        Returns
        -------
        args_1d : list
            ``[x0, x1, x2, d1d, n1d, w1d, bkg_1d]`` used for the 1D-mode
            residual/Jacobian calls.
        args_2d : list
            ``[x0, x1, x2, d2d, n2d, w2d, bkg_2d]`` used for the 2D-mode
            residual/Jacobian calls.
        args_3d : list
            ``[x0, x1, x2, d3d, n3d, w3d, bkg_3d]`` used for the 3D-mode
            residual/Jacobian calls.

            If the data shape is too small (any axis length < 3) or any
            mode's `quick_gaussian` estimate fails, ``None`` is returned
            instead of this tuple.
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
        n_loop = 3

        protocol = [False] * 9

        self.sweep(args_1d, args_2d, args_3d, protocol, n_iter, report_fit)

        for _ in range(n_loop):
            self.update_adaptive_prior(args_1d, args_2d, args_3d)

            protocol = [True] * 3 + [False] * 6

            self.sweep(args_1d, args_2d, args_3d, None, n_iter, report_fit)

            self.update_adaptive_prior(args_1d, args_2d, args_3d)

            protocol = [False] * 3 + [True] * 6

            self.sweep(args_1d, args_2d, args_3d, None, n_iter, report_fit)

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
        prior widths via `prior_widths_from_snr`, and stores them as
        `self.prior_center_sigma`, `self.prior_cov_sigma`,
        `self.prior_distortion_sigma`, `self.prior_rot_sigma`. A width of
        0 fixes ("freezes") the corresponding parameter group (center,
        radii, or orientation) at its predicted resolution-model value
        (`self._prior_r`, `self._prior_u`) by setting ``vary=False``;
        otherwise the group is allowed to vary.

        Parameters
        ----------
        args_1d : sequence
            1D-mode residual arguments (unused directly; kept for a
            uniform call signature alongside `args_3d`).
        args_2d : sequence
            2D-mode residual arguments (unused directly; kept for a
            uniform call signature).
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

        for name, value in zip(("r0", "r1", "r2"), self._prior_r):
            self.params[name].set(vary=radius_vary)
            if not radius_vary:
                self.params[name].set(value=float(value))

        for name, value in zip(("u0", "u1", "u2"), self._prior_u):
            self.params[name].set(vary=orient_vary)
            if not orient_vary:
                self.params[name].set(value=float(value))

    def prior_widths_from_snr(self, snr):
        """
        Map an SNR estimate to SNR-adaptive prior widths via `smooth_saturating`.

        Lower SNR yields tighter (smaller) prior widths, pulling the fit
        more strongly toward the resolution-model prediction; higher SNR
        relaxes the widths, letting the fit rely more on the data.

        Parameters
        ----------
        snr : float
            Estimated intensity/sigma (I/sigma) signal-to-noise ratio,
            e.g. from `_isig_3d`.

        Returns
        -------
        sigma_c : float
            Center prior width, in Mahalanobis units of the prior
            covariance.
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
            Input coordinate. The model is intended for x >= 1.
        f_min : float, optional
            Minimum value at x = 1. Default is 0.0.
        f_max : float, optional
            Asymptotic maximum value as x -> infinity. Default is 1.0.
        x0 : float, optional
            Location where the function reaches fraction p of the way
            from f_min to f_max. Must be greater than 1. Default is 2.0.
        p : float, optional
            Fraction of the maximum reached at x0. Must satisfy 0 < p < 1.
            For example, p=0.9 means f(x0) is 90% of the way to f_max.
            Default is 0.5.
        n : float, optional
            Shape parameter. n=1 gives an exponential rise; n=2 gives
            a smoother minimum at x=1. Must be positive. Default is 1.0.
        clip_below : bool, optional
            If True, values with x < 1 are clipped to x = 1. Default is
            True.

        Returns
        -------
        y : float or ndarray
            Smooth saturating function value.

        Raises
        ------
        ValueError
            If `x0` is not greater than 1, if `p` is not in ``(0, 1)``,
            or if `n` is not positive.
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
        Quick 3D I/sigma estimate, tried at a few radii scales.

        The peak/background split depends on how well the current radii
        match the true peak, so a single scale can underestimate the SNR
        (e.g. if the fit radii are currently too tight or too loose). Trying
        a few scales and taking the best I/sigma gives the SNR-adaptive
        protocol a fairer chance of recognizing a strong peak. For each
        scale, voxels within Mahalanobis distance 1 of the center (using
        radii scaled by `scale`) are treated as peak, and voxels between
        distance 1 and ``2**(1/3)`` are treated as background; the
        background is scaled by the peak/background voxel-count ratio
        before subtraction.

        Parameters
        ----------
        params : lmfit.Parameters
            Current fit parameters; must contain ``c0, c1, c2, r0, r1,
            r2, u0, u1, u2``.
        args_3d : sequence
            ``(x0, x1, x2, d3d, n3d, w3d, bkg_3d)`` (or a shorter
            sequence without `bkg_3d`); `bkg_3d`, if present, is
            subtracted from `d3d` before computing the SNR.
        scales : tuple of float, optional
            Radii scale factors to try. Default is ``(0.5, 1.0, 2.0)``.

        Returns
        -------
        best_isig : float
            The best (maximum) I/sigma estimate found across `scales`,
            or ``-inf`` if no scale yields a positive-uncertainty
            estimate.
        """
        x0, x1, x2, d3d, n3d, *rest = args_3d
        bkg_3d = rest[1] if len(rest) > 1 else None

        c0 = params["c0"].value
        c1 = params["c1"].value
        c2 = params["c2"].value
        r0 = params["r0"].value
        r1 = params["r1"].value
        r2 = params["r2"].value
        u0 = params["u0"].value
        u1 = params["u1"].value
        u2 = params["u2"].value

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

        Optionally sets which parameters vary via `protocol`, then
        minimizes `self.residual`/`self.jacobian` with `lmfit.Minimizer`
        using the ``"leastsq"`` method (MINPACK's Levenberg-Marquardt,
        via `scipy.optimize.leastsq`; bounds are handled by `lmfit`'s own
        internal parameter transform, since MINPACK's `leastsq` has no
        native bounds support). The resulting fit is accepted (written
        back to `self.params`) only if the post-fit 3D I/sigma (`_isig_3d`
        at ``scales=[1]``) is not worse than before the fit; otherwise
        `self.params` is left unchanged.

        Parameters
        ----------
        args_1d : sequence
            Positional arguments for `residual_1d`/`jacobian_1d`.
        args_2d : sequence
            Positional arguments for `residual_2d`/`jacobian_2d`.
        args_3d : sequence
            Positional arguments for `residual_3d`/`jacobian_3d`.
        protocol : sequence of bool, optional
            9-element sequence of ``vary`` flags for
            ``c0, c1, c2, r0, r1, r2, u0, u1, u2`` in that order. If
            ``None``, the current `vary` settings on `self.params` are
            left unchanged. Default is ``None``.
        n_iter : int, optional
            Maximum number of function evaluations (`max_nfev`) for the
            least-squares solver. Default is 50.
        report_fit : bool, optional
            If ``True``, print an `lmfit` fit report after minimizing.
            Default is ``False``.
        """
        if protocol is not None:
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
        Total fitted intensity for a mode from its Gaussian amplitude and shape parameters.

        Computes ``A * gaussian_integral(inv_S, mode)``, i.e. the
        amplitude times the normalization volume of the unit-peak
        Gaussian, giving the integrated intensity under the fitted
        profile.

        Parameters
        ----------
        A : float
            Fitted Gaussian amplitude for this mode.
        H : float
            Unused parameter (reserved, e.g. for a peak height
            convention not currently applied).
        r0 : float
            Ellipsoid principal radius along the first rotated axis.
        r1 : float
            Ellipsoid principal radius along the second rotated axis.
        r2 : float
            Ellipsoid principal radius along the third rotated axis.
        u0 : float
            First component of the orientation rotation vector.
        u1 : float
            Second component of the orientation rotation vector.
        u2 : float
            Third component of the orientation rotation vector.
        mode : str, optional
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``. Default is ``"3d"``.

        Returns
        -------
        intensity : float
            Total integrated intensity ``A * gaussian_integral(...)``.
        """
        inv_S = self.inv_S_matrix(r0, r1, r2, u0, u1, u2)
        g = self.gaussian_integral(inv_S, mode)

        return A * g

    def voxels(self, x0, x1, x2):
        """
        Voxel spacing along each axis, from adjacent-element differences of the meshgrid.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.

        Returns
        -------
        dx0 : float
            Voxel spacing along axis 0.
        dx1 : float
            Voxel spacing along axis 1.
        dx2 : float
            Voxel spacing along axis 2.
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
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.

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
            3D background counts array. Non-finite entries are treated
            as zero.
        mode : str
            One of ``"3d"``, ``"2d_0"``, ``"2d_1"``, ``"2d_2"``,
            ``"1d_0"``, ``"1d_1"``, ``"1d_2"``.

        Returns
        -------
        bkg_proj : ndarray
            `bkg` summed over the axes integrated out by `mode` (a copy
            of `bkg` for `"3d"`).
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
        then subtracts that level (rescaled by `n`) from the raw counts.

        Parameters
        ----------
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        perc : float, optional
            Percentile (0-100) of the normalized intensity used as the
            background estimate for each axis-0 slice. Default is 30.

        Returns
        -------
        d_sub : ndarray
            Background-subtracted raw counts, same shape as `d`.
        n : ndarray
            The (unmodified) normalization counts, returned for a
            convenient ``d, n = subtract_profile(d, n)`` call pattern.
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

        Converts raw counts/normalization to normalized intensity,
        rejects too-small or all-invalid input, crops all arrays to the
        bounding box of voxels with positive normalization, then calls
        `estimate_envelope` to perform the actual iterative fit. On
        success, snapshots the resulting parameters via `copy_combine`.
        Prints and logs the elapsed wall-clock time and outcome for each
        early-exit path or success.

        Parameters
        ----------
        x0_prof : ndarray
            3D meshgrid coordinate array along axis 0, in the original
            (not peak-centered) coordinate frame.
        x1_proj : ndarray
            3D meshgrid coordinate array along axis 1.
        x2_proj : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        dx : object
            Unused parameter (voxel spacing is instead recomputed from
            the coordinate arrays via `voxels`).
        xmod : float
            Offset subtracted from `x0_prof` to produce the fit's
            internal axis-0 coordinate (e.g. to re-center a periodic
            axis); stored as `self._fit_xmod`.
        voxel_weights : ndarray
            Per-voxel fit weight, same shape as `d`.
        b : ndarray, optional
            Measured background counts, same shape as `d`. Default is
            ``None``.
        m : ndarray, optional
            Measured background monitor/normalization counts, same shape
            as `d`. Default is ``None``.

        Returns
        -------
        weights : tuple or None
            The ``(args_1d, args_2d, args_3d)`` tuple returned by
            `estimate_envelope` on success. Returns ``None`` if the data
            shape is too small, if there is insufficient valid data
            (fewer than 11 valid voxels or no positive normalized
            intensity), if all data is invalid after cropping, if
            `estimate_envelope` raises an exception, or if
            `estimate_envelope` itself returns ``None``.
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

        The peak mask is the 1-sigma ellipsoid region (Mahalanobis
        distance <= 1 under `S`) dilated by one voxel in each direction;
        the combined mask is grown by further dilations until the
        background shell (combined mask minus peak mask) has at least
        twice as many voxels as the peak, up to 3 dilation attempts. The
        returned kernel `y` is a Gaussian evaluated with inverse
        covariance rescaled to the `p` confidence contour (via the
        chi-squared quantile at 3 degrees of freedom), independent of the
        mask-growing logic.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        S : ndarray of shape (3, 3)
            Ellipsoid covariance-like matrix.
        scale : float
            Unused parameter (a local variable of the same name is
            recomputed from `p` and overwrites this argument before use).
        p : float, optional
            Confidence-contour probability (0-1) used to build the
            Gaussian kernel `y`. Default is 0.997.

        Returns
        -------
        pk : ndarray of bool
            Peak region mask (1-sigma ellipsoid, dilated by one voxel).
        bkg : ndarray of bool
            Background shell mask (`mask` minus `pk`).
        mask : ndarray of bool
            Combined peak-plus-background mask after dilation growth.
        y : ndarray
            Gaussian kernel evaluated at the `p`-confidence-contour
            inverse covariance, normalized to integrate to 1.
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
            Raw observed event counts, 3D array. Infinite values are
            treated as NaN.
        pk : ndarray of bool
            Peak region mask, same shape as `counts`.
        bkg : ndarray of bool
            Background region mask, same shape as `counts`.

        Returns
        -------
        intens : float
            Background-subtracted summed raw intensity.
        sig : float
            Propagated Poisson uncertainty on `intens`; ``inf`` if the
            propagated variance is non-positive.
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

        Sums raw counts and normalization separately over the peak
        (`pk`) and background-shell (`bkg`) regions (restricted to
        voxels with positive normalization). If a measured/fitted
        background model `bkg_meas` is supplied, it is used directly
        (assumed to have negligible variance) instead of the background
        shell; otherwise the shell counts are scaled by the peak/shell
        voxel-count ratio, as in `extract_raw_intensity` but additionally
        normalized by monitor counts. The final intensity/uncertainty are
        rescaled by the peak voxel count `vol` and by `vol_frac` to
        correct for the fraction of the theoretical ellipsoid volume
        actually present on the detector.

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
            Fraction (0-1] of the theoretical ellipsoid volume actually
            covered by valid detector voxels; `intens`/`sig` are divided
            by this to correct for edge clipping. Default is 1.0.
        bkg_meas : ndarray, optional
            Measured/fitted background counts, same shape as `d`, used
            in place of the background shell if provided (assumed exact,
            zero variance). Default is ``None``.

        Returns
        -------
        intens : float
            Background-subtracted, volume- and edge-corrected intensity.
        sig : float
            Propagated uncertainty on `intens`; ``-inf`` if the
            propagated variance is non-positive.
        b : float
            Estimated background rate (counts per monitor count).
        b_err : float
            Uncertainty on `b`.
        vol : float
            Number of voxels in the peak region (`core`).
        vol_frac : float
            The (unmodified) `vol_frac` argument, echoed back.
        vol_pk : int
            Integer number of voxels in the peak region, same as `vol`.
        pk_cnts : float
            Summed raw counts in the peak region (NaN if zero).
        pk_norm : float
            Summed normalization counts in the peak region (NaN if zero).
        bkg_cnts : float
            Summed background counts used (from `bkg_meas` or the shell).
        bkg_norm : float
            Summed normalization counts used for the background estimate.
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

        Fits ``mu = n*(I*kernel + b) + bkg_meas`` to the raw counts `d`
        by maximizing the Poisson log-likelihood (minimizing
        ``sum(mu - d*log(mu))``) over intensity `I` and background `b`
        using L-BFGS-B, seeded from percentile-based initial estimates.
        The uncertainty on `I` is obtained from the observed Fisher
        information at the optimum.

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
            Mask of voxels to include in the fit.
        bkg_meas : ndarray, optional
            Additional fixed measured background counts, same shape as
            `d`, added to the model. Default is ``None``.

        Returns
        -------
        I : float
            Fitted matched-filter intensity scale.
        I_err : float
            Uncertainty on `I` from the inverse Fisher information;
            ``inf`` if the Fisher information is non-positive.
        A : float
            Peak amplitude, `I` scaled by the kernel's maximum value
            within `mask`.
        b : float
            Fitted background rate.

            If `mask` selects no voxels, ``(0.0, inf, 0.0, 0.0)`` is
            returned instead.
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

        Projects the data onto the axis-0 1D profile, then alternates
        `matched_filter` fits of a Gaussian kernel with re-estimating the
        kernel's sigma from the weighted second moment of the
        (background-subtracted, clipped-positive) net profile, damped by
        `eta` and floored at the voxel size `dx0`. The initial sigma
        comes from ``sqrt(S[0,0])`` rescaled from the 99.7%-contour
        convention using the 1D chi-squared quantile at `p`.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        c : sequence of float
            Peak center ``(c0, c1, c2)``.
        S : ndarray of shape (3, 3)
            Ellipsoid covariance-like matrix (99.7%-contour convention).
        p : float, optional
            Confidence-contour probability (0-1) used to convert `S` to
            an initial 1-sigma width and to define the peak/background
            split for the matched filter. Default is 0.997.
        eta : float, optional
            Damping factor (0-1) for updating sigma between iterations;
            0 keeps the initial sigma, 1 jumps directly to the new
            estimate. Default is 0.5.
        bkg_meas : ndarray, optional
            Measured/fitted background counts, 3D array, same shape as
            `d`, summed onto the axis-0 profile and passed to
            `matched_filter`. Default is ``None``.

        Returns
        -------
        I : float
            Fitted matched-filter intensity from the final iteration.
        I_err : float
            Uncertainty on `I`.
        b : float
            Fitted background rate from the final iteration.
        x : ndarray
            Axis-0 coordinate relative to the center, `x0[:, 0, 0] - c0`.
        y_fit : ndarray
            Fitted profile ``I*kernel + b`` (plus the background-rate
            offset if `bkg_meas` was given).
        y : ndarray
            Observed normalized intensity profile.
        e : ndarray
            Uncertainty on `y`.
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

        Crops all inputs to the same region as the fit (`self._fit_crop`,
        set by `fit`) so that the returned x-coordinates match
        `self.best_prof`. The background is rescaled from monitor units
        to signal units by the median ratio ``m/n``, then projected onto
        each of the three 1D axes and normalized by the projected
        normalization counts.

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0 (uncropped).
        x1 : ndarray
            3D meshgrid coordinate array along axis 1 (uncropped).
        x2 : ndarray
            3D meshgrid coordinate array along axis 2 (uncropped).
        d : ndarray
            Raw observed event counts, 3D array (uncropped).
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`
            (uncropped).
        b : ndarray or None
            Measured background counts, same shape as `d` (uncropped).
            If ``None``, no background profile can be computed.
        m : ndarray or None
            Measured background monitor/normalization counts, same shape
            as `d` (uncropped). If ``None``, no background profile can
            be computed.

        Returns
        -------
        result : list of tuple or None
            A list of 3 tuples ``(x, y_bkg, e_bkg)``, one per axis
            (``1d_0, 1d_1, 1d_2``), each with the axis coordinate array
            and the corresponding background intensity/uncertainty
            arrays. Returns ``None`` if `b` or `m` is ``None``, or if the
            median monitor-count scale factor is not finite and
            positive.
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

        Builds peak/background masks (`peak_roi`) for both the final
        fitted ellipsoid and the original estimated-fit ellipsoid (used
        to compute an edge-clipping volume-fraction correction
        `vol_frac`), applies that correction to the previously computed
        3D box-sum intensity (`self.intensity[2]`/`self.sigma[2]`, from
        `extract_result`), then computes and stores the normalization-aware
        box-sum estimator (`extract_intensity`, `I_ell`), the 1D
        profile-fit estimator (`fitted_profile`, `I_prof`), and a
        chi-squared-inflated matched-filter estimator (`matched_filter`,
        `self.filter`). Also computes the simple raw box-sum estimator
        (`extract_raw_intensity`) for diagnostics. Populates
        `self.intensity`, `self.sigma`, `self.weights`,
        `self.data_norm_fit`, `self.peak_background_mask`,
        `self.integral`, `self.filter`, and `self.info` (a dict
        summarizing all intensity estimators: `I_1d_*`, `I_2d_*`, `I_3d`,
        `I_ell`, `I_prof` and their uncertainties, plus background and
        volume diagnostics).

        Parameters
        ----------
        x0 : ndarray
            3D meshgrid coordinate array along axis 0.
        x1 : ndarray
            3D meshgrid coordinate array along axis 1.
        x2 : ndarray
            3D meshgrid coordinate array along axis 2.
        d : ndarray
            Raw observed event counts, 3D array.
        n : ndarray
            Normalization (monitor/solid-angle) counts, same shape as `d`.
        c : sequence of float
            Fitted peak center ``(c0, c1, c2)``.
        S : ndarray of shape (3, 3)
            Fitted ellipsoid covariance-like matrix.
        b : ndarray, optional
            Measured background counts, same shape as `d`. Default is
            ``None``.
        m : ndarray, optional
            Measured background monitor/normalization counts, same shape
            as `d`. Default is ``None``.

        Returns
        -------
        intens : float
            Final normalization-aware box-sum intensity (`I_ell`),
            scaled by the voxel volume.
        sig : float
            Uncertainty on `intens`; ``inf`` if not finite.
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
