import numpy as np

import scipy.spatial.transform

from lmfit import Parameters

from garnet.reduction.ellipsoid import PeakEllipsoid, SHAPE_BASIS


def _skew(v):
    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def test_inv_S_deriv_r():
    """Verify inv_S_deriv_r against central finite differences of inv_S_matrix."""
    ellipsoid = PeakEllipsoid()

    r0, r1, r2 = 0.2, 0.3, 0.4
    u0, u1, u2 = 0.2, 0.1, 0.4
    delta = 1e-6

    dr = ellipsoid.inv_S_deriv_r(r0, r1, r2, u0, u1, u2)
    for i in range(3):
        rs_p, rs_m = [r0, r1, r2], [r0, r1, r2]
        rs_p[i] += delta
        rs_m[i] -= delta
        fd = (
            ellipsoid.inv_S_matrix(*rs_p, u0, u1, u2)
            - ellipsoid.inv_S_matrix(*rs_m, u0, u1, u2)
        ) / (2 * delta)
        assert np.allclose(dr[i], fd, atol=1e-6, rtol=1e-6)


def test_inv_S_deriv_size_shape():
    """Verify inv_S_deriv_size_shape against finite differences along the log_size/shape_1/shape_2 directions.

    d(r_i)/d(log_size) = r_i and d(r_i)/d(shape_k) = r_i * SHAPE_BASIS[i, k]
    hold at any current radii (independent of any reference ellipsoid), so
    this can be checked directly without going through `update_estimate`.
    """
    ellipsoid = PeakEllipsoid()

    r0, r1, r2 = 0.2, 0.3, 0.4
    u0, u1, u2 = 0.2, 0.1, 0.4
    delta = 1e-6

    d_size, d_shape1, d_shape2 = ellipsoid.inv_S_deriv_size_shape(
        r0, r1, r2, u0, u1, u2
    )

    directions = {
        0: np.array([1.0, 1.0, 1.0]),
        1: SHAPE_BASIS[:, 0],
        2: SHAPE_BASIS[:, 1],
    }
    analytic = [d_size, d_shape1, d_shape2]

    r = np.array([r0, r1, r2])
    for k, direction in directions.items():
        rs_p = r * np.exp(delta * direction)
        rs_m = r * np.exp(-delta * direction)
        fd = (
            ellipsoid.inv_S_matrix(*rs_p, u0, u1, u2)
            - ellipsoid.inv_S_matrix(*rs_m, u0, u1, u2)
        ) / (2 * delta)
        assert np.allclose(analytic[k], fd, atol=1e-6, rtol=1e-5)


def test_inv_S_deriv_domega_at_zero():
    """Verify inv_S_deriv_domega at domega=0 against the closed-form SO(3) generator.

    At domega=0, R(domega) = prior_U @ Exp(domega), so
    dR/d(domega_k)|_0 = prior_U @ [e_k]_x, giving the closed-form
    d(inv_S)/d(domega_k)|_0 = prior_U @ [e_k]_x @ V @ prior_U.T
                              - prior_U @ V @ [e_k]_x @ prior_U.T
    where V = diag(1/r0^2, 1/r1^2, 1/r2^2). This is independent of the
    method's own finite-difference implementation.
    """
    ellipsoid = PeakEllipsoid()

    r0, r1, r2 = 0.2, 0.3, 0.4
    prior_U = ellipsoid.U_matrix(0.3, -0.2, 0.1)
    ellipsoid._prior_U = prior_U

    V = np.diag([1 / r0**2, 1 / r1**2, 1 / r2**2])

    dw = ellipsoid.inv_S_deriv_domega(r0, r1, r2, np.zeros(3))

    for k in range(3):
        ek_x = _skew(np.eye(3)[k])
        expected = (
            prior_U @ ek_x @ V @ prior_U.T - prior_U @ V @ ek_x @ prior_U.T
        )
        assert np.allclose(dw[k], expected, atol=1e-6, rtol=1e-5)


def test_gaussian_derivatives():
    """Verify gaussian_integral_jac_S, gaussian_jac_c, and gaussian_jac_S against finite differences."""
    ellipsoid = PeakEllipsoid()

    modes = ["3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2"]

    r0, r1, r2 = 0.2, 0.3, 0.4
    u0, u1, u2 = 0.2, 0.1, 0.4
    delta = 1e-6

    ellipsoid._prior_U = ellipsoid.U_matrix(u0, u1, u2)

    dr = ellipsoid.inv_S_deriv_size_shape(r0, r1, r2, u0, u1, u2)
    du = ellipsoid.inv_S_deriv_domega(r0, r1, r2, np.zeros(3))

    inv_S0 = ellipsoid.inv_S_matrix(r0, r1, r2, u0, u1, u2)

    size_shape_directions = [
        np.array([1.0, 1.0, 1.0]),
        SHAPE_BASIS[:, 0],
        SHAPE_BASIS[:, 1],
    ]
    r = np.array([r0, r1, r2])

    for mode in modes:
        dg_r = ellipsoid.gaussian_integral_jac_S(inv_S0, dr, mode=mode)
        for i, direction in enumerate(size_shape_directions):
            rs_p = r * np.exp(delta * direction)
            rs_m = r * np.exp(-delta * direction)
            g_p = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(*rs_p, u0, u1, u2), mode=mode
            )
            g_m = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(*rs_m, u0, u1, u2), mode=mode
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.isclose(dg_r[i], fd, atol=1e-6, rtol=1e-5)

        dg_u = ellipsoid.gaussian_integral_jac_S(inv_S0, du, mode=mode)
        for i in range(3):
            dp, dm = np.zeros(3), np.zeros(3)
            dp[i] += delta
            dm[i] -= delta
            Rp = (
                ellipsoid._prior_U
                @ scipy.spatial.transform.Rotation.from_rotvec(dp).as_matrix()
            )
            Rm = (
                ellipsoid._prior_U
                @ scipy.spatial.transform.Rotation.from_rotvec(dm).as_matrix()
            )
            up = scipy.spatial.transform.Rotation.from_matrix(Rp).as_rotvec()
            um = scipy.spatial.transform.Rotation.from_matrix(Rm).as_rotvec()
            g_p = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(r0, r1, r2, *up), mode=mode
            )
            g_m = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(r0, r1, r2, *um), mode=mode
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.isclose(dg_u[i], fd, atol=1e-6, rtol=1e-5)

    x0 = np.linspace(1, 2, 5)
    x1 = np.linspace(-3, -2, 6)
    x2 = np.linspace(0, 1, 7)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    c0, c1, c2 = 1.5, -2.5, 0.5
    c = (c0, c1, c2)

    for mode in modes:
        dgc = ellipsoid.gaussian_jac_c(x0, x1, x2, c, inv_S0, mode=mode)
        for i in range(3):
            c_p, c_m = list(c), list(c)
            c_p[i] += delta
            c_m[i] -= delta
            g_p = ellipsoid.gaussian(x0, x1, x2, c_p, inv_S0, mode=mode)
            g_m = ellipsoid.gaussian(x0, x1, x2, c_m, inv_S0, mode=mode)
            fd = (g_p - g_m) / (2 * delta)
            assert np.allclose(dgc[i], fd, atol=1e-6, rtol=1e-5)

        dgr = ellipsoid.gaussian_jac_S(x0, x1, x2, c, inv_S0, dr, mode=mode)
        for i, direction in enumerate(size_shape_directions):
            rs_p = r * np.exp(delta * direction)
            rs_m = r * np.exp(-delta * direction)
            g_p = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(*rs_p, u0, u1, u2),
                mode=mode,
            )
            g_m = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(*rs_m, u0, u1, u2),
                mode=mode,
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.allclose(dgr[i], fd, atol=1e-6, rtol=1e-5)

        dgu = ellipsoid.gaussian_jac_S(x0, x1, x2, c, inv_S0, du, mode=mode)
        for i in range(3):
            dp, dm = np.zeros(3), np.zeros(3)
            dp[i] += delta
            dm[i] -= delta
            Rp = (
                ellipsoid._prior_U
                @ scipy.spatial.transform.Rotation.from_rotvec(dp).as_matrix()
            )
            Rm = (
                ellipsoid._prior_U
                @ scipy.spatial.transform.Rotation.from_rotvec(dm).as_matrix()
            )
            up = scipy.spatial.transform.Rotation.from_matrix(Rp).as_rotvec()
            um = scipy.spatial.transform.Rotation.from_matrix(Rm).as_rotvec()
            g_p = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(r0, r1, r2, *up),
                mode=mode,
            )
            g_m = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(r0, r1, r2, *um),
                mode=mode,
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.allclose(dgu[i], fd, atol=1e-6, rtol=1e-5)


def _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes):
    ellipsoid.update_constraints(x0, x1, x2, dx=None)
    ellipsoid.update_estimate((0, 0, 0, r0, r1, r2, *axes))


def test_zero_change_reproduces_reference():
    """All six size/shape/orientation parameters at zero must reproduce the reference ellipsoid exactly."""
    ellipsoid = PeakEllipsoid()

    x0 = np.linspace(-1, 1, 5)
    x1 = np.linspace(-1, 1, 5)
    x2 = np.linspace(-1, 1, 5)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    r0, r1, r2 = 0.2, 0.3, 0.4
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)

    for name in (
        "log_size",
        "shape_1",
        "shape_2",
        "domega_x",
        "domega_y",
        "domega_z",
    ):
        assert ellipsoid.params[name].value == 0.0

    (
        r0_fit,
        r1_fit,
        r2_fit,
        u0_fit,
        u1_fit,
        u2_fit,
    ) = ellipsoid.shape_from_params(ellipsoid.params)
    S = ellipsoid.S_matrix(r0_fit, r1_fit, r2_fit, u0_fit, u1_fit, u2_fit)
    assert np.allclose(S, ellipsoid.prior_cov, atol=1e-10)


def test_pure_size_scales_all_radii_uniformly():
    """Changing only log_size scales every radius by the same factor and leaves axis ratios unchanged."""
    ellipsoid = PeakEllipsoid()

    x0 = np.linspace(-1, 1, 5)
    x1 = np.linspace(-1, 1, 5)
    x2 = np.linspace(-1, 1, 5)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    r0, r1, r2 = 0.2, 0.3, 0.4
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)

    t = 0.25
    ellipsoid.params["log_size"].set(value=t)

    r0_fit, r1_fit, r2_fit, *_ = ellipsoid.shape_from_params(ellipsoid.params)

    scale = np.exp(t)
    assert np.isclose(r0_fit, r0 * scale)
    assert np.isclose(r1_fit, r1 * scale)
    assert np.isclose(r2_fit, r2 * scale)
    assert np.isclose(r0_fit / r1_fit, r0 / r1)
    assert np.isclose(r0_fit / r2_fit, r0 / r2)


def test_pure_shape_preserves_geometric_mean_radius():
    """Changing only shape_1 or shape_2 preserves the geometric-mean radius and matches the expected log-ratio change."""
    ellipsoid = PeakEllipsoid()

    x0 = np.linspace(-1, 1, 5)
    x1 = np.linspace(-1, 1, 5)
    x2 = np.linspace(-1, 1, 5)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    r0, r1, r2 = 0.2, 0.3, 0.4
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    g0 = (r0 * r1 * r2) ** (1 / 3)

    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)
    ellipsoid.params["shape_1"].set(value=0.4)
    r0_fit, r1_fit, r2_fit, *_ = ellipsoid.shape_from_params(ellipsoid.params)
    g = (r0_fit * r1_fit * r2_fit) ** (1 / 3)
    assert np.isclose(g, g0)
    assert np.isclose(
        np.log(r0_fit / r1_fit) - np.log(r0 / r1), np.sqrt(2) * 0.4
    )

    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)
    ellipsoid.params["shape_2"].set(value=-0.3)
    r0_fit, r1_fit, r2_fit, *_ = ellipsoid.shape_from_params(ellipsoid.params)
    g = (r0_fit * r1_fit * r2_fit) ** (1 / 3)
    assert np.isclose(g, g0)
    assert np.isclose(
        np.log((r0_fit * r1_fit) / r2_fit**2) - np.log((r0 * r1) / r2**2),
        np.sqrt(6) * -0.3,
    )


def test_pure_orientation_leaves_radii_unchanged():
    """Changing only domega leaves the radii unchanged and rotates the orientation by exactly Exp(domega)."""
    ellipsoid = PeakEllipsoid()

    x0 = np.linspace(-1, 1, 5)
    x1 = np.linspace(-1, 1, 5)
    x2 = np.linspace(-1, 1, 5)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    r0, r1, r2 = 0.2, 0.3, 0.4
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)

    domega = np.array([0.1, -0.2, 0.05])
    ellipsoid.params["domega_x"].set(value=domega[0])
    ellipsoid.params["domega_y"].set(value=domega[1])
    ellipsoid.params["domega_z"].set(value=domega[2])

    (
        r0_fit,
        r1_fit,
        r2_fit,
        u0_fit,
        u1_fit,
        u2_fit,
    ) = ellipsoid.shape_from_params(ellipsoid.params)
    assert np.isclose(r0_fit, r0)
    assert np.isclose(r1_fit, r1)
    assert np.isclose(r2_fit, r2)

    U_fit = ellipsoid.U_matrix(u0_fit, u1_fit, u2_fit)
    R_rel = ellipsoid._prior_U.T @ U_fit
    expected = scipy.spatial.transform.Rotation.from_rotvec(domega).as_matrix()
    assert np.allclose(R_rel, expected, atol=1e-10)


def test_prior_jacobian():
    """Verify prior_jacobian against central finite differences of prior_residual."""
    ellipsoid = PeakEllipsoid()

    x0 = np.linspace(1, 2, 5)
    x1 = np.linspace(-3, -2, 6)
    x2 = np.linspace(0, 1, 7)
    x0, x1, x2 = np.meshgrid(x0, x1, x2, indexing="ij")

    r0, r1, r2 = 0.2, 0.3, 0.4
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    _make_reference(ellipsoid, x0, x1, x2, r0, r1, r2, axes)

    delta = 1e-6

    prior_params = Parameters()
    prior_params.add("c0", value=0.05)
    prior_params.add("c1", value=-0.03)
    prior_params.add("c2", value=0.02)
    prior_params.add("log_size", value=0.1)
    prior_params.add("shape_1", value=-0.05)
    prior_params.add("shape_2", value=0.03)
    prior_params.add("domega_x", value=0.02)
    prior_params.add("domega_y", value=-0.01)
    prior_params.add("domega_z", value=0.04)

    prior_jac = ellipsoid.prior_jacobian(prior_params)

    for i, name in enumerate(
        [
            "c0",
            "c1",
            "c2",
            "log_size",
            "shape_1",
            "shape_2",
            "domega_x",
            "domega_y",
            "domega_z",
        ]
    ):
        params_p, params_m = prior_params.copy(), prior_params.copy()
        params_p[name].set(value=prior_params[name].value + delta)
        params_m[name].set(value=prior_params[name].value - delta)
        res_p = ellipsoid.prior_residual(params_p)
        res_m = ellipsoid.prior_residual(params_m)
        fd = (res_p - res_m) / (2 * delta)
        assert np.allclose(prior_jac[i], fd, atol=1e-6, rtol=1e-5)


def test_full_jacobian():
    """Verify the full stacked jacobian (1d/2d/3d + prior) against finite differences of the full residual."""
    ellipsoid = PeakEllipsoid()

    modes = ["3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2"]

    np.random.seed(0)

    nx, ny, nz = 5, 6, 7
    xg0, xg1, xg2 = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1.2, 1.3, ny),
        np.linspace(-0.8, 0.9, nz),
        indexing="ij",
    )

    r0, r1, r2 = 0.5, 0.6, 0.7
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    _make_reference(ellipsoid, xg0, xg1, xg2, r0, r1, r2, axes)

    def rand_counts(shape):
        return np.random.randint(2, 20, size=shape).astype(float)

    def rand_norm(shape):
        return np.random.uniform(0.5, 2.0, size=shape)

    d1d = [rand_counts(shape) for shape in [(nx,), (ny,), (nz,)]]
    n1d = [rand_norm(shape) for shape in [(nx,), (ny,), (nz,)]]
    args_1d = [xg0, xg1, xg2, d1d, n1d, None, None]

    d2d = [rand_counts(shape) for shape in [(ny, nz), (nx, nz), (nx, ny)]]
    n2d = [rand_norm(shape) for shape in [(ny, nz), (nx, nz), (nx, ny)]]
    args_2d = [xg0, xg1, xg2, d2d, n2d, None, None]

    args_3d = [
        xg0,
        xg1,
        xg2,
        rand_counts((nx, ny, nz)),
        rand_norm((nx, ny, nz)),
        None,
        None,
    ]

    full_params = ellipsoid.params.copy()
    full_params["c0"].set(value=0.05)
    full_params["c1"].set(value=-0.03)
    full_params["c2"].set(value=0.02)
    full_params["log_size"].set(value=0.1)
    full_params["shape_1"].set(value=-0.05)
    full_params["shape_2"].set(value=0.03)
    full_params["domega_x"].set(value=0.02)
    full_params["domega_y"].set(value=-0.01)
    full_params["domega_z"].set(value=0.04)
    for mode in modes:
        full_params.add("A" + mode, value=5.0)
        full_params.add("B" + mode, value=1.0)

    delta = 1e-6

    full_jac = ellipsoid.jacobian(full_params, args_1d, args_2d, args_3d)

    varying = [name for name, par in full_params.items() if par.vary]
    for i, name in enumerate(varying):
        params_p, params_m = full_params.copy(), full_params.copy()
        params_p[name].set(value=full_params[name].value + delta)
        params_m[name].set(value=full_params[name].value - delta)
        res_p = ellipsoid.residual(params_p, args_1d, args_2d, args_3d)
        res_m = ellipsoid.residual(params_m, args_1d, args_2d, args_3d)
        fd = (res_p - res_m) / (2 * delta)
        assert np.allclose(full_jac[:, i], fd, atol=1e-4, rtol=1e-3)


def _grid(n=41, half_extent=3.0):
    ax = np.linspace(-half_extent, half_extent, n)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def test_missing_support_fraction_no_gap():
    """With normalization valid everywhere and the box comfortably covering the peak, missing_fraction must be ~zero."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    c = (0.0, 0.0, 0.0)
    inv_S = ellipsoid.inv_S_matrix(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    n = np.ones_like(x0)

    frac, missing, total = ellipsoid.missing_support_fraction(
        x0, x1, x2, c, inv_S, n
    )
    assert total > 0
    # missing = total_support(analytic) - valid_support(box sum); with no
    # gap and the box comfortably covering the peak, these differ only by
    # floating-point noise, not by an exact 0 (unlike a pure box-sum
    # denominator, where an all-valid mask trivially zeroes this term).
    assert missing < 1e-12
    assert frac < 1e-12


def test_missing_support_fraction_gap_far_away():
    """Invalid voxels far outside the peak's support must leave the fraction negligible."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    c = (0.0, 0.0, 0.0)
    inv_S = ellipsoid.inv_S_matrix(0.3, 0.3, 0.3, 0.0, 0.0, 0.0)
    n = np.ones_like(x0)
    n[x0 > 2.5] = 0.0

    frac, _, total = ellipsoid.missing_support_fraction(
        x0, x1, x2, c, inv_S, n
    )
    assert total > 0
    assert frac < 1e-6


def test_missing_support_fraction_all_missing():
    """When no voxel has valid normalization, the fraction must be 1."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    c = (0.0, 0.0, 0.0)
    inv_S = ellipsoid.inv_S_matrix(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    n = np.zeros_like(x0)

    frac, missing, total = ellipsoid.missing_support_fraction(
        x0, x1, x2, c, inv_S, n
    )
    assert total > 0
    assert np.isclose(frac, 1.0)
    assert np.isclose(missing, total)


def test_missing_support_fraction_increases_as_center_approaches_gap():
    """Moving the ellipsoid center toward an invalid half-space must increase the fraction monotonically."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    inv_S = ellipsoid.inv_S_matrix(0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    n = np.ones_like(x0)
    n[x0 > 0.0] = 0.0

    fractions = []
    for c0 in (-1.0, -0.5, 0.0, 0.5, 1.0):
        frac, _, _ = ellipsoid.missing_support_fraction(
            x0, x1, x2, (c0, 0.0, 0.0), inv_S, n
        )
        fractions.append(frac)

    assert np.all(np.diff(fractions) > 0)


def test_missing_support_fraction_grows_with_size_beyond_the_box():
    """A peak growing large enough to spill past the box edge must be penalized, even with zero explicit gaps."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    c = (0.0, 0.0, 0.0)
    n = np.ones_like(x0)  # valid everywhere: no internal gaps at all

    fractions = []
    for r in (0.3, 1.0, 3.0, 10.0, 50.0):
        inv_S = ellipsoid.inv_S_matrix(r, r, r, 0.0, 0.0, 0.0)
        frac, _, _ = ellipsoid.missing_support_fraction(
            x0, x1, x2, c, inv_S, n
        )
        fractions.append(frac)

    # Small peaks comfortably inside the box: negligible fraction.
    assert fractions[0] < 1e-9
    # As the peak grows past the box, the fraction climbs toward 1 --
    # unlike a box-sum denominator, which would saturate at the box's own
    # (here zero) invalid-voxel fraction and never flag this at all.
    assert np.all(np.diff(fractions) > 0)
    assert fractions[-1] > 0.9


def test_missing_support_fraction_amplitude_invariant():
    """The unit-amplitude profile means missing_fraction must not depend on any amplitude scaling."""
    ellipsoid = PeakEllipsoid()
    x0, x1, x2 = _grid()

    c = (0.3, 0.0, 0.0)
    inv_S = ellipsoid.inv_S_matrix(0.5, 0.5, 0.5, 0.0, 0.0, 0.0)
    n = np.ones_like(x0)
    n[x0 > 0.5] = 0.0

    frac1, _, total1 = ellipsoid.missing_support_fraction(
        x0, x1, x2, c, inv_S, n
    )

    assert np.isclose(total1, ellipsoid.gaussian_integral(inv_S, mode="3d"))

    for amplitude in (1e-3, 1.0, 50.0):
        scaled_total = amplitude * total1
        assert np.isclose(scaled_total / amplitude, total1)
    assert frac1 > 0.0 and frac1 < 1.0
