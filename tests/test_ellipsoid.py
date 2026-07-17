import numpy as np

from lmfit import Parameters

from garnet.reduction.ellipsoid import PeakEllipsoid


def test_ellipsoid_jacobians():
    """Verify every analytic Jacobian in PeakEllipsoid against central finite differences."""
    ellipsoid = PeakEllipsoid()

    modes = ["3d", "2d_0", "2d_1", "2d_2", "1d_0", "1d_1", "1d_2"]

    r0, r1, r2 = 0.2, 0.3, 0.4
    u0, u1, u2 = 0.2, 0.1, 0.4

    delta = 1e-6

    # inv_S_deriv_r / inv_S_deriv_u vs finite differences of inv_S_matrix
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

    du = ellipsoid.inv_S_deriv_u(r0, r1, r2, u0, u1, u2)
    for i in range(3):
        us_p, us_m = [u0, u1, u2], [u0, u1, u2]
        us_p[i] += delta
        us_m[i] -= delta
        fd = (
            ellipsoid.inv_S_matrix(r0, r1, r2, *us_p)
            - ellipsoid.inv_S_matrix(r0, r1, r2, *us_m)
        ) / (2 * delta)
        assert np.allclose(du[i], fd, atol=1e-6, rtol=1e-6)

    inv_S0 = ellipsoid.inv_S_matrix(r0, r1, r2, u0, u1, u2)

    # gaussian_integral_jac_S vs finite differences of gaussian_integral
    for mode in modes:
        dg_r = ellipsoid.gaussian_integral_jac_S(inv_S0, dr, mode=mode)
        for i in range(3):
            rs_p, rs_m = [r0, r1, r2], [r0, r1, r2]
            rs_p[i] += delta
            rs_m[i] -= delta
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
            us_p, us_m = [u0, u1, u2], [u0, u1, u2]
            us_p[i] += delta
            us_m[i] -= delta
            g_p = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(r0, r1, r2, *us_p), mode=mode
            )
            g_m = ellipsoid.gaussian_integral(
                ellipsoid.inv_S_matrix(r0, r1, r2, *us_m), mode=mode
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.isclose(dg_u[i], fd, atol=1e-6, rtol=1e-5)

    # gaussian_jac_c / gaussian_jac_S vs finite differences of gaussian
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
        for i in range(3):
            rs_p, rs_m = [r0, r1, r2], [r0, r1, r2]
            rs_p[i] += delta
            rs_m[i] -= delta
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
            us_p, us_m = [u0, u1, u2], [u0, u1, u2]
            us_p[i] += delta
            us_m[i] -= delta
            g_p = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(r0, r1, r2, *us_p),
                mode=mode,
            )
            g_m = ellipsoid.gaussian(
                x0,
                x1,
                x2,
                c,
                ellipsoid.inv_S_matrix(r0, r1, r2, *us_m),
                mode=mode,
            )
            fd = (g_p - g_m) / (2 * delta)
            assert np.allclose(dgu[i], fd, atol=1e-6, rtol=1e-5)

    # prior_jacobian vs finite differences of prior_residual
    ellipsoid.update_constraints(x0, x1, x2, dx=None)
    axes = (np.eye(3)[0], np.eye(3)[1], np.eye(3)[2])
    ellipsoid.update_estimate((0, 0, 0, r0, r1, r2, *axes))

    prior_params = Parameters()
    prior_params.add("c0", value=0.05)
    prior_params.add("c1", value=-0.03)
    prior_params.add("c2", value=0.02)
    prior_params.add("r0", value=r0 * 1.1)
    prior_params.add("r1", value=r1 * 0.9)
    prior_params.add("r2", value=r2 * 1.05)
    prior_params.add("u0", value=u0)
    prior_params.add("u1", value=u1)
    prior_params.add("u2", value=u2)

    prior_jac = ellipsoid.prior_jacobian(prior_params)

    for i, name in enumerate(
        ["c0", "c1", "c2", "r0", "r1", "r2", "u0", "u1", "u2"]
    ):
        params_p, params_m = prior_params.copy(), prior_params.copy()
        params_p[name].set(value=prior_params[name].value + delta)
        params_m[name].set(value=prior_params[name].value - delta)
        res_p = ellipsoid.prior_residual(params_p)
        res_m = ellipsoid.prior_residual(params_m)
        fd = (res_p - res_m) / (2 * delta)
        assert np.allclose(prior_jac[i], fd, atol=1e-6, rtol=1e-5)

    # Full stacked jacobian (jacobian_1d/2d/3d, jacobian_mode_poisson, and
    # prior_jacobian together) vs finite differences of the full residual.
    np.random.seed(0)

    nx, ny, nz = 5, 6, 7
    xg0, xg1, xg2 = np.meshgrid(
        np.linspace(-1, 1, nx),
        np.linspace(-1.2, 1.3, ny),
        np.linspace(-0.8, 0.9, nz),
        indexing="ij",
    )

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

    full_params = Parameters()
    full_params.add("c0", value=0.05)
    full_params.add("c1", value=-0.03)
    full_params.add("c2", value=0.02)
    full_params.add("r0", value=0.5)
    full_params.add("r1", value=0.6)
    full_params.add("r2", value=0.7)
    full_params.add("u0", value=0.2)
    full_params.add("u1", value=0.1)
    full_params.add("u2", value=0.4)
    for mode in modes:
        full_params.add("A" + mode, value=5.0)
        full_params.add("B" + mode, value=1.0)

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
