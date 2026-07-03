import os
import numpy as np

import scipy.linalg
import scipy.spatial
import scipy.spatial.transform

from garnet.reduction.integration import PeakEllipsoid
from garnet.reduction.ellipsoid import _R_SCALE_3D
from garnet.plots.peaks import PeakPlot
from garnet.plots.volume import SlicePlot

filepath = os.path.dirname(os.path.abspath(__file__))


def test_slice_plot():
    np.random.seed(13)

    U = scipy.spatial.transform.Rotation.random().as_matrix()

    a, b, c = 5, 5, 7
    alpha, beta, gamma = np.deg2rad([90, 90, 120])

    G = np.array(
        [
            [a**2, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b**2, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c**2],
        ]
    )

    B = scipy.linalg.cholesky(np.linalg.inv(G), lower=False)

    UB = U @ B

    W = np.eye(3)

    x0 = np.linspace(-5, 10, 31)
    x1 = np.linspace(-3, 9, 25)
    x2 = np.linspace(-7, 8, 61)

    axes = x0, x1, x2

    X0, X1, X2 = np.meshgrid(*axes, indexing="ij")

    signal = np.ones_like(X0)

    signal[(X0 % 1 == 0) & (X1 % 1 == 0) & (X2 % 1 == 0)] = 1000

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [0, 0, 1])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice2.png"))

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [0, 1, 0])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice1.png"))

    plot = SlicePlot(UB, W)
    plot.calculate_transforms(axes, ["h", "k", "l"], [1, 0, 0])
    plot.make_slice(signal, 0)
    plot.save_plot(os.path.join(filepath, "slice0.png"))


# def test_radius_plot():
#     r_cut = 0.25

#     A = 1.2
#     s = 0.1

#     r = np.linspace(0, r_cut, 51)

#     I = A * np.tanh((r / s) ** 3)

#     sphere = PeakSphere(r_cut)

#     radius = sphere.fit(r, I)

#     I_fit, *vals = sphere.best_fit(r)

#     plot = RadiusPlot(r, I, I_fit)

#     plot.add_sphere(radius, *vals)

#     plot.save_plot(os.path.join(filepath, "sphere.png"))


def test_init_peak_plot():
    plot = PeakPlot()

    file = os.path.join(filepath, "ellipsoid_init.png")

    plot.save_plot(file)

    assert os.path.exists(file)


def test_peak_plot():
    np.random.seed(13)

    nx, ny, nz = 21, 24, 31

    Qx_min, Qx_max = -1, 1
    Qy_min, Qy_max = -1, 1
    Qz_min, Qz_max = -1, 1

    # true peak center offset slightly from the predicted (0, 0, 0) position
    Q0_x, Q0_y, Q0_z = 0.05, -0.03, 0.02

    sigma_x, sigma_y, sigma_z = 0.15, 0.2, 0.12
    rho_yz, rho_xz, rho_xy = 0.3, -0.1, 0.15

    sigma_yz = sigma_y * sigma_z
    sigma_xz = sigma_x * sigma_z
    sigma_xy = sigma_x * sigma_y

    cov = np.array(
        [
            [sigma_x**2, rho_xy * sigma_xy, rho_xz * sigma_xz],
            [rho_xy * sigma_xy, sigma_y**2, rho_yz * sigma_yz],
            [rho_xz * sigma_xz, rho_yz * sigma_yz, sigma_z**2],
        ]
    )

    Q0 = np.array([Q0_x, Q0_y, Q0_z])

    signal = np.random.multivariate_normal(Q0, cov, size=200000)

    counts, bins = np.histogramdd(
        signal,
        density=False,
        bins=[nx, ny, nz],
        range=[(Qx_min, Qx_max), (Qy_min, Qy_max), (Qz_min, Qz_max)],
    )

    x_bin_edges, y_bin_edges, z_bin_edges = bins

    Qx = 0.5 * (x_bin_edges[1:] + x_bin_edges[:-1])
    Qy = 0.5 * (y_bin_edges[1:] + y_bin_edges[:-1])
    Qz = 0.5 * (z_bin_edges[1:] + z_bin_edges[:-1])

    a = 5.0
    b = 20.0

    counts = counts * 5000.0 + b + a * (2 * np.random.random(counts.shape) - 1)
    norm = np.full_like(counts, 1.0)

    Qx, Qy, Qz = np.meshgrid(Qx, Qy, Qz, indexing="ij")

    dQ = Qx[1, 0, 0] - Qx[0, 0, 0]
    xmod = 0.0

    weights = np.ones_like(counts)

    # predicted ("resolution model") ellipsoid, deliberately a bit off from
    # the truth so the fit has something to refine
    eigval, eigvec = np.linalg.eigh(cov)
    r_true = np.sqrt(eigval) * _R_SCALE_3D
    r_pred = r_true * 1.3

    rot_perturb = scipy.spatial.transform.Rotation.from_rotvec(
        [0.1, -0.05, 0.05]
    ).as_matrix()
    U_pred = rot_perturb @ eigvec

    shape = (
        0.0,
        0.0,
        0.0,
        r_pred[0],
        r_pred[1],
        r_pred[2],
        U_pred[:, 0],
        U_pred[:, 1],
        U_pred[:, 2],
    )

    ellipsoid = PeakEllipsoid()
    ellipsoid.update_constraints(Qx, Qy, Qz, dQ)
    ellipsoid.update_estimate(shape)

    args = (Qx, Qy, Qz, counts, norm, dQ, xmod, weights)
    fit_params = ellipsoid.fit(*args)

    assert fit_params is not None

    ellipsoid.extract_result(*fit_params, xmod)

    c, S, *best_fit = ellipsoid.best_fit

    hkl = [1, 2, 3]
    d = 3.14

    wavelength = 3.2887

    angles = 60, 0
    goniometer = [0, 0, 0]

    plot = PeakPlot()

    norm_params = Qx, Qy, Qz, counts, norm, c, S

    ellipsoid.integrate(*norm_params)

    plot.add_ellipsoid_fit(best_fit)

    plot.add_profile_fit(ellipsoid.best_prof)

    plot.add_projection_fit(ellipsoid.best_proj)

    plot.add_ellipsoid(c, S)

    plot.add_peak_info(hkl, d, wavelength, angles, goniometer)

    plot.add_peak_stats(ellipsoid.reddev, ellipsoid.intensity, ellipsoid.sigma)

    plot.add_data_norm_fit(*ellipsoid.data_norm_fit)

    plot.update_envelope(*ellipsoid.peak_background_mask)

    plot.add_integral_fit(ellipsoid.integral)

    file = os.path.join(filepath, "ellipsoid.png")

    plot.save_plot(file)

    assert os.path.exists(file)


def test_peak_plot_reuses_figure_between_saves():
    plot = PeakPlot()

    file_0 = os.path.join(filepath, "ellipsoid_reuse_0.png")
    file_1 = os.path.join(filepath, "ellipsoid_reuse_1.png")

    fig_id = id(plot.fig)

    plot.save_plot(file_0)
    plot.save_plot(file_1)

    assert os.path.exists(file_0)
    assert os.path.exists(file_1)
    assert id(plot.fig) == fig_id

    plot.close()
