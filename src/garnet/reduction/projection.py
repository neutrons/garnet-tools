import numpy as np

import scipy.optimize

from garnet.reduction.plan import SubPlan


def align_shape_axes(radii, axes, ref_axes):
    """
    Permute and sign-fix principal axes to best align with reference axes.

    Eigenvectors are only defined up to sign, and up to permutation
    when eigenvalues are close (near-spherical peaks), so without an
    anchor the axes returned from one fit to the next can flip sign or
    swap order on numerical noise alone. Assigns each reference axis
    the shape axis it overlaps most with (via a linear-sum-assignment
    matching on the pairwise dot products), flips signs so each match
    is positive, then re-enforces a right-handed triple.

    Parameters
    ----------
    radii : sequence of float
        (r0, r1, r2) principal radii, matching the order of `axes`.
    axes : sequence of 1d-array
        (v0, v1, v2) principal axis vectors to reorder/sign-fix.
    ref_axes : sequence of 1d-array
        (n, u, v) reference directions; output axis j is the `axes`
        entry closest aligned with `ref_axes[j]`.

    Returns
    -------
    radii : tuple of float
        Reordered to match the returned axes.
    axes : tuple of 1d-array
        Reordered and sign-fixed, right-handed.

    """

    V = np.column_stack(axes)
    N = np.column_stack(ref_axes)

    overlap = V.T @ N

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(-np.abs(overlap))

    order = np.empty(3, dtype=int)
    for i, j in zip(row_ind, col_ind):
        order[j] = i

    new_radii = tuple(radii[order[j]] for j in range(3))

    new_axes = np.empty((3, 3))
    for j in range(3):
        i = order[j]
        sign = np.sign(overlap[i, j])
        new_axes[:, j] = (sign if sign != 0 else 1.0) * V[:, i]

    if np.linalg.det(new_axes) < 0:
        new_axes[:, 2] *= -1

    return new_radii, (new_axes[:, 0], new_axes[:, 1], new_axes[:, 2])


def bin_axes(R, two_theta, az_phi):
    """
    Compute local Q-sample resolution axes aligned to the scattering geometry.

    These are the axes the anisotropic Q-resolution model (see
    `data.get_resolution_in_Q`) is defined along, not necessarily the
    axes used to bin the data (see `bin_extent`).

    Parameters
    ----------
    R : 2d-array
        Goniometer rotation matrix.
    two_theta : float
        Scattering angle in degrees.
    az_phi : float
        Azimuthal angle in degrees.

    Returns
    -------
    n, u, v : 1d-array
        Orthogonal resolution axes in Q-sample frame.

    """

    two_theta = np.deg2rad(two_theta)
    az_phi = np.deg2rad(az_phi)

    kf_hat = np.array(
        [
            np.sin(two_theta) * np.cos(az_phi),
            np.sin(two_theta) * np.sin(az_phi),
            np.cos(two_theta),
        ]
    )

    ki_hat = np.array([0, 0, 1])

    n = kf_hat - ki_hat
    n /= np.linalg.norm(n)

    u = kf_hat + ki_hat
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    return R.T @ n, R.T @ u, R.T @ v


def project_ellipsoid_parameters(params, projections):
    """
    Project peak shape parameters into local projection frame.

    Parameters
    ----------
    params : tuple
        c0, c1, c2, r0, r1, r2, v0, v1, v2 in Q-sample frame.
    projections : list
        Three orthonormal projection vectors.

    Returns
    -------
    tuple
        Projected c0, c1, c2, r0, r1, r2, v0, v1, v2.

    """

    W = np.column_stack(projections)

    c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

    V = np.column_stack([v0, v1, v2])

    return *np.dot(W.T, [c0, c1, c2]), r0, r1, r2, *np.dot(W.T, V).T


def revert_ellipsoid_parameters(params, projections):
    """
    Revert peak shape parameters from local projection frame to Q-sample.

    Parameters
    ----------
    params : tuple
        c0, c1, c2, r0, r1, r2, v0, v1, v2 in projection frame.
    projections : list
        Three orthonormal projection vectors.

    Returns
    -------
    tuple
        Reverted c0, c1, c2, r0, r1, r2, v0, v1, v2.

    """

    W = np.column_stack(projections)

    c0, c1, c2, r0, r1, r2, v0, v1, v2 = params

    V = np.column_stack([v0, v1, v2])

    return *np.dot(W, [c0, c1, c2]), r0, r1, r2, *np.dot(W, V).T


def transform_Q(Q0, Q1, Q2, projections):
    """
    Transform Q-coordinates from projection frame to Q-sample.

    Parameters
    ----------
    Q0, Q1, Q2 : array
        Coordinate grids in projection frame.
    projections : list
        Three orthonormal projection vectors.

    Returns
    -------
    array
        3 x ... array of Q-sample coordinates.

    """

    W = np.column_stack(projections)

    return np.einsum("ij,j...->i...", W, [Q0, Q1, Q2])


def bin_extent(
    UB, hkl, lamda, R, two_theta, az_phi, shape, dQ, bin_min=19, bin_max=21
):
    """
    Compute bin extents and projection transform for one peak.

    Parameters
    ----------
    UB : 2d-array
        UB matrix.
    hkl : list
        Miller indices.
    lamda : float
        Wavelength in angstroms.
    R : 2d-array
        Goniometer rotation matrix.
    two_theta : float
        Scattering angle in degrees.
    az_phi : float
        Azimuthal angle in degrees.
    shape : tuple
        Peak shape parameters (ci, ri, vi). The peak's own principal axes
        (vi) are used as the projection basis, so the projection frame
        coincides with the ellipsoid's principal frame.
    dQ : array
        Q-resolution (3-element), along the scattering-geometry axes
        from `bin_axes` (not the projection frame).
    bin_min, bin_max : int
        Minimum and maximum bin count per dimension.

    Returns
    -------
    bins : array
        Number of bins per dimension.
    extents : array
        Min/max extents (3 x 2).
    projections : list
        Three orthonormal projection vectors.
    transform : list
        Rows of the HKL to Q projection matrix.
    conversion : array
        Matrix converting HKL offsets to projection-frame Q.

    """

    _, _, _, r0, r1, r2, v0, v1, v2 = shape

    n, u, v = bin_axes(R, two_theta, az_phi)
    (r0, r1, r2), (v0, v1, v2) = align_shape_axes(
        (r0, r1, r2), (v0, v1, v2), (n, u, v)
    )

    projections = [v0, v1, v2]

    r_cut = 3 * np.array([r0, r1, r2])

    N = np.column_stack([n, u, v])
    W = np.column_stack(projections)

    Sres = W.T @ (N @ np.diag(np.asarray(dQ) ** 2) @ N.T) @ W
    dQ_proj = np.sqrt(np.diag(Sres))

    r_cut = np.where(r_cut < 3 * dQ_proj, 3 * dQ_proj, r_cut)

    A = 2 * np.pi * (W.T @ UB)

    miller_half_width = 0.5 * np.sum(np.abs(A), axis=1)
    r_cut = np.minimum(r_cut, miller_half_width)

    bins = np.clip(1 + np.floor(r_cut / dQ_proj).astype(int), bin_min, bin_max)

    Wp = np.linalg.inv(W.T @ (2 * np.pi * UB)).T
    transform = Wp.tolist()

    h, k, l = hkl

    Q0_c, Q1_c, Q2_c = A @ np.array([h, k, l])

    extents = np.array(
        [
            [Q0_c - r_cut[0], Q0_c + r_cut[0]],
            [Q1_c - r_cut[1], Q1_c + r_cut[1]],
            [Q2_c - r_cut[2], Q2_c + r_cut[2]],
        ]
    )

    conversion = A.copy()

    return bins, extents, projections, transform, conversion


def voxel_weights(Q0, Q1, Q2, c, neighbors, t_max=0.95, k_nearest=12):
    """
    Compute per-voxel weights that down-weight voxels nearer to a neighbour peak.

    Parameters
    ----------
    Q0, Q1, Q2 : array
        Coordinate grids.
    c : array
        Peak center in projection frame.
    neighbors : list
        Neighboring peak centers in projection frame.
    t_max : float
        Clipping threshold for proximity parameter.
    k_nearest : int
        Maximum number of neighbours to consider.

    Returns
    -------
    weights : array
        Weight array with same shape as Q2.

    """

    weights = np.ones_like(Q2)

    if neighbors:
        dx = np.stack([Q0 - c[0], Q1 - c[1], Q2 - c[2]], axis=-1)

        box_radius = np.sqrt(np.einsum("...i,...i->...", dx, dx).max())
        deltas = np.array([q_j - c for q_j in neighbors])
        dist_sqs = np.einsum("ji,ji->j", deltas, deltas)

        nearby = dist_sqs < (2.0 * box_radius) ** 2
        if nearby.any():
            deltas = deltas[nearby]
            dist_sqs = dist_sqs[nearby]

            if len(dist_sqs) > k_nearest:
                idx = np.argpartition(dist_sqs, k_nearest)[:k_nearest]
                deltas = deltas[idx]
                dist_sqs = dist_sqs[idx]

            dots = np.einsum("...i,ji->...j", dx, deltas)
            t_all = np.clip(dots / (0.5 * dist_sqs), 0.0, t_max)
            weights = (1.0 - t_all).min(axis=-1)

    return weights


class PeakProjection(SubPlan):
    """
    Base class providing shared geometry and binning utilities for
    integration workflows.
    """

    def project_ellipsoid_parameters(self, params, projections):
        return project_ellipsoid_parameters(params, projections)

    def revert_ellipsoid_parameters(self, params, projections):
        return revert_ellipsoid_parameters(params, projections)

    def transform_Q(self, Q0, Q1, Q2, projections):
        return transform_Q(Q0, Q1, Q2, projections)

    def bin_extent(self, *args, **kwargs):
        return bin_extent(*args, **kwargs)

    def voxel_weights(self, *args, **kwargs):
        return voxel_weights(*args, **kwargs)
