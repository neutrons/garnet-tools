from mantid.api import (
    PythonAlgorithm,
    AlgorithmFactory,
    IPeaksWorkspaceProperty,
    WorkspaceProperty,
)
from mantid.kernel import (
    Direction,
    StringListValidator,
    FloatBoundedValidator,
    IntBoundedValidator,
)
from mantid.dataobjects import TableWorkspaceProperty
from mantid.simpleapi import CreateEmptyTableWorkspace, SetUB
from mantid.geometry import UnitCell

import numpy as np

from garnet.reduction.search import (
    normalize,
    direct_basis_from_lattice,
    reciprocal_basis_from_direct_basis,
    conventional_to_primitive_lattice,
    primitive_ub_to_conventional_ub,
)


def enumerate_reciprocal_directions(max_index):
    """
    Enumerate primitive integer Miller index directions up to a cutoff.

    Every non-zero integer triple `(h, k, l)` with `max(|h|, |k|, |l|)`
    at most `max_index` is included, except those sharing a common
    factor (e.g. `(2, 0, 0)` is dropped since `(1, 0, 0)` already
    covers that direction). Both a triple and its negation are kept,
    since they point in opposite directions.

    Parameters
    ----------
    max_index : int
        Largest Miller index magnitude to enumerate.

    Returns
    -------
    hkls : ndarray of shape (n_directions, 3)
        Primitive integer Miller index triples.
    """
    grid = np.arange(-max_index, max_index + 1)
    h, k, l = np.meshgrid(grid, grid, grid, indexing="ij")
    hkls = np.column_stack([h.ravel(), k.ravel(), l.ravel()])

    nonzero = np.any(hkls != 0, axis=1)
    hkls = hkls[nonzero]

    gcd = np.gcd.reduce(np.abs(hkls), axis=1)
    primitive = gcd == 1

    return hkls[primitive]


def scattering_directions_from_kf_ki(kf_ki_dir):
    """
    Convert Laue `kf - ki` vectors to reciprocal-vector directions.

    In this facility's Q convention, a peak's `kf_ki_dir` (from
    `Peak.getDetectorDirectionSampleFrame() + getSourceDirectionSampleFrame()`)
    relates to its true reflection by `kf_ki_dir = -wavelength * UB @ hkl`,
    so the direction of `UB @ hkl` is `normalize(-kf_ki_dir)`.

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `extract_kf_ki_directions`.

    Returns
    -------
    d_hat : ndarray of shape (n_peaks, 3)
        Unit vectors in the direction of each peak's true `UB @ hkl`.
    """
    q_dir = -np.asarray(kf_ki_dir, dtype=float)
    return q_dir / np.linalg.norm(q_dir, axis=1, keepdims=True)


def candidate_zone_axes(d_hat):
    """
    Build candidate zone-axis directions from every pair of observations.

    Two Laue reflections are in the same zone (coplanar with the
    origin) when their directions are both perpendicular to a common
    real-space zone-axis direction. The cross product of any two
    observed directions is a candidate for that axis; genuine zones
    with several members will have many pairs voting for nearly the
    same candidate.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Unit vectors, as returned by `scattering_directions_from_kf_ki`.

    Returns
    -------
    z_candidates : ndarray of shape (n_pairs, 3)
        Unit vectors, one per pair of observed directions whose cross
        product is well-defined (excludes near-parallel pairs).
    """
    i_idx, j_idx = np.triu_indices(len(d_hat), k=1)
    cross = np.cross(d_hat[i_idx], d_hat[j_idx])
    norms = np.linalg.norm(cross, axis=1)

    well_defined = norms > 1e-8
    return cross[well_defined] / norms[well_defined, None]


def zone_axis_scores(d_hat, z_candidates, tol_deg=1.0):
    """
    Score candidate zone axes by how tightly observations hug their plane.

    With tens of thousands of candidate axes (one per pair of observed
    directions) tested against a modest peak count, a hard "within
    tolerance" count is vulnerable to spurious candidates that
    accumulate many barely-passing near-boundary matches, occasionally
    outscoring a true zone with fewer but exactly-aligned members. A
    Gaussian-weighted density in the deviation from exactly in-plane
    (`dot == 0`) instead rewards exactness, so noise accumulated near
    the tolerance boundary contributes far less than genuine, tightly
    clustered zone members.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Unit vectors, as returned by `scattering_directions_from_kf_ki`.
    z_candidates : ndarray of shape (n_candidates, 3)
        Candidate zone-axis unit vectors, as returned by
        `candidate_zone_axes`.
    tol_deg : float, optional
        Angular tolerance, in degrees; deviations at this scale are
        down-weighted well below deviations near zero.

    Returns
    -------
    scores : ndarray of shape (n_candidates,)
        Summed Gaussian weight of observed directions' closeness to
        each candidate zone's great circle.
    """
    sigma = np.sin(np.deg2rad(tol_deg)) / 2.0
    deviation = d_hat @ z_candidates.T
    weight = np.exp(-0.5 * (deviation / sigma) ** 2)
    return weight.sum(axis=0)


def find_two_zone_axes(d_hat, tol_deg=1.0, min_separation_deg=15.0):
    """
    Find two well-separated, densely populated zone axes.

    Builds candidate zone axes from every pair of observed directions
    (see `candidate_zone_axes`), scores them by zone occupancy (see
    `zone_axis_scores`), and picks the best-scoring candidate as the
    first zone axis, then the highest-scoring remaining candidate at
    least `min_separation_deg` away from it as the second.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Unit vectors, as returned by `scattering_directions_from_kf_ki`.
    tol_deg : float, optional
        Passed to `zone_axis_scores`.
    min_separation_deg : float, optional
        Minimum line-to-line angle, in degrees, between the two zone
        axes.

    Returns
    -------
    zone1_hat, zone2_hat : ndarray of shape (3,)
        Directions of the two chosen zone axes.
    zone1_score, zone2_score : float
        Occupancy of the two chosen zone axes.
    """
    z_candidates = candidate_zone_axes(d_hat)
    scores = zone_axis_scores(d_hat, z_candidates, tol_deg=tol_deg)

    i1 = np.argmax(scores)
    zone1_hat = z_candidates[i1]

    cos_ang = np.abs(z_candidates @ zone1_hat)
    far_enough = cos_ang <= np.cos(np.deg2rad(min_separation_deg))
    if not np.any(far_enough):
        raise ValueError(
            "No second zone axis found beyond min_separation_deg; "
            "try a smaller min_separation_deg."
        )

    i2 = np.arange(len(scores))[far_enough][np.argmax(scores[far_enough])]
    zone2_hat = z_candidates[i2]

    return zone1_hat, zone2_hat, scores[i1], scores[i2]


def rotation_from_two_correspondences(lab1, lab2, crystal1, crystal2):
    """
    Build the rotation mapping two crystal-frame directions to two
    lab-frame directions.

    Constructs a right-handed orthonormal frame from each pair via
    Gram-Schmidt, then returns the rotation between them. `lab2` and
    `crystal2` need not be exactly reproduced if the two input pairs'
    mutual angles don't exactly match; only their component
    perpendicular to `lab1`/`crystal1` is used.

    Parameters
    ----------
    lab1, lab2 : ndarray of shape (3,)
        Two directions in the lab frame.
    crystal1, crystal2 : ndarray of shape (3,)
        Corresponding two directions in the crystal frame.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Rotation matrix such that `R @ crystal1` is parallel to `lab1`
        and `R @ crystal2` is close to `lab2`.
    """

    def _frame(e1, e2):
        e1 = normalize(e1)
        e2 = normalize(e2 - np.dot(e2, e1) * e1)
        e3 = np.cross(e1, e2)
        return np.column_stack([e1, e2, e3])

    M_lab = _frame(lab1, lab2)
    M_crystal = _frame(crystal1, crystal2)

    return M_lab @ M_crystal.T


def match_directions_to_lattice(d_hat, R, B, candidate_hkls):
    """
    Match observed directions to the closest candidate lattice direction.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Observed unit directions.
    R : ndarray of shape (3, 3)
        Trial crystal-to-lab rotation.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    candidate_hkls : ndarray of shape (n_directions, 3)
        Candidate integer Miller index directions, as returned by
        `enumerate_reciprocal_directions`.

    Returns
    -------
    best_hkl : ndarray of shape (n_peaks, 3)
        Best-matching candidate Miller indices for each peak.
    best_cos : ndarray of shape (n_peaks,)
        Cosine of the angle between each peak's observed direction and
        its best-matching candidate direction.
    """
    lattice_dirs = (R @ B @ candidate_hkls.T).T
    lattice_dirs = lattice_dirs / np.linalg.norm(
        lattice_dirs, axis=1, keepdims=True
    )

    cos_ang = d_hat @ lattice_dirs.T
    best_idx = np.argmax(cos_ang, axis=1)
    rows = np.arange(len(d_hat))

    return candidate_hkls[best_idx], cos_ang[rows, best_idx]


def score_orientation(d_hat, R, B, candidate_hkls, tol_deg=1.0):
    """
    Score a trial orientation by how many observed directions it explains.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Observed unit directions.
    R : ndarray of shape (3, 3)
        Trial crystal-to-lab rotation.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    candidate_hkls : ndarray of shape (n_directions, 3)
        Candidate integer Miller index directions, as returned by
        `enumerate_reciprocal_directions`.
    tol_deg : float, optional
        Angular tolerance, in degrees, for a match to count.

    Returns
    -------
    n_matched : int
        Number of peaks matched within `tol_deg`.
    best_hkl : ndarray of shape (n_peaks, 3)
        Best-matching candidate Miller indices for each peak.
    best_cos : ndarray of shape (n_peaks,)
        Cosine of the angle to each peak's best-matching direction.
    """
    best_hkl, best_cos = match_directions_to_lattice(
        d_hat, R, B, candidate_hkls
    )
    n_matched = int(np.sum(best_cos >= np.cos(np.deg2rad(tol_deg))))
    return n_matched, best_hkl, best_cos


def index_zone_pair(
    d_hat,
    zone1_hat,
    zone2_hat,
    A,
    B,
    max_zone_index,
    max_hkl_index,
    zone_angle_tol_deg=1.0,
    match_tol_deg=1.0,
    max_candidate_pairs=300,
):
    """
    Find the orientation best explaining two observed zone axes.

    Enumerates candidate integer real-space directions up to
    `max_zone_index`, finds pairs whose crystal-frame mutual angle
    matches the observed angle between `zone1_hat` and `zone2_hat`,
    builds the rotation implied by each such pair (see
    `rotation_from_two_correspondences`), and keeps whichever rotation
    explains the most observed reciprocal-space peak directions
    overall (see `score_orientation`, using indices up to
    `max_hkl_index`).

    Real-space zone-axis indices are almost always small integers
    regardless of unit cell size, while the observed diffraction
    indices can be large for large unit cells; keeping `max_zone_index`
    small is what keeps the pairwise angle comparison below tractable,
    independently of how large `max_hkl_index` needs to be for scoring.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Observed unit directions (reciprocal-space).
    zone1_hat, zone2_hat : ndarray of shape (3,)
        Directions of the two zone axes to index, as returned by
        `find_two_zone_axes`.
    A : ndarray of shape (3, 3)
        Direct-lattice basis with a, b, c as columns.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    max_zone_index : int
        Largest index magnitude to consider for either zone axis.
    max_hkl_index : int
        Largest Miller index magnitude to consider when scoring a
        candidate rotation against the observed peaks.
    zone_angle_tol_deg : float, optional
        Tolerance, in degrees, for a candidate pair's crystal-frame
        angle to match the observed angle between the two zone axes.
    match_tol_deg : float, optional
        Passed to `score_orientation` for ranking candidate rotations.
    max_candidate_pairs : int, optional
        Largest number of candidate index pairs to score. Symmetric
        crystals can have very many pairs sharing the same
        crystal-frame angle (e.g. cubic point-group equivalents); only
        the closest-matching pairs are tried, which bounds the search
        cost without favoring any one symmetry-equivalent solution.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Best-scoring crystal-to-lab rotation.
    uvw1, uvw2 : ndarray of shape (3,)
        Real-space indices assigned to `zone1_hat` and `zone2_hat`.
    n_matched : int
        Number of peaks matched by the best rotation.
    """
    candidate_uvw = enumerate_reciprocal_directions(max_zone_index)
    real_dirs = (A @ candidate_uvw.T).T
    real_dirs = real_dirs / np.linalg.norm(real_dirs, axis=1, keepdims=True)

    angle_obs = np.arccos(np.clip(np.dot(zone1_hat, zone2_hat), -1.0, 1.0))

    cos_crystal = np.clip(real_dirs @ real_dirs.T, -1.0, 1.0)
    angle_crystal = np.arccos(cos_crystal)

    angle_tol = np.deg2rad(zone_angle_tol_deg)
    angle_dev = np.abs(angle_crystal - angle_obs)
    i_idx, j_idx = np.nonzero(angle_dev <= angle_tol)
    same_direction = i_idx == j_idx
    i_idx = i_idx[~same_direction]
    j_idx = j_idx[~same_direction]

    if len(i_idx) == 0:
        raise ValueError(
            "No candidate index pair matches the observed zone-axis "
            "angle; try a larger max_zone_index or zone_angle_tol_deg."
        )

    # Symmetric crystals can have very many candidate pairs sharing the
    # same crystal-frame angle (e.g. cubic point-group equivalents);
    # trying only the closest-matching ones bounds the search cost
    # without favoring any one symmetry-equivalent solution over another.
    if len(i_idx) > max_candidate_pairs:
        order = np.argsort(angle_dev[i_idx, j_idx])[:max_candidate_pairs]
        i_idx, j_idx = i_idx[order], j_idx[order]

    candidate_hkls = enumerate_reciprocal_directions(max_hkl_index)

    best = None
    for i, j in zip(i_idx, j_idx):
        R = rotation_from_two_correspondences(
            zone1_hat, zone2_hat, real_dirs[i], real_dirs[j]
        )
        n_matched, _, _ = score_orientation(
            d_hat, R, B, candidate_hkls, tol_deg=match_tol_deg
        )
        if best is None or n_matched > best[0]:
            best = (n_matched, R, candidate_uvw[i], candidate_uvw[j])

    n_matched, R, uvw1, uvw2 = best
    return R, uvw1, uvw2, n_matched


def refine_orientation(d_hat, R0, B, candidate_hkls, tol_deg=1.0):
    """
    Refine a trial orientation using all matched peak directions.

    Matches every observed direction to its closest candidate lattice
    direction under `R0`, then finds the least-squares rotation (via
    the Kabsch algorithm) mapping the matched crystal directions onto
    the observed directions.

    Parameters
    ----------
    d_hat : ndarray of shape (n_peaks, 3)
        Observed unit directions.
    R0 : ndarray of shape (3, 3)
        Initial crystal-to-lab rotation.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    candidate_hkls : ndarray of shape (n_directions, 3)
        Candidate integer Miller index directions, as returned by
        `enumerate_reciprocal_directions`.
    tol_deg : float, optional
        Angular tolerance, in degrees, for a match to be used in the
        refinement.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Refined crystal-to-lab rotation.
    n_matched : int
        Number of peaks used in the refinement.
    """
    best_hkl, best_cos = match_directions_to_lattice(
        d_hat, R0, B, candidate_hkls
    )
    matched = best_cos >= np.cos(np.deg2rad(tol_deg))

    if np.sum(matched) < 2:
        return R0, int(np.sum(matched))

    c_dirs = (B @ best_hkl[matched].T).T
    c_dirs = c_dirs / np.linalg.norm(c_dirs, axis=1, keepdims=True)

    H = c_dirs.T @ d_hat[matched]
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)])
    R = Vt.T @ D @ U.T

    return R, int(np.sum(matched))


def find_orientation_from_kf_ki(
    kf_ki_dir,
    A,
    B,
    max_zone_index,
    max_hkl_index,
    zone_tol_deg=1.0,
    min_zone_separation_deg=15.0,
    zone_angle_tol_deg=1.0,
    match_tol_deg=1.0,
    max_candidate_pairs=300,
):
    """
    Find the crystal orientation from Laue `kf - ki` vectors alone.

    Locates two well-separated zone axes in the observed directions
    (see `find_two_zone_axes`), indexes them against the known lattice
    metric (see `index_zone_pair`), then refines the resulting rotation
    against all matched peaks (see `refine_orientation`). Wavelength is
    never used — orientation is determined purely from directions,
    since a Laue reflection's direction is independent of wavelength.

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `extract_kf_ki_directions`.
    A : ndarray of shape (3, 3)
        Direct-lattice basis with a, b, c as columns.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    max_zone_index : int
        Largest index magnitude to consider when indexing the two
        zone axes. Real-space zone axes are almost always small
        integers regardless of unit cell size, so this can stay small.
    max_hkl_index : int
        Largest Miller index magnitude to consider when scoring
        candidate rotations and refining the final one. Must cover the
        actual range of observed diffraction orders, which can be
        large for large unit cells.
    zone_tol_deg : float, optional
        Passed to `find_two_zone_axes`.
    min_zone_separation_deg : float, optional
        Passed to `find_two_zone_axes`.
    zone_angle_tol_deg : float, optional
        Passed to `index_zone_pair`.
    match_tol_deg : float, optional
        Angular tolerance, in degrees, used both to rank candidate
        rotations and to select peaks for the refinement step.
    max_candidate_pairs : int, optional
        Passed to `index_zone_pair`.

    Returns
    -------
    R : ndarray of shape (3, 3)
        Refined crystal-to-lab rotation.
    info : dict
        Diagnostics with keys "zone1_uvw", "zone2_uvw",
        "n_matched_coarse", and "n_matched_refined".
    """
    d_hat = scattering_directions_from_kf_ki(kf_ki_dir)

    zone1_hat, zone2_hat, _, _ = find_two_zone_axes(
        d_hat,
        tol_deg=zone_tol_deg,
        min_separation_deg=min_zone_separation_deg,
    )

    R0, uvw1, uvw2, n_matched_coarse = index_zone_pair(
        d_hat,
        zone1_hat,
        zone2_hat,
        A,
        B,
        max_zone_index,
        max_hkl_index,
        zone_angle_tol_deg=zone_angle_tol_deg,
        match_tol_deg=match_tol_deg,
        max_candidate_pairs=max_candidate_pairs,
    )

    candidate_hkls = enumerate_reciprocal_directions(max_hkl_index)
    R, n_matched_refined = refine_orientation(
        d_hat, R0, B, candidate_hkls, tol_deg=match_tol_deg
    )

    return R, {
        "zone1_uvw": uvw1,
        "zone2_uvw": uvw2,
        "n_matched_coarse": n_matched_coarse,
        "n_matched_refined": n_matched_refined,
    }


def resolve_hkl_and_wavelength(
    kf_ki_dir,
    UB,
    candidate_hkls,
    wavelength_range,
    tol_deg=1.0,
    max_harmonic=20,
):
    """
    Resolve each peak's Miller indices and wavelength from a known UB.

    For each peak, finds the primitive candidate direction (see
    `enumerate_reciprocal_directions`) that best matches its observed
    direction. A Laue reflection's direction is shared by every integer
    multiple of that primitive vector — different harmonics simply
    diffract at different wavelengths — so every order up to
    `max_harmonic` is checked, and the lowest order whose implied
    wavelength falls within `wavelength_range` is assigned. A peak is
    left unindexed if the best direction match falls outside `tol_deg`
    or no harmonic's implied wavelength falls in range.

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `extract_kf_ki_directions`.
    UB : ndarray of shape (3, 3)
        UB matrix indexed on the conventional cell.
    candidate_hkls : ndarray of shape (n_directions, 3)
        Candidate primitive integer Miller index directions, as
        returned by `enumerate_reciprocal_directions`.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    tol_deg : float, optional
        Angular tolerance, in degrees, for a direction match to be
        accepted.
    max_harmonic : int, optional
        Largest integer multiple of the matched primitive direction to
        consider as a candidate order.

    Returns
    -------
    hkl : ndarray of shape (n_peaks, 3)
        Assigned Miller indices; `(0, 0, 0)` where unindexed.
    wavelength : ndarray of shape (n_peaks,)
        Resolved wavelength, in angstroms; `inf` where unindexed.
    indexed : ndarray of bool, shape (n_peaks,)
        True where a peak was successfully indexed.
    """
    kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
    d_hat = scattering_directions_from_kf_ki(kf_ki_dir)

    lattice_vecs = (UB @ candidate_hkls.T).T
    lattice_lengths = np.linalg.norm(lattice_vecs, axis=1)
    lattice_dirs = lattice_vecs / lattice_lengths[:, None]

    cos_ang = d_hat @ lattice_dirs.T
    best_idx = np.argmax(cos_ang, axis=1)
    rows = np.arange(len(d_hat))
    best_cos = cos_ang[rows, best_idx]

    kf_ki_len = np.linalg.norm(kf_ki_dir, axis=1)
    wavelength_order_1 = kf_ki_len / lattice_lengths[best_idx]

    wl_min, wl_max = wavelength_range
    orders = np.arange(1, max_harmonic + 1)
    wavelength_by_order = wavelength_order_1[:, None] / orders[None, :]
    in_range = (wavelength_by_order >= wl_min) & (
        wavelength_by_order <= wl_max
    )

    has_valid_order = np.any(in_range, axis=1)
    best_order_idx = np.argmax(in_range, axis=1)
    best_order = orders[best_order_idx]

    direction_ok = best_cos >= np.cos(np.deg2rad(tol_deg))
    indexed = direction_ok & has_valid_order

    wavelength = wavelength_by_order[rows, best_order_idx]
    hkl = candidate_hkls[best_idx] * best_order[:, None]

    hkl = np.where(indexed[:, None], hkl, 0).astype(float)
    wavelength = np.where(indexed, wavelength, np.inf)

    return hkl, wavelength, indexed


class FindUBFromLauePeaks(PythonAlgorithm):
    """
    Determine a UB matrix from unindexed Laue peaks and known
    conventional-cell lattice parameters.

    Finds two well-separated zone axes directly from peak scattering
    directions (which are independent of the unknown per-peak
    wavelength), indexes them against the known lattice metric, and
    refines the resulting orientation using every peak it explains.
    Once a UB is found, each peak's Miller indices and wavelength are
    resolved directly from its scattering direction and the
    wavelength band, with no optimization required.
    """

    def category(self):
        return "Crystal\\UBMatrix"

    def name(self):
        return "FindUBFromLauePeaks"

    def summary(self):
        return (
            "Determine UB from unindexed Laue peaks and "
            "conventional-cell lattice parameters."
        )

    def PyInit(self):
        """
        Declare the algorithm's input and output properties.
        """
        self.declareProperty(
            IPeaksWorkspaceProperty("PeaksWorkspace", "", Direction.InOut)
        )

        positive = FloatBoundedValidator(lower=0.1)
        self.declareProperty("a", 10.0, positive)
        self.declareProperty("b", 10.0, positive)
        self.declareProperty("c", 10.0, positive)

        self.declareProperty("alpha", 90.0)
        self.declareProperty("beta", 90.0)
        self.declareProperty("gamma", 90.0)

        self.declareProperty(
            "Centering",
            "P",
            StringListValidator(["P", "A", "B", "C", "I", "F", "R"]),
        )

        wavelength_positive = FloatBoundedValidator(lower=0.01)
        self.declareProperty("WavelengthMin", 0.5, wavelength_positive)
        self.declareProperty("WavelengthMax", 4.0, wavelength_positive)

        self.declareProperty("MaxZoneIndex", 8, IntBoundedValidator(lower=1))
        self.declareProperty("MaxHklIndex", 25, IntBoundedValidator(lower=1))
        self.declareProperty("ZoneAngleTolerance", 1.0)
        self.declareProperty("MinZoneSeparation", 15.0)
        self.declareProperty("IndexingTolerance", 1.0)
        self.declareProperty(
            "MaxCandidatePairs", 300, IntBoundedValidator(lower=1)
        )

        self.declareProperty(
            TableWorkspaceProperty(
                "DiagnosticTable", "diagnostics_table", Direction.Output
            )
        )

    def validateInputs(self):
        """
        Check property values and peak count before execution.

        Returns
        -------
        issues : dict
            Mapping of property name to an error message, for
            properties that fail validation.
        """
        issues = {}

        peaks = self.getProperty("PeaksWorkspace").value
        if peaks is None:
            issues["PeaksWorkspace"] = "A PeaksWorkspace is required."
        elif peaks.getNumberPeaks() < 6:
            issues["PeaksWorkspace"] = "At least 6 peaks are required."

        wl_min = self.getProperty("WavelengthMin").value
        wl_max = self.getProperty("WavelengthMax").value
        if wl_min >= wl_max:
            issues[
                "WavelengthMax"
            ] = "WavelengthMax must be greater than WavelengthMin."

        for name in ["alpha", "beta", "gamma"]:
            ang = self.getProperty(name).value
            if not (10.0 < ang < 170.0):
                issues[name] = "Angle must satisfy 10 < angle < 170 degrees."

        return issues

    def _extract_kf_ki_directions(self, peaks_ws):
        """
        Read each peak's scattered-minus-incident beam direction.

        Parameters
        ----------
        peaks_ws : PeaksWorkspace
            Workspace containing the peaks to index.

        Returns
        -------
        kf_ki_dir : ndarray of shape (n_peaks, 3)
            `kf - ki` direction vectors, with magnitude `2 sin(theta)`.
        """
        kf_ki_dir = []
        for i in range(peaks_ws.getNumberPeaks()):
            pk = peaks_ws.getPeak(i)
            kf = pk.getDetectorDirectionSampleFrame()
            ki = pk.getSourceDirectionSampleFrame()
            kf_ki_dir.append(
                [kf.X() + ki.X(), kf.Y() + ki.Y(), kf.Z() + ki.Z()]
            )
        return np.asarray(kf_ki_dir, dtype=float)

    def _assign_ub_to_workspace(self, peaks_ws, UB_conv):
        """
        Attach a UB matrix to a peaks workspace's oriented lattice.

        Parameters
        ----------
        peaks_ws : PeaksWorkspace
            Workspace to update.
        UB_conv : ndarray of shape (3, 3)
            UB matrix indexed on the conventional cell.
        """
        sample = peaks_ws.mutableSample()
        try:
            ol = sample.getOrientedLattice()
        except RuntimeError:
            SetUB(Workspace=peaks_ws, UB=UB_conv)

        ol = sample.getOrientedLattice()
        ol.setUB(UB_conv)

    def _make_diagnostic_table(self, info, n_total, n_indexed):
        """
        Build a table workspace summarizing the fit diagnostics.

        Parameters
        ----------
        info : dict
            Diagnostics returned by `find_orientation_from_kf_ki`.
        n_total : int
            Total number of peaks in the workspace.
        n_indexed : int
            Number of peaks assigned Miller indices and a wavelength.

        Returns
        -------
        table : TableWorkspace
            Table with one "Metric"/"Value" row per diagnostic.
        """
        table = CreateEmptyTableWorkspace()
        table.addColumn("str", "Metric")
        table.addColumn("double", "Value")

        rows = [
            ("n_total", float(n_total)),
            ("n_indexed", float(n_indexed)),
            ("n_matched_coarse", float(info["n_matched_coarse"])),
            ("n_matched_refined", float(info["n_matched_refined"])),
            ("zone1_u", float(info["zone1_uvw"][0])),
            ("zone1_v", float(info["zone1_uvw"][1])),
            ("zone1_w", float(info["zone1_uvw"][2])),
            ("zone2_u", float(info["zone2_uvw"][0])),
            ("zone2_v", float(info["zone2_uvw"][1])),
            ("zone2_w", float(info["zone2_uvw"][2])),
        ]

        for metric, value in rows:
            table.addRow([metric, value])

        return table

    def PyExec(self):
        """
        Find the crystal orientation and index the Laue peaks.
        """
        peaks_ws = self.getProperty("PeaksWorkspace").value

        a = self.getProperty("a").value
        b = self.getProperty("b").value
        c = self.getProperty("c").value
        alpha_deg = self.getProperty("alpha").value
        beta_deg = self.getProperty("beta").value
        gamma_deg = self.getProperty("gamma").value
        centering = self.getProperty("Centering").value.upper()

        alpha = np.deg2rad(alpha_deg)
        beta = np.deg2rad(beta_deg)
        gamma = np.deg2rad(gamma_deg)

        wl_min = self.getProperty("WavelengthMin").value
        wl_max = self.getProperty("WavelengthMax").value
        max_zone_index = self.getProperty("MaxZoneIndex").value
        max_hkl_index = self.getProperty("MaxHklIndex").value
        zone_angle_tol = self.getProperty("ZoneAngleTolerance").value
        min_zone_sep = self.getProperty("MinZoneSeparation").value
        index_tol = self.getProperty("IndexingTolerance").value
        max_candidate_pairs = self.getProperty("MaxCandidatePairs").value

        uc_conv = UnitCell(a, b, c, alpha_deg, beta_deg, gamma_deg)
        self.log().information(
            "Input conventional cell: "
            f"a={uc_conv.a():.6f}, b={uc_conv.b():.6f}, c={uc_conv.c():.6f}, "
            f"alpha={uc_conv.alpha():.6f}, beta={uc_conv.beta():.6f}, "
            f"gamma={uc_conv.gamma():.6f}, centering={centering}"
        )

        if centering == "P":
            lattice_solve = (a, b, c, alpha, beta, gamma)
            T_cp = np.eye(3)
        else:
            lattice_solve, T_cp = conventional_to_primitive_lattice(
                a, b, c, alpha, beta, gamma, centering
            )

        a_p, b_p, c_p, alpha_p, beta_p, gamma_p = lattice_solve

        A_p = direct_basis_from_lattice(
            a_p, b_p, c_p, alpha_p, beta_p, gamma_p
        )
        B_p = reciprocal_basis_from_direct_basis(A_p)

        kf_ki_dir = self._extract_kf_ki_directions(peaks_ws)

        R, info = find_orientation_from_kf_ki(
            kf_ki_dir,
            A_p,
            B_p,
            max_zone_index,
            max_hkl_index,
            zone_tol_deg=zone_angle_tol,
            min_zone_separation_deg=min_zone_sep,
            zone_angle_tol_deg=zone_angle_tol,
            match_tol_deg=index_tol,
            max_candidate_pairs=max_candidate_pairs,
        )

        UB_p = R @ B_p
        UB_conv = primitive_ub_to_conventional_ub(UB_p, T_cp)

        candidate_hkls = enumerate_reciprocal_directions(max_hkl_index)
        hkl, wavelength, indexed = resolve_hkl_and_wavelength(
            kf_ki_dir,
            UB_conv,
            candidate_hkls,
            (wl_min, wl_max),
            tol_deg=index_tol,
        )

        self._assign_ub_to_workspace(peaks_ws, UB_conv)

        for i in range(peaks_ws.getNumberPeaks()):
            pk = peaks_ws.getPeak(i)
            if indexed[i]:
                pk.setHKL(*hkl[i])
                pk.setWavelength(float(wavelength[i]))
            else:
                pk.setHKL(0.0, 0.0, 0.0)

        diag_table = self._make_diagnostic_table(
            info, peaks_ws.getNumberPeaks(), int(np.sum(indexed))
        )
        self.setProperty("DiagnosticTable", diag_table)

        self.setProperty("PeaksWorkspace", peaks_ws)


AlgorithmFactory.subscribe(FindUBFromLauePeaks)
