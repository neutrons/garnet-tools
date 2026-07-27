from mantid.api import (
    PythonAlgorithm,
    AlgorithmFactory,
    IPeaksWorkspaceProperty,
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

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d

from garnet.reduction.search import (
    direct_basis_from_lattice,
    reciprocal_basis_from_direct_basis,
    conventional_to_primitive_lattice,
    primitive_ub_to_conventional_ub,
)


def prepare_peak_rays(kf_ki_dir):
    """
    Split raw Laue `kf - ki` vectors into direction and magnitude.

    In this facility's Q convention, `kf_ki_dir = -wavelength * UB @ hkl`
    (see `garnet.reduction.search`), so the direction of `UB @ hkl` is
    `normalize(-kf_ki_dir)`; its magnitude, `wavelength`-independent, is
    `||kf_ki_dir|| = 2 * sin(theta)`.

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `FindUBFromLauePeaks._extract_kf_ki_directions`.

    Returns
    -------
    qhat : ndarray of shape (n_peaks, 3)
        Unit vectors in the direction of each peak's true `UB @ hkl`.
    m : ndarray of shape (n_peaks,)
        Wavelength-independent ray magnitude, `2 * sin(theta)`.
    """
    kf_ki_dir = np.asarray(kf_ki_dir, dtype=float)
    m = np.linalg.norm(kf_ki_dir, axis=1)
    qhat = -kf_ki_dir / m[:, None]
    return qhat, m


def enumerate_reciprocal_rays(B, gmag_max, max_hkl_index=None):
    """
    Enumerate candidate integer reflections reachable within a length bound.

    Every non-zero integer triple is included, along with its
    reciprocal-space vector, length, and direction, provided its length
    does not exceed `gmag_max`. This must cover the full range of
    diffraction orders actually observable, which can be large for
    large unit cells, so no primitive-only reduction is applied here;
    each integer multiple is a physically distinct reflection at its
    own wavelength.

    Parameters
    ----------
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    gmag_max : float
        Largest reciprocal-vector length to include, in inverse
        angstroms.
    max_hkl_index : int, optional
        Largest Miller index magnitude to consider. If omitted, it is
        derived from `gmag_max` and the shortest reciprocal lattice
        vector, with a small margin.

    Returns
    -------
    hkl : ndarray of shape (n_rays, 3)
        Candidate integer Miller indices.
    g : ndarray of shape (n_rays, 3)
        Reciprocal-space vectors, `B @ hkl`.
    gmag : ndarray of shape (n_rays,)
        Lengths of `g`, in inverse angstroms.
    ghat : ndarray of shape (n_rays, 3)
        Unit directions of `g`.
    """
    if max_hkl_index is None:
        min_recip_length = np.min(np.linalg.norm(B, axis=0))
        max_hkl_index = int(np.ceil(gmag_max / min_recip_length)) + 1

    grid = np.arange(-max_hkl_index, max_hkl_index + 1)
    h, k, l = np.meshgrid(grid, grid, grid, indexing="ij")
    hkl = np.column_stack([h.ravel(), k.ravel(), l.ravel()])
    hkl = hkl[np.any(hkl != 0, axis=1)]

    g = (B @ hkl.T).T
    gmag = np.linalg.norm(g, axis=1)

    within = gmag <= gmag_max
    hkl, g, gmag = hkl[within], g[within], gmag[within]
    ghat = g / gmag[:, None]

    return hkl, g, gmag, ghat


def score_orientation_from_rays(
    qhat, m, U, hkl, gmag, ghat, wavelength_range, tol_deg=1.0
):
    """
    Score a trial orientation against every measured ray, gated by wavelength.

    For each peak and each candidate reflection, the wavelength implied
    by that assignment is `m_i / gmag_hkl` (see `prepare_peak_rays`);
    candidates whose implied wavelength falls outside
    `wavelength_range` are excluded before ranking by angular match, so
    a peak is never assigned a reflection that would require an
    unphysical wavelength.

    Parameters
    ----------
    qhat : ndarray of shape (n_peaks, 3)
        Unit vectors, as returned by `prepare_peak_rays`.
    m : ndarray of shape (n_peaks,)
        Ray magnitudes, as returned by `prepare_peak_rays`.
    U : ndarray of shape (3, 3)
        Trial crystal-to-lab rotation.
    hkl : ndarray of shape (n_rays, 3)
        Candidate integer Miller indices, as returned by
        `enumerate_reciprocal_rays`.
    gmag : ndarray of shape (n_rays,)
        Candidate reciprocal-vector lengths, as returned by
        `enumerate_reciprocal_rays`.
    ghat : ndarray of shape (n_rays, 3)
        Candidate reciprocal-vector directions, as returned by
        `enumerate_reciprocal_rays`.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    tol_deg : float, optional
        Angular tolerance, in degrees, for a match to be accepted.

    Returns
    -------
    hkl_assigned : ndarray of shape (n_peaks, 3)
        Assigned Miller indices; `(0, 0, 0)` where unindexed.
    wavelength : ndarray of shape (n_peaks,)
        Resolved wavelength, in angstroms; `inf` where unindexed.
    indexed : ndarray of bool, shape (n_peaks,)
        True where a peak was successfully indexed.
    info : dict
        Diagnostics with keys "n_indexed", "indexed_fraction",
        "n_unique_hkl", "median_angular_error_deg", and
        "rms_angular_error_deg".
    """
    wl_min, wl_max = wavelength_range

    qhat_pred = (U @ ghat.T).T
    lam = m[:, None] / gmag[None, :]
    wavelength_ok = (lam >= wl_min) & (lam <= wl_max)

    cos_ang = qhat @ qhat_pred.T
    cos_ang = np.where(wavelength_ok, cos_ang, -2.0)

    best_idx = np.argmax(cos_ang, axis=1)
    rows = np.arange(len(qhat))
    best_cos = cos_ang[rows, best_idx]

    indexed = best_cos >= np.cos(np.deg2rad(tol_deg))

    hkl_assigned = np.where(indexed[:, None], hkl[best_idx], 0).astype(float)
    wavelength = np.where(indexed, lam[rows, best_idx], np.inf)

    n_indexed = int(np.sum(indexed))
    if n_indexed > 0:
        theta_deg = np.rad2deg(
            np.arccos(np.clip(best_cos[indexed], -1.0, 1.0))
        )
        n_unique_hkl = len(np.unique(hkl_assigned[indexed], axis=0))
        median_angular_error_deg = float(np.median(theta_deg))
        rms_angular_error_deg = float(np.sqrt(np.mean(theta_deg**2)))
    else:
        n_unique_hkl = 0
        median_angular_error_deg = np.inf
        rms_angular_error_deg = np.inf

    info = {
        "n_indexed": n_indexed,
        "indexed_fraction": n_indexed / len(qhat),
        "n_unique_hkl": n_unique_hkl,
        "median_angular_error_deg": median_angular_error_deg,
        "rms_angular_error_deg": rms_angular_error_deg,
    }

    return hkl_assigned, wavelength, indexed, info


def refine_orientation_from_rays(
    qhat, m, U0, B, hkl, gmag, ghat, wavelength_range, tol_deg=1.0, max_iter=10
):
    """
    Iteratively refine a trial orientation by reassignment and refitting.

    Alternates between assigning each peak to its best allowed
    reflection under the current orientation (see
    `score_orientation_from_rays`) and refitting the rotation from
    those assignments (via the Kabsch algorithm), stopping once the
    assignment and rotation both stabilize.

    Parameters
    ----------
    qhat : ndarray of shape (n_peaks, 3)
        Unit vectors, as returned by `prepare_peak_rays`.
    m : ndarray of shape (n_peaks,)
        Ray magnitudes, as returned by `prepare_peak_rays`.
    U0 : ndarray of shape (3, 3)
        Initial crystal-to-lab rotation.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    hkl, gmag, ghat : ndarray
        Candidate reflections, as returned by
        `enumerate_reciprocal_rays`.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    tol_deg : float, optional
        Passed to `score_orientation_from_rays`.
    max_iter : int, optional
        Maximum number of reassign-refit iterations.

    Returns
    -------
    U : ndarray of shape (3, 3)
        Refined crystal-to-lab rotation.
    hkl_assigned : ndarray of shape (n_peaks, 3)
        Assigned Miller indices; `(0, 0, 0)` where unindexed.
    wavelength : ndarray of shape (n_peaks,)
        Resolved wavelength, in angstroms; `inf` where unindexed.
    indexed : ndarray of bool, shape (n_peaks,)
        True where a peak was successfully indexed.
    info : dict
        Diagnostics, as returned by `score_orientation_from_rays`.
    """
    U = U0.copy()
    previous_hkl = None

    for _ in range(max_iter):
        hkl_assigned, wavelength, indexed, info = score_orientation_from_rays(
            qhat, m, U, hkl, gmag, ghat, wavelength_range, tol_deg=tol_deg
        )

        if info["n_indexed"] < 2:
            break

        ghat_assigned = (B @ hkl_assigned[indexed].T).T
        ghat_assigned = ghat_assigned / np.linalg.norm(
            ghat_assigned, axis=1, keepdims=True
        )

        H = ghat_assigned.T @ qhat[indexed]
        P, _, Vt = np.linalg.svd(H)
        D = np.diag([1.0, 1.0, np.linalg.det(Vt.T @ P.T)])
        U_new = Vt.T @ D @ P.T

        converged = previous_hkl is not None and np.array_equal(
            hkl_assigned, previous_hkl
        )
        previous_hkl = hkl_assigned
        U = U_new

        if converged:
            break

    hkl_assigned, wavelength, indexed, info = score_orientation_from_rays(
        qhat, m, U, hkl, gmag, ghat, wavelength_range, tol_deg=tol_deg
    )
    return U, hkl_assigned, wavelength, indexed, info


def build_omega_interpolator(n_samples=4096):
    """
    Build the inverse-CDF map from a uniform [0, 1] draw to a rotation angle.

    Sampling a rotation uniformly over SO(3) via a uniformly-distributed
    axis and an independently-uniform [0, 1] draw for the angle would
    NOT give a uniform (Haar) measure over the rotation group: the
    angle must instead be drawn with density `(1 - cos(omega)) / pi` on
    `[0, pi]` to account for the group manifold's volume element. This
    builds the inverse of that distribution's CDF, `(omega -
    sin(omega)) / pi`, via linear interpolation, so a uniform [0, 1]
    draw can be mapped to a correctly-distributed angle.

    Parameters
    ----------
    n_samples : int, optional
        Number of interpolation points spanning `[0, pi]`.

    Returns
    -------
    omega_interp : callable
        Maps a uniform [0, 1] value (or array of values) to a rotation
        angle in `[0, pi]`, in radians.
    """
    omega = np.linspace(0.0, np.pi, n_samples)
    cdf = (omega - np.sin(omega)) / np.pi
    return interp1d(cdf, omega)


def orientation_from_unit_cube(u, omega_interp):
    """
    Map a point in the unit cube to a uniformly-distributed rotation.

    The three coordinates parametrize, respectively: the polar angle
    of the rotation axis (via `arccos(1 - 2 * u0)`, uniform on the
    sphere), the azimuthal angle of the rotation axis (`2 * pi * u1`),
    and the rotation angle (via the inverse-CDF map from
    `build_omega_interpolator`, uniform under the SO(3) Haar measure).
    This gives `differential_evolution` a bounded, uniformly-weighted
    search space that still covers every possible orientation.

    Parameters
    ----------
    u : ndarray of shape (..., 3)
        Points in `[0, 1]^3`.
    omega_interp : callable
        As returned by `build_omega_interpolator`.

    Returns
    -------
    U : ndarray of shape (..., 3, 3)
        Proper rotation matrices, one per leading index of `u`.
    """
    u = np.asarray(u, dtype=float)
    u0, u1, u2 = u[..., 0], u[..., 1], u[..., 2]

    theta = np.arccos(np.clip(1.0 - 2.0 * u0, -1.0, 1.0))
    phi = 2.0 * np.pi * u1
    omega = omega_interp(u2)

    axis = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )
    rotvec = omega[..., None] * axis
    flat_rotvec = rotvec.reshape(-1, 3)
    U = Rotation.from_rotvec(flat_rotvec).as_matrix()
    return U.reshape(rotvec.shape[:-1] + (3, 3))


def laue_orientation_cost(
    x, kf_ki_dir, B, wavelength_range, omega_interp, n_wavelength_samples=100
):
    """
    Smooth, wavelength-marginalized misfit of a trial orientation.

    For each peak, the true wavelength is unknown, so for a trial
    orientation this scans a grid of candidate wavelengths spanning
    `wavelength_range`, converts each `(peak, wavelength)` pair to a
    fractional Miller index, and measures its distance to the nearest
    integer with the smooth proxy `sin(pi * hkl) / pi` (periodic,
    zero at integers, avoiding the discontinuity of rounding). That
    residual is transformed through the trial `UB` into Cartesian
    reciprocal space before taking its norm, so errors along different
    crystallographic directions are weighted by the actual reciprocal
    metric rather than treated as equally "sized" in fractional-index
    units. Each peak then contributes its best (minimum) distance over
    the wavelength grid — a peak genuinely explained by this
    orientation, at any in-band wavelength, contributes near zero. The
    total cost is the sum over peaks, suitable as a
    `differential_evolution` objective (lower is better).

    Accepts either scipy's vectorized calling convention (`x` of shape
    `(3, n_trials)`, one column per population member) or a single
    trial (`x` of shape `(3,)`).

    Parameters
    ----------
    x : ndarray of shape (3,) or (3, n_trials)
        Unit-cube orientation parameters (see
        `orientation_from_unit_cube`).
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `FindUBFromLauePeaks._extract_kf_ki_directions`.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    omega_interp : callable
        As returned by `build_omega_interpolator`.
    n_wavelength_samples : int, optional
        Number of candidate wavelengths sampled per peak.

    Returns
    -------
    cost : float or ndarray of shape (n_trials,)
        Total misfit, matching the leading shape of `x`.
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 1
    u = x[None, :] if scalar else x.T

    U = orientation_from_unit_cube(u, omega_interp)
    UB = U @ B[None, :, :]
    UB_inv = np.linalg.inv(UB)

    hkl_lambda = -np.einsum("sij,pj->spi", UB_inv, kf_ki_dir)

    wl_min, wl_max = wavelength_range
    wavelengths = np.linspace(wl_min, wl_max, n_wavelength_samples)

    hkl = hkl_lambda[:, :, None, :] / wavelengths[None, None, :, None]
    diff = np.sin(np.pi * hkl) / np.pi
    dist_vec = np.einsum("sij,spwj->spwi", UB, diff)
    dist2 = np.sum(dist_vec**2, axis=-1)

    cost = np.sum(np.min(dist2, axis=-1), axis=-1)
    return cost[0] if scalar else cost


def fit_orientation_differential_evolution(
    kf_ki_dir,
    B,
    wavelength_range,
    n_wavelength_samples=60,
    popsize=60,
    maxiter=200,
    mutation=(0.5, 1.5),
    recombination=0.7,
    tol=1e-7,
    rng=None,
):
    """
    Globally search for the crystal orientation via differential evolution.

    Minimizes `laue_orientation_cost` over the unit cube (see
    `orientation_from_unit_cube`) using `scipy.optimize.
    differential_evolution` in vectorized mode, so every population
    member is scored in a single batched call per generation.

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `FindUBFromLauePeaks._extract_kf_ki_directions`.
    B : ndarray of shape (3, 3)
        Reciprocal-lattice basis with a*, b*, c* as columns.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    n_wavelength_samples : int, optional
        Passed to `laue_orientation_cost`.
    popsize, maxiter, mutation, recombination, tol : optional
        Passed to `scipy.optimize.differential_evolution`.
    rng : numpy.random.Generator, optional
        Random number generator for the optimizer.

    Returns
    -------
    U : ndarray of shape (3, 3)
        Best-found crystal-to-lab rotation.
    result : scipy.optimize.OptimizeResult
        Raw optimizer result, for diagnostics (`result.fun`,
        `result.nit`, `result.nfev`).
    """
    omega_interp = build_omega_interpolator()

    def objective(x):
        return laue_orientation_cost(
            x,
            kf_ki_dir,
            B,
            wavelength_range,
            omega_interp,
            n_wavelength_samples=n_wavelength_samples,
        )

    result = differential_evolution(
        objective,
        bounds=[(0.0, 1.0)] * 3,
        popsize=popsize,
        maxiter=maxiter,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        polish=False,
        vectorized=True,
        rng=rng,
    )

    U = orientation_from_unit_cube(result.x, omega_interp)
    return U, result


def _run_single_restart(
    kf_ki_dir,
    B,
    wavelength_range,
    qhat,
    m,
    hkl,
    gmag,
    ghat,
    n_wavelength_samples,
    popsize,
    maxiter,
    match_tol_deg,
    max_refine_iter,
    seed,
):
    """
    Run one differential-evolution restart plus polish, given a seed.

    A module-level (picklable) helper so `estimate_orientation_from_rays`
    can dispatch independent restarts to a `ProcessPoolExecutor` — each
    restart shares no mutable state with any other, so this is
    embarrassingly parallel.

    Parameters
    ----------
    kf_ki_dir, B, wavelength_range, qhat, m, hkl, gmag, ghat :
        As used by `fit_orientation_differential_evolution` and
        `refine_orientation_from_rays`.
    n_wavelength_samples, popsize, maxiter, match_tol_deg,
    max_refine_iter :
        Passed through to those functions.
    seed : int
        Seeds this restart's random number generator.

    Returns
    -------
    candidate : dict
        Keys "U", "UB", "hkl", "wavelength", "indexed", "de_cost",
        "de_iterations", and the diagnostics from
        `score_orientation_from_rays`.
    """
    rng = np.random.default_rng(seed)
    U0, de_result = fit_orientation_differential_evolution(
        kf_ki_dir,
        B,
        wavelength_range,
        n_wavelength_samples=n_wavelength_samples,
        popsize=popsize,
        maxiter=maxiter,
        rng=rng,
    )

    U, hkl_assigned, wavelength, indexed, info = refine_orientation_from_rays(
        qhat,
        m,
        U0,
        B,
        hkl,
        gmag,
        ghat,
        wavelength_range,
        tol_deg=match_tol_deg,
        max_iter=max_refine_iter,
    )

    return {
        "U": U,
        "UB": U @ B,
        "hkl": hkl_assigned,
        "wavelength": wavelength,
        "indexed": indexed,
        "de_cost": float(de_result.fun),
        "de_iterations": int(de_result.nit),
        **info,
    }


def estimate_orientation_from_rays(
    kf_ki_dir,
    A,
    wavelength_range,
    n_restarts=5,
    n_wavelength_samples=60,
    popsize=60,
    maxiter=200,
    match_tol_deg=1.0,
    max_refine_iter=10,
    rng=None,
    n_workers=None,
):
    """
    Estimate crystal orientation from Laue `kf - ki` vectors alone.

    Each restart runs an independent global search over the full
    rotation group via `fit_orientation_differential_evolution`
    (following the wavelength-marginalized cost in
    `laue_orientation_cost`), then polishes that estimate by iterative
    reassignment against the full, wavelength-gated set of candidate
    reflections (`refine_orientation_from_rays`).

    Differential evolution's cost landscape can have a spurious
    attractor competing with the true orientation — most pronounced
    for low-symmetry (e.g. triclinic) cells, where no lattice symmetry
    redundancy helps distinguish them — that a single restart
    occasionally converges to instead. The iterative polish step
    reliably completes a restart that is already close to the true
    orientation, but cannot rescue one that converged to a genuinely
    different attractor; running several independent restarts and
    keeping whichever gives the most peaks indexed (see
    `score_orientation_from_rays`) is what actually distinguishes a
    true convergence from a spurious one. Higher-symmetry cells
    typically converge on the first restart; harder, low-symmetry
    cells may need more (see `n_restarts`).

    Parameters
    ----------
    kf_ki_dir : ndarray of shape (n_peaks, 3)
        Raw `kf - ki` vectors, as returned by
        `FindUBFromLauePeaks._extract_kf_ki_directions`.
    A : ndarray of shape (3, 3)
        Direct-lattice basis with a, b, c as columns.
    wavelength_range : tuple of float
        `(wavelength_min, wavelength_max)`, in angstroms.
    n_restarts : int, optional
        Number of independent differential-evolution searches to run;
        the best-indexing result across all restarts is kept.
    n_wavelength_samples : int, optional
        Passed to `fit_orientation_differential_evolution`.
    popsize, maxiter : optional
        Passed to `fit_orientation_differential_evolution`.
    match_tol_deg : float, optional
        Angular tolerance, in degrees, used to rank candidate
        rotations and gate peak assignment.
    max_refine_iter : int, optional
        Passed to `refine_orientation_from_rays`.
    rng : numpy.random.Generator, optional
        Random number generator; used to derive one independent seed
        per restart.
    n_workers : int, optional
        Number of restarts to run concurrently, via
        `concurrent.futures.ProcessPoolExecutor`. Restarts share no
        mutable state, so this parallelizes without changing results.
        Defaults to `os.cpu_count()`.

    Returns
    -------
    best : dict
        The best-indexing orientation across all restarts, with keys
        "U", "UB", "hkl", "wavelength", "indexed", "de_cost",
        "de_iterations", and the diagnostics from
        `score_orientation_from_rays`.
    """
    if rng is None:
        rng = np.random.default_rng(1234)

    B = reciprocal_basis_from_direct_basis(A)
    qhat, m = prepare_peak_rays(kf_ki_dir)

    wl_min, _ = wavelength_range
    gmag_max = m.max() / wl_min
    hkl, _, gmag, ghat = enumerate_reciprocal_rays(B, gmag_max)

    seeds = rng.integers(0, 2**31 - 1, size=n_restarts)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(
                _run_single_restart,
                kf_ki_dir,
                B,
                wavelength_range,
                qhat,
                m,
                hkl,
                gmag,
                ghat,
                n_wavelength_samples,
                popsize,
                maxiter,
                match_tol_deg,
                max_refine_iter,
                int(seed),
            )
            for seed in seeds
        ]
        candidates = [future.result() for future in futures]

    candidates.sort(
        key=lambda cand: (
            cand["n_indexed"],
            -cand["rms_angular_error_deg"],
        ),
        reverse=True,
    )
    return candidates[0]


class FindUBFromLauePeaks(PythonAlgorithm):
    """
    Determine a UB matrix from unindexed Laue peaks and known
    conventional-cell lattice parameters.

    Searches for the crystal orientation via differential evolution
    over the full rotation group, scored by a wavelength-marginalized
    misfit against the observed scattering directions (the per-peak
    wavelength is unknown, so every candidate wavelength within the
    supplied band is considered), then polishes and finalizes each
    peak's Miller indices and wavelength together by iterative
    reassignment against the full set of candidate reflections. See
    `estimate_orientation_from_rays` for why several independent
    restarts are used.
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

        self.declareProperty("NumRestarts", 5, IntBoundedValidator(lower=1))
        self.declareProperty("NumWorkers", 0, IntBoundedValidator(lower=0))
        self.declareProperty(
            "PopulationSize", 60, IntBoundedValidator(lower=4)
        )
        self.declareProperty(
            "MaxIterations", 200, IntBoundedValidator(lower=1)
        )
        self.declareProperty(
            "NumWavelengthSamples", 60, IntBoundedValidator(lower=2)
        )
        self.declareProperty("IndexingTolerance", 1.0)
        self.declareProperty("RandomSeed", 1234)

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

    def _make_diagnostic_table(self, best, n_total):
        """
        Build a table workspace summarizing the fit diagnostics.

        Parameters
        ----------
        best : dict
            Best orientation, as returned by
            `estimate_orientation_from_rays`.
        n_total : int
            Total number of peaks in the workspace.

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
            ("n_indexed", float(best["n_indexed"])),
            ("indexed_fraction", float(best["indexed_fraction"])),
            ("n_unique_hkl", float(best["n_unique_hkl"])),
            (
                "median_angular_error_deg",
                float(best["median_angular_error_deg"]),
            ),
            ("rms_angular_error_deg", float(best["rms_angular_error_deg"])),
            ("de_cost", best["de_cost"]),
            ("de_iterations", float(best["de_iterations"])),
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
        n_restarts = self.getProperty("NumRestarts").value
        n_workers = self.getProperty("NumWorkers").value or None
        popsize = self.getProperty("PopulationSize").value
        maxiter = self.getProperty("MaxIterations").value
        n_wl_samples = self.getProperty("NumWavelengthSamples").value
        index_tol = self.getProperty("IndexingTolerance").value
        seed = self.getProperty("RandomSeed").value

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

        kf_ki_dir = self._extract_kf_ki_directions(peaks_ws)

        best = estimate_orientation_from_rays(
            kf_ki_dir,
            A_p,
            (wl_min, wl_max),
            n_restarts=n_restarts,
            n_wavelength_samples=n_wl_samples,
            popsize=popsize,
            maxiter=maxiter,
            match_tol_deg=index_tol,
            rng=np.random.default_rng(seed),
            n_workers=n_workers,
        )

        UB_p = best["UB"]
        UB_conv = primitive_ub_to_conventional_ub(UB_p, T_cp)

        # Miller indices reindex as hkl_p = T_cp.T @ hkl_c (see
        # garnet.reduction.search.primitive_ub_to_conventional_ub), so
        # hkl_c = inv(T_cp.T) @ hkl_p.
        T_cp_inv_T = np.linalg.inv(T_cp).T
        hkl_conv = (T_cp_inv_T @ best["hkl"].T).T

        self._assign_ub_to_workspace(peaks_ws, UB_conv)

        for i in range(peaks_ws.getNumberPeaks()):
            pk = peaks_ws.getPeak(i)
            if best["indexed"][i]:
                pk.setHKL(*hkl_conv[i])
                pk.setWavelength(float(best["wavelength"][i]))
            else:
                pk.setHKL(0.0, 0.0, 0.0)

        diag_table = self._make_diagnostic_table(
            best, peaks_ws.getNumberPeaks()
        )
        self.setProperty("DiagnosticTable", diag_table)

        self.setProperty("PeaksWorkspace", peaks_ws)


AlgorithmFactory.subscribe(FindUBFromLauePeaks)
