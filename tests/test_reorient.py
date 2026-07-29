from mantid.simpleapi import CreatePeaksWorkspace, SetUB, mtd

import numpy as np

from garnet.reduction.ub import Reorient


def make_scrambled_peaks(a, b, c, alpha, beta, gamma, u, v, perm, negate_axis):
    """
    Build a "ref" workspace with the given cell/orientation, and a
    "peaks" workspace describing the *same* crystal but with its UB
    axes permuted (and, for odd permutations, one axis sign-flipped to
    keep the scramble itself proper/right-handed -- a real from-scratch
    UB determination never produces a left-handed result).

    Returns (UB_ref, perm_determinant).
    """

    CreatePeaksWorkspace(
        NumberOfPeaks=0, OutputType="LeanElasticPeak", OutputWorkspace="ref"
    )
    SetUB(
        Workspace="ref",
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        u=u,
        v=v,
    )
    UB_ref = mtd["ref"].sample().getOrientedLattice().getUB().copy()

    P = np.eye(3)[:, perm]
    if negate_axis is not None:
        P[:, negate_axis] *= -1
    assert np.linalg.det(P) > 0

    UB_scrambled = UB_ref @ np.linalg.inv(P)

    CreatePeaksWorkspace(
        NumberOfPeaks=0,
        OutputType="LeanElasticPeak",
        OutputWorkspace="peaks",
    )
    SetUB(Workspace="peaks", UB=UB_scrambled)

    return UB_ref


def assert_reoriented(UB_ref, system):
    Reorient("peaks", UB_ref, system)

    UB_after = mtd["peaks"].sample().getOrientedLattice().getUB()
    assert np.allclose(UB_after, UB_ref, atol=1e-5)


def test_orthorhombic_swap_and_flip():
    UB_ref = make_scrambled_peaks(
        5, 6, 7, 90, 90, 90, [0.3, 0.7, 0.1], [-0.6, 0.2, 0.9], (1, 0, 2), 2
    )
    assert_reoriented(UB_ref, "Orthorhombic")


def test_orthorhombic_even_permutation():
    UB_ref = make_scrambled_peaks(
        5,
        6,
        7,
        90,
        90,
        90,
        [0.3, 0.7, 0.1],
        [-0.6, 0.2, 0.9],
        (2, 0, 1),
        None,
    )
    assert_reoriented(UB_ref, "Orthorhombic")


def test_monoclinic_unique_axis_remap():
    UB_ref = make_scrambled_peaks(
        5,
        6,
        7,
        90,
        100,
        90,
        [0.1, 0.2, 0.9],
        [0.8, -0.1, 0.3],
        (0, 2, 1),
        0,
    )
    assert_reoriented(UB_ref, "Monoclinic")


def test_triclinic_odd_permutation():
    UB_ref = make_scrambled_peaks(
        5,
        6,
        7,
        80,
        95,
        110,
        [0.5, 0.5, 0.5],
        [0.2, -0.8, 0.1],
        (2, 1, 0),
        1,
    )
    assert_reoriented(UB_ref, "Triclinic")


def test_triclinic_even_permutation():
    UB_ref = make_scrambled_peaks(
        5,
        6,
        7,
        80,
        95,
        110,
        [0.5, 0.5, 0.5],
        [0.2, -0.8, 0.1],
        (1, 2, 0),
        None,
    )
    assert_reoriented(UB_ref, "Triclinic")


def test_cubic_unaffected_by_axis_resolution():
    # Cubic's own point group already covers axis permutation (a=b=c is
    # a real symmetry, not just a labeling convention), so no separate
    # permutation search should be needed -- resolve_axis_ambiguity must
    # be a no-op (identity) here.
    UB_ref = make_scrambled_peaks(
        6, 6, 6, 90, 90, 90, [1, 0, 0], [0, 1, 0], (0, 1, 2), None
    )

    r = Reorient.__new__(Reorient)
    r.UB = mtd["peaks"].sample().getOrientedLattice().getUB().copy()
    r.UB_ref = UB_ref.copy()
    assert np.array_equal(r.resolve_axis_ambiguity("Cubic"), np.eye(3))

    assert_reoriented(UB_ref, "Cubic")
