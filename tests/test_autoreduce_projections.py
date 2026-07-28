import numpy as np

from garnet.utilities.autoreduce import AutoReduce


def make_autoreduce():
    return AutoReduce.__new__(AutoReduce)


def orthorhombic_UB(a=5.0, b=6.0, c=7.0):
    return np.diag([1 / a, 1 / b, 1 / c])


def triclinic_UB():
    # Skewed reciprocal lattice vectors, none pairwise orthogonal.
    return np.array(
        [
            [0.2, 0.05, 0.03],
            [0.01, 0.18, 0.06],
            [0.02, 0.04, 0.15],
        ]
    )


def test_title_group_key_enumerated():
    ar = make_autoreduce()

    assert ar._title_group_key("Si_powder_5") == "Si_powder_*"
    assert ar._title_group_key("Si_powder5") == "Si_powder*"


def test_title_group_key_not_enumerated():
    ar = make_autoreduce()

    assert ar._title_group_key("Si_powder") == "Si_powder"
    assert ar._title_group_key("5") == "5"


def test_axis_aligned_orthorhombic():
    ar = make_autoreduce()

    assert ar._is_axis_aligned(orthorhombic_UB()) is True


def test_axis_aligned_triclinic():
    ar = make_autoreduce()

    assert ar._is_axis_aligned(triclinic_UB()) is False


def test_candidate_projections_count():
    ar = make_autoreduce()

    aligned_UB = orthorhombic_UB()
    not_aligned_UB = triclinic_UB()

    aligned = ar._candidate_projections(aligned_UB, True)
    not_aligned = ar._candidate_projections(not_aligned_UB, False)

    assert len(aligned) == 3
    assert [name for name, _ in aligned] == ["hk0", "h0l", "0kl"]

    assert len(not_aligned) == 4
    assert [name for name, _ in not_aligned] == [
        "hk0",
        "h0l",
        "0kl",
        "equatorial",
    ]

    for _, W in aligned + not_aligned:
        W = np.asarray(W, dtype=float)
        assert np.column_stack(W).shape == (3, 3)
        assert not np.isclose(np.linalg.det(np.column_stack(W)), 0)


def test_projection_extents_scales_with_d_min():
    ar = make_autoreduce()

    UB = orthorhombic_UB()
    projections = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    extents_coarse, bins = ar._projection_extents(UB, projections, d_min=1.0)
    extents_fine, _ = ar._projection_extents(UB, projections, d_min=0.5)

    assert bins == [801, 801, 1]

    # Thin axis is always the fixed +/-0.1 integration range.
    assert extents_coarse[2] == [-0.1, 0.1]
    assert extents_fine[2] == [-0.1, 0.1]

    # Smaller d_min reaches further into reciprocal space.
    assert extents_fine[0][1] > extents_coarse[0][1]
    assert extents_fine[1][1] > extents_coarse[1][1]


def test_heatmap_div_is_plotly_content():
    ar = make_autoreduce()

    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    z = np.random.default_rng(0).normal(size=(5, 5))

    div = ar._heatmap_div(x, y, z, x_title="x", y_title="y", title="t")

    assert "plotly-graph-div" in div
    assert "Plotly.newPlot" in div
