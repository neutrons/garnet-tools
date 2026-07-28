import warnings

import numpy as np
import pytest

from garnet.plots.monitor import SlicePlot


def make_slice_plot():
    UB = np.diag([0.2, 0.18, 0.15])
    W = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    plot = SlicePlot(UB, W)
    axes = [
        np.linspace(-1, 1, 40),
        np.linspace(-1, 1, 50),
        np.array([0.0]),
    ]
    titles = ["h", "k", "l"]
    plot.calculate_transforms(
        axes, titles, [0, 0, 1], oversample=1.0, interpolation="nearest"
    )
    return plot


def test_trace_stays_linked_to_figure():
    # fig.add_trace() clones the trace it's given rather than keeping it
    # linked -- self.im must be re-pointed at the live fig.data[0] object
    # afterwards, or every later mutation in make_slice() updates an
    # orphaned copy that never gets serialized (axes render, heatmap
    # stays empty).
    plot = make_slice_plot()

    assert plot.im is plot.fig.data[0]

    data = np.full((40, 50, 1), 5.0)
    norm = np.full((40, 50, 1), 2.0)
    plot.make_slice(data, norm, 0.0)

    assert plot.im is plot.fig.data[0]
    assert np.isfinite(plot.fig.data[0].z).any()


def test_make_slice_no_runtime_warnings():
    plot = make_slice_plot()

    rng = np.random.default_rng(2)
    data = rng.poisson(lam=5, size=(40, 50, 1)).astype(float)
    norm = rng.uniform(1, 10, size=(40, 50, 1))

    # Real zero-count region: must survive as a genuine zero, not be
    # masked away like the (unrelated) masked-normalization case below.
    data[0:5, 0:5, 0] = 0.0
    # No-coverage region: zero normalization must become NaN, not a
    # divide-by-zero warning or an inf/garbage value.
    norm[10:15, 10:15, 0] = 0.0
    # Non-finite inputs should be excluded cleanly, without warnings.
    data[20, 20, 0] = np.nan
    norm[25, 25, 0] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot.make_slice(data, norm, 0.0)

    z = plot.fig.data[0].z
    assert np.isfinite(z).any()
    assert not np.isinf(z).any()


def test_make_slice_zero_counts_are_not_masked():
    plot = make_slice_plot()

    # Uniformly zero counts everywhere, but validly covered (positive
    # normalization) -- the true ratio is 0 everywhere, not NaN.
    data = np.zeros((40, 50, 1))
    norm = np.full((40, 50, 1), 2.0)

    plot.make_slice(data, norm, 0.0)

    # log10(0) is masked for the log-color display (expected -- you can't
    # show zero on a log scale), but the customdata linear ratio must
    # still carry the true zero, not NaN, since it *was* validly sampled.
    ratio = plot.fig.data[0].customdata[..., 2]
    assert np.isfinite(ratio).all()
    assert np.all(ratio == 0.0)
    assert np.isnan(plot.fig.data[0].z).all()


def test_compute_clim_ignores_nonpositive_values():
    plot = make_slice_plot()

    data = np.array([0.0, 0.0, 3.0, 7.0, np.nan])

    vmin, vmax = plot._compute_clim(data)

    assert vmin > 0
    assert vmax >= vmin
    # Must be finite and log-able -- this is what broke before the fix.
    assert np.isfinite(np.log10(vmin))
    assert np.isfinite(np.log10(vmax))


def test_compute_clim_all_nonpositive_falls_back():
    plot = make_slice_plot()

    data = np.array([0.0, 0.0, np.nan])

    assert plot._compute_clim(data) == (0.01, 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
