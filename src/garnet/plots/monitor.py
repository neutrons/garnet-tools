import numpy as np
import scipy.linalg
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go


class SlicePlot:
    """
    Plotly version of the reciprocal-space slice plot.

    This version resamples the oblique slice grid onto a rectangular
    display grid and draws it with a standard Plotly Heatmap.

    Parameters
    ----------
    UB : ndarray, shape (3, 3)
        UB matrix.
    W : ndarray, shape (3, 3)
        Basis or transform matrix used to construct the metric tensor
        in the slice frame.
    """

    # Hard cap on the display grid per axis, independent of oversample --
    # a full-resolution slice (e.g. 801x801) plus its 3x customdata
    # channel can exceed livedata.sns.gov's upload size limit (HTTP 413).
    MAX_DISPLAY_BINS = 201

    def __init__(self, UB, W):
        G = UB.T @ UB

        B = scipy.linalg.cholesky(G, lower=False)
        Bp = B @ W

        _, R = scipy.linalg.qr(Bp)

        self.V = R.T @ R

        self.fig = go.Figure()

        self.im = None

        self.x = None
        self.y = None
        self.z = None

        self.xlabel = None
        self.ylabel = None
        self.z_label = None

        self.slice_ind = None

        self.shear = None
        self.aspect = None

        self.u = None
        self.v = None
        self.X_source = None
        self.Y_source = None
        self.sample_points = None

        self.interpolation_method = "nearest"

    def calculate_transforms(
        self,
        axes,
        labels,
        normal,
        oversample=1.0,
        interpolation="nearest",
    ):
        """
        Precompute the display transform and build the Plotly heatmap.

        Parameters
        ----------
        axes : sequence of ndarray
            Axis arrays for the 3 dimensions.
        labels : sequence of str
            Axis labels for the 3 dimensions.
        normal : array-like of length 3
            Slice normal selector. This follows the same convention
            as the original code: the dimension with value 1 is taken
            as the sliced dimension.
        oversample : float, optional
            Display oversampling factor for the rectangular grid.
            A value of 1.0 is usually fine.
        interpolation : {"nearest", "linear"}, optional
            Interpolation used when resampling onto the display grid.
        """
        ind = np.asarray(normal) != 1

        axes_ind = np.arange(3)[ind]
        slice_ind = np.arange(3)[~ind][0]

        x, y = [np.asarray(axes[i]) for i in axes_ind]
        xlabel, ylabel = [labels[i] for i in axes_ind]

        self.z = np.asarray(axes[slice_ind])
        self.z_label = labels[slice_ind]
        self.slice_ind = slice_ind

        self.x = x
        self.y = y
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.interpolation_method = interpolation

        # Same 2x2 metric reduction as Matplotlib code.
        v = scipy.linalg.cholesky(self.V[np.ix_(ind, ind)], lower=False)
        v /= v[0, 0]

        # Affine part:
        # u = x + shear * (y - y_min)
        # displayed y-axis aspect handled separately.
        self.shear = v[0, 1]
        self.aspect = v[1, 1]

        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()

        # Circumscribed rectangle in transformed display coordinates.
        u_corners = np.array(
            [
                xmin,
                xmax,
                xmin + self.shear * (ymax - ymin),
                xmax + self.shear * (ymax - ymin),
            ]
        )

        umin = u_corners.min()
        umax = u_corners.max()

        dx = np.median(np.abs(np.diff(x))) if len(x) > 1 else 1.0
        dy = np.median(np.abs(np.diff(y))) if len(y) > 1 else 1.0

        nu = max(2, int(np.ceil(oversample * (umax - umin) / dx)) + 1)
        nv = max(2, int(np.ceil(oversample * (ymax - ymin) / dy)) + 1)

        nu = min(nu, self.MAX_DISPLAY_BINS)
        nv = min(nv, self.MAX_DISPLAY_BINS)

        self.u = np.linspace(umin, umax, nu)
        self.v = np.linspace(ymin, ymax, nv)

        U, V = np.meshgrid(self.u, self.v)

        # Inverse mapping from display coordinates (u, v) back to
        # logical slice coordinates (x, y):
        #
        #   u = x + shear * (y - y_min)
        #   v = y
        #
        # so
        #
        #   x = u - shear * (v - y_min)
        #   y = v
        # float32: halves the JSON payload of the customdata/z arrays
        # embedded in the exported plot, with no visible loss of
        # precision for display purposes.
        self.X_source = (U - self.shear * (V - ymin)).astype(np.float32)
        self.Y_source = V.astype(np.float32)

        # RegularGridInterpolator expects points in the order of
        # the data dimensions. Since slice data will have shape
        # (len(y), len(x)), the order is (y, x).
        self.sample_points = np.column_stack(
            [
                self.Y_source.ravel(),
                self.X_source.ravel(),
            ]
        )

        initial = np.full((nv, nu), np.nan, dtype=np.float32)
        customdata = np.stack(
            [self.X_source, self.Y_source, initial],
            axis=-1,
        )

        self.fig.data = []

        self.im = go.Heatmap(
            x=self.u,
            y=self.v,
            z=initial,
            customdata=customdata,
            colorscale="Turbo",
            zsmooth=False,
            colorbar=dict(title="Intensity"),
            hovertemplate=(
                f"{xlabel}: %{{customdata[0]:.3f}}<br>"
                f"{ylabel}: %{{customdata[1]:.3f}}<br>"
                "Intensity: %{customdata[2]:.5g}"
                "<extra></extra>"
            ),
        )

        self.fig.add_trace(self.im)

        # add_trace clones the trace onto the figure rather than keeping
        # it linked -- fig.data[0] is a different object from self.im at
        # this point. Re-point self.im at the live trace so later
        # mutations in make_slice() actually reach the figure that gets
        # serialized, instead of updating an orphaned copy.
        self.im = self.fig.data[0]

        self.fig.update_layout(
            margin=dict(l=70, r=90, b=60, t=60),
            xaxis=dict(
                title=xlabel,
                dtick=1,
                showgrid=False,
                zeroline=False,
            ),
            yaxis=dict(
                title=ylabel,
                dtick=1,
                showgrid=False,
                zeroline=False,
                scaleanchor="x",
                scaleratio=self.aspect,
            ),
        )

    def _extract_slice(self, signal, i):
        """
        Extract one 2D slice from the 3D signal array.
        """
        if self.slice_ind == 0:
            data = signal[i, :, :].T
        elif self.slice_ind == 1:
            data = signal[:, i, :].T
        else:
            data = signal[:, :, i].T

        return np.asarray(data, dtype=float)

    def _compute_clim(self, data):
        """
        Compute robust lower and upper bounds for log coloring.
        """
        finite = data[np.isfinite(data) & (data > 0)]

        if finite.size == 0:
            return 0.01, 1.0

        vmin = finite.min()
        vmax = finite.max()

        if np.isclose(vmin, vmax):
            if np.isclose(vmax, 0):
                return 0.01, 1.0
            return vmax / 100.0, vmax

        return vmin, vmax

    def _interpolate(self, signal_2d, mask_zero):
        """
        Resample one 2D array (counts or normalization) onto the display
        grid via the same (y, x) -> (u, v) mapping for every array, so
        counts and normalization interpolate consistently and can be
        divided afterwards on the display grid.

        mask_zero : bool
            Also treat zero/negative values as invalid (for the
            normalization array, where zero means no coverage) rather
            than a genuine value (for raw counts, where zero is valid).
        """
        valid = np.isfinite(signal_2d)
        if mask_zero:
            valid &= signal_2d > 0

        masked = np.where(valid, signal_2d, np.nan)

        interpolator = RegularGridInterpolator(
            (self.y, self.x),
            masked,
            method=self.interpolation_method,
            bounds_error=False,
            fill_value=np.nan,
        )

        return interpolator(self.sample_points).reshape(self.Y_source.shape)

    def make_slice(self, data_signal, norm_signal, value):
        """
        Update the figure to show the slice nearest the requested value.

        Counts and normalization are resampled onto the display grid
        separately, then divided -- resampling their ratio directly
        produces artifacts near masked/zero-exposure bins, since a
        single missing/zero normalization value would otherwise corrupt
        the interpolated ratio in its whole interpolation neighborhood.

        Parameters
        ----------
        data_signal : ndarray, shape (n0, n1, n2)
            3D raw-counts signal array.
        norm_signal : ndarray, shape (n0, n1, n2)
            3D normalization signal array, matching data_signal's shape.
        value : float
            Requested slice coordinate.
        """
        if self.z is None:
            raise RuntimeError(
                "calculate_transforms must be called before make_slice."
            )

        i = np.argmin(np.abs(self.z - value))

        data_2d = self._extract_slice(data_signal, i)
        norm_2d = self._extract_slice(norm_signal, i)

        sampled_data = self._interpolate(data_2d, mask_zero=False)
        sampled_norm = self._interpolate(norm_2d, mask_zero=True)

        with np.errstate(invalid="ignore", divide="ignore"):
            sampled = sampled_data / sampled_norm

        vmin, vmax = self._compute_clim(sampled)

        with np.errstate(divide="ignore", invalid="ignore"):
            log_sampled = np.where(
                sampled > 0, np.log10(sampled), np.nan
            ).astype(np.float32)
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)

        if np.isclose(log_min, log_max):
            log_min = log_max - 2.0

        exponent_min = int(np.ceil(log_min))
        exponent_max = int(np.floor(log_max))

        if exponent_max >= exponent_min:
            tickvals = np.arange(exponent_min, exponent_max + 1)
            ticktext = [f"1e{p}" for p in tickvals]
        else:
            tickvals = np.array([log_min, log_max])
            ticktext = [f"{vmin:.2g}", f"{vmax:.2g}"]

        customdata = np.stack(
            [self.X_source, self.Y_source, sampled.astype(np.float32)],
            axis=-1,
        )

        self.im.z = log_sampled
        self.im.customdata = customdata
        self.im.zmin = log_min
        self.im.zmax = log_max
        self.im.colorbar.tickvals = tickvals
        self.im.colorbar.ticktext = ticktext

        self.fig.update_layout(title=f"{self.z_label}={self.z[i]:.3f}")
