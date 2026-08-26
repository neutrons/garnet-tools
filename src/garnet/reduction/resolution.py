import csv

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from mantid.simpleapi import mtd
from mantid.kernel import V3D
from mantid.dataobjects import PeakShapeEllipsoid

from scipy.optimize import least_squares
from scipy.stats import chi2

# Stored/fitted ellipsoid radii and covariance-like "S" matrices throughout
# this module (and ellipsoid.py) are on a 99.7%-contour convention, not a
# literal 1-sigma covariance -- see ellipsoid.py's `gaussian`/
# `ellipsoid_covariance`. This is the same rescale factor applied there
# (`ellipsoid_covariance`'s `scale = chi2.ppf(perc/100, df=3)`), needed
# anywhere a stored/predicted S is used as a literal Gaussian covariance.
_CONTAINMENT_SCALE_3D = chi2.ppf(0.997, df=3)


def stoica_wilkinson_transform(two_theta, phi):
    """
    Local instrument-frame resolution axes (n, u, v) in lab (Q-lab) frame.

    Parameters
    ----------
    two_theta : float
        Scattering angle in radians.
    phi : float
        Azimuthal angle in radians.

    Returns
    -------
    T : ndarray, shape (3, 3)
        Rows are the orthonormal n, u, v directions.

    """
    ki_hat = np.array([0.0, 0.0, 1.0])

    kf_hat = np.array(
        [
            np.sin(two_theta) * np.cos(phi),
            np.sin(two_theta) * np.sin(phi),
            np.cos(two_theta),
        ]
    )

    n = kf_hat - ki_hat
    n /= np.linalg.norm(n)

    u = kf_hat + ki_hat
    u /= np.linalg.norm(u)

    v = np.cross(n, u)
    v /= np.linalg.norm(v)

    return np.vstack([n, u, v])


def _plot_resolution_diagnostics(rows, filename, sn_label="$I/\\sigma$"):
    """
    Render the obs-vs-predicted resolution diagnostic figure.

    Used by `ResolutionEllipsoid.plot_diagnostics` (obs = per-peak
    ellipsoid shapes already fit onto the workspace), factored out so a
    future second "obs" source can reuse the same rendering without
    duplicating it.

    Parameters
    ----------
    rows : list of dict
        Per-peak rows with keys "gamma", "nu", "lambda", "signal_noise",
        "Q", "obs_x0"/"x1"/"x2", "pred_x0"/"x1"/"x2",
        "offset_x0"/"x1"/"x2", "offset_norm_x0"/"x1"/"x2", and optionally
        "outlier" (bool) -- drawn as an open circle instead of a filled,
        colored point when True. Missing/absent defaults to False.
    filename : str
        Output image path.
    sn_label : str
        Colorbar label for the "signal_noise" field (callers may not all
        mean literal I/sigma).

    """
    gamma = np.array([r["gamma"] for r in rows])
    nu = np.array([r["nu"] for r in rows])
    lamda = np.array([r["lambda"] for r in rows])
    Q = np.array([r["Q"] for r in rows])
    signal_noise = np.array([r["signal_noise"] for r in rows])
    outlier = np.array([bool(r.get("outlier", False)) for r in rows])
    # If every single point is flagged, marking them all "outliers" is
    # meaningless (nothing left to contrast against) and would leave the
    # colored scatter with no data at all, which breaks log-scale
    # colorbars outright (no range to autoscale from) -- fall back to
    # plotting everything normally in that degenerate case.
    if outlier.all():
        outlier = np.zeros_like(outlier)

    s = 1 if Q.size > 1000 else 10

    def scatter_with_outliers(ax, x, y, c, **kwargs):
        # Points flagged as outliers (down-weighted by robust_nnls when
        # fitting the model) are drawn as open circles on top, so they
        # read as "excluded from the fit" rather than blending in with
        # the normal colored points.
        sc = ax.scatter(x[~outlier], y[~outlier], c=c[~outlier], **kwargs)
        if outlier.any():
            ax.scatter(
                x[outlier],
                y[outlier],
                facecolors="none",
                edgecolors="red",
                s=kwargs.get("s", 10) * 3,
                marker="o",
                linewidths=0.8,
                rasterized=kwargs.get("rasterized", False),
                zorder=0,
            )
        return sc

    obs = {
        k: np.array([r[f"obs_{k}"] for r in rows]) for k in ["x0", "x1", "x2"]
    }
    pred = {
        k: np.array([r[f"pred_{k}"] for r in rows]) for k in ["x0", "x1", "x2"]
    }
    offset = {
        k: np.array([r[f"offset_{k}"] for r in rows])
        for k in ["x0", "x1", "x2"]
    }

    hi = np.sqrt(
        max(
            max(obs[k].max() for k in obs),
            max(pred[k].max() for k in pred),
        )
    )

    max_res = max((abs(obs[k] - pred[k]) / Q).max() for k in obs) * 100

    max_trans = max(abs(offset[k]).max() for k in offset)

    names = {
        "x0": "{|Q|}",
        "x1": "{\\Delta{Q}_1}",
        "x2": "{\\Delta{Q}_2}",
    }

    fig, axes = plt.subplots(2, 6, figsize=(18, 7), constrained_layout=True)

    sc_lambda = sc_sn = sc_resid = None

    for k, lab in enumerate(["x0", "x1", "x2"]):
        c0 = k
        c1 = k + 3
        name = names[lab]

        sig_obs = np.sqrt(np.maximum(obs[lab], 0.0))
        sig_pred = np.sqrt(np.maximum(pred[lab], 0.0))

        # [0, c0] obs vs calc – lambda color
        ax = axes[0, c0]
        sc_lambda = scatter_with_outliers(
            ax,
            sig_pred,
            sig_obs,
            lamda,
            s=s,
            cmap="viridis",
            marker=".",
            rasterized=True,
        )
        ax.plot([0, hi], [0, hi], "k--", lw=1)
        ax.set_xlabel("$r(\\mathrm{calc})$ [$\\AA^{-1}$]")
        if k == 0:
            ax.set_ylabel("$r(\\mathrm{obs})$ [$\\AA^{-1}$]")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"${name}$-axis")
        ax.set_aspect("equal", adjustable="box")
        ax.minorticks_on()

        # [0, c1] residual vs Q – S/N color
        ax = axes[0, c1]
        resid = (sig_obs - sig_pred) / Q * 100
        sc_sn = scatter_with_outliers(
            ax,
            Q,
            resid,
            signal_noise,
            s=s,
            cmap="plasma",
            norm="log",
            marker=".",
            rasterized=True,
        )
        ax.axhline(0, color="k", lw=1, linestyle="--")
        ax.set_ylim(-max_res, max_res)
        ax.tick_params(labelbottom=False)
        if k == 0:
            ax.set_ylabel("$[r_{\\rm obs}-r_{\\rm calc}]/|Q|$ [%]")
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(f"${name}$-axis")
        ax.minorticks_on()

        # [1, c0] gamma/nu map – relative residual % color
        ax = axes[1, c0]
        resid_map = (
            np.abs(sig_obs - sig_pred) / np.maximum(sig_pred, 1e-12) * 100
        )
        sc_resid = scatter_with_outliers(
            ax,
            np.rad2deg(gamma),
            np.rad2deg(nu),
            resid_map,
            s=s,
            cmap="binary",
            norm="linear",
            marker=".",
            rasterized=True,
        )
        ax.set_xlabel("$\\gamma$ [$^\\circ$]")
        if k == 0:
            ax.set_ylabel("$\\nu$ [$^\\circ$]")
        else:
            ax.tick_params(labelleft=False)
        ax.set_aspect("equal", adjustable="box")
        ax.minorticks_on()

        # [1, c1] center offset vs Q – S/N color
        ax = axes[1, c1]
        resid = offset[lab] / Q * 100
        sc_sn = scatter_with_outliers(
            ax,
            Q,
            resid,
            signal_noise,
            s=s,
            cmap="plasma",
            norm="log",
            marker=".",
            rasterized=True,
        )
        ax.axhline(0, color="k", lw=1, linestyle="--")
        ax.set_ylim(-max_trans, max_trans)
        ax.set_xlabel("$|Q|$ [$\\AA^{-1}$]")
        if k == 0:
            ax.set_ylabel("$\Delta{c}/|Q|$ [%]")
        else:
            ax.tick_params(labelleft=False)
        ax.minorticks_on()

    cb = fig.colorbar(
        sc_lambda, ax=list(axes[0, [0, 1, 2]]), label="$\\lambda$ [$\\AA$]"
    )
    cb.ax.minorticks_on()
    cb = fig.colorbar(
        sc_sn,
        ax=axes[:, [3, 4, 5]].ravel().tolist(),
        label=sn_label,
    )
    cb.ax.minorticks_on()
    cb = fig.colorbar(
        sc_resid,
        ax=list(axes[1, [2]]),
        label="$|r_{\\rm obs}/r_{\\rm calc}-1|$ [%]",
    )
    cb.ax.minorticks_on()

    fig.savefig(filename, bbox_inches="tight")


def _ellipse_patch(S_2d, center_2d, **kwargs):
    """
    A `matplotlib.patches.Ellipse` for a 2D containment-scale covariance
    (same convention as `_CONTAINMENT_SCALE_3D` -- radii, not sigma).
    """
    eigvals, eigvecs = np.linalg.eigh(0.5 * (S_2d + S_2d.T))
    eigvals = np.maximum(eigvals, 0.0)

    minor, major = np.sqrt(eigvals)
    vec_major = eigvecs[:, 1]
    angle = np.degrees(np.arctan2(vec_major[1], vec_major[0]))

    return Ellipse(
        center_2d,
        width=2 * major,
        height=2 * minor,
        angle=angle,
        **kwargs,
    )


def _plot_peak_shape_diagnostics(samples, filename, axis_labels=None):
    """
    Render the per-peak box + overlaid ellipse diagnostic figure -- a
    spatial complement to `_plot_resolution_diagnostics`'s
    population-level obs-vs-pred scatter, showing the actual counts
    box each peak's shape estimate(s) came from.

    Used by `Integration.plot_peak_shape_diagnostics` -- observed vs.
    population-model-predicted shape, and (when an instrument prior
    model was used to size the integration radii) prior vs. observed
    shape too -- on a small, signal/noise- and |Q|-stratified sample of
    peaks (plotting every peak this way doesn't scale).

    Parameters
    ----------
    samples : list of dict
        Per-peak rows with keys "label", "Q0"/"Q1"/"Q2" (dense grids
        from `data.bin_in_Q`, ij-indexed), "counts" (matching 3D
        array), and "ellipses": a list of dict, each with "center"
        (3-tuple, same frame as "Q0"/"Q1"/"Q2"), "S" (3x3,
        containment-scale, same frame), "label", "color", and
        "linestyle" -- one dashed/solid pair overlaid on the counts
        image per sample, in the order given.
    filename : str
        Output image path.
    axis_labels : list of (str, str), optional
        Per-projection axis labels, one pair per dimension dropped
        (i.e. projections onto the other two). Defaults to the
        Q-sample convention `[("Q_1", "Q_2"), ("Q_0", "Q_2"),
        ("Q_0", "Q_1")]`; pass e.g. local rotated-frame labels when
        "Q0"/"Q1"/"Q2" aren't literal Q-sample components.

    """
    n_rows = len(samples)

    if axis_labels is None:
        axis_labels = [("Q_1", "Q_2"), ("Q_0", "Q_2"), ("Q_0", "Q_1")]

    fig, axes = plt.subplots(
        n_rows, 3, figsize=(12, 4 * n_rows), squeeze=False
    )

    for row, sample in enumerate(samples):
        xs = [
            sample["Q0"][:, 0, 0],
            sample["Q1"][0, :, 0],
            sample["Q2"][0, 0, :],
        ]
        counts = sample["counts"]
        ellipses = sample["ellipses"]

        for k in range(3):
            a, b = [d for d in range(3) if d != k]

            ax = axes[row, k]

            proj = counts.sum(axis=k)

            ax.imshow(
                proj.T,
                origin="lower",
                extent=[xs[a].min(), xs[a].max(), xs[b].min(), xs[b].max()],
                aspect="auto",
                cmap="viridis",
            )

            idx = [a, b]

            for ellipse in ellipses:
                center_2d = (ellipse["center"][a], ellipse["center"][b])

                ax.add_patch(
                    _ellipse_patch(
                        ellipse["S"][np.ix_(idx, idx)],
                        center_2d,
                        edgecolor=ellipse["color"],
                        facecolor="none",
                        linestyle=ellipse["linestyle"],
                        linewidth=1.2,
                    )
                )

            lab_a, lab_b = axis_labels[k]
            ax.set_xlabel("${}$ [$\\AA^{{-1}}$]".format(lab_a))
            if k == 0:
                ax.set_ylabel(
                    "{}\n${}$ [$\\AA^{{-1}}$]".format(sample["label"], lab_b)
                )
            else:
                ax.set_ylabel("${}$ [$\\AA^{{-1}}$]".format(lab_b))
            if row == 0:
                ax.set_title("{} vs {}".format(lab_a, lab_b))

    seen = set()
    handles = []
    for sample in samples:
        for ellipse in sample["ellipses"]:
            key = (ellipse["label"], ellipse["color"], ellipse["linestyle"])
            if key not in seen:
                seen.add(key)
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        color=ellipse["color"],
                        linestyle=ellipse["linestyle"],
                        label=ellipse["label"],
                    )
                )
    fig.legend(handles=handles, loc="upper center", ncol=len(handles))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


class ResolutionEllipsoid:
    def __init__(
        self,
        peaks_ws,
        r_cut=np.inf,
        sig_noise_cut=5.0,
        min_peaks=10,
        scale_bounds=(0.5, 2.0),
        mosaic="isotropic",
        peak_shape_frame="s",
    ):
        if peak_shape_frame not in ("l", "s"):
            raise ValueError(
                "peak_shape_frame must be 'l' (lab) or 's' (sample)"
            )

        self.peaks_ws = peaks_ws
        self.r_cut = r_cut
        self.sig_noise_cut = sig_noise_cut
        self.min_peaks = min_peaks
        self.scale_bounds = scale_bounds
        self.mosaic = mosaic
        self.peak_shape_frame = peak_shape_frame
        self.model = None
        self.prior_S_sigma = None
        self.prior_center_sigma = None
        self.lamda_min = np.inf
        self.lamda_max = 0
        self.two_theta_min = np.inf
        self.two_theta_max = 0

    def _transverse_directions(self, qhat):
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(qhat, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        m1 = np.cross(qhat, ref)
        m1 /= np.linalg.norm(m1)
        m2 = np.cross(qhat, m1)
        m2 /= np.linalg.norm(m2)
        return m1, m2

    def _normalize_columns(self, V):
        V = np.asarray(V, dtype=float)
        n = np.linalg.norm(V, axis=0, keepdims=True)
        n[n == 0] = 1.0
        return V / n

    def _vech6(self, M):
        return np.array(
            [M[0, 0], M[1, 1], M[2, 2], M[1, 2], M[0, 2], M[0, 1]], dtype=float
        )

    def _outer6(self, a):
        M = np.outer(a, a)
        return self._vech6(M)

    def _get_peak_offset(self, ws, no):
        shape = ws.getPeak(no).getPeakShape()

        UB = ws.sample().getOrientedLattice().getUB()

        hkl = ws.getPeak(no).getHKL()
        R = ws.getPeak(no).getGoniometerMatrix()

        Qobs = ws.getPeak(no).getQLabFrame()
        Qcalc = 2 * np.pi * R @ UB @ hkl

        d0, d1, d2 = Qobs - Qcalc

        if shape.shapeName() == "ellipsoid":
            try:
                d = eval(shape.toJSON())
            except:
                d = None

            if d is not None:
                if "translation0" in d.keys():
                    d0 += d["translation0"]
                    d1 += d["translation1"]
                    d2 += d["translation2"]

        return np.array([d0, d1, d2])

    def _get_peak_shape(self, ws, no):
        """
        A peak's stored ellipsoid shape, or NaN radii if it was never
        given a real one (`shapeName() != "ellipsoid"` -- e.g. a peak
        `estimate_peak_shapes` skipped as no-signal/degenerate). NaN
        radii fail every caller's existing `np.isfinite(radii)` check,
        so such peaks are excluded rather than silently fit/plotted
        against a fabricated `r_cut`-sized shape that was never
        actually observed.
        """
        shape = ws.getPeak(no).getPeakShape()

        radii = np.full(3, np.nan, dtype=float)
        v0, v1, v2 = np.eye(3).tolist()

        if shape.shapeName() == "ellipsoid":
            try:
                d = eval(shape.toJSON())
            except:
                d = None

            if d is not None:
                v0 = [float(x) for x in d["direction0"].split()]
                v1 = [float(x) for x in d["direction1"].split()]
                v2 = [float(x) for x in d["direction2"].split()]

                radii = np.array(
                    [
                        float(d["radius0"]),
                        float(d["radius1"]),
                        float(d["radius2"]),
                    ]
                )

        V = np.column_stack([v0, v1, v2])
        V = self._normalize_columns(V)

        return radii, V, self.peak_shape_frame

    def _set_peak_shape(self, ws, no, radii, V_lab, R=None):
        if self.peak_shape_frame == "s":
            V = self._lab_axes_to_sample(R, V_lab)
        else:
            V = V_lab

        V = self._normalize_columns(V)
        v0, v1, v2 = V.T

        shape = PeakShapeEllipsoid(
            [V3D(*v0), V3D(*v1), V3D(*v2)],
            list(radii),
            list(radii),
            list(radii),
        )
        ws.getPeak(no).setPeakShape(shape)

    def _S_from_ellipsoid(self, radii, V):
        # S = V @ diag(r²) @ V.T  (r² = containment-radii squared)
        r = np.asarray(radii, dtype=float)
        return V @ np.diag(r**2) @ V.T

    def _ellipsoid_from_S(self, S):
        S = 0.5 * (S + S.T)

        r_sq, V = np.linalg.eigh(S)

        radii = np.sqrt(np.maximum(r_sq, 0.0))
        V = self._normalize_columns(V)

        if np.linalg.det(V) < 0:
            V[:, -1] = -V[:, -1]

        return radii, V

    def _peak_params(self, peak):
        two_theta = peak.getScattering()
        phi = peak.getAzimuthal()

        lamda = peak.getWavelength()

        kf_x = np.sin(two_theta) * np.cos(phi)
        kf_y = np.sin(two_theta) * np.sin(phi)
        kf_z = np.cos(two_theta)

        nu = np.arcsin(kf_y)
        gamma = np.arctan2(kf_x, kf_z)

        return gamma, nu, lamda

    def _Q_magnitude(self, two_theta, lamda):
        return (4.0 * np.pi / lamda) * np.sin(0.5 * two_theta)

    def _tof_path_length(self, lamda, tof):
        return 0.003956034 * tof / lamda

    def _stoica_wilkinson_transform_from_peak(self, peak):
        return stoica_wilkinson_transform(
            peak.getScattering(), peak.getAzimuthal()
        )

    def _sample_axes_to_lab(self, R, V_sample):
        return R @ V_sample

    def _lab_axes_to_sample(self, R, V_lab):
        return R.T @ V_lab

    def _model_design_lab(
        self, two_theta, phi, lamda, R, lambda_0=None, p=1.0
    ):
        """
        Approximation to the resolution function.

        J. B. Forsyth, Single Crystal Pulsed Neutron Diffraction, in
        Chemical Crystallography with Pulsed Neutrons and Synchroton X-Rays,
        edited by M. A. Carrondo and G. A. Jeffrey (Springer Netherlands,
        Dordrecht, 1988), pp. 117–135.

        A. D. Stoica, On the resolution of slow-neutron spectrometers. II. The
        resolution function for time-of-flight diffractometry, Acta Cryst A 31,
        193 (1975).

        Parameters
        ----------
        two_theta : float
            Scattering angle in radians.
        phi : float
            Azimuthal angle in radians.
        lamda : float
            Wavelength in angstroms.
        R : ndarray
            Goniometer matrix (only used for the "diagonal"/"full" mosaic
            models).
        lambda_0 : float or None, optional
            Saturation wavelength scale for the isotropic incident-
            divergence model sigma_i(lambda) = sigma_i * x^p /
            sqrt(1 + x^(2p)), x = lambda / lambda_0 -- a smooth curve
            that grows like x^p for lambda << lambda_0 and saturates to
            sigma_i for lambda >> lambda_0. None (default) disables the
            scaling -- sigma_i is then a plain wavelength-independent
            sigma, as before this model existed.
        p : float, optional
            Growth-rate exponent for the sigma_i(lambda) curve above.
            Fit jointly with lambda_0 rather than assumed -- empirically
            close to 2 (not the linear growth a tanh(lambda/lambda_0)
            curve implies) for TOPAZ/CORELLI, ~1.75 for MANDI. Unused
            when lambda_0 is None.

        Returns
        -------
        A : ndarray
            Design matrix.

        """
        k = 2.0 * np.pi / lamda

        s = np.sin(two_theta)
        c = np.cos(two_theta)
        cp = np.cos(phi)
        sp = np.sin(phi)

        # Incident-beam divergence directions. k_i is always exactly
        # the beam axis (no elevation/azimuth to vary), so the
        # horizontal/vertical angular coordinates (gamma, nu) and
        # literal tangent-plane directions coincide exactly here.
        # gamma_i/nu_i were fit as separate parameters, but came out
        # strongly correlated (0.5-0.9 across TOPAZ/CORELLI/MANDI) --
        # not separately identifiable from this data -- so they're
        # combined into one isotropic incident-divergence term.
        gamma_i = np.array([1.0, 0.0, 0.0])
        nu_i = np.array([0.0, 1.0, 0.0])

        ki = np.array([0.0, 0.0, 1.0])
        kf = np.array([s * cp, s * sp, c])

        # Final-beam divergence directions, in terms of the detector's
        # horizontal/vertical angular coordinates (gamma, nu) -- not
        # the (two_theta, phi) polar-around-the-beam-axis angles kf is
        # built from above. Away from the equatorial plane (nu != 0)
        # these two angular bases rotate apart, so a divergence model
        # built from (two_theta, phi) tangents mixes the true
        # horizontal/vertical spread into the wrong directions.
        #
        # gamma = atan2(x, z), nu = arcsin(y) for kf = (x, y, z); see
        # diagnostics() for the same (gamma, nu) convention. rho =
        # sqrt(x^2 + z^2) = cos(nu).
        #
        # gamma_f = cos(nu) * h_f (h_f = unit horizontal tangent,
        # d(kf)/d(gamma) = cos(nu) * h_f) -- the cos(nu) simplifies
        # away to (z, 0, -x) exactly, avoiding a 1/rho division. This
        # makes sigma_gamma_f a coordinate-angle uncertainty (in
        # gamma itself), not a literal tangent-plane spread.
        # nu_f = v_f (unit vertical tangent, d(kf)/d(nu) = v_f).
        x, y, z = kf
        rho = np.hypot(x, z)

        gamma_f = np.array([z, 0.0, -x])
        nu_f = np.array([-x * y, rho**2, -y * z]) / rho

        q_lambda = kf - ki

        Q_vec = k * q_lambda
        Q2 = np.dot(Q_vec, Q_vec)

        # (sigma_lambda/lambda)^2 = sigma_dl_mod^2 + (sigma_dl_mod_lam * lambda)^2
        # -- a wavelength-independent floor (sigma_dl_mod, as before)
        # plus a term that grows with lambda (sigma_dl_mod_lam).
        #
        # Incident divergence saturates with wavelength instead of
        # growing without bound: sigma_i(lambda) = sigma_i * x^p /
        # sqrt(1 + x^(2p)), x = lambda / lambda_0. Physical motivation:
        # if the incident beam is guide-fed, the guide's supermirror
        # critical angle scales with lambda, so short-lambda neutrons
        # are guide-limited (divergence rises with lambda) until the
        # guide's acceptance exceeds the source aperture's own
        # divergence, at which point the guide stops being the
        # limiting factor and sigma_i saturates at its asymptotic
        # value. lambda_0 and p are fit jointly, nonlinearly, alongside
        # the (still linear/bounded) sigma parameters -- see
        # robust_nnls_saturation. p empirically comes out close to 2,
        # not the linear (p=1) growth a tanh(lambda/lambda_0) curve
        # would imply.
        if lambda_0 is not None:
            x = lamda / lambda_0
            sat2 = x ** (2 * p) / (1.0 + x ** (2 * p))
        else:
            sat2 = 1.0

        cols = [
            (k**2 * sat2)
            * (self._outer6(gamma_i) + self._outer6(nu_i)),  # sigma_i^2
            k**2 * self._outer6(gamma_f),  # sigma_gamma_f^2
            k**2 * self._outer6(nu_f),  # sigma_nu_f^2
            k**2 * self._outer6(q_lambda),  # sigma_dl_mod^2 (σ_λ/λ)
            (k**2 * lamda**2)
            * self._outer6(q_lambda),  # sigma_dl_mod_lam^2
        ]

        if self.mosaic == "isotropic":
            mosaic_iso = Q2 * np.eye(3) - np.outer(Q_vec, Q_vec)
            cols.append(self._vech6(mosaic_iso))

        elif self.mosaic == "diagonal":
            for j in range(3):
                cols.append(self._outer6(np.cross(Q_vec, R[:, j])))

        else:  # "full"
            s2 = np.sqrt(0.5)
            mosaic_dirs = [
                R[:, 0],
                R[:, 1],
                R[:, 2],
                (R[:, 0] + R[:, 1]) * s2,
                (R[:, 0] + R[:, 2]) * s2,
                (R[:, 1] + R[:, 2]) * s2,
            ]
            for v in mosaic_dirs:
                cols.append(self._outer6(np.cross(Q_vec, v)))

        return np.column_stack(cols)

    def _predict_S_lab(self, peak):
        A = self._model_design_lab(
            peak.getScattering(),
            peak.getAzimuthal(),
            peak.getWavelength(),
            peak.getGoniometerMatrix(),
            lambda_0=self.model.get("lambda_0"),
            p=self.model.get("p", 1.0),
        )
        y = A @ self.model["variance_parameters"]
        S = np.array(
            [[y[0], y[5], y[4]], [y[5], y[1], y[3]], [y[4], y[3], y[2]]]
        )
        return 0.5 * (S + S.T)

    def _signed_sqrt(self, v, eps):
        """
        sign(v) * sqrt(|v| + eps).

        `vech6`'s three diagonal rows (Q_xx, Q_yy, Q_zz) are always
        >= 0, where this is equivalent to the naive sqrt(max(v, eps))
        width transform. Its three off-diagonal rows (Q_yz, Q_xz,
        Q_xy) are signed -- empirically ~2/3 negative, comparable in
        magnitude to the diagonal rows -- and a plain max(v, eps)
        floor collapses all of that sign/magnitude information to
        ~sqrt(eps), destroying real data about how each peak's
        ellipsoid is tilted in the lab frame instead of just guarding
        against sqrt of a tiny/negative model-prediction rounding
        error. This is a smooth, sign-preserving generalization of the
        same width transform, applied uniformly since it reduces to
        the old behavior on the always-nonnegative rows anyway.
        """
        return np.sign(v) * np.sqrt(np.abs(v) + eps)

    def robust_nnls(self, A, y, max_iter=20, c=1.345, eps=1e-12):
        """
        Robust bounded least squares (x >= 0) using a Huber loss.

        Replaces a previous plain-NNLS (Lawson-Hanson) reweighting
        scheme: NNLS's active-set solve pins parameters to exactly
        zero, and which parameters get pinned can flip between
        iterations as the Huber weights change, so the fit could
        oscillate instead of converging. `least_squares` with
        `method="trf"` enforces x >= 0 continuously instead, and
        supports the Huber loss directly.
        """

        finite = np.all(np.isfinite(A), axis=1) & np.isfinite(y)
        A = A[finite]
        y = y[finite]

        n = A.shape[1]
        bounds = (np.zeros(n), np.full(n, np.inf))

        def width_resid(x):
            # compare widths rather than variances
            y_fit = A @ x
            return self._signed_sqrt(y_fit, eps) - self._signed_sqrt(y, eps)

        # initial unweighted fit for a feasible starting point
        x = least_squares(
            width_resid, np.zeros(n), bounds=bounds, method="trf"
        ).x

        w = np.ones(A.shape[0])
        result = None

        for _ in range(max_iter):
            r = width_resid(x)

            # robust scale estimate
            mad = np.median(np.abs(r - np.median(r)))
            scale = 1.4826 * mad + eps

            result = least_squares(
                width_resid,
                x,
                bounds=bounds,
                method="trf",
                loss="huber",
                f_scale=c * scale,
            )
            x_new = result.x

            z = width_resid(x_new) / scale
            w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z))

            if np.linalg.norm(x_new - x) < 1e-10 * (np.linalg.norm(x) + eps):
                x = x_new
                break

            x = x_new

        residual_norm = np.linalg.norm(A @ x - y)
        return x, residual_norm, w, result

    def robust_nnls_saturation(
        self,
        A,
        y,
        lamda,
        tanh_cols,
        max_iter=20,
        c=1.345,
        eps=1e-12,
        lambda_0_bounds=(0.01, 100.0),
        p_bounds=(0.2, 6.0),
    ):
        """
        Like `robust_nnls`, but `tanh_cols` columns additionally get
        scaled by x^(2p) / (1 + x^(2p)) (x = lamda / lambda_0) before
        every residual evaluation, with lambda_0 and p fit jointly and
        nonlinearly alongside the (still nonnegative) linear sigma^2
        parameters. `least_squares` handles a mix of nonlinear
        parameters and several bounded linear ones fine -- lambda_0
        and p are just appended to the parameter vector.

        Parameters
        ----------
        A, y : as in `robust_nnls` -- A's `tanh_cols` columns must be
            the raw (lambda_0=None) columns from `_model_design_lab`,
            not yet saturation-scaled.
        lamda : ndarray, shape (A.shape[0],)
            Wavelength (Angstrom) for each row of A.
        tanh_cols : sequence of int
            Column indices of A to scale by the saturation factor.
        lambda_0_bounds : tuple of float, optional
            Bounds for lambda_0 (Angstrom). Deliberately wide -- a
            lambda_0 much larger than the data's wavelength range
            recovers pure power-law growth, and one much smaller
            recovers a wavelength-independent constant, so the data
            is free to prefer either limit.
        p_bounds : tuple of float, optional
            Bounds for the growth-rate exponent p.

        Returns
        -------
        x : ndarray
            Fitted sigma^2 parameters (same meaning/order as
            `robust_nnls`, i.e. without lambda_0/p).
        lambda_0 : float
            Fitted saturation wavelength scale.
        p : float
            Fitted growth-rate exponent.
        residual_norm : float
        w : ndarray
            Final robust (Huber) weights.
        result : OptimizeResult
            `result.x`/`result.jac` have two extra (final) entries/
            columns, for lambda_0 then p, for `_parameter_covariance`.

        """

        finite = (
            np.all(np.isfinite(A), axis=1)
            & np.isfinite(y)
            & np.isfinite(lamda)
        )
        A = A[finite]
        y = y[finite]
        lamda = lamda[finite]

        n = A.shape[1]
        lo = np.concatenate([np.zeros(n), [lambda_0_bounds[0], p_bounds[0]]])
        hi = np.concatenate(
            [np.full(n, np.inf), [lambda_0_bounds[1], p_bounds[1]]]
        )

        def scaled_A(lambda_0, p):
            x = lamda / lambda_0
            factor = x ** (2 * p) / (1.0 + x ** (2 * p))
            A_s = A.copy()
            for j in tanh_cols:
                A_s[:, j] = A_s[:, j] * factor
            return A_s

        def width_resid(v):
            x, lambda_0, p = v[:-2], v[-2], v[-1]
            y_fit = scaled_A(lambda_0, p) @ x
            return self._signed_sqrt(y_fit, eps) - self._signed_sqrt(y, eps)

        v0 = np.concatenate(
            [
                np.zeros(n),
                [np.sqrt(lambda_0_bounds[0] * lambda_0_bounds[1])],
                [1.0],
            ]
        )

        v = least_squares(width_resid, v0, bounds=(lo, hi), method="trf").x

        w = np.ones(A.shape[0])
        result = None

        for _ in range(max_iter):
            r = width_resid(v)

            mad = np.median(np.abs(r - np.median(r)))
            scale = 1.4826 * mad + eps

            result = least_squares(
                width_resid,
                v,
                bounds=(lo, hi),
                method="trf",
                loss="huber",
                f_scale=c * scale,
            )
            v_new = result.x

            z = width_resid(v_new) / scale
            w = np.where(np.abs(z) <= c, 1.0, c / np.abs(z))

            if np.linalg.norm(v_new - v) < 1e-10 * (np.linalg.norm(v) + eps):
                v = v_new
                break

            v = v_new

        x, lambda_0, p = v[:-2], v[-2], v[-1]
        residual_norm = np.linalg.norm(scaled_A(lambda_0, p) @ x - y)
        return x, lambda_0, p, residual_norm, w, result

    def _parameter_covariance(self, result):
        """
        Approximate covariance of the fitted (variance-space) parameters
        from the final robust least-squares solve in `robust_nnls`, using
        the same Jacobian/SVD approach `scipy.optimize.curve_fit` uses
        internally (Gauss-Newton approximation to the Hessian of the cost).

        This is an asymptotic estimate: it assumes the solution is not
        pinned against the x >= 0 bound and that the Huber-reweighted
        Jacobian is a reasonable stand-in for the true curvature. Returns
        None if the Jacobian is unavailable or degenerate (e.g. more free
        parameters than usable peaks, or a parameter pinned at zero).
        """
        if result is None:
            return None

        J = result.jac
        m, n = J.shape
        if m <= n:
            return None

        try:
            _, s, VT = np.linalg.svd(J, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        threshold = np.finfo(float).eps * max(J.shape) * s[0]
        keep = s > threshold
        if not np.any(keep):
            return None

        s = s[keep]
        VT = VT[keep]
        pcov = (VT.T / s**2) @ VT

        dof = m - n
        s_sq = np.sum(result.fun**2) / dof

        return pcov * s_sq

    def _label_variance_uncertainties(self, x, x_cov):
        """
        Propagate variance-space parameter uncertainties (diagonal of
        `x_cov`, i.e. Var(sigma^2)) to standard errors on sigma itself via
        the delta method: Var(sigma) = Var(sigma^2) / (4 * sigma^2).

        Returns a dict keyed like `_label_variance_parameters` with a
        "_stderr" suffix, e.g. "sigma_gamma_i_stderr". Empty if `x_cov`
        is None.
        """
        if x_cov is None:
            return {}

        var_x = np.clip(np.diag(x_cov), 0.0, None)
        sigma = np.sqrt(np.clip(x, 0.0, None))

        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_err = np.where(
                sigma > 0, np.sqrt(var_x) / (2.0 * sigma), np.nan
            )

        labels = list(self._label_variance_parameters(x).keys())
        return {
            f"{label}_stderr": err for label, err in zip(labels, sigma_err)
        }

    def _parameter_correlation(self, x_cov):
        """
        Correlation matrix of the fitted sigma parameters.

        Note this is identical to the correlation matrix of the
        underlying variance-space parameters `x_cov`: sigma = sqrt(x) is
        a per-parameter (diagonal) reparametrization, and a correlation
        matrix is invariant under positive diagonal rescaling of each
        variable. Returns None if `x_cov` is None or any parameter has
        zero variance (e.g. pinned at the x >= 0 bound).
        """
        if x_cov is None:
            return None

        d = np.sqrt(np.clip(np.diag(x_cov), 0.0, None))
        if np.any(d == 0):
            return None

        return x_cov / np.outer(d, d)

    def fit(
        self, fixed_instrumental=None, weighting="sn_over_q2", exclude=None
    ):
        """
        Fit the variance parameters against the workspace's peak shapes.

        Parameters
        ----------
        fixed_instrumental : dict or None
            Prior instrumental sigmas (radians, "sigma_dl_mod"/
            "sigma_dl_mod_lam" dimensionless -- same shape as
            `set_variance_parameters` expects), e.g. from a beamline's
            characterized `DivergenceParams`. When given, the
            instrumental columns (sigma_i, gamma_f, nu_f, dl_mod,
            dl_mod_lam) are held at this prior shape and only a
            single overall scale factor plus the mosaic term(s) are
            fit, rather than letting all instrumental terms float
            independently -- lambda_0 (if present in the prior) is
            baked into the design matrix directly rather than fit.
            "sigma_dl_mod_lam" defaults to 0 if absent from an older
            prior.
        weighting : str, optional
            How to weight each peak's row:
            "sn_over_q2" (default) -- signal/noise / Q^2.
            "sn" -- signal/noise alone (no Q^2 down-weighting of the
            high-Q tail).
            "none" -- every peak weighted equally.
            A diagnostic knob to check whether the Q^2 term is masking
            a real trend at the tails of the Q or wavelength range,
            where weight is otherwise systematically smaller/larger.
        exclude : set of int or None, optional
            Peak indices (workspace row index, same as `diagnostics()`'s
            "i") to hard-exclude from this fit entirely -- e.g. peaks a
            prior iteration of `fit_iterative` flagged as sigma-clipped
            outliers. Unlike the internal Huber down-weighting (which
            still lets an outlier's row pull the fit a little), these
            peaks contribute nothing.

        """
        ws = mtd[self.peaks_ws]
        n_excluded = 0

        # None (free calibration fit) -- built with lambda_0=None (raw,
        # unscaled gamma_i/nu_i columns) and lambda_0/p are fit
        # nonlinearly alongside the linear sigma^2 parameters
        # (robust_nnls_saturation). Given (per-run science-data fit
        # against a fixed instrumental prior) -- that prior's own
        # calibrated lambda_0/p are baked directly into A via
        # _model_design_lab, and the rest of the fit stays fully
        # linear (robust_nnls).
        lambda_0_prior = (
            fixed_instrumental.get("lambda_0")
            if fixed_instrumental is not None
            else None
        )
        p_prior = (
            fixed_instrumental.get("p", 1.0)
            if fixed_instrumental is not None
            else None
        )

        A_blocks = []
        y_blocks = []
        lamda_blocks = []
        used = []

        n_low_sig_noise = 0
        n_bad_shape = 0
        n_bad_Q = 0
        n_nonfinite_row = 0
        sig_noise_seen = []

        for i, peak in enumerate(ws):
            if exclude is not None and i in exclude:
                n_excluded += 1
                continue

            sig_noise = peak.getIntensityOverSigma()

            radii_s, V_s, frame = self._get_peak_shape(ws, i)
            if np.all(np.isfinite(radii_s)):
                sig_noise_seen.append(sig_noise)

            if not np.isfinite(sig_noise) or sig_noise < self.sig_noise_cut:
                n_low_sig_noise += 1
                continue

            two_theta = peak.getScattering()
            lamda = peak.getWavelength()

            if lamda < self.lamda_min:
                self.lamda_min = lamda
            if lamda > self.lamda_max:
                self.lamda_max = lamda

            if two_theta < self.two_theta_min:
                self.two_theta_min = two_theta
            if two_theta > self.two_theta_max:
                self.two_theta_max = two_theta

            R = peak.getGoniometerMatrix()

            if not np.all(np.isfinite(radii_s)) or np.any(radii_s <= 0):
                n_bad_shape += 1
                continue

            V_lab = self._sample_axes_to_lab(R, V_s) if frame == "s" else V_s
            V_lab = self._normalize_columns(V_lab)

            S_lab_obs = self._S_from_ellipsoid(radii_s, V_lab)
            S_lab_obs = 0.5 * (S_lab_obs + S_lab_obs.T)

            y_p = self._vech6(S_lab_obs)
            A_p = self._model_design_lab(
                two_theta,
                peak.getAzimuthal(),
                lamda,
                R,
                lambda_0=lambda_0_prior,
                p=p_prior if p_prior is not None else 1.0,
            )

            Q = self._Q_magnitude(two_theta, lamda)

            if not (np.isfinite(Q) and Q > 0):
                n_bad_Q += 1
                continue

            if weighting == "sn_over_q2":
                w = sig_noise / Q**2
            elif weighting == "sn":
                w = sig_noise
            elif weighting == "none":
                w = 1.0
            else:
                raise ValueError(f"Unknown weighting: {weighting!r}")

            if not (np.all(np.isfinite(y_p)) and np.all(np.isfinite(A_p))):
                n_nonfinite_row += 1
                continue

            A_blocks.append(w * A_p)
            y_blocks.append(w * y_p)
            lamda_blocks.append(lamda)
            used.append(i)

        sig_noise_str = (
            "min={:.2f} median={:.2f} max={:.2f}".format(
                np.min(sig_noise_seen),
                np.median(sig_noise_seen),
                np.max(sig_noise_seen),
            )
            if sig_noise_seen
            else "n/a (no peak had a valid shape at all)"
        )
        print(
            "ResolutionEllipsoid.fit: {} used of {} peaks (hard-excluded "
            "{}, sig_noise_cut={:.1f} excluded {}, bad-shape excluded {}, "
            "bad-Q excluded {}, nonfinite-row excluded {}); sig_noise "
            "among peaks with a valid shape: {}".format(
                len(used),
                ws.getNumberPeaks(),
                n_excluded,
                self.sig_noise_cut,
                n_low_sig_noise,
                n_bad_shape,
                n_bad_Q,
                n_nonfinite_row,
                sig_noise_str,
            )
        )

        if not A_blocks:
            return

        A = np.vstack(A_blocks)
        y = np.concatenate(y_blocks)
        # Each peak contributes 6 rows (vech6), so its wavelength must
        # be repeated 6x to line up with A/y's rows.
        lamda_arr = np.repeat(lamda_blocks, 6)

        n_instrumental = 5
        instrumental_scale = None
        lambda_0 = lambda_0_prior
        p = p_prior

        if fixed_instrumental is not None:
            x_prior = (
                np.array(
                    [
                        fixed_instrumental[k]
                        for k in (
                            "sigma_i",
                            "sigma_gamma_f",
                            "sigma_nu_f",
                            "sigma_dl_mod",
                        )
                    ]
                    # sigma_dl_mod_lam defaults to 0 -- older priors
                    # (e.g. beamline DivergenceParams) predate it.
                    + [fixed_instrumental.get("sigma_dl_mod_lam", 0.0)],
                    dtype=float,
                )
                ** 2
            )

            A_scale = (A[:, :n_instrumental] @ x_prior)[:, None]
            A_fit = np.column_stack([A_scale, A[:, n_instrumental:]])

            x_fit, residual_norm, robust_weights, _ = self.robust_nnls(
                A_fit, y
            )

            instrumental_scale = x_fit[0]
            x = np.concatenate([instrumental_scale * x_prior, x_fit[1:]])
            x_cov = None
        else:
            (
                x,
                lambda_0,
                p,
                residual_norm,
                robust_weights,
                lsq_result,
            ) = self.robust_nnls_saturation(A, y, lamda_arr, tanh_cols=(0,))

            # lsq_result's last two parameters are lambda_0 then p, not
            # part of x -- keep their own uncertainties separately and
            # drop them from the sigma-parameter covariance/correlation.
            x_cov_ext = self._parameter_covariance(lsq_result)
            if x_cov_ext is not None:
                x_cov = x_cov_ext[:-2, :-2]
                lambda_0_stderr = np.sqrt(max(x_cov_ext[-2, -2], 0.0))
                p_stderr = np.sqrt(max(x_cov_ext[-1, -1], 0.0))
            else:
                x_cov = None
                lambda_0_stderr = None
                p_stderr = None

        x_corr = self._parameter_correlation(x_cov)

        self.model = {
            **self._label_variance_parameters(x.ravel()),
            **self._label_variance_uncertainties(x.ravel(), x_cov),
            "variance_parameters": x,
            "variance_covariance": x_cov,
            "variance_correlation": x_corr,
            "residual_norm": residual_norm,
            "used_peaks": used,
            # Huber weight per used peak, same order as "used_peaks";
            # exactly 1.0 for peaks within the robust threshold (see
            # robust_nnls), <1.0 for peaks it down-weighted as outliers.
            "robust_weights": robust_weights,
            "instrumental_scale": instrumental_scale,
            # Saturation wavelength scale and growth-rate exponent for
            # sigma_i (lambda_0=None means the saturation scaling isn't
            # in effect -- i.e. sigma_i is a plain constant).
            "lambda_0": lambda_0,
            "p": p,
        }
        if fixed_instrumental is None:
            self.model["lambda_0_stderr"] = lambda_0_stderr
            self.model["p_stderr"] = p_stderr
        return self.model

    def fit_iterative(
        self,
        n_iter=5,
        clip_sigma=5.0,
        weighting="sn_over_q2",
        fixed_instrumental=None,
    ):
        """
        Repeatedly fit(), then hard-exclude peaks whose observed shape
        disagrees with the just-fitted model by more than `clip_sigma`
        robust-scaled residuals, and refit without them. Converges
        when a pass finds no new peaks to exclude (or after `n_iter`
        passes).

        Unlike the Huber down-weighting already inside fit() (bounded
        but nonzero influence for every peak, every pass), this drops
        flagged peaks entirely and for good -- useful when a genuine
        trend in the data is being obscured by a persistent minority
        of badly-integrated peaks that Huber alone doesn't fully
        suppress.

        Parameters
        ----------
        n_iter : int, optional
            Maximum number of fit-then-clip passes.
        clip_sigma : float, optional
            Peaks whose largest-axis relative residual
            (|obs_r - pred_r| / pred_r, max over the three
            Stoica-Wilkinson axes) is more than `clip_sigma` robust
            (MAD-based) scaled deviations above the median are
            excluded on the next pass. Large by default (5) -- this
            is a coarse "drop the badly-integrated tail", not a
            general-purpose robust estimator (fit() already does
            that internally).
        weighting, fixed_instrumental :
            Passed through to every fit() call.

        Returns
        -------
        model : dict
            Same shape as fit()'s return, from the final pass.
        excluded : set of int
            Peak indices excluded by the final pass.

        """
        exclude = set()

        for iteration in range(n_iter):
            model = self.fit(
                fixed_instrumental=fixed_instrumental,
                weighting=weighting,
                exclude=exclude,
            )
            if model is None:
                break

            rows = self.diagnostics()
            if not rows:
                break

            idx = np.array([r["i"] for r in rows])
            resid = np.array(
                [
                    max(
                        abs(r["err_r0"]) / max(r["pred_r0"], 1e-12),
                        abs(r["err_r1"]) / max(r["pred_r1"], 1e-12),
                        abs(r["err_r2"]) / max(r["pred_r2"], 1e-12),
                    )
                    for r in rows
                ]
            )

            median = np.median(resid)
            mad = np.median(np.abs(resid - median))
            scale = 1.4826 * mad + 1e-12
            z = (resid - median) / scale

            newly_bad = set(idx[z > clip_sigma].tolist())

            print(
                "ResolutionEllipsoid.fit_iterative: pass {} flagged {} "
                "new outlier peaks (of {} fit, {} previously "
                "excluded)".format(
                    iteration + 1,
                    len(newly_bad - exclude),
                    len(rows),
                    len(exclude),
                )
            )

            if newly_bad <= exclude:
                break

            exclude |= newly_bad

        return self.model, exclude

    def _label_variance_parameters(self, x):
        sq = lambda v: np.sqrt(max(v, 0.0))
        base = {
            "sigma_i": sq(x[0]),
            "sigma_gamma_f": sq(x[1]),
            "sigma_nu_f": sq(x[2]),
            "sigma_dl_mod": sq(x[3]),
            "sigma_dl_mod_lam": sq(x[4]),
        }
        if self.mosaic == "isotropic":
            base["sigma_mosaic"] = sq(x[5])
        elif self.mosaic == "diagonal":
            base["sigma_mosaic_0"] = sq(x[5])
            base["sigma_mosaic_1"] = sq(x[6])
            base["sigma_mosaic_2"] = sq(x[7])
        else:  # full
            labels = ["00", "11", "22", "01", "02", "12"]
            for k, lab in enumerate(labels):
                base[f"sigma_mosaic_{lab}"] = sq(x[5 + k])
        return base

    def set_variance_parameters(self, sigmas):
        """
        Initialize the model directly from known/prior sigma parameters,
        bypassing fit() -- e.g. instrument-characteristic divergence
        parameters (`config.instruments.beamlines[...]["DivergenceParams"]`)
        used to seed peak shapes before per-run peaks exist to fit an
        instrument-specific model against.

        Parameters
        ----------
        sigmas : dict
            Same keys `_label_variance_parameters` returns for the
            current `self.mosaic` model (e.g. for "isotropic":
            sigma_i, sigma_gamma_f, sigma_nu_f, sigma_dl_mod,
            sigma_mosaic).

        Returns
        -------
        model : dict
            Same shape `fit()` sets on `self.model`, minus the fit-only
            keys "residual_norm", "used_peaks", "robust_weights".

        """
        if (
            "sigma_i" not in sigmas
            and "sigma_gamma_i" in sigmas
            and "sigma_nu_i" in sigmas
        ):
            # combine an older separate gamma_i/nu_i prior into the
            # single isotropic sigma_i term (quadrature mean) -- they
            # were found to be too correlated to identify separately.
            sigmas = {
                **sigmas,
                "sigma_i": np.sqrt(
                    0.5
                    * (
                        sigmas["sigma_gamma_i"] ** 2
                        + sigmas["sigma_nu_i"] ** 2
                    )
                ),
            }

        if self.mosaic == "full" and "sigma_mosaic" in sigmas:
            # expand an isotropic prior (sigma_mosaic) into the full
            # tensor: same sigma on the three axis-aligned directions,
            # zero on the off-diagonal directions -- reproduces the
            # isotropic model exactly (M = sigma_mosaic^2 * I).
            s = sigmas["sigma_mosaic"]
            sigmas = {
                **sigmas,
                "sigma_mosaic_00": s,
                "sigma_mosaic_11": s,
                "sigma_mosaic_22": s,
                "sigma_mosaic_01": 0.0,
                "sigma_mosaic_02": 0.0,
                "sigma_mosaic_12": 0.0,
            }

        if self.mosaic == "isotropic":
            keys = [
                "sigma_i",
                "sigma_gamma_f",
                "sigma_nu_f",
                "sigma_dl_mod",
                "sigma_dl_mod_lam",
                "sigma_mosaic",
            ]
        elif self.mosaic == "diagonal":
            keys = [
                "sigma_i",
                "sigma_gamma_f",
                "sigma_nu_f",
                "sigma_dl_mod",
                "sigma_dl_mod_lam",
                "sigma_mosaic_0",
                "sigma_mosaic_1",
                "sigma_mosaic_2",
            ]
        else:  # full
            keys = [
                "sigma_i",
                "sigma_gamma_f",
                "sigma_nu_f",
                "sigma_dl_mod",
                "sigma_dl_mod_lam",
            ] + [
                f"sigma_mosaic_{lab}"
                for lab in ("00", "11", "22", "01", "02", "12")
            ]

        # sigma_dl_mod_lam defaults to 0 -- older priors predate this
        # term and don't have it.
        sigmas = {**{"sigma_dl_mod_lam": 0.0}, **sigmas}

        x = np.array([sigmas[k] for k in keys], dtype=float) ** 2

        self.model = {k: sigmas[k] for k in keys}
        self.model["variance_parameters"] = x
        # Saturation wavelength scale and growth-rate exponent for
        # sigma_i (see _model_design_lab); lambda_0 is None if this
        # prior predates that model. p defaults to 1.0 for priors that
        # predate the power-law saturation model.
        self.model["lambda_0"] = sigmas.get("lambda_0")
        self.model["p"] = sigmas.get("p", 1.0)

        return self.model

    def set_variance_parameters_deg(self, sigmas_deg):
        """
        Same as `set_variance_parameters`, but with the angular sigmas
        given in degrees instead of radians. "sigma_dl_mod"/
        "sigma_dl_mod_lam" are dimensionless (sigma_lambda/lambda, see
        `_model_design_lab`) rather than an angle, so they are passed
        through unchanged.

        Parameters
        ----------
        sigmas_deg : dict
            Same keys as `set_variance_parameters`, with every key but
            "sigma_dl_mod"/"sigma_dl_mod_lam"/"lambda_0" in degrees.
            "lambda_0" (Angstrom) is passed through unchanged too, if
            present -- it isn't an angle at all.

        Returns
        -------
        model : dict
            Same as `set_variance_parameters`.

        """
        pass_through = {"sigma_dl_mod", "sigma_dl_mod_lam", "lambda_0", "p"}
        sigmas = {
            key: (value if key in pass_through else np.radians(value))
            for key, value in sigmas_deg.items()
        }
        return self.set_variance_parameters(sigmas)

    def renumber_by_size(self, n=None):
        """
        Renumber peak runNumbers 1..n ordered by model ellipsoid size.

        Each peak's predicted largest principal radius is computed from the
        resolution model.  Peaks are sorted from smallest to largest and
        divided into n equal-count bins; all peaks in the same bin get the
        same runNumber (1 = smallest, n = largest).  When n is None each
        peak gets its own unique rank (equivalent to n = n_peaks).  Call
        after fit() (and optionally apply()).

        Parameters
        ----------
        n : int or None
            Number of size bins.  Defaults to one bin per peak.

        Returns
        -------
        radii_bins : ndarray, shape (n_bins,)
            Mean largest radius of the peaks assigned to each bin,
            ordered from smallest (bin 1) to largest (bin n).
        """
        if self.model is None:
            raise RuntimeError("Call fit() before renumber_by_size().")

        ws = mtd[self.peaks_ws]
        n_peaks = ws.getNumberPeaks()

        max_radii = np.empty(n_peaks)
        for i in range(n_peaks):
            peak = ws.getPeak(i)
            S_lab = self._predict_S_lab(peak)
            radii, _ = self._ellipsoid_from_S(S_lab)
            max_radii[i] = float(np.max(radii))

        n_bins = n_peaks if (n is None) else int(n)
        n_bins = max(1, min(n_bins, n_peaks))

        order = np.argsort(max_radii)
        sorted_radii = max_radii[order]

        # Assign each sorted position to a bin 1..n_bins (equal-count split)
        bin_number = np.empty(n_peaks, dtype=int)
        bin_indices = [[] for _ in range(n_bins)]
        for rank, peak_idx in enumerate(order):
            b = rank * n_bins // n_peaks
            bin_number[peak_idx] = b + 1
            bin_indices[b].append(rank)

        for i in range(n_peaks):
            ws.getPeak(i).setRunNumber(int(bin_number[i]))

        # Representative radius for each bin: mean of the sorted radii in that bin
        radii_bins = np.array(
            [sorted_radii[idx].mean() for idx in bin_indices]
        )
        return radii_bins

    def apply(self):
        if self.model is None:
            raise RuntimeError("Call fit() before apply().")

        lo, hi = self.scale_bounds
        ws = mtd[self.peaks_ws]

        for i, peak in enumerate(ws):
            R = peak.getGoniometerMatrix()

            S_lab = self._predict_S_lab(peak)
            radii, V_lab = self._ellipsoid_from_S(S_lab)

            self._set_peak_shape(ws, i, radii, V_lab, R=R)

    def predict_lab_S(self, peak_index):
        ws = mtd[self.peaks_ws]
        peak = ws.getPeak(peak_index)
        S_lab = self._predict_S_lab(peak)
        return 0.5 * (S_lab + S_lab.T)

    def predict_sample_S(self, peak_index):
        S_lab = self.predict_lab_S(peak_index)
        ws = mtd[self.peaks_ws]
        peak = ws.getPeak(peak_index)
        R = peak.getGoniometerMatrix()
        S_sam = R.T @ S_lab @ R
        return 0.5 * (S_sam + S_sam.T)

    def predict_roi_radii(self, r_cut, margin=2.0, min_frac=0.3):
        """
        Per-peak box half-width for a second `extract_peak_counts` pass,
        sized from this model's predicted ellipsoid instead of the one
        global `r_cut` every peak used on the first pass.

        A single global `r_cut` is simultaneously too large for compact/
        weak peaks (their box is mostly background, biasing the direct
        centroid/covariance moment low via `estimate_peak_shapes`'s
        SNR shrinkage) and potentially too small for genuinely broad
        ones. Resizing each peak's box from the model's own prediction
        targets both at once. Call after `fit()`.

        Parameters
        ----------
        r_cut : float
            Original (first-pass) box half-width -- also the upper
            bound returned here, since it's presumably sized to safely
            avoid absorbing neighboring peaks.
        margin : float
            Multiple of the predicted largest containment radius used
            as the new box half-width, leaving room around the peak
            for `estimate_peak_shapes`'s per-box background estimate.
        min_frac : float
            Lower bound on the returned radius, as a fraction of
            `r_cut` -- guards against a pathologically tight box where
            the model predicts a near-zero radius.

        Returns
        -------
        radii : ndarray, shape (n_peaks,)
            Per-peak box half-width, in the same order as the peaks
            workspace.

        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_roi_radii().")

        ws = mtd[self.peaks_ws]
        n_peaks = ws.getNumberPeaks()

        radii = np.full(n_peaks, float(r_cut))

        for i in range(n_peaks):
            S_sample = self.predict_sample_S(i)
            r, _ = self._ellipsoid_from_S(S_sample)
            radii[i] = np.clip(margin * r.max(), min_frac * r_cut, r_cut)

        return radii

    def build_design_matrices(self, peak_indices):
        """
        Return per-peak design matrices for the nonlinear optimizer.

        Parameters
        ----------
        peak_indices : list of int

        Returns
        -------
        dict : {peak_index: (6, n_params) ndarray}
            Maps each index to the matrix A such that vech6(S_lab) ≈ A @ x.
        """
        ws = mtd[self.peaks_ws]
        result = {}
        for i in peak_indices:
            peak = ws.getPeak(i)
            result[i] = self._model_design_lab(
                peak.getScattering(),
                peak.getAzimuthal(),
                peak.getWavelength(),
                peak.getGoniometerMatrix(),
                lambda_0=self.model.get("lambda_0") if self.model else None,
                p=self.model.get("p", 1.0) if self.model else 1.0,
            )
        return result

    def estimate_prior_sigmas(self):
        """
        Estimate prior width for S matrix and centroid parameters.

        Uses the peak shapes stored in the workspace (set by apply()) against
        the model predictions to compute RMS residuals for both the covariance
        and centroid.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before estimate_prior_sigmas().")

        ws = mtd[self.peaks_ws]
        S_obs_vecs = []
        offset_vecs = []

        moment_S_obs = {}

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)
            R = peak.getGoniometerMatrix()
            Q_mag = np.linalg.norm(peak.getQLabFrame())
            if Q_mag == 0:
                continue

            T = self._stoica_wilkinson_transform_from_peak(peak)

            radii, V_s, frame = self._get_peak_shape(ws, i)
            V_lab = self._sample_axes_to_lab(R, V_s) if frame == "s" else V_s
            S_lab = self._S_from_ellipsoid(radii, V_lab)

            S_lab = 0.5 * (S_lab + S_lab.T)
            S_w = T @ S_lab @ T.T
            moment_S_obs[i] = S_w  # store actual obs for diagnostics

            # prior_S_sigma from residual (obs − model), not raw magnitude
            S_w_pred = T @ self._predict_S_lab(peak) @ T.T
            S_w_resid = S_w - S_w_pred
            S_obs_vecs.append(
                np.array(
                    [
                        S_w_resid[0, 0],
                        S_w_resid[1, 1],
                        S_w_resid[2, 2],
                        S_w_resid[1, 2],
                        S_w_resid[0, 2],
                        S_w_resid[0, 1],
                    ]
                )
                / Q_mag**2
            )

        self._moment_S_obs = moment_S_obs

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)
            R = peak.getGoniometerMatrix()
            Q_mag = np.linalg.norm(np.array(peak.getQLabFrame()))
            if Q_mag == 0:
                continue
            T = self._stoica_wilkinson_transform_from_peak(peak)
            offset_s = self._get_peak_offset(ws, i)
            offset_vecs.append(T @ (R @ offset_s) / Q_mag)

        S_obs_vecs = np.array(S_obs_vecs)  # (n_peaks, 6)
        offset_vecs = np.array(offset_vecs)  # (n_peaks, 3)

        self.prior_ellipsoid_sigma = np.sqrt(np.mean(S_obs_vecs**2, axis=0))
        self.prior_center_sigma = np.sqrt(np.mean(offset_vecs**2, axis=0))

        self._offset_vecs = offset_vecs

        return self.prior_center_sigma, self.prior_ellipsoid_sigma

    def diagnostics(self):
        if self.model is None:
            raise RuntimeError("Call fit() before diagnostics().")

        ws = mtd[self.peaks_ws]
        rows = []

        moment_S_obs = getattr(self, "_moment_S_obs", {})

        # Peaks robust_nnls down-weighted (w < 1) when fitting the model
        # -- same peaks, same order as "used_peaks" -- flagged here so
        # the plot can mark them distinctly.
        robust_weights = self.model.get("robust_weights")
        outlier_by_peak = (
            {
                i: w < 0.999
                for i, w in zip(self.model["used_peaks"], robust_weights)
            }
            if robust_weights is not None
            else {}
        )

        for i in self.model["used_peaks"]:
            peak = ws.getPeak(i)

            R = peak.getGoniometerMatrix()

            S_lab_pred = self._predict_S_lab(peak)

            T = self._stoica_wilkinson_transform_from_peak(peak)

            if i in moment_S_obs:
                S_w_obs = moment_S_obs[i]
            else:
                radii, V_sample, frame = self._get_peak_shape(ws, i)
                V_lab = (
                    self._sample_axes_to_lab(R, V_sample)
                    if frame == "s"
                    else V_sample
                )
                S_lab_obs = self._S_from_ellipsoid(radii, V_lab)
                S_lab_obs = 0.5 * (S_lab_obs + S_lab_obs.T)
                S_w_obs = T @ S_lab_obs @ T.T
            S_w_pred = T @ S_lab_pred @ T.T

            # Project translation offset along predicted ellipsoid axes
            offset_s = self._get_peak_offset(ws, i)
            offset_lab = R @ offset_s
            radii_pred, V_lab_pred = self._ellipsoid_from_S(S_lab_pred)
            projs = [
                float(np.dot(offset_lab, V_lab_pred[:, k])) for k in range(3)
            ]
            norms = [projs[k] / max(radii_pred[k], 1e-12) for k in range(3)]

            two_theta = peak.getScattering()
            phi = peak.getAzimuthal()

            kx_hat = np.sin(two_theta) * np.cos(phi)
            ky_hat = np.sin(two_theta) * np.sin(phi)
            kz_hat = np.cos(two_theta)

            gamma = np.arctan2(kx_hat, kz_hat)
            nu = np.arcsin(ky_hat)

            obs_r = np.sqrt(np.maximum([S_w_obs[k, k] for k in range(3)], 0.0))
            pred_r = np.sqrt(
                np.maximum([S_w_pred[k, k] for k in range(3)], 0.0)
            )

            rows.append(
                {
                    "i": i,
                    "gamma": gamma,
                    "nu": nu,
                    "lambda": peak.getWavelength(),
                    "signal_noise": peak.getIntensityOverSigma(),
                    "Q": peak.getQSampleFrame().norm(),
                    "outlier": outlier_by_peak.get(i, False),
                    "obs_x0": S_w_obs[0, 0],
                    "obs_x1": S_w_obs[1, 1],
                    "obs_x2": S_w_obs[2, 2],
                    "pred_x0": S_w_pred[0, 0],
                    "pred_x1": S_w_pred[1, 1],
                    "pred_x2": S_w_pred[2, 2],
                    "obs_r0": obs_r[0],
                    "obs_r1": obs_r[1],
                    "obs_r2": obs_r[2],
                    "pred_r0": pred_r[0],
                    "pred_r1": pred_r[1],
                    "pred_r2": pred_r[2],
                    "err_r0": obs_r[0] - pred_r[0],
                    "err_r1": obs_r[1] - pred_r[1],
                    "err_r2": obs_r[2] - pred_r[2],
                    "offset_x0": projs[0],
                    "offset_x1": projs[1],
                    "offset_x2": projs[2],
                    "offset_norm_x0": norms[0],
                    "offset_norm_x1": norms[1],
                    "offset_norm_x2": norms[2],
                }
            )

        return rows

    def plot_diagnostics(self, filename):
        _plot_resolution_diagnostics(self.diagnostics(), filename)

    def write_diagnostics_csv(self, filename):
        """
        Write the per-peak diagnostics table (Q, observed/predicted/error
        radii along the Stoica-Wilkinson axes, angles, S/N, outlier flag)
        to a CSV file. Call after fit().
        """
        rows = self.diagnostics()
        if not rows:
            raise RuntimeError("No peaks available to write diagnostics for.")

        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_resolution_parameters(self, filename):
        if self.model is None:
            raise RuntimeError(
                "Call fit() before write_resolution_parameters()."
            )

        lines = [
            "mosaic model: {}\n".format(self.mosaic),
            "peak shape frame: {}\n".format(self.peak_shape_frame),
            "peaks used: {}\n".format(len(self.model["used_peaks"])),
            "residual norm: {:.4e}\n".format(self.model["residual_norm"]),
            "wavelength range: {:.4f} - {:.4f} Ang\n".format(
                self.lamda_min, self.lamda_max
            ),
            "scattering angle range: {:.4f} - {:.4f} deg\n".format(
                np.degrees(self.two_theta_min), np.degrees(self.two_theta_max)
            ),
        ]

        instrumental_scale = self.model.get("instrumental_scale")
        if instrumental_scale is not None:
            lines.append(
                "instrumental_scale: {:.6e} (dimensionless)\n".format(
                    instrumental_scale
                )
            )

        lambda_0 = self.model.get("lambda_0")
        if lambda_0 is not None:
            lambda_0_stderr = self.model.get("lambda_0_stderr")
            if lambda_0_stderr is None:
                lines.append("lambda_0: {:.6e} Ang\n".format(lambda_0))
            else:
                lines.append(
                    "lambda_0: {:.6e} +/- {:.6e} Ang\n".format(
                        lambda_0, lambda_0_stderr
                    )
                )

            p = self.model.get("p")
            p_stderr = self.model.get("p_stderr")
            if p_stderr is None:
                lines.append("p: {:.6e} (dimensionless)\n".format(p))
            else:
                lines.append(
                    "p: {:.6e} +/- {:.6e} (dimensionless)\n".format(
                        p, p_stderr
                    )
                )

        skip = {
            "variance_parameters",
            "variance_covariance",
            "variance_correlation",
            "residual_norm",
            "used_peaks",
            "robust_weights",
            "instrumental_scale",
            "lambda_0",
            "p",
        }

        for key, val in self.model.items():
            if key in skip or key.endswith("_stderr"):
                continue

            stderr = self.model.get(f"{key}_stderr")

            if key in ("sigma_dl_mod", "sigma_dl_mod_lam"):
                if stderr is None:
                    lines.append(
                        "{}: {:.6e} (dimensionless)\n".format(key, val)
                    )
                else:
                    lines.append(
                        "{}: {:.6e} +/- {:.6e} (dimensionless)\n".format(
                            key, val, stderr
                        )
                    )
            else:
                if stderr is None:
                    lines.append(
                        "{}: {:.6e} deg\n".format(key, np.degrees(val))
                    )
                else:
                    lines.append(
                        "{}: {:.6e} +/- {:.6e} deg\n".format(
                            key, np.degrees(val), np.degrees(stderr)
                        )
                    )

        x_corr = self.model.get("variance_correlation")
        if x_corr is not None:
            labels = list(
                self._label_variance_parameters(
                    self.model["variance_parameters"]
                ).keys()
            )
            col_w = max(10, max(len(lab) for lab in labels) + 2)
            row_w = max(len(lab) for lab in labels) + 2

            lines.append("\nparameter correlation matrix:\n")
            lines.append(
                "{:>{row_w}}".format("", row_w=row_w)
                + "".join(
                    "{:>{col_w}}".format(lab, col_w=col_w) for lab in labels
                )
                + "\n"
            )
            for i, lab in enumerate(labels):
                lines.append(
                    "{:>{row_w}}".format(lab, row_w=row_w)
                    + "".join(
                        "{:>{col_w}.3f}".format(x_corr[i, j], col_w=col_w)
                        for j in range(len(labels))
                    )
                    + "\n"
                )

        for line in lines:
            print(line)

        with open(filename, "w") as f:
            for line in lines:
                f.write(line)
