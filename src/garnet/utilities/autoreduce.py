import os
import re
import sys
import glob
import itertools

directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(directory)

sys.path.append(os.path.abspath(os.path.join(directory, "../..")))

import yaml

import numpy as np

import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot

from mantid import logger
from mantid.simpleapi import (
    LoadEventNexus,
    LoadNexus,
    ExtractMonitors,
    GroupDetectors,
    CompressEvents,
    MaskBTP,
    LoadInstrument,
    SaveNexus,
    PreprocessDetectorsToMD,
    CorelliCrossCorrelate,
    ClearUB,
    LoadIsawUB,
    SetGoniometer,
    ConvertUnits,
    CropWorkspaceForMDNorm,
    Rebin,
    MaskDetectorsIf,
    MaskDetectors,
    ExtractMask,
    ConvertToMD,
    RecalculateTrajectoriesExtents,
    MDNorm,
    LoadMD,
    SaveMD,
    PlusMD,
    CopySample,
    CreateSingleValuedWorkspace,
    AddSampleLog,
    DeleteWorkspace,
    mtd,
)

from garnet.plots.monitor import SlicePlot

try:
    from plot_publisher import publish_plot
except ImportError:
    webmonplot = False
else:
    webmonplot = True

from garnet.config.instruments import beamlines

instrument_dict = {
    beamlines[key]["InstrumentName"]: key for key in beamlines.keys()
}

AUTOLITE = "/SNS/software/scd/lite/"


class AutoReduce:
    # Display-grid oversample for the interactive monitor plots, decoupled
    # from the underlying MDNorm bin count (kept fine on disk for the
    # accumulated data/norm nxs files). At oversample=1.0 the display grid
    # tracks the source resolution 1:1, which for an 801x801 slice plus its
    # 3x customdata channel made the combined monitor-page payload exceed
    # livedata.sns.gov's upload size limit (HTTP 413). This keeps the
    # display grid to roughly 1/5 the source resolution per axis.
    MONITOR_OVERSAMPLE = 0.2

    def __init__(self, filename):
        self.filename = filename

        facility, self.inst, *_ = self.filename.split("/")[1:]

        LoadEventNexus(
            Filename=self.filename, OutputWorkspace="data", NumberOfBins=1
        )

        ExtractMonitors(
            InputWorkspace="data",
            DetectorWorkspace="data",
            MonitorWorkspace="monitors",
        )

        self.run = mtd["data"].getRunNumber()

        self.instrument = instrument_dict[self.inst]

        name = beamlines[self.instrument]["Name"]

        idf = glob.glob(
            os.path.join(AUTOLITE, "{}_Definition*.xml").format(name)
        )

        self.idf = idf[0] if len(idf) == 1 else None

        self.files = {}
        self.plot_html = ""

        self.cc = False

    def elastic(self, time_offset=14000):
        if self.instrument == "CORELLI":
            try:
                CorelliCrossCorrelate(
                    InputWorkspace="data",
                    OutputWorkspace="elastic",
                    TimingOffset=time_offset,
                )
            except RuntimeError as e:
                logger.warning("Cross Correlation failed: {}".format(e))
            else:
                output = self.filename.replace(
                    ".nxs.h5", "_elastic.nxs"
                ).replace("nexus", "shared/autoreduce")
                SaveNexus(InputWorkspace="elastic", Filename=output)
                self.cc = True
                self.compress("elastic")
                self.plot_instrument()

    def compress(self, ws):
        beamline = beamlines[self.instrument]

        c, r = [int(val) for val in beamline["Grouping"].split("x")]
        cols, rows = beamline["BankPixels"]
        mask_cols, mask_rows = beamline["MaskEdges"]

        PreprocessDetectorsToMD(
            InputWorkspace="data", OutputWorkspace="detectors"
        )

        det_map = np.asarray(mtd["detectors"].column(5)).reshape(
            -1, cols, rows
        )

        nb, nc, nr = det_map.shape

        gc = (np.arange(nc) // c).astype(np.int32)
        gr = (np.arange(nr) // r).astype(np.int32)

        ngc = (nc + c - 1) // c
        ngr = (nr + r - 1) // r

        group_id = (
            (np.arange(nb, dtype=np.int32)[:, None, None] * (ngc * ngr))
            + (gc[None, :, None] * ngr)
            + gr[None, None, :]
        ).ravel()

        det_ids = det_map.ravel()

        order = np.argsort(group_id, kind="stable")
        g_sorted = group_id[order]
        d_sorted = det_ids[order]

        starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
        ends = np.r_[starts[1:], g_sorted.size]

        parts = []
        for s, e in zip(starts, ends):
            parts.append("+".join(map(str, d_sorted[s:e])))

        detector_list = ",".join(parts)

        GroupDetectors(
            InputWorkspace="data",
            GroupingPattern=detector_list,
            OutputWorkspace="lite",
        )

        CompressEvents(InputWorkspace="lite", OutputWorkspace="lite")

        LoadInstrument(
            Workspace="lite",
            Filename=self.idf,
            RewriteSpectraMap="True",
        )

        cols //= c
        rows //= r
        mask_cols //= c
        mask_rows //= r

        inst = beamline["Name"]

        MaskBTP(
            Workspace="lite",
            Instrument=inst,
            Pixel="0-{},{}-{}".format(mask_rows, rows - mask_rows, rows),
        )
        MaskBTP(
            Workspace="lite",
            Instrument=inst,
            Tube="0-{},{}-{}".format(mask_cols, cols - mask_cols, cols),
        )

        mask_lost = beamline.get("MaskLost")

        if mask_lost is not None:
            for btp in mask_lost:
                bank, tube, pixel = btp
                tube = [val // c for val in tube]
                pixel = [val // r for val in pixel]
                MaskBTP(
                    Workspace="lite",
                    Instrument=inst,
                    Bank=bank,
                    Tube="{}-{}".format(*tube),
                    Pixel="{}-{}".format(*pixel),
                )
        banks = beamline["MaskBanks"]

        for bank in banks:
            MaskBTP(
                Workspace="lite",
                Instrument=inst,
                Bank=bank,
            )

        out = "_" + ws if ws != "data" else ""

        fname, *exts = self.filename.split(os.extsep)

        ext = ".lite." + ".".join(exts)

        output = fname.replace("nexus", "shared/autoreduce") + out + ext

        SaveNexus(
            InputWorkspace="lite",
            Filename=output,
        )

    def _sigma_clim(self, data, n_sigma=3.0):
        """
        +/-n_sigma clip color limits, mirroring NeuXtalViz's default
        "normal" calculate_clim (models/ub_tools.py): mean/std are
        computed excluding non-finite and near-zero values, then the
        +/-n_sigma spread is clipped to the actual (non-excluded) data
        range.
        """

        trans = np.asarray(data, dtype=float).copy()
        trans[~np.isfinite(trans)] = np.nan
        trans[np.isclose(trans, 0)] = np.nan

        if np.all(np.isnan(trans)):
            return 0.0, 1.0

        vmin, vmax = np.nanmin(trans), np.nanmax(trans)

        mu, sigma = np.nanmean(trans), np.nanstd(trans)
        spread = n_sigma * sigma
        cmin, cmax = mu - spread, mu + spread

        if np.isclose(cmin, cmax) or cmax < cmin:
            cmin, cmax = vmin, vmax

        zmin = max(cmin, vmin)
        zmax = min(cmax, vmax)

        if np.isclose(zmin, zmax):
            return 0.0, 1.0

        return float(zmin), float(zmax)

    def _heatmap_div(
        self,
        x,
        y,
        z,
        x_title="",
        y_title="",
        title="",
        aspect=None,
        zmin=None,
        zmax=None,
    ):
        """
        Build a Plotly heatmap HTML div, mirroring plot_publisher's
        plot_heatmap layout style exactly (so publish_plot's Plotly-div
        detection/version-injection still applies), without requiring
        that optional package to be installed just to render a plot.

        aspect : float, optional
            If given, locks the y-axis to this data-unit ratio relative
            to x (e.g. 1 for equal x/y scaling), like matplotlib's
            set_aspect.
        zmin, zmax : float, optional
            Color limits; if omitted, Plotly auto-ranges from the data.
        """

        axis_style = dict(
            zeroline=False,
            exponentformat="power",
            showexponent="all",
            showgrid=True,
            showline=True,
            mirror="all",
            ticks="inside",
        )

        yaxis = dict(title=y_title, **axis_style)
        if aspect is not None:
            yaxis.update(scaleanchor="x", scaleratio=aspect)

        layout = go.Layout(
            showlegend=False,
            autosize=True,
            width=600,
            height=500,
            margin=dict(t=40, b=40, l=80, r=40),
            hovermode="closest",
            bargap=0,
            xaxis=dict(title=x_title, **axis_style),
            yaxis=yaxis,
            title=title,
        )

        fig = go.Figure(
            data=[
                go.Heatmap(
                    z=z, x=x, y=y, colorscale="Jet", zmin=zmin, zmax=zmax
                )
            ],
            layout=layout,
        )

        return plotly_plot(
            fig, output_type="div", include_plotlyjs=False, show_link=False
        )

    def plot_instrument(self):
        """
        Bin the detector-space (gamma, nu) view into a 2D histogram
        (mean intensity per bin, mirroring NeuXtalViz's
        ub_tools.py calculate_instrument_view), and render it as an
        interactive Plotly heatmap for the monitor page.
        """

        PreprocessDetectorsToMD(
            InputWorkspace="lite", OutputWorkspace="detectors"
        )
        tt = np.array(mtd["detectors"].column(2))
        az = np.array(mtd["detectors"].column(3))

        counts = mtd["lite"].extractY().ravel()

        kf_x = np.sin(tt) * np.cos(az)
        kf_y = np.sin(tt) * np.sin(az)
        kf_z = np.cos(tt)

        nu = np.rad2deg(np.arcsin(kf_y))
        gamma = np.rad2deg(np.arctan2(kf_x, kf_z))

        n_bins = 400
        xedges = np.linspace(gamma.min(), gamma.max(), n_bins + 1)
        yedges = np.linspace(nu.min(), nu.max(), n_bins + 1)

        sum_I, xedges, yedges = np.histogram2d(
            gamma, nu, bins=[xedges, yedges], weights=counts
        )
        count_map, _, _ = np.histogram2d(gamma, nu, bins=[xedges, yedges])

        with np.errstate(invalid="ignore"):
            img = sum_I / count_map
        img[count_map == 0] = np.nan

        x = 0.5 * (xedges[1:] + xedges[:-1])
        y = 0.5 * (yedges[1:] + yedges[:-1])

        out = "_elastic" if self.cc else ""
        run_title = os.path.basename(self.filename.replace(".nxs.h5", out))

        zmin, zmax = self._sigma_clim(img)

        div = self._heatmap_div(
            x,
            y,
            img.T,
            x_title="γ [°]",
            y_title="ν [°]",
            zmin=zmin,
            zmax=zmax,
            title=run_title,
            aspect=1,
        )

        self.plot_html += "<div>{}</div>\n".format(div)

    def find_autoreduce_config(self):
        """
        Locate the most recently modified autoreduce yaml config, if any.

        Requires UBFile, VanadiumFile, and FluxFile keys; missing file or
        keys means the slice preview workflow is skipped entirely.
        """

        self.autoreduce_dir = os.path.dirname(
            self.filename.replace("nexus", "shared/autoreduce")
        )

        candidates = glob.glob(os.path.join(self.autoreduce_dir, "*.yaml"))

        if not candidates:
            self.slice_config = None
            return False

        newest = max(candidates, key=os.path.getmtime)

        with open(newest) as f:
            config = yaml.safe_load(f)

        required = ("UBFile", "VanadiumFile", "FluxFile")

        if not config or not all(config.get(key) for key in required):
            self.slice_config = None
            return False

        self.slice_config = config

        return True

    def _title_group_key(self, title):
        """
        Group a run title into a key, collapsing a trailing enumeration.

        Mirrors garnet.utilities.ipts.Model._title_group_key.
        """

        text = str(title).strip()
        match = re.match(r"^(.*?)([_\-\s]?)(\d+)$", text)
        if match is None:
            return text

        base, sep, _ = match.groups()
        if base == "":
            return text

        return "{}{}*".format(base, sep)

    def _is_axis_aligned(self, UB, tol_deg=1.0):
        """
        Check whether UB's reciprocal-lattice vectors are pairwise
        orthogonal in Cartesian (lab) space, within tol_deg of 90 degrees.
        """

        UB = np.asarray(UB, dtype=float)
        cols = [UB[:, i] / np.linalg.norm(UB[:, i]) for i in range(3)]

        tol = np.sin(np.deg2rad(tol_deg))

        pairs = [(0, 1), (0, 2), (1, 2)]

        return all(abs(np.dot(cols[i], cols[j])) < tol for i, j in pairs)

    def _candidate_projections(self, UB, aligned):
        """
        Default projection bases: always hk0/h0l/0kl, plus one "main
        equatorial" zone when the lattice isn't axis-aligned (4 total),
        picked the same way application.py's FormModel.autoproj scores
        its candidates: whichever thin/integrated axis (v2), transformed
        into Cartesian lab space by UB, best aligns with the vertical lab
        direction -- i.e. the zone whose in-plane (v0, v1) axes span the
        horizontal/equatorial scattering plane.

        Each entry is (name, [v0, v1, v2]) where v2 is the thin/integrated
        axis and v0, v1 are the wide in-plane axes.
        """

        base = [
            ("hk0", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            ("h0l", [[1, 0, 0], [0, 0, 1], [0, 1, 0]]),
            ("0kl", [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
        ]

        if aligned:
            return base

        candidates = [
            ("hk_plus", [[1, 1, 0], [0, 0, 1], [-1, 1, 0]]),
            ("hk_minus", [[-1, 1, 0], [0, 0, 1], [1, 1, 0]]),
            ("hl_plus", [[1, 0, 1], [0, 1, 0], [-1, 0, 1]]),
            ("hl_minus", [[-1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            ("kl_plus", [[0, 1, 1], [1, 0, 0], [0, -1, 1]]),
            ("kl_minus", [[0, -1, 1], [1, 0, 0], [0, 1, 1]]),
        ]

        target_axis = np.array([0.0, 1.0, 0.0])

        best_W, best_score = None, -np.inf

        for _, W in candidates:
            thin_cart = UB @ np.asarray(W[2], dtype=float)
            thin_cart /= np.linalg.norm(thin_cart)
            score = abs(np.dot(thin_cart, target_axis))

            if score > best_score:
                best_W, best_score = W, score

        return base + [("equatorial", best_W)]

    def _projection_extents(self, UB, projections, d_min):
        """
        Wide in-plane extents from d_min (cube-corner-through-inv(UB@W),
        same technique as application.py's FormModel.autolim), fixed
        +/-0.1 integration on the thin (third) axis. Bins are fixed at
        NxNx1.
        """

        UB = np.asarray(UB, dtype=float)
        W = np.column_stack(projections)

        s_max = 1.0 / float(d_min) / np.sqrt(3.0)

        T = np.linalg.inv(UB @ W)

        corners = np.array(
            list(
                itertools.product(
                    [-s_max, s_max], [-s_max, s_max], [-s_max, s_max]
                )
            )
        )

        limits = np.max(np.abs((T @ corners.T).T), axis=0)

        extents = [
            [-limits[0], limits[0]],
            [-limits[1], limits[1]],
            [-0.1, 0.1],
        ]
        bins = [801, 801, 1]

        return extents, bins

    def _load_solid_angle(self, vanadium_file):
        """
        Load the solid-angle workspace directly: it already matches the
        raw data's per-pixel geometry, no rebinning/matching needed.
        Extract a mask from its non-finite pixels for apply_mask-style
        use (see slice_workflow).
        """

        if mtd.doesExist("sa"):
            return

        LoadNexus(Filename=vanadium_file, OutputWorkspace="sa")

        MaskDetectorsIf(
            InputWorkspace="sa",
            Mode="SelectIf",
            Operator="NotFinite",
            OutputWorkspace="sa",
        )

        ExtractMask(
            InputWorkspace="sa",
            UngroupDetectors=True,
            OutputWorkspace="sa_mask",
        )

    def _load_flux(self, flux_file):
        """
        Load the flux workspace directly. Flux is defined per-bank (not
        per-pixel/spectrum) and MDNorm resolves each detector's flux by
        bank internally, so no expansion to the raw data's per-pixel
        spectra is needed.
        """

        if mtd.doesExist("flux"):
            return

        LoadNexus(Filename=flux_file, OutputWorkspace="flux")

        self.k_min = mtd["flux"].getXDimension().getMinimum()
        self.k_max = mtd["flux"].getXDimension().getMaximum()

    def _convert_to_Q_sample(self, d_min):
        """
        Convert the raw "data" workspace to a Q-sample MD event workspace
        ("raw_md"), bounded by a generous +/-2*pi/d_min box on each axis.
        """

        ConvertUnits(
            InputWorkspace="data", OutputWorkspace="data", Target="Momentum"
        )

        CropWorkspaceForMDNorm(
            InputWorkspace="data",
            XMin=self.k_min,
            XMax=self.k_max,
            OutputWorkspace="data",
        )

        Rebin(
            InputWorkspace="data",
            Params=[self.k_min, self.k_max, self.k_max],
            OutputWorkspace="data",
            PreserveEvents=True,
        )

        CompressEvents(
            InputWorkspace="data", OutputWorkspace="data", Tolerance=1e-2
        )

        PreprocessDetectorsToMD(
            InputWorkspace="data", OutputWorkspace="slice_detectors"
        )

        Q_max = 2 * np.pi / d_min

        ConvertToMD(
            InputWorkspace="data",
            QDimensions="Q3D",
            dEAnalysisMode="Elastic",
            Q3DFrames="Q_sample",
            QConversionScales="Q in A^-1",
            LorentzCorrection=False,
            MinValues=[-Q_max, -Q_max, -Q_max],
            MaxValues=[Q_max, Q_max, Q_max],
            OutputWorkspace="raw_md",
            PreprocDetectorsWS="slice_detectors",
            SplitInto=2,
            MaxRecursionDepth=10,
        )

        RecalculateTrajectoriesExtents(
            InputWorkspace="raw_md", OutputWorkspace="raw_md"
        )

    def _run_mdnorm(self, md, projections, extents, bins):
        """
        MDNorm a projection out of "raw_md", mirroring the exact call in
        LaueData.normalize_to_hkl (no background, no symmetry).
        """

        v0, v1, v2 = projections

        (Q0_min, Q0_max), (Q1_min, Q1_max), (Q2_min, Q2_max) = extents
        n0, n1, n2 = bins

        dQ0 = (Q0_max - Q0_min) / n0
        dQ1 = (Q1_max - Q1_min) / n1
        dQ2 = (Q2_max - Q2_min) / n2

        MDNorm(
            InputWorkspace=md,
            SolidAngleWorkspace="sa",
            FluxWorkspace="flux",
            QDimension0=v0,
            QDimension1=v1,
            QDimension2=v2,
            Dimension0Name="QDimension0",
            Dimension1Name="QDimension1",
            Dimension2Name="QDimension2",
            Dimension0Binning=[Q0_min, dQ0, Q0_max],
            Dimension1Binning=[Q1_min, dQ1, Q1_max],
            Dimension2Binning=[Q2_min, dQ2, Q2_max],
            OutputWorkspace=md + "_result",
            OutputDataWorkspace=md + "_data",
            OutputNormalizationWorkspace=md + "_norm",
        )

        return md + "_data", md + "_norm", md + "_result"

    def _attach_ub_w(self, ws, ub_file, projections):
        """
        Attach the OrientedLattice UB and a numeric "W_MATRIX" run property
        to ws, mirroring DataModel.add_UBW.
        """

        if not mtd.doesExist(ws):
            return

        CreateSingleValuedWorkspace(OutputWorkspace="ubw")

        LoadIsawUB(
            InputWorkspace="ubw",
            Filename=ub_file.replace("*", str(self.run)),
        )

        W = np.column_stack(projections)

        AddSampleLog(
            Workspace=ws,
            LogName="W_MATRIX",
            LogText=",".join(9 * ["{}"]).format(*W.flatten()),
            LogType="String",
        )

        run = mtd[ws].getExperimentInfo(0).run()
        run.addProperty("W_MATRIX", list(W.flatten() * 1.0), True)

        CopySample(
            InputWorkspace="ubw",
            OutputWorkspace=ws,
            CopyName=False,
            CopyMaterial=False,
            CopyEnvironment=False,
            CopyLattice=True,
            CopyOrientationOnly=False,
        )

    def _bin_center_axes(self, ws):
        dims = [mtd[ws].getDimension(i) for i in range(3)]

        axes = []
        for d in dims:
            width = (d.getMaximum() - d.getMinimum()) / d.getNBins()
            axes.append(
                d.getMinimum() + width * (np.arange(d.getNBins()) + 0.5)
            )

        titles = ["{} {}".format(d.name, d.getUnits()) for d in dims]

        return axes, titles

    def _plot_slice(self, data_ws, norm_ws, base):
        """
        Render a 2D slice (thin axis is always index 2) as an
        interactive Plotly heatmap (resampled onto a rectangular
        display grid, see garnet.plots.monitor.SlicePlot) and append
        it to the shared monitor-page HTML.

        Counts and normalization are resampled separately and divided
        by SlicePlot.make_slice, rather than dividing the MD workspaces
        (DivideMD) before resampling -- see garnet.plots.monitor for why.
        """

        data_signal = mtd[data_ws].getSignalArray().copy()
        norm_signal = mtd[norm_ws].getSignalArray().copy()

        ei = mtd[data_ws].getExperimentInfo(0)
        UB = ei.sample().getOrientedLattice().getUB()
        W = np.array(ei.run().getProperty("W_MATRIX").value).reshape(3, 3)

        axes, titles = self._bin_center_axes(data_ws)

        norm = np.zeros(3, dtype=int)
        norm[2] = 1

        plot = SlicePlot(UB, W)
        plot.calculate_transforms(
            axes, titles, norm, oversample=self.MONITOR_OVERSAMPLE
        )
        plot.make_slice(data_signal, norm_signal, 0.0)

        plot.fig.update_layout(
            title="{}: {}".format(base, plot.fig.layout.title.text)
        )

        div = plotly_plot(
            plot.fig,
            output_type="div",
            include_plotlyjs=False,
            show_link=False,
        )

        self.plot_html += "<div>{}</div>\n".format(div)

    def slice_workflow(self):
        """
        Compute default UB-driven HKL-projection slices, MDNorm-normalize
        against the configured vanadium solid-angle/flux, accumulate
        on-disk with any prior run of the same title, and plot/upload.
        """

        config = self.slice_config
        d_min = float(config.get("DMin", 0.7))

        title = mtd["data"].getTitle()
        title_key = self._title_group_key(title)
        safe_key = title_key.replace(" ", "").replace("/", "_")

        ClearUB(Workspace="data")
        LoadIsawUB(
            InputWorkspace="data",
            Filename=config["UBFile"].replace("*", str(self.run)),
        )
        SetGoniometer(Workspace="data", Goniometers="Universal")

        UB = mtd["data"].sample().getOrientedLattice().getUB()
        aligned = self._is_axis_aligned(UB)
        projections = self._candidate_projections(UB, aligned)

        self._load_solid_angle(config["VanadiumFile"])
        self._load_flux(config["FluxFile"])

        if mtd.doesExist("sa_mask"):
            MaskDetectors(Workspace="data", MaskedWorkspace="sa_mask")

        self._convert_to_Q_sample(d_min)

        for name, W in projections:
            extents, bins = self._projection_extents(UB, W, d_min)

            data_ws, norm_ws, _ = self._run_mdnorm("raw_md", W, extents, bins)

            base = "{}_{}_{}".format(self.instrument, safe_key, name)
            data_file = os.path.join(self.autoreduce_dir, base + "_data.nxs")
            norm_file = os.path.join(self.autoreduce_dir, base + "_norm.nxs")

            if os.path.exists(data_file) and os.path.exists(norm_file):
                LoadMD(Filename=data_file, OutputWorkspace="prev_data")
                LoadMD(Filename=norm_file, OutputWorkspace="prev_norm")

                PlusMD(
                    LHSWorkspace=data_ws,
                    RHSWorkspace="prev_data",
                    OutputWorkspace=data_ws,
                )
                PlusMD(
                    LHSWorkspace=norm_ws,
                    RHSWorkspace="prev_norm",
                    OutputWorkspace=norm_ws,
                )

                DeleteWorkspace(Workspace="prev_data")
                DeleteWorkspace(Workspace="prev_norm")

            self._attach_ub_w(data_ws, config["UBFile"], W)
            self._attach_ub_w(norm_ws, config["UBFile"], W)

            SaveMD(
                Filename=data_file,
                InputWorkspace=data_ws,
                SaveHistory=False,
                SaveInstrument=False,
                SaveSample=True,
                SaveLogs=False,
            )
            SaveMD(
                Filename=norm_file,
                InputWorkspace=norm_ws,
                SaveHistory=False,
                SaveInstrument=False,
                SaveSample=True,
                SaveLogs=False,
            )

            self._plot_slice(data_ws, norm_ws, base)

    def publish_plots(self):
        if webmonplot:
            self.files["file"] = self.plot_html
            request = publish_plot(self.inst, self.run, files=self.files)
            print(request)


if __name__ == "__main__":
    filename = sys.argv[1]
    ar = AutoReduce(filename)
    ar.compress("data")
    ar.plot_instrument()
    ar.elastic()
    if ar.find_autoreduce_config():
        ar.slice_workflow()
    ar.publish_plots()
