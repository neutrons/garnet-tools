import os
import sys
import tempfile
import traceback

import numpy as np
import itertools
import scipy.linalg

import pyvista as pv
from pyvistaqt import QtInteractor

os.environ["QT_API"] = "pyqt6"

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QPlainTextEdit,
    QSizePolicy,
    QMessageBox,
    QSplitter,
    QFrame,
)

from qtpy.QtGui import (
    QDoubleValidator,
    QIntValidator,
    QFont,
    QIcon,
    QPixmap,
    QPainter,
    QColor,
)
from qtpy.QtCore import Qt, QProcess, QElapsedTimer, QSettings

_local_cfg = os.path.join(
    tempfile.gettempdir(), os.environ.get("USER", "user"), "qt"
)
os.makedirs(_local_cfg, exist_ok=True)
QSettings.setPath(
    QSettings.Format.NativeFormat, QSettings.Scope.UserScope, _local_cfg
)

from qdarkstyle.light.palette import LightPalette
import qdarkstyle
import qtawesome as qta

from garnet._version import __version__

from garnet.config.instruments import beamlines
from garnet.reduction.plan import ReductionPlan
from garnet.reduction.crystallography import (
    space_point,
    point_laue,
    space_number,
)


def _compute_preview_transforms(UB, W):
    """Compute (P, T, S) projection/transform/scale from UB @ W via QR + Cholesky."""
    Bp = np.asarray(UB, dtype=float) @ np.column_stack(
        [np.asarray(row, dtype=float) for row in W]
    )
    _, R = scipy.linalg.qr(Bp)
    v = scipy.linalg.cholesky(R.T @ R, lower=False)
    s = np.linalg.norm(v, axis=0)
    T = v / s
    P = v / v[0, 0]
    S = np.linalg.norm(P, axis=0)
    return P, T, S


def _centering_mask(hkl, centering):
    h, k, l = hkl[:, 0], hkl[:, 1], hkl[:, 2]
    if centering == "I":
        return (h + k + l) % 2 == 0
    elif centering == "F":
        return ((h + k) % 2 == 0) & ((h + l) % 2 == 0) & ((k + l) % 2 == 0)
    elif centering == "A":
        return (k + l) % 2 == 0
    elif centering == "B":
        return (h + l) % 2 == 0
    elif centering == "C":
        return (h + k) % 2 == 0
    elif centering == "R":
        return (-h + k + l) % 3 == 0
    elif centering == "H":
        return (h - k) % 3 == 0
    return np.ones(len(hkl), dtype=bool)


class FormView(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()
        load_save_layout = QHBoxLayout()
        process_layout = QHBoxLayout()

        exp_tab = self.init_plan()

        setup_widget = QTabWidget(self)
        setup_widget.addTab(exp_tab, "Instrument/Experiment")

        layout.addWidget(setup_widget)

        norm_tab = self.norm_plan()
        param_tab = self.param_plan()
        int_tab = self.int_plan()

        self.plan_widget = QTabWidget(self)
        self.plan_widget.addTab(norm_tab, "Normalization")
        self.plan_widget.addTab(param_tab, "Parametrization")
        self.plan_widget.addTab(int_tab, "Integration")

        self._plotter_frame = QFrame(self)
        self.plotter = QtInteractor(self._plotter_frame)
        self.plotter.set_background("white")
        self.plotter.enable_parallel_projection()
        self._preview_T = None
        self._preview_UB = None

        self._reset_view_btn = QPushButton(
            qta.icon("fa6s.house"), "Isometric View", self
        )
        self._reset_view_btn.setToolTip("Reset to isometric view")
        self._reset_view_btn.clicked.connect(self._on_reset_view)

        self._camera_btn = QPushButton(
            qta.icon("fa6s.camera"), "Reset Camera", self
        )
        self._camera_btn.setToolTip("Reset camera (fit to scene)")
        self._camera_btn.clicked.connect(self._on_reset_camera)

        self._plot_btn = QPushButton(
            qta.icon("fa6s.rotate"), "Relot Preview", self
        )
        self._plot_btn.setToolTip("Refresh preview")

        self._parallel_box = QCheckBox("Parallel Projection", self)
        self._parallel_box.setChecked(True)
        self._parallel_box.setToolTip("Toggle parallel projection")
        self._parallel_box.stateChanged.connect(self._on_projection_changed)

        self._recip_box = QCheckBox("Reciprocal Lattice", self)
        self._recip_box.setChecked(True)
        self._recip_box.setToolTip(
            "Toggle reciprocal lattice compass labels (a*/b*/c* vs a/b/c)"
        )
        self._recip_box.stateChanged.connect(self._show_compass)

        # --- Group 1: camera controls (2×2) + plot button ---
        cam_widget = QWidget(self)
        cam_grid = QGridLayout(cam_widget)
        cam_grid.setContentsMargins(0, 0, 0, 0)
        cam_grid.setSpacing(2)
        cam_grid.addWidget(self._reset_view_btn, 0, 0)
        cam_grid.addWidget(self._camera_btn, 1, 0)
        cam_grid.addWidget(self._parallel_box, 0, 2)
        cam_grid.addWidget(self._recip_box, 1, 2)
        cam_grid.addWidget(self._plot_btn, 0, 1)

        def _vsep():
            s = QFrame(self)
            s.setFrameShape(QFrame.VLine)
            s.setFrameShadow(QFrame.Sunken)
            return s

        # --- Group 2: ±Qx/y/z Cartesian views (2×3) ---
        qxyz_widget = QWidget(self)
        qxyz_layout = QGridLayout(qxyz_widget)
        qxyz_layout.setContentsMargins(0, 0, 0, 0)
        qxyz_layout.setSpacing(2)
        for label, tip, slot, row, col in [
            (
                "+Qx",
                "View along +Qx",
                lambda: self._q_axis_view(0, +1, 1),
                0,
                0,
            ),
            (
                "+Qy",
                "View along +Qy",
                lambda: self._q_axis_view(1, +1, 2),
                0,
                1,
            ),
            (
                "+Qz",
                "View along +Qz",
                lambda: self._q_axis_view(2, +1, 1),
                0,
                2,
            ),
            (
                "-Qx",
                "View along -Qx",
                lambda: self._q_axis_view(0, -1, 1),
                1,
                0,
            ),
            (
                "-Qy",
                "View along -Qy",
                lambda: self._q_axis_view(1, -1, 2),
                1,
                1,
            ),
            (
                "-Qz",
                "View along -Qz",
                lambda: self._q_axis_view(2, -1, 1),
                1,
                2,
            ),
        ]:
            btn = QPushButton(label, qxyz_widget)
            btn.setToolTip(tip)
            btn.setFixedHeight(24)
            btn.setIcon(qta.icon("fa6s.right-long"))
            btn.clicked.connect(slot)
            qxyz_layout.addWidget(btn, row, col)

        # --- Group 3: crystal axis views (2×3, colored) ---
        crys_widget = QWidget(self)
        crys_layout = QGridLayout(crys_widget)
        crys_layout.setContentsMargins(0, 0, 0, 0)
        crys_layout.setSpacing(2)
        for label, tip, slot, color, row, col in [
            (
                "a*",
                "View along a*",
                lambda: self._axis_view(0, 1),
                QColor("red"),
                0,
                0,
            ),
            (
                "b*",
                "View along b*",
                lambda: self._axis_view(1, 2),
                QColor("green"),
                0,
                1,
            ),
            (
                "c*",
                "View along c*",
                lambda: self._axis_view(2, 0),
                QColor("blue"),
                0,
                2,
            ),
            (
                "a",
                "View along a",
                lambda: self._real_axis_view(0),
                QColor("red"),
                1,
                0,
            ),
            (
                "b",
                "View along b",
                lambda: self._real_axis_view(1),
                QColor("green"),
                1,
                1,
            ),
            (
                "c",
                "View along c",
                lambda: self._real_axis_view(2),
                QColor("blue"),
                1,
                2,
            ),
        ]:
            btn = QPushButton(label, crys_widget)
            btn.setToolTip(tip)
            btn.setFixedHeight(24)
            btn.setMaximumWidth(36)
            btn.setIcon(qta.icon("fa6s.right-long", color=color))
            btn.clicked.connect(slot)
            crys_layout.addWidget(btn, row, col)

        ctrl_bar = QHBoxLayout()
        ctrl_bar.setContentsMargins(4, 2, 4, 2)
        ctrl_bar.setSpacing(4)
        ctrl_bar.addWidget(cam_widget)
        ctrl_bar.addStretch(1)
        ctrl_bar.addWidget(_vsep())
        ctrl_bar.addWidget(qxyz_widget)
        ctrl_bar.addWidget(_vsep())
        ctrl_bar.addWidget(crys_widget)

        _pl = QVBoxLayout()
        _pl.setContentsMargins(0, 0, 0, 0)
        _pl.setSpacing(0)
        _pl.addLayout(ctrl_bar)
        _pl.addWidget(self.plotter.interactor, stretch=1)
        self._plotter_frame.setLayout(_pl)

        name_label = QLabel("Config File:")

        self.load_button = QPushButton("Load", self)
        self.save_button = QPushButton("Save", self)
        self.save_as_button = QPushButton("Save As", self)
        self.stop_button = QPushButton("Stop Process", self)
        self.stop_button.setIcon(qta.icon("fa6s.stop"))

        self.load_button.setIcon(qta.icon("fa6s.folder-open"))
        self.save_button.setIcon(qta.icon("fa6s.floppy-disk"))
        self.save_as_button.setIcon(qta.icon("fa6s.file-export"))

        load_save_layout.addWidget(name_label)
        load_save_layout.addWidget(self.output_line)
        load_save_layout.addWidget(self.load_button)
        load_save_layout.addWidget(self.save_button)
        load_save_layout.addWidget(self.save_as_button)

        layout.addLayout(load_save_layout)
        layout.addWidget(self.plan_widget)

        self.cpu_line = QLineEdit("1")
        self.cpu_line.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)

        validator = QIntValidator(1, 64, self)
        self.cpu_line.setValidator(validator)
        self.dev_box = QCheckBox("vDev", self)

        self.mem_label = QLabel("~? GB")
        self.mem_label.setToolTip(
            "Estimated memory: instrument workspaces + task\n"
            "Instrument: 6 ws × processes × banks × rows × cols × 8 B\n"
            "Norm/Param: processes × 4 arrays × bins × 8 B\n"
            "Integration: N_peaks(d_min, centering) × 21³ × 8 B"
        )

        process_layout.addStretch(1)
        process_layout.addWidget(QLabel("Processes:"))
        process_layout.addWidget(self.cpu_line)
        process_layout.addWidget(self.mem_label)
        process_layout.addWidget(self.dev_box)
        process_layout.addWidget(self.stop_button)

        layout.addLayout(process_layout)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)

        font = QFont("Courier New")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(10)
        self.output.setFont(font)

        layout.addWidget(self.output)

        layout.addStretch(1)

        form_widget = QWidget()
        form_widget.setLayout(layout)

        outer_splitter = QSplitter(Qt.Horizontal)
        outer_splitter.addWidget(form_widget)
        outer_splitter.addWidget(self._plotter_frame)
        outer_splitter.setStretchFactor(0, 2)
        outer_splitter.setStretchFactor(1, 3)

        outer_layout = QHBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(outer_splitter)
        self.setLayout(outer_layout)

        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        self.stop_button.clicked.connect(self.stop_process)

        self.cpu_line.editingFinished.connect(self.update_mem_estimate)
        self.instrument_combo.activated.connect(
            lambda *_: self.update_mem_estimate()
        )
        self.plan_widget.currentChanged.connect(
            lambda *_: self.update_mem_estimate()
        )
        self.centering_combo.activated.connect(
            lambda *_: self.update_mem_estimate()
        )
        self.min_d_line.editingFinished.connect(self.update_mem_estimate)
        for _line in [
            self.norm_bins_1_line,
            self.norm_bins_2_line,
            self.norm_bins_3_line,
            self.param_bins_1_line,
            self.param_bins_2_line,
            self.param_bins_3_line,
            self.param_bins_4_line,
        ]:
            _line.editingFinished.connect(self.update_mem_estimate)

        self.update_mem_estimate()

    def stop_process(self):
        self.process.terminate()
        if not self.process.waitForFinished(3000):
            self.process.kill()

    @staticmethod
    def _format_bytes(n):
        if n >= 1e12:
            return f"~{n / 1e12:.1f} TB"
        if n >= 1e9:
            return f"~{n / 1e9:.1f} GB"
        return f"~{n / 1e6:.1f} MB"

    @staticmethod
    def _estimate_n_peaks(d_min, centering):
        """Estimate predicted peak count from d_min and centering.

        Uses (4π/3)×(1/d_min)³×V_typical×f_centering with V_typical=500 Å³.
        """
        centering_frac = {
            "P": 1.0,
            "I": 0.5,
            "F": 0.25,
            "A": 0.5,
            "B": 0.5,
            "C": 0.5,
            "R": 1 / 3,
            "H": 2 / 3,
        }.get(centering, 1.0)
        V_typical = 500.0
        return int(
            (4 * np.pi / 3) * (1.0 / d_min) ** 3 * V_typical * centering_frac
        )

    def update_mem_estimate(self):
        instrument = self.get_instrument()
        bl = beamlines.get(instrument, {})
        rows, cols = bl.get("BankPixels", [256, 256])
        banks = bl.get("Banks", 1)
        processes = self.get_processes()

        inst_bytes = 12 * processes * banks * rows * cols * 8

        tab = self.get_plan_tab_index()
        if tab == 0:  # Normalization
            b1 = self.get_norm_bins_1() or 1
            b2 = self.get_norm_bins_2() or 1
            b3 = self.get_norm_bins_3() or 1
            task_bytes = processes * 4 * b1 * b2 * b3 * 8
        elif tab == 1:  # Parametrization
            b1 = self.get_param_bins_1() or 1
            b2 = self.get_param_bins_2() or 1
            b3 = self.get_param_bins_3() or 1
            b4 = self.get_param_bins_4() or 1
            task_bytes = processes * 4 * b1 * b2 * b3 * b4 * 8
        else:  # Integration
            d_min = self.get_min_d() or 0.7
            centering = self.get_centering() or "P"
            n_peaks = self._estimate_n_peaks(d_min, centering)
            task_bytes = n_peaks * (21**3) * 16

        self.mem_label.setText(self._format_bytes(inst_bytes + task_bytes))

    def int_plan(self):
        tab = QWidget()

        layout = QVBoxLayout()

        int_layout = QGridLayout()
        profile_layout = QHBoxLayout()

        dim_1_label = QLabel("1:")
        dim_2_label = QLabel("2:")
        dim_3_label = QLabel("3:")

        cell_label = QLabel("Cell")
        centering_label = QLabel("Centering")

        dh_label = QLabel("Δh")
        dk_label = QLabel("Δk")
        dl_label = QLabel("Δl")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.mod_11_line = QLineEdit("0.0")
        self.mod_12_line = QLineEdit("0.0")
        self.mod_13_line = QLineEdit("0.0")

        self.mod_21_line = QLineEdit("0.0")
        self.mod_22_line = QLineEdit("0.0")
        self.mod_23_line = QLineEdit("0.0")

        self.mod_31_line = QLineEdit("0.0")
        self.mod_32_line = QLineEdit("0.0")
        self.mod_33_line = QLineEdit("0.0")

        self.mod_11_line.setValidator(validator)
        self.mod_12_line.setValidator(validator)
        self.mod_13_line.setValidator(validator)

        self.mod_21_line.setValidator(validator)
        self.mod_22_line.setValidator(validator)
        self.mod_23_line.setValidator(validator)

        self.mod_31_line.setValidator(validator)
        self.mod_32_line.setValidator(validator)
        self.mod_33_line.setValidator(validator)

        self.mod_11_line.setEnabled(False)
        self.mod_12_line.setEnabled(False)
        self.mod_13_line.setEnabled(False)

        self.mod_21_line.setEnabled(False)
        self.mod_22_line.setEnabled(False)
        self.mod_23_line.setEnabled(False)

        self.mod_31_line.setEnabled(False)
        self.mod_32_line.setEnabled(False)
        self.mod_33_line.setEnabled(False)

        self.cell_combo = QComboBox(self)
        self.cell_combo.addItem("Triclinic")
        self.cell_combo.addItem("Monoclinic")
        self.cell_combo.addItem("Orthorhombic")
        self.cell_combo.addItem("Tetragonal")
        self.cell_combo.addItem("Rhombohedral")
        self.cell_combo.addItem("Hexagonal")
        self.cell_combo.addItem("Cubic")

        self.auto_scale_dropdown(self.cell_combo)

        self.centering_combo = QComboBox(self)
        self.centering_combo.addItem("P")
        self.centering_combo.addItem("I")
        self.centering_combo.addItem("F")
        self.centering_combo.addItem("R")
        self.centering_combo.addItem("A")
        self.centering_combo.addItem("B")
        self.centering_combo.addItem("C")
        self.centering_combo.addItem("H")

        self.auto_scale_dropdown(self.centering_combo)

        radius_label = QLabel("Radius:")
        radius_unit_label = QLabel("Å⁻¹")

        validator = QDoubleValidator(0.001, 1, 3, notation=notation)

        self.radius_line = QLineEdit("0.2")
        self.radius_line.setValidator(validator)

        min_d_label = QLabel("Min d-spacing:")
        d_unit_lael = QLabel("Å")

        self.satellite_box = QCheckBox("Satellite", self)
        self.satellite_box.setChecked(False)

        self.cross_box = QCheckBox("Cross Terms", self)
        self.cross_box.setChecked(False)
        self.cross_box.setEnabled(False)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.4, 100, 3, notation=notation)

        self.min_d_line = QLineEdit("0.7")
        self.min_d_line.setValidator(validator)

        self.min_sat_d_line = QLineEdit("1.0")
        self.min_sat_d_line.setValidator(validator)
        self.min_sat_d_line.setEnabled(False)

        self.max_order_line = QLineEdit("0")
        self.max_order_line.setEnabled(False)

        max_order_label = QLabel("Max Order")

        int_layout.addWidget(cell_label, 0, 0, Qt.AlignCenter)
        int_layout.addWidget(centering_label, 0, 1, Qt.AlignCenter)
        int_layout.addWidget(max_order_label, 0, 2, Qt.AlignCenter)
        int_layout.addWidget(self.satellite_box, 0, 3)
        int_layout.addWidget(dh_label, 0, 5, Qt.AlignCenter)
        int_layout.addWidget(dk_label, 0, 6, Qt.AlignCenter)
        int_layout.addWidget(dl_label, 0, 7, Qt.AlignCenter)

        int_layout.addWidget(self.cell_combo, 1, 0)
        int_layout.addWidget(self.centering_combo, 1, 1)
        int_layout.addWidget(self.max_order_line, 1, 2)
        int_layout.addWidget(self.cross_box, 1, 3)
        int_layout.addWidget(dim_1_label, 1, 4)
        int_layout.addWidget(self.mod_11_line, 1, 5)
        int_layout.addWidget(self.mod_12_line, 1, 6)
        int_layout.addWidget(self.mod_13_line, 1, 7)

        int_layout.addWidget(min_d_label, 2, 0)
        int_layout.addWidget(self.min_d_line, 2, 1)
        int_layout.addWidget(self.min_sat_d_line, 2, 2)
        int_layout.addWidget(d_unit_lael, 2, 3)
        int_layout.addWidget(dim_2_label, 2, 4)
        int_layout.addWidget(self.mod_21_line, 2, 5)
        int_layout.addWidget(self.mod_22_line, 2, 6)
        int_layout.addWidget(self.mod_23_line, 2, 7)

        int_layout.addWidget(radius_label, 3, 0)
        int_layout.addWidget(self.radius_line, 3, 1)
        int_layout.addWidget(radius_unit_label, 3, 2)
        int_layout.addWidget(dim_3_label, 3, 4)
        int_layout.addWidget(self.mod_31_line, 3, 5)
        int_layout.addWidget(self.mod_32_line, 3, 6)
        int_layout.addWidget(self.mod_33_line, 3, 7)

        self.int_run_button = QPushButton("Run Integration", self)
        self.int_run_button.setIcon(qta.icon("fa6s.play"))

        profile_layout.addStretch(1)
        profile_layout.addWidget(self.int_run_button)

        layout.addLayout(int_layout)
        layout.addStretch(1)
        layout.addLayout(profile_layout)

        tab.setLayout(layout)

        return tab

    def auto_scale_dropdown(self, combo):
        """
        Autoscale a combobox width to fit text plus any icons/checks.

        This keeps the drop-down (and closed state) wide enough for the
        longest item label while leaving extra room for icons or check
        indicators drawn on the left-hand side.
        """

        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        fm = combo.fontMetrics()
        max_width = 0

        digit = all(
            [combo.itemText(i).isdigit() for i in range(combo.count())]
        )

        keys = space_number.keys()

        for i in range(combo.count()):
            text = combo.itemText(i)
            icon = qta.icon("fa6s.hashtag" if digit else "fa6s.minus")
            if text == "TOPAZ":
                icon = qta.icon("fa6s.gem")
            elif text == "CORELLI":
                icon = qta.icon("fa6s.scissors")
            elif text == "MANDI":
                icon = qta.icon("fa6s.dna")
            elif text == "WAND²":
                icon = qta.icon("fa6s.wand-magic")
            elif text == "DEMAND":
                icon = qta.icon("fa6s.magnet")
            elif text == "SNAP":
                icon = qta.icon("fa6s.weight-scale")
            elif text == "IMAGINE":
                icon = qta.icon("fa6s.lightbulb")
            elif text in keys:
                no = str(space_number[text])
                pixmap = QPixmap(64, 64)
                pixmap.fill(Qt.GlobalColor.transparent)
                p = QPainter(pixmap)
                p.setFont(QFont("Arial", 32))
                p.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, no)
                p.end()
                icon = QIcon(pixmap)

            combo.setItemIcon(i, icon)

        for i in range(combo.count()):
            text = combo.itemText(i)
            text_width = fm.horizontalAdvance(text)

            icon = combo.itemIcon(i)
            icon_width = 0
            if not icon.isNull():
                size = icon.actualSize(combo.iconSize())
                icon_width = size.width() + 8

            max_width = max(max_width, text_width + icon_width)

        if max_width:
            padding = 40
            combo.setMinimumWidth(max_width + padding)

    def connect_satellite_box(self, update):
        self.satellite_box.stateChanged.connect(update)

    def set_satellite(self, state):
        self.satellite_box.setChecked(state)

    def set_centering(self, centering):
        index = self.centering_combo.findText(centering)
        if index != -1:
            self.centering_combo.setCurrentIndex(index)

    def get_centering(self):
        return self.centering_combo.currentText()

    def set_cell(self, cell):
        index = self.cell_combo.findText(cell)
        if index != -1:
            self.cell_combo.setCurrentIndex(index)

    def get_cell(self):
        return self.cell_combo.currentText()

    def set_mod_vec_1(self, vec):
        vec = [""] * 3 if vec is None else vec
        self.mod_11_line.setText(str(vec[0]))
        self.mod_12_line.setText(str(vec[1]))
        self.mod_13_line.setText(str(vec[2]))

    def set_mod_vec_2(self, vec):
        vec = [""] * 3 if vec is None else vec
        self.mod_21_line.setText(str(vec[0]))
        self.mod_22_line.setText(str(vec[1]))
        self.mod_23_line.setText(str(vec[2]))

    def set_mod_vec_3(self, vec):
        vec = [""] * 3 if vec is None else vec
        self.mod_31_line.setText(str(vec[0]))
        self.mod_32_line.setText(str(vec[1]))
        self.mod_33_line.setText(str(vec[2]))

    def get_mod_vec_1(
        self,
    ):
        params = self.mod_11_line, self.mod_12_line, self.mod_13_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_mod_vec_2(
        self,
    ):
        params = self.mod_21_line, self.mod_22_line, self.mod_23_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_mod_vec_3(
        self,
    ):
        params = self.mod_31_line, self.mod_32_line, self.mod_33_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_cross_terms(self):
        return self.cross_box.isChecked()

    def set_cross_terms(self, state):
        self.cross_box.setChecked(state)

    def get_max_order(self):
        if self.max_order_line.hasAcceptableInput():
            return int(self.max_order_line.text())
        else:
            return 0

    def set_max_order(self, order):
        self.max_order_line.setText(str(order))

    def get_sat_min_d(self):
        if self.min_sat_d_line.hasAcceptableInput():
            return float(self.min_sat_d_line.text())

    def set_sat_min_d(self, min_d):
        min_d = 1.0 if min_d is None else min_d
        self.min_sat_d_line.setText(str(min_d))

    def get_min_d(self):
        if self.min_d_line.hasAcceptableInput():
            return float(self.min_d_line.text())

    def set_min_d(self, min_d):
        self.min_d_line.setText(str(min_d))

    def get_radius(self):
        if self.radius_line.hasAcceptableInput():
            return float(self.radius_line.text())

    def set_radius(self, r):
        self.radius_line.setText(str(r))

    def clear_satellite(self):
        check = self.satellite_box.isChecked()
        self.mod_11_line.setEnabled(check)
        self.mod_12_line.setEnabled(check)
        self.mod_13_line.setEnabled(check)
        self.mod_21_line.setEnabled(check)
        self.mod_22_line.setEnabled(check)
        self.mod_23_line.setEnabled(check)
        self.mod_31_line.setEnabled(check)
        self.mod_32_line.setEnabled(check)
        self.mod_33_line.setEnabled(check)
        self.cross_box.setEnabled(check)
        self.max_order_line.setEnabled(check)
        self.min_sat_d_line.setEnabled(check)

        if not check:
            self.set_mod_vec_1([0.0, 0.0, 0.0])
            self.set_mod_vec_2([0.0, 0.0, 0.0])
            self.set_mod_vec_3([0.0, 0.0, 0.0])
            self.set_cross_terms(False)
            self.set_max_order(0)
        else:
            self.set_max_order(1)

        self.set_sat_min_d(1.0)

    def param_plan(self):
        tab = QWidget()

        layout = QVBoxLayout()

        bin_layout = QGridLayout()
        miller_layout = QHBoxLayout()

        dim_1_label = QLabel("1:")
        dim_2_label = QLabel("2:")
        dim_3_label = QLabel("3:")
        dim_4_label = QLabel("4:")

        min_label = QLabel("Min")
        max_label = QLabel("Max")
        step_label = QLabel("Step")
        bins_label = QLabel("Bins")

        self.p1_label = QLabel("h")
        self.p2_label = QLabel("k")
        self.p3_label = QLabel("l")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-1001, 1001, 5, notation=notation)

        self.param_min_1_line = QLineEdit("-0.1")
        self.param_min_2_line = QLineEdit("-0.1")
        self.param_min_3_line = QLineEdit("0.5")
        self.param_min_4_line = QLineEdit("5")

        self.param_max_1_line = QLineEdit("0.1")
        self.param_max_2_line = QLineEdit("0.1")
        self.param_max_3_line = QLineEdit("1.5")
        self.param_max_4_line = QLineEdit("300")

        self.param_min_1_line.setValidator(validator)
        self.param_min_2_line.setValidator(validator)
        self.param_min_3_line.setValidator(validator)
        self.param_min_4_line.setValidator(validator)

        self.param_max_1_line.setValidator(validator)
        self.param_max_2_line.setValidator(validator)
        self.param_max_3_line.setValidator(validator)
        self.param_max_4_line.setValidator(validator)

        validator = QIntValidator(1, 10000, self)

        self.param_bins_1_line = QLineEdit("1")
        self.param_bins_2_line = QLineEdit("1")
        self.param_bins_3_line = QLineEdit("51")
        self.param_bins_4_line = QLineEdit("60")

        self.param_bins_1_line.setValidator(validator)
        self.param_bins_2_line.setValidator(validator)
        self.param_bins_3_line.setValidator(validator)

        validator = QIntValidator(0, 10000, self)

        self.param_bins_4_line.setValidator(validator)

        self.param_step_1_line = QLineEdit("0.2")
        self.param_step_2_line = QLineEdit("0.2")
        self.param_step_3_line = QLineEdit("0.02")
        self.param_step_4_line = QLineEdit("5.0")

        self.param_step_1_line.setReadOnly(True)
        self.param_step_2_line.setReadOnly(True)
        self.param_step_3_line.setReadOnly(True)
        self.param_step_4_line.setReadOnly(True)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.param_proj_11_line = QLineEdit("1")
        self.param_proj_12_line = QLineEdit("0")
        self.param_proj_13_line = QLineEdit("0")

        self.param_proj_21_line = QLineEdit("0")
        self.param_proj_22_line = QLineEdit("1")
        self.param_proj_23_line = QLineEdit("0")

        self.param_proj_31_line = QLineEdit("0")
        self.param_proj_32_line = QLineEdit("0")
        self.param_proj_33_line = QLineEdit("1")

        self.param_proj_11_line.setValidator(validator)
        self.param_proj_12_line.setValidator(validator)
        self.param_proj_13_line.setValidator(validator)

        self.param_proj_21_line.setValidator(validator)
        self.param_proj_22_line.setValidator(validator)
        self.param_proj_23_line.setValidator(validator)

        self.param_proj_31_line.setValidator(validator)
        self.param_proj_32_line.setValidator(validator)
        self.param_proj_33_line.setValidator(validator)

        self.param_log_combo = QComboBox(self)
        self.param_log_line = QLineEdit("sample_temperature")

        bin_layout.addWidget(min_label, 0, 1, Qt.AlignCenter)
        bin_layout.addWidget(max_label, 0, 2, Qt.AlignCenter)
        bin_layout.addWidget(bins_label, 0, 3, Qt.AlignCenter)
        bin_layout.addWidget(step_label, 0, 4, Qt.AlignCenter)
        bin_layout.addWidget(self.p1_label, 0, 5, Qt.AlignCenter)
        bin_layout.addWidget(self.p2_label, 0, 6, Qt.AlignCenter)
        bin_layout.addWidget(self.p3_label, 0, 7, Qt.AlignCenter)

        bin_layout.addWidget(dim_1_label, 1, 0)
        bin_layout.addWidget(self.param_min_1_line, 1, 1)
        bin_layout.addWidget(self.param_max_1_line, 1, 2)
        bin_layout.addWidget(self.param_bins_1_line, 1, 3)
        bin_layout.addWidget(self.param_step_1_line, 1, 4)
        bin_layout.addWidget(self.param_proj_11_line, 1, 5)
        bin_layout.addWidget(self.param_proj_12_line, 1, 6)
        bin_layout.addWidget(self.param_proj_13_line, 1, 7)

        bin_layout.addWidget(dim_2_label, 2, 0)
        bin_layout.addWidget(self.param_min_2_line, 2, 1)
        bin_layout.addWidget(self.param_max_2_line, 2, 2)
        bin_layout.addWidget(self.param_bins_2_line, 2, 3)
        bin_layout.addWidget(self.param_step_2_line, 2, 4)
        bin_layout.addWidget(self.param_proj_21_line, 2, 5)
        bin_layout.addWidget(self.param_proj_22_line, 2, 6)
        bin_layout.addWidget(self.param_proj_23_line, 2, 7)

        bin_layout.addWidget(dim_3_label, 3, 0)
        bin_layout.addWidget(self.param_min_3_line, 3, 1)
        bin_layout.addWidget(self.param_max_3_line, 3, 2)
        bin_layout.addWidget(self.param_bins_3_line, 3, 3)
        bin_layout.addWidget(self.param_step_3_line, 3, 4)
        bin_layout.addWidget(self.param_proj_31_line, 3, 5)
        bin_layout.addWidget(self.param_proj_32_line, 3, 6)
        bin_layout.addWidget(self.param_proj_33_line, 3, 7)

        param_layout = QHBoxLayout()
        param_layout.addWidget(self.param_log_combo)
        param_layout.addWidget(self.param_log_line)

        bin_layout.addWidget(dim_4_label, 4, 0)
        bin_layout.addWidget(self.param_min_4_line, 4, 1)
        bin_layout.addWidget(self.param_max_4_line, 4, 2)
        bin_layout.addWidget(self.param_bins_4_line, 4, 3)
        bin_layout.addWidget(self.param_step_4_line, 4, 4)
        bin_layout.addLayout(param_layout, 4, 5, 1, 3)

        self.miller_box = QCheckBox("Miller Index")
        self.miller_box.setChecked(False)

        miller_h_label = QLabel("h:")
        miller_k_label = QLabel("k:")
        miller_l_label = QLabel("l:")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100, 100, 5, notation=notation)

        self.miller_h_line = QLineEdit("")
        self.miller_k_line = QLineEdit("")
        self.miller_l_line = QLineEdit("")

        self.miller_h_line.setEnabled(False)
        self.miller_k_line.setEnabled(False)
        self.miller_l_line.setEnabled(False)

        self.miller_h_line.setValidator(validator)
        self.miller_k_line.setValidator(validator)
        self.miller_l_line.setValidator(validator)

        self.param_run_button = QPushButton("Run Parametrization", self)
        self.param_run_button.setIcon(qta.icon("fa6s.play"))

        miller_layout.addWidget(self.miller_box)
        miller_layout.addWidget(miller_h_label)
        miller_layout.addWidget(self.miller_h_line)
        miller_layout.addWidget(miller_k_label)
        miller_layout.addWidget(self.miller_k_line)
        miller_layout.addWidget(miller_l_label)
        miller_layout.addWidget(self.miller_l_line)
        miller_layout.addStretch(1)
        miller_layout.addWidget(self.param_run_button)

        layout.addLayout(bin_layout)
        layout.addStretch(1)
        layout.addLayout(miller_layout)

        tab.setLayout(layout)

        return tab

    def connect_miller_box(self, update):
        self.miller_box.stateChanged.connect(update)

    def connect_param_min_1_line(self, update):
        self.param_min_1_line.editingFinished.connect(update)

    def connect_param_min_2_line(self, update):
        self.param_min_2_line.editingFinished.connect(update)

    def connect_param_min_3_line(self, update):
        self.param_min_3_line.editingFinished.connect(update)

    def connect_param_min_4_line(self, update):
        self.param_min_4_line.editingFinished.connect(update)

    def connect_param_max_1_line(self, update):
        self.param_max_1_line.editingFinished.connect(update)

    def connect_param_max_2_line(self, update):
        self.param_max_2_line.editingFinished.connect(update)

    def connect_param_max_3_line(self, update):
        self.param_max_3_line.editingFinished.connect(update)

    def connect_param_max_4_line(self, update):
        self.param_max_4_line.editingFinished.connect(update)

    def connect_param_bins_1_line(self, update):
        self.param_bins_1_line.editingFinished.connect(update)

    def connect_param_bins_2_line(self, update):
        self.param_bins_2_line.editingFinished.connect(update)

    def connect_param_bins_3_line(self, update):
        self.param_bins_3_line.editingFinished.connect(update)

    def connect_param_bins_4_line(self, update):
        self.param_bins_4_line.editingFinished.connect(update)

    def connect_log_option_combo(self, update):
        self.param_log_combo.activated.connect(lambda *_: update())

    def set_log_options(self, log_options):
        self.param_log_combo.clear()
        for option in log_options:
            self.param_log_combo.addItem(option)
        self.auto_scale_dropdown(self.param_log_combo)

    def get_log_option(self):
        return self.param_log_combo.currentText()

    def get_log_name(self):
        return self.param_log_line.text()

    def set_log_name(self, name):
        self.param_log_line.setText(name)
        index = self.param_log_combo.findText(name)
        self.param_log_combo.setCurrentIndex(index)

    def get_miller_index(self):
        check = self.miller_box.isChecked()
        params = self.miller_h_line, self.miller_k_line, self.miller_l_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params and check:
            return [self.str_to_number(param.text()) for param in params]

    def update_miller(self, state):
        self.miller_box.setChecked(state)

    def set_miller_index(self, hkl):
        hkl = [""] * 3 if hkl is None else hkl
        self.miller_h_line.setText(str(hkl[0]))
        self.miller_k_line.setText(str(hkl[1]))
        self.miller_l_line.setText(str(hkl[2]))

    def clear_miller(self):
        check = self.miller_box.isChecked()
        self.miller_h_line.setEnabled(check)
        self.miller_k_line.setEnabled(check)
        self.miller_l_line.setEnabled(check)
        self.param_proj_11_line.setEnabled(not check)
        self.param_proj_12_line.setEnabled(not check)
        self.param_proj_13_line.setEnabled(not check)
        self.param_proj_21_line.setEnabled(not check)
        self.param_proj_22_line.setEnabled(not check)
        self.param_proj_23_line.setEnabled(not check)
        self.param_proj_31_line.setEnabled(not check)
        self.param_proj_32_line.setEnabled(not check)
        self.param_proj_33_line.setEnabled(not check)

        if check:
            self.set_miller_index([0, 0, 1])
            self.set_param_projections([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.p1_label.setText("Δ|Q|")  # x₁	x₂
            self.p2_label.setText("ΔQ₁")
            self.p3_label.setText("ΔQ₂")
        else:
            self.set_miller_index([""] * 3)
            self.p1_label.setText("h")
            self.p2_label.setText("k")
            self.p3_label.setText("l")

    def get_param_bins_1(self):
        param = self.param_bins_1_line
        if param.hasAcceptableInput():
            return int(param.text())

    def get_param_bins_2(self):
        param = self.param_bins_2_line
        if param.hasAcceptableInput():
            return int(param.text())

    def get_param_bins_3(self):
        param = self.param_bins_3_line
        if param.hasAcceptableInput():
            return int(param.text())

    def get_param_bins_4(self):
        param = self.param_bins_4_line
        if param.hasAcceptableInput():
            return int(param.text())

    def set_param_bins_1(self, bins):
        self.param_bins_1_line.setText("{}".format(bins))

    def set_param_bins_2(self, bins):
        self.param_bins_2_line.setText("{}".format(bins))

    def set_param_bins_3(self, bins):
        self.param_bins_3_line.setText("{}".format(bins))

    def set_param_bins_4(self, bins):
        self.param_bins_4_line.setText("{}".format(bins))

    def get_param_limits_1(self):
        params = self.param_min_1_line, self.param_max_1_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_param_limits_2(self):
        params = self.param_min_2_line, self.param_max_2_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_param_limits_3(self):
        params = self.param_min_3_line, self.param_max_3_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_param_limits_4(self):
        params = self.param_min_4_line, self.param_max_4_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def set_param_limits_1(self, limits):
        vmin, vmax = limits
        self.param_min_1_line.setText("{:.4f}".format(vmin))
        self.param_max_1_line.setText("{:.4f}".format(vmax))

    def set_param_limits_2(self, limits):
        vmin, vmax = limits
        self.param_min_2_line.setText("{:.4f}".format(vmin))
        self.param_max_2_line.setText("{:.4f}".format(vmax))

    def set_param_limits_3(self, limits):
        vmin, vmax = limits
        self.param_min_3_line.setText("{:.4f}".format(vmin))
        self.param_max_3_line.setText("{:.4f}".format(vmax))

    def set_param_limits_4(self, limits):
        vmin, vmax = limits
        self.param_min_4_line.setText("{:.4f}".format(vmin))
        self.param_max_4_line.setText("{:.4f}".format(vmax))

    def set_param_step_1(self, step):
        self.param_step_1_line.setText("{:.4f}".format(step))

    def set_param_step_2(self, step):
        self.param_step_2_line.setText("{:.4f}".format(step))

    def set_param_step_3(self, step):
        self.param_step_3_line.setText("{:.4f}".format(step))

    def set_param_step_4(self, step):
        self.param_step_4_line.setText("{:.4f}".format(step))

    def get_param_projections(self):
        params = (
            self.param_proj_11_line,
            self.param_proj_12_line,
            self.param_proj_13_line,
            self.param_proj_21_line,
            self.param_proj_22_line,
            self.param_proj_23_line,
            self.param_proj_31_line,
            self.param_proj_32_line,
            self.param_proj_33_line,
        )
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            proj = [self.str_to_number(param.text()) for param in params]
            return [proj[i : i + 3] for i in range(0, len(proj), 3)]

    def set_param_projections(self, proj):
        params = (
            self.param_proj_11_line,
            self.param_proj_12_line,
            self.param_proj_13_line,
            self.param_proj_21_line,
            self.param_proj_22_line,
            self.param_proj_23_line,
            self.param_proj_31_line,
            self.param_proj_32_line,
            self.param_proj_33_line,
        )
        items = [item for row in proj for item in row]
        for item, param in zip(items, params):
            param.setText(str(item))

    def norm_plan(self):
        tab = QWidget()

        layout = QVBoxLayout()

        bin_layout = QGridLayout()
        symmetry_layout = QHBoxLayout()

        dim_1_label = QLabel("1:")
        dim_2_label = QLabel("2:")
        dim_3_label = QLabel("3:")

        min_label = QLabel("Min")
        max_label = QLabel("Max")
        step_label = QLabel("Step")
        bins_label = QLabel("Bins")

        h_label = QLabel("h")
        k_label = QLabel("k")
        l_label = QLabel("l")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-100001, 100001, 5, notation=notation)

        self.norm_min_1_line = QLineEdit("-10")
        self.norm_min_2_line = QLineEdit("-10")
        self.norm_min_3_line = QLineEdit("-10")

        self.norm_max_1_line = QLineEdit("10")
        self.norm_max_2_line = QLineEdit("10")
        self.norm_max_3_line = QLineEdit("10")

        self.norm_min_1_line.setValidator(validator)
        self.norm_min_2_line.setValidator(validator)
        self.norm_min_3_line.setValidator(validator)

        self.norm_max_1_line.setValidator(validator)
        self.norm_max_2_line.setValidator(validator)
        self.norm_max_3_line.setValidator(validator)

        validator = QIntValidator(1, 1000, self)

        self.norm_bins_1_line = QLineEdit("201")
        self.norm_bins_2_line = QLineEdit("201")
        self.norm_bins_3_line = QLineEdit("201")

        self.norm_bins_1_line.setValidator(validator)
        self.norm_bins_2_line.setValidator(validator)
        self.norm_bins_3_line.setValidator(validator)

        self.norm_step_1_line = QLineEdit("0.1")
        self.norm_step_2_line = QLineEdit("0.1")
        self.norm_step_3_line = QLineEdit("0.1")

        self.norm_step_1_line.setReadOnly(True)
        self.norm_step_2_line.setReadOnly(True)
        self.norm_step_3_line.setReadOnly(True)

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(-10, 10, 5, notation=notation)

        self.norm_proj_11_line = QLineEdit("1")
        self.norm_proj_12_line = QLineEdit("0")
        self.norm_proj_13_line = QLineEdit("0")

        self.norm_proj_21_line = QLineEdit("0")
        self.norm_proj_22_line = QLineEdit("1")
        self.norm_proj_23_line = QLineEdit("0")

        self.norm_proj_31_line = QLineEdit("0")
        self.norm_proj_32_line = QLineEdit("0")
        self.norm_proj_33_line = QLineEdit("1")

        self.norm_proj_11_line.setValidator(validator)
        self.norm_proj_12_line.setValidator(validator)
        self.norm_proj_13_line.setValidator(validator)

        self.norm_proj_21_line.setValidator(validator)
        self.norm_proj_22_line.setValidator(validator)
        self.norm_proj_23_line.setValidator(validator)

        self.norm_proj_31_line.setValidator(validator)
        self.norm_proj_32_line.setValidator(validator)
        self.norm_proj_33_line.setValidator(validator)

        bin_layout.addWidget(min_label, 0, 1, Qt.AlignCenter)
        bin_layout.addWidget(max_label, 0, 2, Qt.AlignCenter)
        bin_layout.addWidget(bins_label, 0, 3, Qt.AlignCenter)
        bin_layout.addWidget(step_label, 0, 4, Qt.AlignCenter)
        bin_layout.addWidget(h_label, 0, 5, Qt.AlignCenter)
        bin_layout.addWidget(k_label, 0, 6, Qt.AlignCenter)
        bin_layout.addWidget(l_label, 0, 7, Qt.AlignCenter)

        bin_layout.addWidget(dim_1_label, 1, 0)
        bin_layout.addWidget(self.norm_min_1_line, 1, 1)
        bin_layout.addWidget(self.norm_max_1_line, 1, 2)
        bin_layout.addWidget(self.norm_bins_1_line, 1, 3)
        bin_layout.addWidget(self.norm_step_1_line, 1, 4)
        bin_layout.addWidget(self.norm_proj_11_line, 1, 5)
        bin_layout.addWidget(self.norm_proj_12_line, 1, 6)
        bin_layout.addWidget(self.norm_proj_13_line, 1, 7)

        bin_layout.addWidget(dim_2_label, 2, 0)
        bin_layout.addWidget(self.norm_min_2_line, 2, 1)
        bin_layout.addWidget(self.norm_max_2_line, 2, 2)
        bin_layout.addWidget(self.norm_bins_2_line, 2, 3)
        bin_layout.addWidget(self.norm_step_2_line, 2, 4)
        bin_layout.addWidget(self.norm_proj_21_line, 2, 5)
        bin_layout.addWidget(self.norm_proj_22_line, 2, 6)
        bin_layout.addWidget(self.norm_proj_23_line, 2, 7)

        bin_layout.addWidget(dim_3_label, 3, 0)
        bin_layout.addWidget(self.norm_min_3_line, 3, 1)
        bin_layout.addWidget(self.norm_max_3_line, 3, 2)
        bin_layout.addWidget(self.norm_bins_3_line, 3, 3)
        bin_layout.addWidget(self.norm_step_3_line, 3, 4)
        bin_layout.addWidget(self.norm_proj_31_line, 3, 5)
        bin_layout.addWidget(self.norm_proj_32_line, 3, 6)
        bin_layout.addWidget(self.norm_proj_33_line, 3, 7)

        symmetry_label = QLabel("Apply Symmetry:")

        self.symmetry_combo = QComboBox(self)
        self.symmetry_combo.addItem("None")
        self.symmetry_combo.addItem("Space Group")
        self.symmetry_combo.addItem("Point Group")

        self.auto_scale_dropdown(self.symmetry_combo)

        self.symmetry_options_combo = QComboBox(self)

        self.norm_run_button = QPushButton("Run Normalization", self)
        self.norm_run_button.setIcon(qta.icon("fa6s.play"))

        self.auto_proj_button = QPushButton("Auto Project", self)
        self.auto_bin_button = QPushButton("Auto Bin", self)

        res_label = QLabel("d(min) [Å]:")

        self.res_line = QLineEdit("0.7")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.3, 100, 5, notation=notation)

        self.res_line.setValidator(validator)

        symmetry_layout.addWidget(symmetry_label)
        symmetry_layout.addWidget(self.symmetry_combo)
        symmetry_layout.addWidget(self.symmetry_options_combo)
        symmetry_layout.addStretch(1)
        symmetry_layout.addWidget(self.auto_proj_button)
        symmetry_layout.addWidget(res_label)
        symmetry_layout.addWidget(self.res_line)
        symmetry_layout.addWidget(self.auto_bin_button)
        symmetry_layout.addStretch(1)
        symmetry_layout.addWidget(self.norm_run_button)

        layout.addLayout(bin_layout)
        layout.addStretch(1)
        layout.addLayout(symmetry_layout)

        tab.setLayout(layout)

        return tab

    def connect_symmetry_combo(self, update_symmetry):
        self.symmetry_combo.currentIndexChanged.connect(update_symmetry)

    def connect_norm_min_1_line(self, update):
        self.norm_min_1_line.editingFinished.connect(update)

    def connect_norm_min_2_line(self, update):
        self.norm_min_2_line.editingFinished.connect(update)

    def connect_norm_min_3_line(self, update):
        self.norm_min_3_line.editingFinished.connect(update)

    def connect_norm_max_1_line(self, update):
        self.norm_max_1_line.editingFinished.connect(update)

    def connect_norm_max_2_line(self, update):
        self.norm_max_2_line.editingFinished.connect(update)

    def connect_norm_max_3_line(self, update):
        self.norm_max_3_line.editingFinished.connect(update)

    def connect_norm_bins_1_line(self, update):
        self.norm_bins_1_line.editingFinished.connect(update)

    def connect_norm_bins_2_line(self, update):
        self.norm_bins_2_line.editingFinished.connect(update)

    def connect_norm_bins_3_line(self, update):
        self.norm_bins_3_line.editingFinished.connect(update)

    def connect_auto_projection(self, auto_proj):
        self.auto_proj_button.clicked.connect(auto_proj)

    def connect_auto_binning(self, auto_bin):
        self.auto_bin_button.clicked.connect(auto_bin)

    def get_symmetry(self):
        return self.symmetry_combo.currentText()

    def set_symmetry(self, symmetry):
        index = self.symmetry_combo.findText(symmetry)
        if index != -1:
            self.symmetry_combo.setCurrentIndex(index)

    def set_symmetry_options(self, symmetries):
        self.symmetry_options_combo.clear()
        for symmetry in symmetries:
            self.symmetry_options_combo.addItem(symmetry)
        self.auto_scale_dropdown(self.symmetry_options_combo)

    def set_symmetry_option(self, symmetry):
        index = self.symmetry_options_combo.findText(symmetry)
        if index != -1:
            self.symmetry_options_combo.setCurrentIndex(index)

    def get_symmetry_option(self):
        symmetry = self.symmetry_options_combo.currentText()
        if symmetry != "":
            return symmetry

    def get_norm_bins_1(self):
        param = self.norm_bins_1_line
        if param.hasAcceptableInput():
            return int(param.text())

    def get_norm_bins_2(self):
        param = self.norm_bins_2_line
        if param.hasAcceptableInput():
            return int(param.text())

    def get_norm_bins_3(self):
        param = self.norm_bins_3_line
        if param.hasAcceptableInput():
            return int(param.text())

    def set_norm_bins_1(self, bins):
        self.norm_bins_1_line.setText("{}".format(bins))

    def set_norm_bins_2(self, bins):
        self.norm_bins_2_line.setText("{}".format(bins))

    def set_norm_bins_3(self, bins):
        self.norm_bins_3_line.setText("{}".format(bins))

    def get_norm_limits_1(self):
        params = self.norm_min_1_line, self.norm_max_1_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_norm_limits_2(self):
        params = self.norm_min_2_line, self.norm_max_2_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def get_norm_limits_3(self):
        params = self.norm_min_3_line, self.norm_max_3_line
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            return [float(param.text()) for param in params]

    def set_norm_limits_1(self, limits):
        vmin, vmax = limits
        self.norm_min_1_line.setText("{:.4f}".format(vmin))
        self.norm_max_1_line.setText("{:.4f}".format(vmax))

    def set_norm_limits_2(self, limits):
        vmin, vmax = limits
        self.norm_min_2_line.setText("{:.4f}".format(vmin))
        self.norm_max_2_line.setText("{:.4f}".format(vmax))

    def set_norm_limits_3(self, limits):
        vmin, vmax = limits
        self.norm_min_3_line.setText("{:.4f}".format(vmin))
        self.norm_max_3_line.setText("{:.4f}".format(vmax))

    def set_norm_step_1(self, step):
        self.norm_step_1_line.setText("{:.4f}".format(step))

    def set_norm_step_2(self, step):
        self.norm_step_2_line.setText("{:.4f}".format(step))

    def set_norm_step_3(self, step):
        self.norm_step_3_line.setText("{:.4f}".format(step))

    def get_norm_projections(self):
        params = (
            self.norm_proj_11_line,
            self.norm_proj_12_line,
            self.norm_proj_13_line,
            self.norm_proj_21_line,
            self.norm_proj_22_line,
            self.norm_proj_23_line,
            self.norm_proj_31_line,
            self.norm_proj_32_line,
            self.norm_proj_33_line,
        )
        valid_params = all([param.hasAcceptableInput() for param in params])
        if valid_params:
            proj = [self.str_to_number(param.text()) for param in params]
            return [proj[i : i + 3] for i in range(0, len(proj), 3)]

    def set_norm_projections(self, proj):
        params = (
            self.norm_proj_11_line,
            self.norm_proj_12_line,
            self.norm_proj_13_line,
            self.norm_proj_21_line,
            self.norm_proj_22_line,
            self.norm_proj_23_line,
            self.norm_proj_31_line,
            self.norm_proj_32_line,
            self.norm_proj_33_line,
        )
        items = [item for row in proj for item in row]
        for item, param in zip(items, params):
            param.setText(str(item))

    def get_norm_resolution_min(self):
        param = self.res_line
        if param.hasAcceptableInput():
            return float(param.text())

    def set_norm_resolution_min(self, d_min):
        self.res_line.setText(str(d_min))

    def str_to_number(self, s):
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        elif s.replace(".", "", 1).replace("-", "", 1).isdigit():
            return float(s)

    def init_plan(self):
        tab = QWidget()

        layout = QVBoxLayout()

        experiment_params_layout = QHBoxLayout()
        run_params_layout = QGridLayout()
        instrument_params_layout = QGridLayout()

        self.instrument_combo = QComboBox(self)
        self.instrument_combo.addItem("TOPAZ")
        self.instrument_combo.addItem("MANDI")
        self.instrument_combo.addItem("CORELLI")
        self.instrument_combo.addItem("SNAP")

        self.auto_scale_dropdown(self.instrument_combo)

        self.grouping_combo = QComboBox(self)

        self.auto_scale_dropdown(self.grouping_combo)

        self.elastic_box = QCheckBox("Elastic")

        ipts_label = QLabel("IPTS:")
        exp_label = QLabel("Experiment:")
        run_label = QLabel("Runs:")
        angstrom_label = QLabel("Å")

        validator = QIntValidator(1, 1000000000, self)

        self.runs_line = QLineEdit("")

        self.ipts_line = QLineEdit("")
        self.ipts_line.setValidator(validator)

        self.exp_line = QLineEdit("")
        self.exp_line.setValidator(validator)

        self.ub_line = QLineEdit("")
        self.bkg_line = QLineEdit("")
        self.van_line = QLineEdit("")
        self.flux_line = QLineEdit("")
        self.eff_line = QLineEdit("")
        self.spec_line = QLineEdit("")
        self.cal_line = QLineEdit("")
        self.tube_line = QLineEdit("")
        self.mask_line = QLineEdit("")
        self.output_line = QLineEdit("")
        self.gonio_line = QLineEdit("")

        self.wl_min_line = QLineEdit("0.3")
        self.wl_max_line = QLineEdit("3.5")

        wl_label = QLabel("λ:")

        notation = QDoubleValidator.StandardNotation

        validator = QDoubleValidator(0.2, 10, 5, notation=notation)

        self.wl_min_line.setValidator(validator)
        self.wl_max_line.setValidator(validator)

        validator = QIntValidator(1, 1000, self)

        self.ub_browse_button = QPushButton("UB", self)
        self.bkg_browse_button = QPushButton("Background", self)
        self.van_browse_button = QPushButton("Solid Angle", self)
        self.flux_browse_button = QPushButton("Flux", self)
        self.cal_browse_button = QPushButton("Detector", self)
        self.tube_browse_button = QPushButton("Tube", self)
        self.mask_browse_button = QPushButton("Mask", self)
        self.gonio_browse_button = QPushButton("Goniometer", self)

        browse_icon = qta.icon("fa6s.folder-open")
        self.ub_browse_button.setIcon(browse_icon)
        self.bkg_browse_button.setIcon(browse_icon)
        self.van_browse_button.setIcon(browse_icon)
        self.flux_browse_button.setIcon(browse_icon)
        self.cal_browse_button.setIcon(browse_icon)
        self.tube_browse_button.setIcon(browse_icon)
        self.mask_browse_button.setIcon(browse_icon)
        self.gonio_browse_button.setIcon(browse_icon)

        experiment_params_layout.addWidget(self.instrument_combo)
        experiment_params_layout.addWidget(ipts_label)
        experiment_params_layout.addWidget(self.ipts_line)
        experiment_params_layout.addWidget(exp_label)
        experiment_params_layout.addWidget(self.exp_line)
        experiment_params_layout.addWidget(self.elastic_box)

        run_params_layout.addWidget(run_label, 0, 0)
        run_params_layout.addWidget(self.runs_line, 0, 1)

        experiment_params_layout.addStretch(1)
        experiment_params_layout.addWidget(wl_label)
        experiment_params_layout.addWidget(self.wl_min_line)
        experiment_params_layout.addWidget(self.wl_max_line)
        experiment_params_layout.addWidget(angstrom_label)
        experiment_params_layout.addWidget(self.grouping_combo)

        instrument_params_layout.addWidget(self.ub_line, 1, 0)
        instrument_params_layout.addWidget(self.ub_browse_button, 1, 1)
        instrument_params_layout.addWidget(self.van_line, 2, 0)
        instrument_params_layout.addWidget(self.van_browse_button, 2, 1)
        instrument_params_layout.addWidget(self.flux_line, 3, 0)
        instrument_params_layout.addWidget(self.flux_browse_button, 3, 1)
        instrument_params_layout.addWidget(self.bkg_line, 4, 0)
        instrument_params_layout.addWidget(self.bkg_browse_button, 4, 1)
        instrument_params_layout.addWidget(self.mask_line, 5, 0)
        instrument_params_layout.addWidget(self.mask_browse_button, 5, 1)
        instrument_params_layout.addWidget(self.cal_line, 6, 0)
        instrument_params_layout.addWidget(self.cal_browse_button, 6, 1)
        instrument_params_layout.addWidget(self.tube_line, 7, 0)
        instrument_params_layout.addWidget(self.tube_browse_button, 7, 1)
        instrument_params_layout.addWidget(self.gonio_line, 8, 0)
        instrument_params_layout.addWidget(self.gonio_browse_button, 8, 1)

        layout.addLayout(experiment_params_layout)
        layout.addLayout(run_params_layout)
        layout.addLayout(instrument_params_layout)

        tab.setLayout(layout)

        return tab

    def run_command(self, command):
        self.output.appendPlainText("Running shell command...\n")
        script, *args = command.split(" ")
        if self.process.state() == QProcess.NotRunning:
            self._elapsed_timer = QElapsedTimer()
            self._elapsed_timer.start()
            self.process.start(script, args)
        else:
            print("Process already running!")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        text = bytes(data).decode("utf-8")
        self.output.appendPlainText(text)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        text = bytes(data).decode("utf-8")
        self.output.appendPlainText(f"[stderr] {text}")

    def process_finished(self):
        if hasattr(self, "_elapsed_timer"):
            ms = self._elapsed_timer.elapsed()
            h = ms // 3_600_000
            m = (ms % 3_600_000) // 60_000
            s = (ms % 60_000) // 1_000
            self.output.appendPlainText(
                f"Command finished. Elapsed time: {h:02d}:{m:02d}:{s:02d}\n"
            )
        else:
            self.output.appendPlainText("Command finished.\n")

    def connect_int_run_button(self, run):
        self.int_run_button.clicked.connect(run)

    def connect_param_run_button(self, run):
        self.param_run_button.clicked.connect(run)

    def connect_norm_run_button(self, run):
        self.norm_run_button.clicked.connect(run)

    def connect_load_config(self, load_config):
        self.load_button.clicked.connect(load_config)

    def connect_save_config(self, save_config):
        self.save_button.clicked.connect(save_config)

    def connect_save_as_config(self, save_as_config):
        self.save_as_button.clicked.connect(save_as_config)

    def connect_switch_instrument(self, switch_instrument):
        self.instrument_combo.activated.connect(switch_instrument)

    def connect_wavelength(self, update_wavelength):
        self.wl_min_line.editingFinished.connect(update_wavelength)

    def connect_load_UB(self, load_UB):
        self.ub_browse_button.clicked.connect(load_UB)

    def connect_load_mask(self, load_mask):
        self.mask_browse_button.clicked.connect(load_mask)

    def connect_load_detector(self, load_detector_cal):
        self.cal_browse_button.clicked.connect(load_detector_cal)

    def connect_load_goniometer(self, load_goniometer_cal):
        self.gonio_browse_button.clicked.connect(load_goniometer_cal)

    def connect_load_tube(self, load_tube_cal):
        self.tube_browse_button.clicked.connect(load_tube_cal)

    def connect_load_background(self, load_background):
        self.bkg_browse_button.clicked.connect(load_background)

    def connect_load_vanadium(self, load_vanadium):
        self.van_browse_button.clicked.connect(load_vanadium)

    def connect_load_flux(self, load_flux):
        self.flux_browse_button.clicked.connect(load_flux)

    def get_config(self):
        return self.output_line.text()

    def set_config(self, filename):
        return self.output_line.setText(filename)

    def set_wavelength(self, wavelength):
        if type(wavelength) is list:
            self.wl_min_line.setText(str(wavelength[0]))
            self.wl_max_line.setText(str(wavelength[1]))
            self.wl_max_line.setEnabled(True)
        else:
            self.wl_min_line.setText(str(wavelength))
            self.wl_max_line.setText(str(wavelength))
            self.wl_max_line.setEnabled(False)

    def get_wavelength(self):
        params = self.wl_min_line, self.wl_max_line

        valid_params = all([param.hasAcceptableInput() for param in params])

        if valid_params:
            return [float(param.text()) for param in params]

    def update_wavelength(self, lamda_min):
        if not self.wl_max_line.isEnabled():
            self.wl_max_line.setText(str(lamda_min))

    def set_elastic(self, el):
        return self.elastic_box.setChecked(el)

    def get_elastic(self):
        if self.elastic_box.isEnabled():
            return self.elastic_box.isChecked()

    def get_instrument(self):
        return self.instrument_combo.currentText()

    def set_instrument(self, instrument):
        index = self.instrument_combo.findText(instrument)
        if index != -1:
            self.instrument_combo.setCurrentIndex(index)
            self.update_mem_estimate()

    def clear_run_info(self, filepath):
        self.exp_line.setText("")
        self.cal_line.setText("")
        self.tube_line.setText("")
        self.flux_line.setText("")

        self.elastic_box.setChecked(False)
        self.elastic_box.setEnabled(False)

        if "exp" in filepath:
            self.exp_line.setEnabled(True)
        else:
            self.exp_line.setEnabled(False)

        if "SNS" in filepath:
            self.cal_line.setEnabled(True)
            self.cal_browse_button.setEnabled(True)
            self.gonio_line.setEnabled(True)
            self.gonio_browse_button.setEnabled(True)
            self.tube_line.setEnabled(False)
            self.tube_browse_button.setEnabled(False)
            if "CORELLI" in filepath:
                self.tube_line.setEnabled(True)
                self.tube_browse_button.setEnabled(True)
                self.elastic_box.setEnabled(True)
            self.flux_line.setEnabled(True)
            self.flux_browse_button.setEnabled(True)
        else:
            self.cal_line.setEnabled(False)
            self.cal_browse_button.setEnabled(False)
            self.gonio_line.setEnabled(False)
            self.gonio_browse_button.setEnabled(False)
            self.tube_line.setEnabled(False)
            self.tube_browse_button.setEnabled(False)
            self.flux_line.setEnabled(False)
            self.flux_browse_button.setEnabled(False)

    def get_vanadium(self):
        return self.van_line.text()

    def set_vanadium(self, filename):
        return self.van_line.setText(filename)

    def load_vanadium_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Vanadium files (*.nxs)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load vanadiim file", path, file_filters, options=options
        )

        return filename

    def get_flux(self):
        return self.flux_line.text()

    def set_flux(self, filename):
        return self.flux_line.setText(filename)

    def load_flux_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Flux files (*.nxs)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load flux file", path, file_filters, options=options
        )

        return filename

    def get_mask(self):
        return self.mask_line.text()

    def set_mask(self, filename):
        return self.mask_line.setText(filename)

    def load_mask_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Mask files (*.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load mask file", path, file_filters, options=options
        )

        return filename

    def get_background(self):
        return self.bkg_line.text()

    def set_background(self, filename):
        return self.bkg_line.setText(filename)

    def load_background_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Background files (*.h5 *.nxs)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load background file", path, file_filters, options=options
        )

        return filename

    def get_tube_calibration(self):
        return self.tube_line.text()

    def set_tube_calibration(self, filename):
        return self.tube_line.setText(filename)

    def load_tube_cal_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Tube files (*.h5 *.nxs)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load calibration file", path, file_filters, options=options
        )

        return filename

    def get_detector_calibration(self):
        return self.cal_line.text()

    def set_detector_calibration(self, filename):
        return self.cal_line.setText(filename)

    def load_detector_cal_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Calibration files (*.DetCal *.detcal *.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load calibration file", path, file_filters, options=options
        )

        return filename

    def get_goniometer_calibration(self):
        return self.gonio_line.text()

    def set_goniometer_calibration(self, filename):
        return self.gonio_line.setText(filename)

    def load_goniometer_cal_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        file_filters = "Calibration files (*.xml)"

        filename, _ = file_dialog.getOpenFileName(
            self, "Load goniometer file", path, file_filters, options=options
        )

        return filename

    def get_IPTS(self):
        if self.ipts_line.hasAcceptableInput():
            return int(self.ipts_line.text())

    def set_IPTS(self, IPTS):
        if type(IPTS) in [int, str]:
            self.ipts_line.setText(str(IPTS))

    def get_experiment(self):
        if self.exp_line.hasAcceptableInput():
            return self.exp_line.text()

    def set_experiment(self, exp):
        self.exp_line.setText(exp)

    def get_UB(self):
        return self.ub_line.text()

    def set_UB(self, UB):
        self.ub_line.setText(UB)

    def load_UB_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self, "Load UB file", path, "UB files (*.mat)", options=options
        )

        return filename

    def get_runs(self):
        return self.runs_line.text()

    def set_runs(self, run_str):
        self.runs_line.setText(run_str)

    def set_groupings(self, groupings):
        self.grouping_combo.clear()
        for grouping in groupings:
            self.grouping_combo.addItem(grouping)

        self.auto_scale_dropdown(self.grouping_combo)

    def set_grouping(self, grouping):
        index = self.grouping_combo.findText(grouping)
        if index != -1:
            self.grouping_combo.setCurrentIndex(index)

    def get_grouping(self):
        return self.grouping_combo.currentText()

    def load_config_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getOpenFileName(
            self,
            "Load config file",
            path,
            "Config files (*.yaml)",
            options=options,
        )

        return filename

    def save_config_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)

        filename, _ = file_dialog.getSaveFileName(
            self,
            "Save config file",
            path,
            "Config files (*.yaml)",
            options=options,
        )

        if filename is not None:
            if not filename.endswith(".yaml"):
                filename += ".yaml"

        return filename

    def get_processes(self):
        if self.cpu_line.hasAcceptableInput():
            return int(self.cpu_line.text())
        else:
            return 1

    def set_processes(self, cpu):
        self.cpu_line.setText(str(cpu))
        self.update_mem_estimate()

    def get_development(self):
        return self.dev_box.isChecked()

    def connect_plan_tab_changed(self, callback):
        self.plan_widget.currentChanged.connect(lambda *_: callback())

    def get_plan_tab_index(self):
        return self.plan_widget.currentIndex()

    def connect_preview_update(self, callback):
        for line in [
            self.norm_proj_11_line,
            self.norm_proj_12_line,
            self.norm_proj_13_line,
            self.norm_proj_21_line,
            self.norm_proj_22_line,
            self.norm_proj_23_line,
            self.norm_proj_31_line,
            self.norm_proj_32_line,
            self.norm_proj_33_line,
            self.norm_min_1_line,
            self.norm_max_1_line,
            self.norm_min_2_line,
            self.norm_max_2_line,
            self.norm_min_3_line,
            self.norm_max_3_line,
            self.param_proj_11_line,
            self.param_proj_12_line,
            self.param_proj_13_line,
            self.param_proj_21_line,
            self.param_proj_22_line,
            self.param_proj_23_line,
            self.param_proj_31_line,
            self.param_proj_32_line,
            self.param_proj_33_line,
            self.param_min_1_line,
            self.param_max_1_line,
            self.param_min_2_line,
            self.param_max_2_line,
            self.param_min_3_line,
            self.param_max_3_line,
            self.min_d_line,
        ]:
            line.editingFinished.connect(callback)
        self.centering_combo.currentIndexChanged.connect(lambda *_: callback())
        self._plot_btn.clicked.connect(lambda *_: callback())

    def _on_reset_view(self):
        self.plotter.reset_camera()
        self.plotter.view_isometric()

    def _on_reset_camera(self):
        self.plotter.reset_camera()

    def _q_axis_view(self, col, sign, viewup_col):
        """View along ±Qx/Qy/Qz in the display world frame.

        The display world frame is the lab Cartesian frame rotated by the QR
        normalization (absorbs U).  M = P @ UB^{-1} maps lab Cartesian vectors
        to display world vectors; its columns are the Qx/Qy/Qz directions.
        viewup_col: 1 (Qy) for ±Qx/±Qz views, 2 (Qz) for ±Qy views.
        """
        if self._preview_UB is None:
            return
        P, _, _ = _compute_preview_transforms(self._preview_UB, np.eye(3))
        M = P @ np.linalg.inv(self._preview_UB)
        norms = np.linalg.norm(M, axis=0)
        norms[norms < 1e-10] = 1.0
        M = M / norms
        self.plotter.view_vector(
            (sign * M[:, col]).tolist(), viewup=M[:, viewup_col].tolist()
        )

    def _crystal_T(self):
        """Return T_crystal: crystal axes (a*, b*, c*) in the display world frame.

        The display world frame is set by QR/Cholesky of UB@W, which rotates the
        lab Cartesian frame so the first projection axis aligns with x.  Raw UB
        columns are in lab Cartesian — they must be expressed in the display frame
        before being used for view_vector or the compass.  _compute_preview_transforms
        with W=I gives exactly that transformation.
        """
        _, T, _ = _compute_preview_transforms(self._preview_UB, np.eye(3))
        return T

    def _show_compass(self, *_):
        if self._preview_UB is None:
            return
        T = self._crystal_T()
        self.plotter.hide_axes()
        if self._recip_box.isChecked():
            # T columns are already unit-norm a*, b*, c* in display world frame
            cols = T
            actor = self.plotter.add_axes(
                xlabel="a*", ylabel="b*", zlabel="c*"
            )
        else:
            # Real lattice: a = b*×c*, b = c*×a*, c = a*×b*
            a_dir = np.cross(T[:, 1], T[:, 2])
            b_dir = np.cross(T[:, 2], T[:, 0])
            c_dir = np.cross(T[:, 0], T[:, 1])
            cols_raw = np.column_stack([a_dir, b_dir, c_dir])
            norms = np.linalg.norm(cols_raw, axis=0)
            norms[norms < 1e-10] = 1.0
            cols = cols_raw / norms
            actor = self.plotter.add_axes(xlabel="a", ylabel="b", zlabel="c")
        t = pv._vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                t.SetElement(i, j, cols[i, j])
        actor.SetUserMatrix(t)
        self.plotter.render()

    def _on_projection_changed(self):
        if self._parallel_box.isChecked():
            self.plotter.enable_parallel_projection()
        else:
            self.plotter.disable_parallel_projection()

    def _axis_view(self, col, up_col):
        """View along crystal reciprocal axis col with up = axis up_col.

        Uses T_crystal so directions are in the display world frame, not lab frame.
        """
        if self._preview_UB is None:
            return
        T = self._crystal_T()
        self.plotter.view_vector(
            T[:, col].tolist(), viewup=T[:, up_col].tolist()
        )

    def _real_axis_view(self, col):
        """View along real lattice axis a/b/c (cross-products of T_crystal columns).

        Matches NXV: a = b*×c*, b = c*×a*, c = a*×b*.  All in display world frame.
        """
        if self._preview_UB is None:
            return
        T = self._crystal_T()
        dir_pairs = [(1, 2), (2, 0), (0, 1)]
        up_pairs = [(2, 0), (0, 1), (1, 2)]
        da, db = dir_pairs[col]
        ua, ub_i = up_pairs[col]
        direction = np.cross(T[:, da], T[:, db])
        viewup = np.cross(T[:, ua], T[:, ub_i])
        n = np.linalg.norm(direction)
        vn = np.linalg.norm(viewup)
        if n < 1e-10 or vn < 1e-10:
            return
        self.plotter.view_vector(
            (direction / n).tolist(), viewup=(viewup / vn).tolist()
        )

    def _draw_box_with_grid(
        self, P, T, S, min_lim, max_lim, labels, label_limits=None
    ):
        """Add an oriented wireframe box and oblique-axis grid to the plotter.

        label_limits: optional (3,2) array of [vmin,vmax] per axis to use for
        tick labels when they differ from the display geometry (e.g. a centered
        parameter axis whose labels still show the real physical range).
        """
        box = pv.Box(
            bounds=[
                min_lim[0],
                max_lim[0],
                min_lim[1],
                max_lim[1],
                min_lim[2],
                max_lim[2],
            ]
        )
        M4 = np.eye(4, dtype=float)
        M4[:3, :3] = P
        box.transform(M4, inplace=True)
        self.plotter.add_mesh(
            box, style="wireframe", color="steelblue", line_width=2
        )
        center = P @ ((min_lim + max_lim) / 2.0)

        min_bnd = min_lim * S
        max_bnd = max_lim * S
        bounds = np.array([[min_bnd[i], max_bnd[i]] for i in range(3)])
        disp_limits = np.array([[min_lim[i], max_lim[i]] for i in range(3)])
        tick_limits = label_limits if label_limits is not None else disp_limits

        actor = self.plotter.show_grid(
            xtitle=labels[0],
            ytitle=labels[1],
            ztitle=labels[2],
            font_size=8,
            minor_ticks=True,
        )
        actor.SetAxisBaseForX(*T[:, 0])
        actor.SetAxisBaseForY(*T[:, 1])
        actor.SetAxisBaseForZ(*T[:, 2])
        actor.bounds = bounds.ravel()
        actor.SetXAxisRange(disp_limits[0])
        actor.SetYAxisRange(disp_limits[1])
        actor.SetZAxisRange(disp_limits[2])
        for ax_i, (vmin, vmax) in enumerate(tick_limits):
            n = [actor.n_xlabels, actor.n_ylabels, actor.n_zlabels][ax_i]
            fmt = [
                actor.x_label_format,
                actor.y_label_format,
                actor.z_label_format,
            ][ax_i]
            lbl = pv.plotting.cube_axes_actor.make_axis_labels(
                vmin=vmin, vmax=vmax, n=n, fmt=fmt
            )
            actor.SetAxisLabels(ax_i, lbl)
        return center

    def update_norm_preview(self, UB, W, extents, labels):
        self.plotter.clear()
        if UB is None or W is None:
            self.plotter.render()
            return
        self._preview_UB = np.asarray(UB, dtype=float)
        P, T, S = _compute_preview_transforms(UB, W)
        self._preview_T = T
        min_lim = np.array([e[0] if e else -5.0 for e in extents], dtype=float)
        max_lim = np.array([e[1] if e else 5.0 for e in extents], dtype=float)
        center = self._draw_box_with_grid(P, T, S, min_lim, max_lim, labels)
        self.plotter.reset_camera()
        self.plotter.set_focus(center.tolist())
        self._show_compass()
        self.plotter.render()

    def update_param_preview(
        self, UB, W, extents, bins, labels, log_extents, log_bins, log_name
    ):
        self.plotter.clear()
        if UB is None:
            self.plotter.render()
            return
        self._preview_UB = np.asarray(UB, dtype=float)

        n_dims = sum(1 for b in bins[:3] if b is not None and b > 1)

        if n_dims == 3:
            P, T, S = _compute_preview_transforms(UB, W)
            self._preview_T = T
            min_lim = np.array(
                [e[0] if e else -5.0 for e in extents[:3]], dtype=float
            )
            max_lim = np.array(
                [e[1] if e else 5.0 for e in extents[:3]], dtype=float
            )
            center = self._draw_box_with_grid(
                P, T, S, min_lim, max_lim, labels
            )
            self.plotter.reset_camera()
            self.plotter.set_focus(center.tolist())
            self._show_compass()

        elif n_dims == 2:
            P, T, S = _compute_preview_transforms(UB, W)
            binned = [
                i for i, b in enumerate(bins[:3]) if b is not None and b > 1
            ]
            ia, ib = binned[0], binned[1]

            u = P[:, ia]
            u_norm = np.linalg.norm(u)
            u_hat = u / u_norm if u_norm > 1e-10 else u

            # Gram-Schmidt: always computed for scale reference (v_norm)
            v_raw = P[:, ib]
            v_orth = v_raw - np.dot(v_raw, u_hat) * u_hat
            v_norm = np.linalg.norm(v_orth)
            v_hat = (
                v_orth / v_norm
                if v_norm > 1e-10
                else np.array([0.0, 1.0, 0.0])
            )

            if self._recip_box.isChecked():
                # Physical scattering-geometry frame:
                # ki is along [0,0,1] in the display Cartesian frame (beam direction;
                # QR/Cholesky always places the first projection axis along x so z is
                # a sensible beam direction for horizontal-scattering-plane geometry).
                # n_hat = ki × Q̂  →  normal to the scattering plane
                # p_hat = Q̂ × n_hat  →  in-plane perpendicular to Q
                ki_hat = np.array([0.0, 0.0, 1.0])
                cross = np.cross(ki_hat, u_hat)
                cross_n = np.linalg.norm(cross)
                if cross_n > 1e-6:
                    n_hat = cross / cross_n
                    p_hat = np.cross(u_hat, n_hat)
                else:
                    n_hat = v_hat
                    p_hat = np.cross(u_hat, n_hat)
            else:
                n_hat = v_hat
                p_hat = np.cross(u_hat, n_hat)

            ea = extents[ia] if extents[ia] else [-5.0, 5.0]
            eb = extents[ib] if extents[ib] else [-5.0, 5.0]
            ep = log_extents if log_extents else [0.0, 1.0]

            # Scale parameter axis so the box looks isotropic: match geometric
            # mean of the two spatial visual extents
            size_a = u_norm * abs(ea[1] - ea[0])
            size_b = v_norm * abs(eb[1] - eb[0])
            param_range = abs(ep[1] - ep[0])
            w_scale = (
                np.sqrt(size_a * size_b) / param_range
                if param_range > 1e-10
                else 1.0
            )

            P2 = np.column_stack([u, n_hat * v_norm, p_hat * w_scale])
            T2 = np.column_stack([u_hat, n_hat, p_hat])
            S2 = np.array([u_norm, v_norm, w_scale])
            self._preview_T = T2

            # Center the parameter axis at the origin for display; keep real
            # physical values (ep[0]..ep[1]) as tick labels
            ep_half = (ep[1] - ep[0]) / 2.0
            min_lim = np.array([ea[0], eb[0], -ep_half], dtype=float)
            max_lim = np.array([ea[1], eb[1], +ep_half], dtype=float)
            label_lim = np.array(
                [[ea[0], ea[1]], [eb[0], eb[1]], [ep[0], ep[1]]]
            )
            ax_labels = [labels[ia], labels[ib], log_name or "param"]
            center = self._draw_box_with_grid(
                P2, T2, S2, min_lim, max_lim, ax_labels, label_limits=label_lim
            )
            self.plotter.reset_camera()
            self.plotter.set_focus(center.tolist())
            self._show_compass()

        elif n_dims == 1:
            ia = next(
                i for i, b in enumerate(bins[:3]) if b is not None and b > 1
            )
            ea = extents[ia] if extents[ia] else [-5.0, 5.0]
            ep = log_extents if log_extents else [0.0, 1.0]

            chart = pv.Chart2D()
            rect_x = np.array([ea[0], ea[1], ea[1], ea[0], ea[0]])
            rect_y = np.array([ep[0], ep[0], ep[1], ep[1], ep[0]])
            chart.line(rect_x, rect_y, color="steelblue", width=2)
            chart.x_label = labels[ia] if labels else "dim"
            chart.y_label = log_name or "param"
            self.plotter.add_chart(chart)

        else:
            ep = log_extents if log_extents else [0.0, 1.0]
            chart = pv.Chart2D()
            chart.line(
                np.array([ep[0], ep[1]]),
                np.zeros(2),
                color="steelblue",
                width=2,
            )
            chart.x_label = log_name or "param"
            chart.y_label = "Intensity"
            self.plotter.add_chart(chart)

        self.plotter.render()

    def update_int_preview(
        self, UB, centering, d_min, mod_vecs, max_order, sat_min_d=None
    ):
        self.plotter.clear()
        if UB is None or d_min is None:
            self.plotter.render()
            return

        B = np.asarray(UB, dtype=float)
        q_max = 1.0 / float(d_min)

        col_norms = np.linalg.norm(B, axis=0)
        h_max, k_max, l_max = np.ceil(q_max / col_norms).astype(int) + 1

        h = np.arange(-h_max, h_max + 1)
        k = np.arange(-k_max, k_max + 1)
        l = np.arange(-l_max, l_max + 1)
        H, K, L = np.meshgrid(h, k, l, indexing="ij")
        hkl = np.stack([H.ravel(), K.ravel(), L.ravel()], axis=1)

        hkl = hkl[np.any(hkl != 0, axis=1)]
        q_norms = np.linalg.norm(hkl @ B.T, axis=1)
        hkl = hkl[q_norms <= q_max]
        hkl = hkl[_centering_mask(hkl, centering)]

        if len(hkl) == 0:
            self.plotter.render()
            return

        if len(hkl) > 5000:
            order = np.argsort(np.linalg.norm(hkl @ B.T, axis=1))
            hkl = hkl[order[:5000]]

        self._preview_UB = B
        P, T, S = _compute_preview_transforms(UB, np.eye(3))
        self._preview_T = T
        positions = (P @ hkl.astype(float).T).T

        peak_colors = ["steelblue", "tomato", "darkorange", "gold"]
        pd_main = pv.PolyData(positions)
        self.plotter.add_mesh(
            pd_main,
            color=peak_colors[0],
            point_size=8,
            render_points_as_spheres=True,
        )

        if max_order and max_order > 0:
            for mv_i, q_vec in enumerate(mod_vecs or []):
                if q_vec is None:
                    continue
                q = np.asarray(q_vec, dtype=float)
                if np.linalg.norm(q) < 1e-10:
                    continue
                sat_q_max = 1.0 / float(sat_min_d) if sat_min_d else q_max
                sat_parts = []
                for n in range(1, max_order + 1):
                    sat_parts.append(hkl.astype(float) + n * q)
                    sat_parts.append(hkl.astype(float) - n * q)
                sat_hkl = np.vstack(sat_parts)
                sat_q = np.linalg.norm(sat_hkl @ B.T, axis=1)
                sat_hkl = sat_hkl[sat_q <= sat_q_max]
                if len(sat_hkl) == 0:
                    continue
                if len(sat_hkl) > 5000:
                    sat_hkl = sat_hkl[:5000]
                sat_pos = (P @ sat_hkl.T).T
                sat_pd = pv.PolyData(sat_pos)
                self.plotter.add_mesh(
                    sat_pd,
                    color=peak_colors[(mv_i + 1) % len(peak_colors)],
                    point_size=5,
                    render_points_as_spheres=True,
                    opacity=0.7,
                )

        lims = [h_max, k_max, l_max]

        bounds_arr = np.array(
            [[-lims[i] * S[i], lims[i] * S[i]] for i in range(3)]
        )
        limits = np.array([[-lims[i], lims[i]] for i in range(3)], dtype=float)

        actor = self.plotter.show_grid(
            xtitle="[h,0,0]",
            ytitle="[k,0,0]",
            ztitle="[0,0,l]",
            font_size=8,
            minor_ticks=True,
        )
        actor.SetAxisBaseForX(*T[:, 0])
        actor.SetAxisBaseForY(*T[:, 1])
        actor.SetAxisBaseForZ(*T[:, 2])
        actor.bounds = bounds_arr.ravel()
        actor.SetXAxisRange(limits[0])
        actor.SetYAxisRange(limits[1])
        actor.SetZAxisRange(limits[2])
        for ax_i, (vmin, vmax) in enumerate(limits):
            n = [actor.n_xlabels, actor.n_ylabels, actor.n_zlabels][ax_i]
            fmt = [
                actor.x_label_format,
                actor.y_label_format,
                actor.z_label_format,
            ][ax_i]
            lbl = pv.plotting.cube_axes_actor.make_axis_labels(
                vmin=vmin, vmax=vmax, n=n, fmt=fmt
            )
            actor.SetAxisLabels(ax_i, lbl)

        self.plotter.reset_camera()
        self._show_compass()
        self.plotter.render()


class FormPresenter:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_wavelength(self.update_wavelength)
        self.view.connect_load_UB(self.load_UB)
        self.view.connect_load_mask(self.load_mask)
        self.view.connect_load_detector(self.load_detector)
        self.view.connect_load_goniometer(self.load_goniometer)
        self.view.connect_load_tube(self.load_tube)
        self.view.connect_load_background(self.load_background)
        self.view.connect_load_vanadium(self.load_vanadium)
        self.view.connect_load_flux(self.load_flux)
        self.view.connect_load_config(self.load_config)
        self.view.connect_save_config(self.save_config)
        self.view.connect_save_as_config(self.save_config_as)

        self.view.connect_int_run_button(self.run_integration)
        self.view.connect_param_run_button(self.run_parametrization)
        self.view.connect_norm_run_button(self.run_normalization)

        self.switch_instrument()

        self.view.connect_satellite_box(self.clear_satellite)

        self.view.connect_miller_box(self.clear_miller)

        self.view.connect_param_min_1_line(self.update_param_step_1)
        self.view.connect_param_min_2_line(self.update_param_step_2)
        self.view.connect_param_min_3_line(self.update_param_step_3)
        self.view.connect_param_min_4_line(self.update_param_step_4)

        self.view.connect_param_max_1_line(self.update_param_step_1)
        self.view.connect_param_max_2_line(self.update_param_step_2)
        self.view.connect_param_max_3_line(self.update_param_step_3)
        self.view.connect_param_max_4_line(self.update_param_step_4)

        self.view.connect_param_bins_1_line(self.update_param_step_1)
        self.view.connect_param_bins_2_line(self.update_param_step_2)
        self.view.connect_param_bins_3_line(self.update_param_step_3)
        self.view.connect_param_bins_4_line(self.update_param_step_4)
        self.view.connect_log_option_combo(self.update_log_name)

        self.view.connect_symmetry_combo(self.update_symmetry)

        self.view.connect_norm_min_1_line(self.update_norm_step_1)
        self.view.connect_norm_min_2_line(self.update_norm_step_2)
        self.view.connect_norm_min_3_line(self.update_norm_step_3)

        self.view.connect_norm_max_1_line(self.update_norm_step_1)
        self.view.connect_norm_max_2_line(self.update_norm_step_2)
        self.view.connect_norm_max_3_line(self.update_norm_step_3)

        self.view.connect_norm_bins_1_line(self.update_norm_step_1)
        self.view.connect_norm_bins_2_line(self.update_norm_step_2)
        self.view.connect_norm_bins_3_line(self.update_norm_step_3)

        self.view.connect_auto_binning(self.auto_bin)
        self.view.connect_auto_projection(self.auto_proj)

        self.view.connect_plan_tab_changed(self.update_preview)
        self.view.connect_preview_update(self.update_preview)

    def _projection_labels(self, W):
        if W is None:
            return ["P₁", "P₂", "P₃"]
        labels = []
        for row in W:
            vals = [
                int(v) if float(v) == int(float(v)) else round(float(v), 2)
                for v in row
            ]
            labels.append("[{},{},{}]".format(*vals))
        return labels

    def update_preview(self):
        tab = self.view.get_plan_tab_index()
        UB_file = self.view.get_UB()
        try:
            UB = self.model.load_UB_matrix(UB_file) if UB_file else None
        except Exception:
            UB = None

        if tab == 0:
            W = self.view.get_norm_projections()
            extents = [
                self.view.get_norm_limits_1(),
                self.view.get_norm_limits_2(),
                self.view.get_norm_limits_3(),
            ]
            labels = self._projection_labels(W)
            self.view.update_norm_preview(UB, W, extents, labels)
        elif tab == 1:
            W = self.view.get_param_projections()
            extents = [
                self.view.get_param_limits_1(),
                self.view.get_param_limits_2(),
                self.view.get_param_limits_3(),
            ]
            bins = [
                self.view.get_param_bins_1(),
                self.view.get_param_bins_2(),
                self.view.get_param_bins_3(),
            ]
            log_extents = self.view.get_param_limits_4()
            log_bins = self.view.get_param_bins_4()
            log_name = self.view.get_log_name()
            labels = self._projection_labels(W)
            self.view.update_param_preview(
                UB, W, extents, bins, labels, log_extents, log_bins, log_name
            )
        else:
            centering = self.view.get_centering()
            d_min = self.view.get_min_d()
            mod_vecs = [
                self.view.get_mod_vec_1(),
                self.view.get_mod_vec_2(),
                self.view.get_mod_vec_3(),
            ]
            max_order = self.view.get_max_order()
            sat_min_d = self.view.get_sat_min_d()
            self.view.update_int_preview(
                UB, centering, d_min, mod_vecs, max_order, sat_min_d
            )

    def run_integration(self):
        self.run_command("i")

    def run_parametrization(self):
        self.run_command("p")

    def run_normalization(self):
        self.run_command("n")

    def run_command(self, arg):
        filename = self.view.get_config()
        self.save_config()
        filename = self.view.get_config()
        proc = self.view.get_processes()
        if proc is not None and filename is not None:
            dev = self.view.get_development()
            dev_flag = "-d " if dev else ""
            command = self.model.command.format(dev_flag, arg, filename, proc)
            self.view.run_command(command)

    def clear_satellite(self):
        self.view.clear_satellite()

    def clear_miller(self):
        self.view.clear_miller()

    def update_param_step_1(self):
        limits = self.view.get_param_limits_1()
        bins = self.view.get_param_bins_1()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_param_step_1(step)

    def update_param_step_2(self):
        limits = self.view.get_param_limits_2()
        bins = self.view.get_param_bins_2()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_param_step_2(step)

    def update_param_step_3(self):
        limits = self.view.get_param_limits_3()
        bins = self.view.get_param_bins_3()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_param_step_3(step)

    def update_param_step_4(self):
        limits = self.view.get_param_limits_4()
        bins = self.view.get_param_bins_4()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_param_step_4(step)

    def update_log_name(self):
        self.view.set_log_name(self.view.get_log_option())

    def auto_proj(self):
        UB_file = self.view.get_UB()
        if UB_file is not None:
            UB = self.model.load_UB_matrix(UB_file)
            W = self.model.autoproj(UB)
            self.view.set_norm_projections(W)

    def auto_bin(self):
        UB_file = self.view.get_UB()
        if UB_file is not None:
            UB = self.model.load_UB_matrix(UB_file)
            W = self.view.get_norm_projections()
            d_min = self.view.get_norm_resolution_min()
            bins = self.model.autolim(UB, W, d_min)
            X_max, Y_max, Z_max, X_bins, Y_bins, Z_bins = bins
            self.view.set_norm_bins_1(X_bins)
            self.view.set_norm_bins_2(Y_bins)
            self.view.set_norm_bins_3(Z_bins)
            self.view.set_norm_limits_1([-X_max, X_max])
            self.view.set_norm_limits_2([-Y_max, Y_max])
            self.view.set_norm_limits_3([-Z_max, Z_max])
            self.update_norm_step_1()
            self.update_norm_step_2()
            self.update_norm_step_3()

    def update_symmetry(self):
        symmetry = self.view.get_symmetry()
        if symmetry == "None":
            symmetries = []
        elif symmetry == "Point Group":
            symmetries = point_laue.keys()
        else:
            keys = list(space_point.keys())
            nos = list(space_number.values())
            symmetries = [x for _, x in sorted(zip(nos, keys))]
        self.view.set_symmetry_options(symmetries)

    def find_symmetry(self, option):
        if option is None:
            symmetry, symmetries = "None", []
        elif option in list(point_laue.keys()):
            symmetry, symmetries = "Point Group", point_laue.keys()
        elif option in list(space_point.keys()):
            symmetry, symmetries = "Space Group", space_point.keys()
        else:
            option, symmetry, symmetries = None, "None", []

        self.view.set_symmetry(symmetry)
        self.view.set_symmetry_options(symmetries)
        if option is not None:
            self.view.set_symmetry_option(option)

    def update_norm_step_1(self):
        limits = self.view.get_norm_limits_1()
        bins = self.view.get_norm_bins_1()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_norm_step_1(step)

    def update_norm_step_2(self):
        limits = self.view.get_norm_limits_2()
        bins = self.view.get_norm_bins_2()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_norm_step_2(step)

    def update_norm_step_3(self):
        limits = self.view.get_norm_limits_3()
        bins = self.view.get_norm_bins_3()
        if limits is not None and bins is not None:
            step = self.model.calculate_step(*limits, bins)
            self.view.set_norm_step_3(step)

    def switch_instrument(self):
        instrument = self.view.get_instrument()
        self.model.set_instrument(instrument)

        wavelength = self.model.get_wavelength()
        self.view.set_wavelength(wavelength)

        groupings = self.model.get_groupings()
        self.view.set_groupings(groupings)

        logs = self.model.get_logs()
        self.view.set_log_options(logs)

        grouping = self.model.get_grouping()
        if grouping is not None:
            self.view.set_grouping(grouping)

        filepath = self.model.get_raw_file_path()
        self.view.clear_run_info(filepath)

        cpus = self.model.get_processes()
        self.view.set_processes(cpus)

    def update_wavelength(self):
        wl_min, wl_max = self.view.get_wavelength()
        self.view.update_wavelength(wl_min)

    def load_UB(self):
        ipts = self.view.get_IPTS()
        path = self.model.get_shared_file_path(ipts)
        filename = self.view.load_UB_file_dialog(path)

        if filename:
            self.view.set_UB(filename)
            self.update_preview()

    def load_mask(self):
        ipts = self.view.get_IPTS()
        path = self.model.get_shared_file_path(ipts)
        filename = self.view.load_mask_file_dialog(path)

        if filename:
            self.view.set_mask(filename)

    def load_detector(self):
        path = self.model.get_calibration_file_path()
        filename = self.view.load_detector_cal_dialog(path)

        if filename:
            self.view.set_detector_calibration(filename)

    def load_goniometer(self):
        path = self.model.get_goniometer_file_path()
        filename = self.view.load_goniometer_cal_dialog(path)

        if filename:
            self.view.set_goniometer_calibration(filename)

    def load_tube(self):
        path = self.model.get_calibration_file_path()
        filename = self.view.load_tube_cal_dialog(path)

        if filename:
            self.view.set_tube_calibration(filename)

    def load_background(self):
        path = self.model.get_vanadium_file_path()
        filename = self.view.load_background_file_dialog(path)

        if filename:
            self.view.set_background(filename)

    def load_vanadium(self):
        path = self.model.get_vanadium_file_path()
        filename = self.view.load_vanadium_file_dialog(path)

        if filename:
            self.view.set_vanadium(filename)

    def load_flux(self):
        path = self.model.get_vanadium_file_path()
        filename = self.view.load_flux_file_dialog(path)

        if filename:
            self.view.set_flux(filename)

    def load_config(self):
        ipts = self.view.get_IPTS()
        path = self.model.get_shared_file_path(ipts)
        filename = self.view.load_config_file_dialog(path)

        if filename:
            self.view.set_config(filename)
            self.model.load_config(filename)
            inst = self.model.get_instrument()
            self.view.set_instrument(inst)

            self.switch_instrument()
            self.model.load_config(filename)

            el = self.model.get_elastic()
            if el is not None:
                self.view.set_elastic(el)

            exp = self.model.get_experiment()
            if exp is not None:
                self.view.set_experiment(exp)

            IPTS = self.model.get_IPTS()
            if IPTS is not None:
                self.view.set_IPTS(IPTS)

            runs = self.model.get_runs()
            if runs is not None:
                self.view.set_runs(runs)

            UB = self.model.get_UB()
            if UB is not None:
                self.view.set_UB(UB)

            mask = self.model.get_mask()
            if mask is not None:
                self.view.set_mask(mask)

            background = self.model.get_background()
            if background is not None:
                self.view.set_background(background)

            detector = self.model.get_detector_calibration()
            if detector is not None:
                self.view.set_detector_calibration(detector)

            gonio = self.model.get_goniometer_calibration()
            if gonio is not None:
                self.view.set_goniometer_calibration(gonio)

            tube = self.model.get_tube_calibration()
            if tube is not None:
                self.view.set_tube_calibration(tube)

            flux = self.model.get_flux()
            if flux is not None:
                self.view.set_flux(flux)

            van = self.model.get_vanadium()
            if van is not None:
                self.view.set_vanadium(van)

            grouping = self.model.get_grouping()
            if grouping is not None:
                self.view.set_grouping(grouping)

            self.load_int()
            self.load_param()
            self.load_norm()

    def save_config(self):
        filename = self.view.get_config()

        valid = self.model.validate_file(filename)

        if not valid:
            ipts = self.view.get_IPTS()
            path = self.model.get_shared_file_path(ipts)
            filename = self.view.save_config_file_dialog(path)

            valid = self.model.validate_file(filename)

        if filename and valid:
            self.view.set_config(filename)

            lamda = self.view.get_wavelength()
            if lamda is not None:
                self.model.set_wavelength(lamda)

            el = self.view.get_elastic()
            if el is not None:
                self.model.set_elastic(el)

            exp = self.view.get_experiment()
            if exp is not None:
                self.model.set_experiment(exp)

            IPTS = self.view.get_IPTS()
            if IPTS is not None:
                self.model.set_IPTS(IPTS)

            runs = self.view.get_runs()
            if runs is not None:
                self.model.set_runs(runs)

            UB = self.view.get_UB()
            if UB is not None:
                self.model.set_UB(UB)

            mask = self.view.get_mask()
            if mask is not None:
                self.model.set_mask(mask)

            background = self.view.get_background()
            if background is not None:
                self.model.set_background(background)

            detector = self.view.get_detector_calibration()
            if detector is not None:
                self.model.set_detector_calibration(detector)

            gonio = self.view.get_goniometer_calibration()
            if gonio is not None:
                self.model.set_goniometer_calibration(gonio)

            tube = self.view.get_tube_calibration()
            if tube is not None:
                self.model.set_tube_calibration(tube)

            flux = self.view.get_flux()
            if flux is not None:
                self.model.set_flux(flux)

            van = self.view.get_vanadium()
            if van is not None:
                self.model.set_vanadium(van)

            grouping = self.view.get_grouping()
            if grouping is not None:
                self.model.set_grouping(grouping)

            self.save_int()
            self.save_param()
            self.save_norm()

            self.model.save_config(filename)

    def save_config_as(self):
        ipts = self.view.get_IPTS()
        path = self.model.get_shared_file_path(ipts)
        filename = self.view.save_config_file_dialog(path)

        valid = self.model.validate_file(filename)

        if filename and valid:
            self.view.set_config(filename)
            self.save_config()

    def load_int(self):
        params = self.model.get_int()
        if params is not None:
            (
                cell,
                centering,
                mod_vec_1,
                mod_vec_2,
                mod_vec_3,
                max_order,
                cross_terms,
                min_d,
                sat_min_d,
                radius,
            ) = params
            self.view.set_satellite(max_order > 0)
            self.view.clear_satellite()
            self.view.set_centering(centering)
            self.view.set_cell(cell)
            self.view.set_mod_vec_1(mod_vec_1)
            self.view.set_mod_vec_2(mod_vec_2)
            self.view.set_mod_vec_3(mod_vec_3)
            self.view.set_max_order(max_order)
            self.view.set_cross_terms(cross_terms)
            self.view.set_min_d(min_d)
            self.view.set_sat_min_d(sat_min_d)
            self.view.set_radius(radius)

    def save_int(self):
        centering = self.view.get_centering()
        cell = self.view.get_cell()
        mod_vec_1 = self.view.get_mod_vec_1()
        mod_vec_2 = self.view.get_mod_vec_2()
        mod_vec_3 = self.view.get_mod_vec_3()
        max_order = self.view.get_max_order()
        cross_terms = self.view.get_cross_terms()
        min_d = self.view.get_min_d()
        sat_min_d = self.view.get_sat_min_d()
        radius = self.view.get_radius()
        self.model.set_int(
            cell,
            centering,
            mod_vec_1,
            mod_vec_2,
            mod_vec_3,
            max_order,
            cross_terms,
            min_d,
            sat_min_d,
            radius,
        )

    def load_param(self):
        params = self.model.get_param()
        if params is not None:
            log, log_extents, log_bins, hkl, proj, extents, bins = params
            self.view.set_log_name(log)
            self.view.set_param_projections(proj)
            self.view.set_param_bins_1(bins[0])
            self.view.set_param_bins_2(bins[1])
            self.view.set_param_bins_3(bins[2])
            self.view.set_param_bins_4(log_bins)
            self.view.set_param_limits_1(extents[0])
            self.view.set_param_limits_2(extents[1])
            self.view.set_param_limits_3(extents[2])
            self.view.set_param_limits_4(log_extents)
            self.update_param_step_1()
            self.update_param_step_2()
            self.update_param_step_3()

    def save_param(self):
        log = self.view.get_log_name()
        proj = self.view.get_param_projections()
        bins_1 = self.view.get_param_bins_1()
        bins_2 = self.view.get_param_bins_2()
        bins_3 = self.view.get_param_bins_3()
        log_bins = self.view.get_param_bins_4()
        extents_1 = self.view.get_param_limits_1()
        extents_2 = self.view.get_param_limits_2()
        extents_3 = self.view.get_param_limits_3()
        log_extents = self.view.get_param_limits_4()
        hkl = self.view.get_miller_index()
        bins = [bins_1, bins_2, bins_3]
        extents = [extents_1, extents_2, extents_3]
        self.model.set_param(
            log, log_extents, log_bins, hkl, proj, extents, bins
        )

    def load_norm(self):
        params = self.model.get_norm()
        if params is not None:
            symmetry, proj, extents, bins = params
            self.find_symmetry(symmetry)
            self.view.set_norm_projections(proj)
            self.view.set_norm_bins_1(bins[0])
            self.view.set_norm_bins_2(bins[1])
            self.view.set_norm_bins_3(bins[2])
            self.view.set_norm_limits_1(extents[0])
            self.view.set_norm_limits_2(extents[1])
            self.view.set_norm_limits_3(extents[2])
            self.update_norm_step_1()
            self.update_norm_step_2()
            self.update_norm_step_3()

    def save_norm(self):
        symmetry = self.view.get_symmetry_option()
        proj = self.view.get_norm_projections()
        bins_1 = self.view.get_norm_bins_1()
        bins_2 = self.view.get_norm_bins_2()
        bins_3 = self.view.get_norm_bins_3()
        extents_1 = self.view.get_norm_limits_1()
        extents_2 = self.view.get_norm_limits_2()
        extents_3 = self.view.get_norm_limits_3()
        bins = [bins_1, bins_2, bins_3]
        extents = [extents_1, extents_2, extents_3]
        self.model.set_norm(symmetry, proj, extents, bins)


class FormModel:
    def __init__(self):
        self.command = "/SNS/software/scd/reduce.sh {}-{} {} {}"
        self.reduction = ReductionPlan()

    def validate_file(self, filename, ext=".yaml"):
        path = os.path.dirname(filename)
        file = os.path.basename(filename)
        return os.path.isdir(path) and os.path.splitext(file)[1] == ext

    def calculate_step(self, vmin, vmax, bins):
        return (vmax - vmin) / (bins - 1) if bins > 1 else (vmax - vmin)

    def get_int(self):
        if self.reduction.plan is not None:
            params = self.reduction.plan.get("Integration")
            if params is not None:
                cell = params["Cell"]
                centering = params["Centering"]
                mod_vec_1 = params["ModVec1"]
                mod_vec_2 = params["ModVec2"]
                mod_vec_3 = params["ModVec3"]
                max_order = params["MaxOrder"]
                cross_terms = params["CrossTerms"]
                min_d = params["MinD"]
                sat_min_d = params.get("SatMinD")
                radius = params["Radius"]
                return (
                    cell,
                    centering,
                    mod_vec_1,
                    mod_vec_2,
                    mod_vec_3,
                    max_order,
                    cross_terms,
                    min_d,
                    sat_min_d,
                    radius,
                )

    def set_int(
        self,
        cell,
        centering,
        mod_vec_1,
        mod_vec_2,
        mod_vec_3,
        max_order,
        cross_terms,
        min_d,
        sat_min_d,
        radius,
    ):
        if self.reduction.plan is not None:
            params = {}
            params["Cell"] = cell
            params["Centering"] = centering
            params["ModVec1"] = mod_vec_1
            params["ModVec2"] = mod_vec_2
            params["ModVec3"] = mod_vec_3
            params["MaxOrder"] = max_order
            params["CrossTerms"] = cross_terms
            params["MinD"] = min_d
            params["SatMinD"] = sat_min_d
            params["Radius"] = radius
            self.reduction.plan["Integration"] = params

    def get_param(self):
        if self.reduction.plan is not None:
            params = self.reduction.plan.get("Parametrization")
            if params is not None:
                log = params["LogName"]
                log_extents = params["LogExtents"]
                log_bins = params["LogBins"]
                hkl = params.get("MillerIndex")
                proj = params["Projections"]
                extents = params["Extents"]
                bins = params["Bins"]
                return log, log_extents, log_bins, hkl, proj, extents, bins

    def set_param(self, log, log_extents, log_bins, hkl, proj, extents, bins):
        if self.reduction.plan is not None:
            params = {}
            params["LogName"] = log
            params["LogExtents"] = log_extents
            params["LogBins"] = log_bins
            params["MillerIndex"] = hkl
            params["Projections"] = proj
            params["Extents"] = extents
            params["Bins"] = bins
            self.reduction.plan["Parametrization"] = params

    def get_norm(self):
        if self.reduction.plan is not None:
            params = self.reduction.plan.get("Normalization")
            if params is not None:
                symmetry = params["Symmetry"]
                proj = params["Projections"]
                extents = params["Extents"]
                bins = params["Bins"]
                return symmetry, proj, extents, bins

    def set_norm(self, symmetry, proj, extents, bins):
        if self.reduction.plan is not None:
            params = {}
            params["Symmetry"] = symmetry
            params["Projections"] = proj
            params["Extents"] = extents
            params["Bins"] = bins
            self.reduction.plan["Normalization"] = params

    def get_vanadium(self):
        if self.reduction.plan is not None:
            van = self.reduction.plan.get("VanadiumFile")
            return van

    def set_vanadium(self, van):
        van = None if van == "" else van
        if self.reduction.plan is not None:
            self.reduction.plan.pop("VanadiumFile", None)
            if van is not None:
                self.reduction.plan["VanadiumFile"] = van

    def get_flux(self):
        if self.reduction.plan is not None:
            flux = self.reduction.plan.get("FluxFile")
            return flux

    def set_flux(self, flux):
        flux = None if flux == "" else flux
        if self.reduction.plan is not None:
            self.reduction.plan.pop("FluxFile", None)
            if flux is not None:
                self.reduction.plan["FluxFile"] = flux

    def get_detector_calibration(self):
        if self.reduction.plan is not None:
            cal = self.reduction.plan.get("DetectorCalibration")
            return cal

    def set_detector_calibration(self, cal):
        cal = None if cal == "" else cal
        if self.reduction.plan is not None:
            self.reduction.plan.pop("DetectorCalibration", None)
            if cal is not None:
                self.reduction.plan["DetectorCalibration"] = cal

    def get_goniometer_calibration(self):
        if self.reduction.plan is not None:
            cal = self.reduction.plan.get("GoniometerCalibration")
            return cal

    def set_goniometer_calibration(self, cal):
        cal = None if cal == "" else cal
        if self.reduction.plan is not None:
            self.reduction.plan.pop("GoniometerCalibration", None)
            if cal is not None:
                self.reduction.plan["GoniometerCalibration"] = cal

    def get_tube_calibration(self):
        if self.reduction.plan is not None:
            cal = self.reduction.plan.get("TubeCalibration")
            return cal

    def set_tube_calibration(self, cal):
        cal = None if cal == "" else cal
        if self.reduction.plan is not None:
            self.reduction.plan.pop("TubeCalibration", None)
            if cal is not None:
                self.reduction.plan["TubeCalibration"] = cal

    def get_background(self):
        if self.reduction.plan is not None:
            mask = self.reduction.plan.get("BackgroundFile")
            return mask

    def set_background(self, background):
        background = None if background == "" else background
        if self.reduction.plan is not None:
            self.reduction.plan.pop("BackgroundFile", None)
            if background is not None:
                self.reduction.plan["BackgroundFile"] = background

    def get_mask(self):
        if self.reduction.plan is not None:
            mask = self.reduction.plan.get("MaskFile")
            return mask

    def set_mask(self, mask):
        mask = None if mask == "" else mask
        if self.reduction.plan is not None:
            self.reduction.plan.pop("MaskFile", None)
            if mask is not None:
                self.reduction.plan["MaskFile"] = mask

    def get_UB(self):
        if self.reduction.plan is not None:
            UB = self.reduction.plan.get("UBFile")
            return UB

    def set_UB(self, UB):
        UB = None if UB == "" else UB
        if self.reduction.plan is not None:
            self.reduction.plan["UBFile"] = UB

    def get_runs(self):
        if self.reduction.plan is not None:
            runs = self.reduction.plan.get("Runs")
            if type(runs) is list:
                runs = self.reduction.runs_list_to_string(runs)
            return runs

    def set_runs(self, runs):
        runs = None if runs == "" else runs
        if self.reduction.plan is not None:
            self.reduction.plan["Runs"] = runs

    def get_elastic(self):
        if self.reduction.plan is not None:
            el = self.reduction.plan.get("Elastic")
            return el

    def set_elastic(self, el):
        if self.reduction.plan is not None:
            self.reduction.plan["Elastic"] = el

    def get_experiment(self):
        if self.reduction.plan is not None:
            exp = self.reduction.plan.get("Experiment")
            return exp

    def set_experiment(self, exp):
        exp = None if exp == "" else exp
        if self.reduction.plan is not None:
            self.reduction.plan["Experiment"] = exp

    def get_IPTS(self):
        if self.reduction.plan is not None:
            IPTS = self.reduction.plan.get("IPTS")
            return IPTS

    def set_IPTS(self, IPTS):
        IPTS = None if IPTS == "" else IPTS
        if self.reduction.plan is not None:
            self.reduction.plan["IPTS"] = IPTS

    def load_config(self, filename):
        self.reduction.load_plan(filename)
        self.reduction.plan.pop("OutputPath", None)
        self.reduction.plan.pop("OutputName", None)

    def save_config(self, filename):
        self.reduction.save_plan(filename, False)

    def get_instrument(self):
        if self.reduction.plan is not None:
            return self.reduction.plan.get("Instrument")

    def set_instrument(self, instrument):
        self.beamline = beamlines[instrument]
        self.reduction.generate_plan(instrument)

    def get_processes(self):
        return self.beamline["Processes"]

    def get_wavelength(self):
        wl = self.beamline["Wavelength"]
        if self.reduction is not None:
            lamda = self.reduction.plan.get("Wavelength")
            if lamda is not None:
                wl = lamda
        return wl

    def set_wavelength(self, lamda):
        if self.reduction is not None:
            if abs(lamda[1] - lamda[0]) < 0.001:
                self.reduction.plan["Wavelength"] = lamda[0]
            else:
                self.reduction.plan["Wavelength"] = lamda

    def get_groupings(self):
        return self.beamline["Groupings"]

    def get_logs(self):
        logs = self.beamline.get("Logs", [])
        return list(dict.fromkeys(logs))

    def get_grouping(self):
        if self.reduction.plan is not None:
            return self.reduction.plan.get("Grouping")

    def set_grouping(self, grouping):
        grouping = None if grouping == "" else grouping
        if self.reduction.plan is not None:
            self.reduction.plan["Grouping"] = grouping

    def get_raw_file_path(self):
        return os.path.join(
            "/",
            self.beamline["Facility"],
            self.beamline["InstrumentName"],
            "IPTS-{}",
            self.beamline["RawFile"],
        )

    def get_shared_file_path(self, ipts):
        if ipts is not None:
            filepath = os.path.join(
                "/",
                self.beamline["Facility"],
                self.beamline["InstrumentName"],
                "IPTS-{}".format(ipts),
                "shared",
            )
            if os.path.exists(filepath):
                return filepath

        filepath = os.path.join(
            "/", self.beamline["Facility"], self.beamline["InstrumentName"]
        )

        return filepath

    def get_calibration_file_path(self):
        return os.path.join(
            "/",
            self.beamline["Facility"],
            self.beamline["InstrumentName"],
            "shared",
            "calibration",
        )

    def get_goniometer_file_path(self):
        return os.path.join(
            "/",
            self.beamline["Facility"],
            self.beamline["InstrumentName"],
            "shared",
            "calibration",
        )

    def get_vanadium_file_path(self):
        return os.path.join(
            "/",
            self.beamline["Facility"],
            self.beamline["InstrumentName"],
            "shared",
            "Vanadium",
        )

    def load_UB_matrix(self, filename, beam=2, up=1, back=0):
        UB = np.zeros((3, 3), dtype=float)

        with open(filename, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if len(lines) < 3:
            raise ValueError(
                f"Expected at least 3 non-empty lines, got {len(lines)}"
            )

        for basis in range(3):
            parts = lines[basis].split()
            if len(parts) < 3:
                raise ValueError(
                    f"Line {basis+1} has less than 3 columns: {lines[basis]!r}"
                )

            b, bk, u = map(float, parts[:3])

            UB[beam, basis] = b
            UB[back, basis] = bk
            UB[up, basis] = u

        return UB

    def _unit_vector(self, v, tol=1e-12):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < tol:
            raise ValueError("Cannot normalize near-zero vector.")
        return v / n

    def autoproj(self, UB, target_axis=(0, 1, 0)):
        """
        Choose the projection whose plane normal is most aligned with target_axis
        in Cartesian/lab reciprocal space.

        Returns:
            [u, v, n] where u and v span the plane and n is the plane normal in hkl.
        """
        UB = np.asarray(UB, dtype=float)
        target_axis = self._unit_vector(target_axis)

        projections = {
            "hk0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "h0l": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            "0kl": [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
            "hhl": [[1, 1, 0], [0, 0, 1], [-1, 1, 0]],
            "-hhl": [[-1, 1, 0], [0, 0, 1], [1, 1, 0]],
        }

        best_key = None
        best_similarity = -np.inf

        for key, basis in projections.items():
            n_hkl = self._unit_vector(basis[2])

            n_cart = UB @ n_hkl
            n_cart = self._unit_vector(n_cart)

            similarity = abs(np.dot(n_cart, target_axis))
            if similarity > best_similarity:
                best_key = key
                best_similarity = similarity

        return projections[best_key]

    def autolim(self, UB, W, d_min, n=400):
        """
        Robust limits for the transformed cube q in [-s_max, s_max]^3.
        """
        UB = np.asarray(UB, dtype=float)
        W = np.column_stack(W)

        s_max = 1.0 / float(d_min) / np.sqrt(3.0)
        T = np.linalg.inv(UB @ W)

        corners = np.array(
            list(
                itertools.product(
                    [-s_max, s_max], [-s_max, s_max], [-s_max, s_max]
                )
            )
        )

        proj_coords = (T @ corners.T).T
        limits = np.ceil(np.max(np.abs(proj_coords), axis=0)).astype(int)

        x_max, y_max, z_max = np.maximum(limits, 1)

        return x_max, y_max, z_max, n + 1, n + 1, n // 4 + 1


class Garnet(QMainWindow):
    __instance = None

    def __new__(cls):
        if Garnet.__instance is None:
            Garnet.__instance = QMainWindow.__new__(cls)
        return Garnet.__instance

    def __init__(self, parent=None):
        super().__init__(parent)

        icon = os.path.join(os.path.dirname(__file__), "icons/garnet.png")
        self.setWindowIcon(QIcon(icon))
        self.setWindowTitle("garnet {}".format(__version__))

        main_window = QWidget(self)
        self.setCentralWidget(main_window)

        layout = QVBoxLayout(main_window)

        view = FormView()
        model = FormModel()
        self.form = FormPresenter(view, model)
        layout.addWidget(view)


def handle_exception(exc_type, exc_value, exc_traceback):
    error_message = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    msg_box = QMessageBox()
    msg_box.setWindowTitle("Application Error")
    msg_box.setText("An unexpected error occurred. Please see details below:")
    msg_box.setDetailedText(error_message)
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.exec()


def gui():
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
    window = Garnet()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui()
