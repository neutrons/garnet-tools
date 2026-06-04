import os
import sys
import traceback
import csv
import re

import numpy as np

from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QListWidget,
    QHBoxLayout,
    QVBoxLayout,
    QComboBox,
    QAbstractItemView,
    QFileDialog,
)
from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

try:
    from qdarkstyle.light.palette import LightPalette
    import qdarkstyle

    style = True
except:
    qdarkstyle = None
    style = False

try:
    import qtawesome as qta
except:
    qta = None

import pyoncat

directory = os.path.dirname(os.path.realpath(__file__))
directory = os.path.abspath(os.path.join(directory, "../.."))
sys.path.append(directory)

from garnet.config.instruments import beamlines


class View(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        login_label = QLabel("ONCat Login")
        user_label = QLabel("Username: ")
        pass_label = QLabel("Password: ")
        self.user_line = QLineEdit()
        self.pass_line = QLineEdit()
        self.pass_line.setEchoMode(QLineEdit.Password)
        self.login_button = QPushButton("Sign In")
        self.refresh_button = QPushButton("Refresh")
        self.report_button = QPushButton("Export CSV")
        self.message_label = QLabel("Not Signed In")
        self.message_label.setStyleSheet("color: red;")
        self.message_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        login_layout = QHBoxLayout()
        login_layout.addWidget(login_label)
        login_layout.addWidget(user_label)
        login_layout.addWidget(self.user_line, 2)
        login_layout.addWidget(pass_label)
        login_layout.addWidget(self.pass_line)
        login_layout.addWidget(self.login_button)
        login_layout.addWidget(self.message_label, 2)
        self.layout.addLayout(login_layout)

        instrument_cbox_label = QLabel("Instrument: ")
        self.instrument_cbox = QComboBox(self)
        instruments = ["TOPAZ", "MANDI", "CORELLI", "SNAP", "WAND²", "DEMAND"]
        self.instrument_cbox.addItems(instruments)
        self.auto_scale_dropdown(self.instrument_cbox)

        ipts_label = QLabel("IPTS: ")
        self.ipts_field = QComboBox(self)
        self.auto_scale_dropdown(self.ipts_field)

        self.name_list = QListWidget()
        self.name_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        self.runs_label = QLabel("Run Numbers: ")
        self.runs_list = QLineEdit()

        self.exp_cbox = QComboBox(self)
        self.exp_cbox.setEnabled(False)
        self.auto_scale_dropdown(self.exp_cbox)

        ipts_layout = QHBoxLayout()
        ipts_layout.addWidget(instrument_cbox_label)
        ipts_layout.addWidget(self.instrument_cbox)
        ipts_layout.addWidget(ipts_label)
        ipts_layout.addWidget(self.ipts_field, 1)
        ipts_layout.addWidget(self.exp_cbox)
        self.layout.addLayout(ipts_layout)

        self.plot = FigureCanvas(Figure(figsize=(8, 6)))
        self.toolbar = NavigationToolbar(self.plot, self)
        self._run_guides = []
        self.plot.mpl_connect("button_press_event", self._handle_plot_click)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.plot, 1)
        right_layout.addWidget(self.toolbar)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.name_list, 1)
        content_layout.addLayout(right_layout, 2)
        self.layout.addLayout(content_layout, 1)

        runs_layout = QHBoxLayout()
        runs_layout.addWidget(self.runs_label)
        runs_layout.addWidget(self.runs_list, 1)
        runs_layout.addWidget(self.report_button)
        runs_layout.addWidget(self.refresh_button)
        self.layout.addLayout(runs_layout)

    def ipts_entered(self):
        if self.ipts_field.hasAcceptableInput():
            self.update()

    def get_instrument(self):
        return self.instrument_cbox.currentText()

    def get_name(self):
        return self.name_list.selectedItems()

    def get_ipts(self):
        return self.ipts_field.currentText()

    def get_runs(self):
        return self.runs_list.text()

    def get_experiment(self):
        return self.exp_cbox.currentText()

    def connect_ipts(self, update):
        self.ipts_field.activated.connect(update)

    def connect_switch_instrument(self, update):
        self.instrument_cbox.activated.connect(update)

    def connect_select_name(self, update):
        self.name_list.itemSelectionChanged.connect(update)
        self.name_list.itemClicked.connect(update)

    def connect_adjust_runs(self, update):
        self.runs_list.editingFinished.connect(update)

    def connect_login_button(self, update):
        self.login_button.clicked.connect(update)
        self.pass_line.returnPressed.connect(update)

    def connect_select_experiment(self, update):
        self.exp_cbox.activated.connect(update)

    def connect_refresh_button(self, update):
        self.refresh_button.clicked.connect(update)

    def connect_report_button(self, update):
        self.report_button.clicked.connect(update)

    def save_report_file_dialog(self, path=""):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        default_file = (
            os.path.join(path, "experiment_report.csv") if path else ""
        )

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export experiment report",
            default_file,
            "CSV files (*.csv)",
            options=options,
        )

        if filename and not filename.lower().endswith(".csv"):
            filename += ".csv"

        return filename

    def plot_goniometer(
        self,
        gonio_values,
        gonio_names,
        run_numbers_list,
        scale_values,
        subplot_limits,
        inst_params,
        log_values=None,
        log_names=None,
    ):
        self.plot.figure.clf()
        self._run_guides = []

        if self.get_ipts() == "":
            return

        if log_values is None:
            log_values = []
        if log_names is None:
            log_names = []

        (
            temp_values,
            temp_names,
            other_values,
            other_names,
        ) = self._split_temperature_logs(log_values, log_names)

        has_logs = len(log_values) > 0

        self.plot.figure.subplots_adjust(wspace=0.1, top=0.85, bottom=0.1)
        colors = self._series_colors(len(gonio_values))

        if len(subplot_limits) == 1:
            if has_logs:
                ax1, ax_log = self.plot.figure.subplots(
                    2,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 2]},
                )
            else:
                ax1 = self.plot.figure.subplots()
                ax_log = None

            if self.get_instrument() != "DEMAND":
                for val, lab, c in zip(gonio_values, gonio_names, colors):
                    ax1.plot(run_numbers_list, val, ".", color=c, label=lab)
            else:
                for val, lab, c in zip(gonio_values, gonio_names, colors):
                    v = val[:]
                    for ii in range(len(run_numbers_list)):
                        if ii == 0:
                            ax1.errorbar(
                                run_numbers_list[ii],
                                v[0][ii],
                                yerr=np.array(
                                    [
                                        abs(v[1][ii] - v[0][ii]),
                                        abs(v[2][ii] - v[0][ii]),
                                    ]
                                ),
                                fmt=".",
                                color=c,
                                label=lab,
                                elinewidth=0.5,
                                capsize=2,
                            )
                        else:
                            ax1.errorbar(
                                run_numbers_list[ii],
                                v[0][ii],
                                yerr=np.array(
                                    [
                                        abs(v[1][ii] - v[0][ii]),
                                        abs(v[2][ii] - v[0][ii]),
                                    ]
                                ),
                                fmt=".",
                                color=c,
                                elinewidth=0.5,
                                capsize=2,
                            )

            ax1.set_ylabel("Goniometer (degrees)")
            if not has_logs and self.get_instrument() != "DEMAND":
                ax1.set_xlabel("Run Number")
            elif not has_logs:
                ax1.set_xlabel("Scan Number")
                ax1.set_title(f"exp{self.get_experiment()}")
            ax1.set_xlim(subplot_limits[0][0] - 1, subplot_limits[0][1] + 1)
            ax1.legend(
                fontsize="x-small", loc="upper left", bbox_to_anchor=(0, 1.2)
            )
            ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax1.ticklabel_format(style="plain", axis="x", useOffset=False)

            ax2 = ax1.twinx()
            color = "gray"
            ax2.set_ylabel(
                f'Scale ({inst_params["Scale"].split(".")[-1]})', color=color
            )
            ax2.plot(run_numbers_list, scale_values, ".", color=color)
            ax2.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
            ax2.tick_params(axis="y", labelcolor=color)
            ax2.set_ylim(
                -0.1 * np.max(scale_values), np.max(scale_values) * 1.1
            )
            self._set_plain_y_axis(ax1)
            self._set_plain_y_axis(ax2)
            self._enable_minor_ticks(ax1)
            self._enable_minor_ticks(ax2, x=False)

            if has_logs:
                temp_colors = [
                    self._log_series_color(name, i)
                    for i, name in enumerate(temp_names)
                ]
                for vals, name, c in zip(temp_values, temp_names, temp_colors):
                    ax_log.plot(
                        run_numbers_list, vals, ".", color=c, label=name
                    )

                ax_log.set_ylabel("Temperature (K)")
                if self.get_instrument() != "DEMAND":
                    ax_log.set_xlabel("Run Number")
                else:
                    ax_log.set_xlabel("Scan Number")
                    ax1.set_title(f"exp{self.get_experiment()}")

                ax_log.set_xlim(
                    subplot_limits[0][0] - 1, subplot_limits[0][1] + 1
                )
                ax_log.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax_log.ticklabel_format(
                    style="plain", axis="x", useOffset=False
                )
                self._set_plain_y_axis(ax_log)
                self._enable_minor_ticks(ax_log)

                if len(temp_values) > 0:
                    ax_log.legend(
                        fontsize="x-small",
                        loc="upper left",
                        bbox_to_anchor=(0, 1.2),
                    )

                if len(other_values) > 0:
                    ax_log2 = ax_log.twinx()
                    other_colors = [
                        self._log_series_color(name, i)
                        for i, name in enumerate(other_names)
                    ]
                    for vals, name, c in zip(
                        other_values, other_names, other_colors
                    ):
                        ax_log2.plot(
                            run_numbers_list, vals, ".", color=c, label=name
                        )
                    ax_log2.tick_params(
                        axis="x",
                        which="both",
                        bottom=False,
                        top=False,
                        labelbottom=False,
                    )
                    ax_log2.set_ylabel("Other Logs")
                    self._set_plain_y_axis(ax_log2)
                    self._enable_minor_ticks(ax_log2, x=False)
                    ax_log2.legend(
                        fontsize="x-small",
                        loc="upper right",
                        bbox_to_anchor=(1, 1.2),
                    )

        else:
            if has_logs:
                axs = self.plot.figure.subplots(
                    2,
                    len(subplot_limits),
                    sharey="row",
                    sharex="col",
                    width_ratios=[l[1] - l[0] + 2 for l in subplot_limits],
                    gridspec_kw={"height_ratios": [3, 2]},
                )
                gonio_axs = axs[0]
                log_axs = axs[1]
            else:
                gonio_axs = self.plot.figure.subplots(
                    1,
                    len(subplot_limits),
                    sharey=True,
                    width_ratios=[l[1] - l[0] + 2 for l in subplot_limits],
                )
                log_axs = None

            if self.get_instrument() != "DEMAND":
                self.plot.figure.supxlabel("Run Number", fontsize="medium")
            else:
                self.plot.figure.supxlabel("Scan Number", fontsize="medium")
                self.plot.figure.suptitle(
                    f"exp{self.get_experiment()}", fontsize="medium"
                )

            for i, ax1 in enumerate(gonio_axs):
                lim = subplot_limits[i]
                lim_range = 1
                ax1.set_xlim(lim[0] - lim_range, lim[1] + lim_range)
                ax1.set_ylim(
                    np.min(gonio_values) - 10, np.max(gonio_values) + 10
                )

                if i == 0:
                    ax1.set_ylabel("Goniometer (degrees)")
                    if self.get_instrument() != "DEMAND":
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            ax1.plot(
                                run_numbers_list, val, ".", color=c, label=lab
                            )
                    else:
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            v = val[:]
                            for ii in range(len(run_numbers_list)):
                                if ii == 0:
                                    ax1.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][ii]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        label=lab,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )
                                else:
                                    ax1.errorbar(
                                        run_numbers_list[ii],
                                        v[0][ii],
                                        yerr=np.array(
                                            [
                                                abs(v[1][ii] - v[0][ii]),
                                                abs(v[2][ii] - v[0][ii]),
                                            ]
                                        ),
                                        fmt=".",
                                        color=c,
                                        elinewidth=0.5,
                                        capsize=2,
                                    )
                    ax1.legend(
                        fontsize="x-small",
                        loc="upper left",
                        bbox_to_anchor=(0, 1.2),
                    )
                    ax1.ticklabel_format(
                        style="plain", axis="x", useOffset=False
                    )
                    ax1.spines.right.set_visible(False)
                else:
                    if self.get_instrument() != "DEMAND":
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            ax1.plot(run_numbers_list, val, ".", color=c)
                    else:
                        for val, lab, c in zip(
                            gonio_values, gonio_names, colors
                        ):
                            v = val[:]
                            for ii in range(len(run_numbers_list)):
                                ax1.errorbar(
                                    run_numbers_list[ii],
                                    v[0][ii],
                                    yerr=np.array(
                                        [
                                            abs(v[1][ii] - v[0][ii]),
                                            abs(v[2][ii] - v[0][ii]),
                                        ]
                                    ),
                                    fmt=".",
                                    color=c,
                                    elinewidth=0.5,
                                    capsize=2,
                                )
                    ax1.ticklabel_format(
                        style="plain", axis="x", useOffset=False
                    )
                    ax1.spines.left.set_visible(False)
                    ax1.tick_params(labelleft=False)
                    if i != len(subplot_limits) - 1:
                        ax1.spines.right.set_visible(False)
                        ax1.tick_params(labelright=False)
                        ax1.tick_params(axis="y", length=0)
                    else:
                        ax1.spines.right.set_visible(False)
                        ax1.tick_params(axis="y", length=0)

                ax2 = ax1.twinx()
                color = "gray"
                ax2.plot(run_numbers_list, scale_values, ".", color=color)
                ax2.tick_params(
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )
                ax2.tick_params(axis="y", labelcolor=color)
                ax2.set_ylim(
                    -0.1 * np.max(scale_values), np.max(scale_values) * 1.1
                )
                self._set_plain_y_axis(ax1)
                self._set_plain_y_axis(ax2)
                self._enable_minor_ticks(ax1)
                self._enable_minor_ticks(ax2, x=False)
                if i < len(subplot_limits) - 1:
                    ax2.spines.right.set_visible(False)
                    ax2.spines.left.set_visible(False)
                    ax2.tick_params(labelright=False)
                    ax2.tick_params(labelleft=False)
                    ax2.tick_params(axis="y", length=0)
                else:
                    ax2.tick_params(labelright=True)
                    ax2.spines.left.set_visible(False)
                    ax2.tick_params(labelleft=False)
                    ax2.tick_params(axis="y", color=color, labelright=True)
                    ax2.set_ylabel(
                        f'Scale ({inst_params["Scale"].split(".")[-1]})',
                        color=color,
                    )

                if has_logs:
                    ax_log = log_axs[i]
                    temp_colors = [
                        self._log_series_color(name, i)
                        for i, name in enumerate(temp_names)
                    ]
                    for vals, name, c in zip(
                        temp_values, temp_names, temp_colors
                    ):
                        if i == 0:
                            ax_log.plot(
                                run_numbers_list,
                                vals,
                                ".",
                                color=c,
                                label=name,
                            )
                        else:
                            ax_log.plot(run_numbers_list, vals, ".", color=c)

                    ax_log.set_xlim(lim[0] - lim_range, lim[1] + lim_range)
                    ax_log.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax_log.ticklabel_format(
                        style="plain", axis="x", useOffset=False
                    )
                    self._set_plain_y_axis(ax_log)
                    self._enable_minor_ticks(ax_log)
                    ax_log.tick_params(
                        axis="x", which="both", bottom=True, labelbottom=True
                    )
                    if i == 0:
                        ax_log.set_ylabel("Temperature (K)")
                        if len(temp_values) > 0:
                            ax_log.legend(
                                fontsize="x-small",
                                loc="upper left",
                                bbox_to_anchor=(0, 1.2),
                            )
                    else:
                        ax_log.tick_params(labelleft=False)
                        ax_log.tick_params(axis="y", length=0)
                        ax_log.spines.left.set_visible(False)

                    if i < len(subplot_limits) - 1:
                        ax_log.spines.right.set_visible(False)
                        ax_log.tick_params(labelright=False)

                    if len(other_values) > 0:
                        ax_log2 = ax_log.twinx()
                        other_colors = [
                            self._log_series_color(name, i)
                            for i, name in enumerate(other_names)
                        ]
                        for vals, name, c in zip(
                            other_values, other_names, other_colors
                        ):
                            if i == len(subplot_limits) - 1:
                                ax_log2.plot(
                                    run_numbers_list,
                                    vals,
                                    ".",
                                    color=c,
                                    label=name,
                                )
                            else:
                                ax_log2.plot(
                                    run_numbers_list, vals, ".", color=c
                                )
                        ax_log2.tick_params(
                            axis="x",
                            which="both",
                            bottom=False,
                            top=False,
                            labelbottom=False,
                        )

                        self._set_plain_y_axis(ax_log2)
                        self._enable_minor_ticks(ax_log2, x=False)
                        if i < len(subplot_limits) - 1:
                            ax_log2.spines.right.set_visible(False)
                            ax_log2.spines.left.set_visible(False)
                            ax_log2.tick_params(labelright=False)
                            ax_log2.tick_params(labelleft=False)
                            ax_log2.tick_params(axis="y", length=0)
                            ax_log2.tick_params(
                                axis="x",
                                which="both",
                                bottom=False,
                                labelbottom=False,
                            )
                        else:
                            ax_log2.tick_params(labelright=True)
                            ax_log2.spines.left.set_visible(False)
                            ax_log2.tick_params(labelleft=False)
                            ax_log2.set_ylabel("Other Logs")
                            ax_log2.legend(
                                fontsize="x-small",
                                loc="upper right",
                                bbox_to_anchor=(1, 1.2),
                            )

        self.plot.figure.canvas.draw()

    def _set_plain_y_axis(self, ax):
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
        formatter = ax.yaxis.get_major_formatter()
        if hasattr(formatter, "set_scientific"):
            formatter.set_scientific(False)
        if hasattr(formatter, "set_useOffset"):
            formatter.set_useOffset(False)

    def _enable_minor_ticks(self, ax, x=True, y=True):
        ax.minorticks_on()
        if not x:
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
        if not y:
            ax.tick_params(axis="y", which="minor", left=False, right=False)

    def _series_colors(self, n):
        return ["C{}".format(i) for i in range(n)]

    def _clear_run_guides(self):
        for line in self._run_guides:
            try:
                line.remove()
            except ValueError:
                pass
        self._run_guides = []

    def _guide_axes(self):
        # Keep one axis per subplot panel (ignore twinx duplicates sharing bounds).
        axes = []
        seen = set()
        for ax in self.plot.figure.axes:
            bounds = tuple(np.round(ax.get_position().bounds, 6))
            if bounds in seen:
                continue
            seen.add(bounds)
            axes.append(ax)
        return axes

    def _handle_plot_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return

        # Snap to the nearest run number available in the clicked axes.
        nearest = None
        for line in event.inaxes.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            if len(x) == 0:
                continue

            valid = np.isfinite(x)
            if not np.any(valid):
                continue

            runs = x[valid]
            distances = np.abs(runs - float(event.xdata))
            index = int(np.argmin(distances))
            distance = float(distances[index])

            if nearest is None or distance < nearest["distance"]:
                nearest = {
                    "distance": distance,
                    "run": float(runs[index]),
                }

        if nearest is None:
            return

        run_value = nearest["run"]
        run_text = (
            str(int(round(run_value)))
            if abs(run_value - round(run_value)) < 1e-6
            else "{:.6g}".format(run_value)
        )

        if hasattr(self.toolbar, "set_message"):
            self.toolbar.set_message(f"Run {run_text}")

        self._clear_run_guides()
        for ax in self._guide_axes():
            line = ax.axvline(
                run_value, color="black", linestyle="--", linewidth=1.0
            )
            self._run_guides.append(line)

        self.plot.draw_idle()

    def _log_series_color(self, name, index):
        if "statistics" in str(name).lower():
            return "gray"
        return "C{}".format(index)

    def _split_temperature_logs(self, log_values, log_names):
        temp_values, temp_names = [], []
        other_values, other_names = [], []

        for vals, name in zip(log_values, log_names):
            lname = str(name).lower()
            is_temperature = "(k)" in lname or "temp" in lname
            if is_temperature:
                temp_values.append(vals)
                temp_names.append(name)
            else:
                other_values.append(vals)
                other_names.append(name)

        return temp_values, temp_names, other_values, other_names

    def auto_scale_dropdown(self, combo):
        """
        Autoscale a combobox width to fit item text and item icons.

        This keeps both the closed state and popup comfortably wide for
        currently loaded entries.
        """

        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        fm = combo.fontMetrics()
        max_width = 0
        digit = all(
            [combo.itemText(i).isdigit() for i in range(combo.count())]
        )

        for i in range(combo.count()):
            text = combo.itemText(i)
            icon = QIcon()

            if qta is not None:
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
            combo.setMinimumWidth(max_width + 40)
        else:
            combo.setMinimumWidth(0)

    def clear_ipts(self):
        """Clear the IPTS combo box."""
        self.ipts_field.clear()
        self.auto_scale_dropdown(self.ipts_field)

    def add_ipts_items(self, items):
        """Add items to the IPTS combo box."""
        if items:
            self.ipts_field.addItems(items)
        self.auto_scale_dropdown(self.ipts_field)

    def set_experiment_enabled(self, enabled: bool):
        """Enable or disable the experiment combo box."""
        self.exp_cbox.setEnabled(enabled)

    def set_runs_label(self, text: str):
        """Set the runs label text."""
        self.runs_label.setText(text)

    def show_message(self, text: str, color: str = "red"):
        """Set the message label text and colour."""
        self.message_label.setText(text)
        self.message_label.setStyleSheet(f"color: {color};")

    def add_name_items(self, items):
        """Add names to the selection list."""
        if items:
            self.name_list.addItems(items)

    def clear_name_list(self):
        """Clear the name selection list."""
        self.name_list.clear()

    def clear_experiments(self):
        """Clear the experiments combo box."""
        self.exp_cbox.clear()
        self.auto_scale_dropdown(self.exp_cbox)

    def set_experiment_current(self, text: str):
        """Set the current experiment text."""
        self.exp_cbox.setCurrentText(text)
        self.auto_scale_dropdown(self.exp_cbox)

    def set_runs_text(self, text: str):
        """Set the runs list text."""
        self.runs_list.setText(text)

    def get_credentials(self):
        """Return (username, password) from the UI fields."""
        return self.user_line.text(), self.pass_line.text()

    def clear_password(self):
        """Clear the password field in the UI."""
        self.pass_line.setText("")


class Presenter:
    def __init__(self, view, model):
        self.view = view
        self.model = model

        self.view.connect_switch_instrument(self.switch_instrument)
        self.view.connect_select_name(self.select_name)
        self.view.connect_ipts(self.set_ipts)
        self.view.connect_adjust_runs(self.adjust_runs_list)
        self.view.connect_login_button(self.sign_in)
        self.view.connect_select_experiment(self.set_exp)
        self.view.connect_refresh_button(self.refresh)
        self.view.connect_report_button(self.export_experiment_report)

        self.switch_instrument()
        self.login = None
        self.data_files = None

    def switch_instrument(self):
        instrument = self.view.get_instrument()
        self.view.clear_ipts()
        self.clear()

        if instrument == "DEMAND":
            self.view.set_experiment_enabled(True)
            self.view.set_runs_label("Scan Numbers:")
        else:
            self.view.set_experiment_enabled(False)
            self.view.set_runs_label("Run Numbers: ")

        inst_params = self.model.beamline_info(instrument)

        try:
            available_runs = self.model.list_available(self.login, inst_params)
            self.view.add_ipts_items(available_runs)
        except AttributeError:
            self.view.show_message("Not Signed In", color="red")

        self.inst_params = inst_params

    def set_ipts(self):
        ipts = self.view.get_ipts()
        self.clear()
        try:
            self.data_files = self.model.retrieve_data_files(
                self.login, self.inst_params, ipts
            )
            # model populates the experiments combo-box as before
            self.model.set_experiments(self.view.exp_cbox, self.data_files)
            self.view.auto_scale_dropdown(self.view.exp_cbox)
            self.names = self.model.run_title_dictionary(
                self.data_files, self.inst_params
            )
            if self.view.get_instrument() != "DEMAND":
                self.view.add_name_items(list(self.names.keys()))
            else:
                self.set_exp()

        except AttributeError:
            self.view.show_message("Not Signed In", color="red")
        except pyoncat.InvalidRefreshTokenError:
            self.view.show_message("Login Expired", color="orange")

    def refresh(self):
        self.set_ipts()

    def export_experiment_report(self):
        if not self.data_files:
            self.view.show_message(
                "No experiment data to report", color="orange"
            )
            return

        filename = self.view.save_report_file_dialog(
            self.model.default_export_path(
                self.inst_params, self.view.get_ipts()
            )
        )
        if not filename:
            return

        headers, rows = self.model.experiment_summary_rows(
            self.data_files, self.inst_params
        )
        if len(rows) == 0:
            self.view.show_message(
                "No experiment data to report", color="orange"
            )
            return

        with open(filename, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)

        self.view.show_message("Experiment report exported", color="green")

    def set_exp(self):
        self.data_files = self.model.retrieve_data_files(
            self.login, self.inst_params, self.view.get_ipts()
        )
        data_files = self.data_files
        exp = self.view.get_experiment()

        self.clear()
        self.model.set_experiments(self.view.exp_cbox, self.data_files)
        self.view.auto_scale_dropdown(self.view.exp_cbox)
        self.view.set_experiment_current(exp)

        mask = np.array([f"exp{exp}" in df["id"] for df in data_files])
        dfs = np.array(data_files)[mask]
        self.data_files = list(dfs)

        self.names = self.model.run_title_dictionary(
            self.data_files, self.inst_params
        )

        self.view.add_name_items(list(self.names.keys()))

    def select_name(self):
        names = self.view.get_name()
        runs = []
        run_numbers_list = []
        data_indices = []
        for name in names:
            runs.append(self.names[name.text()])

            rrun_numbers_list, ddata_indices = self.model.run_numbers_indices(
                name.text(), self.data_files, self.names, self.inst_params
            )
            for r in rrun_numbers_list:
                run_numbers_list.append(r)
            for d in ddata_indices:
                data_indices.append(d)

        if len(names) > 0:
            rnl = run_numbers_list[:]
            rnl.sort()
            run_seq = np.split(rnl, np.where(np.diff(rnl) > 1)[0] + 1)
            rs = ",".join(
                [
                    str(s[0]) + ":" + str(s[-1]) if len(s) - 1 else str(s[0])
                    for s in run_seq
                ]
            )

            self.runs = rs
        else:
            self.runs = ",".join(runs)

        self.view.runs_list.setText(self.runs)

        self.adjust_runs_list()

    def adjust_runs_list(self):
        rs = self.view.get_runs()
        try:
            runs_list = self.model.run_numbers_list(rs)
        except:
            # print('Invalid run numbers')
            return

        if self.data_files is None:
            return

        run_numbers_list, data_indices = self.model.run_numbers_indices_1(
            self.data_files, runs_list, self.inst_params
        )

        if len(run_numbers_list) > 0:
            gonio_values, gonio_names = self.model.goniometer_values(
                self.data_files, data_indices, self.inst_params
            )
            scale_values = self.model.scale_values(
                self.data_files, data_indices, self.inst_params
            )
            log_values, log_names = self.model.log_values(
                self.data_files, data_indices, self.inst_params
            )

            self.plot(
                gonio_values,
                gonio_names,
                run_numbers_list,
                scale_values,
                log_values,
                log_names,
            )

    def clear(self):
        self.view.name_list.clear()
        self.view.exp_cbox.clear()
        self.view.runs_list.setText("")
        self.view.plot.figure.clf()
        self.view.plot.figure.canvas.draw()

    def plot(
        self,
        gonio_values,
        gonio_names,
        run_numbers_list,
        scale_values,
        log_values,
        log_names,
    ):
        subplot_limits = getattr(self.model, "subplot_limits", [])
        inst_params = getattr(self, "inst_params", {})

        self.view.plot_goniometer(
            gonio_values,
            gonio_names,
            run_numbers_list,
            scale_values,
            subplot_limits,
            inst_params,
            log_values,
            log_names,
        )

    def sign_in(self):
        user, pw = self.view.get_credentials()

        try:
            oncat = self.model.login_oncat(user, pw)
        except pyoncat.InvalidRefreshTokenError:
            self.view.show_message("Login Expired", color="orange")
            self.view.clear_password()
            return
        except Exception:
            self.view.show_message(
                "Incorrect Username or Password", color="red"
            )
            self.view.clear_password()
            return

        self.login = oncat
        self.view.show_message("Signed In", color="green")
        self.view.clear_password()
        self.switch_instrument()


class Model:
    def __init__(self):
        pass

    def _title_group_key(self, title):
        text = str(title).strip()
        match = re.match(r"^(.*?)([_\-\s]?)(\d+)$", text)
        if match is None:
            return text

        base, sep, _ = match.groups()
        if base == "":
            return text

        return "{}{}*".format(base, sep)

    def login_oncat(self, user: str, password: str):
        """
        Create an ONCat client and perform login.
        """
        ONCAT_URL = "https://oncat.ornl.gov"
        CLIENT_ID = "99025bb3-ce06-4f4b-bcf2-36ebf925cd1d"

        oncat = pyoncat.ONCat(
            ONCAT_URL,
            client_id=CLIENT_ID,
            flow=pyoncat.RESOURCE_OWNER_CREDENTIALS_FLOW,
        )

        oncat.login(user, password)
        return oncat

    def goniometer_entries(self, inst_params):
        goniometer = inst_params["Goniometer"]
        goniometer_engry = inst_params["GoniometerEntry"]

        if inst_params["FancyName"] == "DEMAND":
            goniometer["2theta"] = "2theta"

        projection = []
        for name in goniometer.keys():
            if (
                inst_params["FancyName"] != "DEMAND"
                or inst_params["InstrumentName"] == "WAND"
            ):
                entry = ".".join(
                    [goniometer_engry, name.lower(), "average_value"]
                )
            else:
                min_entry = ".".join(
                    [goniometer_engry, name.lower(), "minimum"]
                )
                entry = ".".join([goniometer_engry, name.lower(), "average"])
                max_entry = ".".join(
                    [goniometer_engry, name.lower(), "maximum"]
                )
                projection.append(min_entry)
                projection.append(max_entry)
            projection.append(entry)

        return projection

    def retrieve_data_files(self, login, inst_params, ipts_number):
        facility = inst_params["Facility"]
        instrument = inst_params["Name"]
        run_number = inst_params["RunNumber"]

        projection = [
            run_number,
            inst_params["Title"],
            inst_params["Scale"],
            "metadata.entry.proton_charge",
            "id",
        ]

        projection += self.goniometer_entries(inst_params)
        projection += self.log_projection_entries(inst_params)

        exts = [inst_params["Extension"]]

        data_files = login.Datafile.list(
            facility=facility,
            instrument=instrument,
            experiment="IPTS-{}".format(ipts_number),
            projection=projection,
            exts=exts,
            tags=["type/raw"],
        )
        return data_files

    def list_available(self, login, inst_params):
        facility = inst_params["Facility"]
        instrument = inst_params["Name"]
        projection = ["id"]

        available_runs = login.Experiment.list(
            facility=facility, instrument=instrument, projection=projection
        )

        available = [""]
        avs = []
        if len(available_runs) != 0:
            for i in available_runs:
                avs.append(int(i["id"].split("-")[-1]))
        avs.sort(reverse=True)
        for i in range(len(avs)):
            available.append(str(avs[i]))

        return available

    def set_experiments(self, cbox, data_files):
        ids = np.array(
            [df["id"].split("/")[-3].strip("expIPTS-") for df in data_files]
        )
        unique = np.unique(ids)
        cbox_entries = []  # ['']
        for i in unique:
            cbox_entries.append(i)
        cbox.addItems(cbox_entries)

    def run_title_dictionary(self, data_files, inst_params):
        title_entry = inst_params["Title"]
        run_number_entry = inst_params["RunNumber"]

        titles = np.array([df[title_entry] for df in data_files])
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )

        grouped_runs = {}
        for title, run in zip(titles, run_numbers):
            key = self._title_group_key(title)
            if key not in grouped_runs:
                grouped_runs[key] = []
            grouped_runs[key].append(int(run))

        run_title_dict = {}
        for key in sorted(grouped_runs.keys()):
            runs = np.array(sorted(set(grouped_runs[key])))
            run_seq = np.split(
                runs.astype(str), np.where(np.diff(runs) > 1)[0] + 1
            )
            rs = ",".join(
                [s[0] + ":" + s[-1] if len(s) - 1 else s[0] for s in run_seq]
            )
            run_title_dict[key] = rs

        return run_title_dict

    def run_numbers_list(self, rs):
        run_seq = [np.array(s.split(":")).astype(int) for s in rs.split(",")]
        run_list = [
            np.arange(r[0], r[-1] + 1) if len(r) - 1 else r for r in run_seq
        ]

        return np.array([r for sub_list in run_list for r in sub_list])

    def _runs_to_string_with_step(self, runs):
        runs = sorted(set(int(r) for r in runs))
        if len(runs) == 0:
            return ""

        parts = []
        i = 0
        n = len(runs)
        while i < n:
            if i == n - 1:
                parts.append(str(runs[i]))
                break

            start = runs[i]
            step = runs[i + 1] - runs[i]
            j = i + 1
            while j + 1 < n and (runs[j + 1] - runs[j]) == step:
                j += 1

            end = runs[j]
            if end == start:
                parts.append(str(start))
            elif step == 1:
                parts.append("{}:{}".format(start, end))
            else:
                parts.append("{}:{};{}".format(start, end, step))

            i = j + 1

        return ",".join(parts)

    def _extract_goniometer_average(self, df, inst_params, axis_name):
        base = inst_params["GoniometerEntry"] + "." + axis_name.lower()
        average_value = base + ".average_value"
        average = base + ".average"

        value = df.get(average_value, None)
        if value is None:
            value = df.get(average, None)

        if value is None:
            return np.nan

        if isinstance(value, list):
            if len(value) == 0 or value[0] is None:
                return np.nan
            value = value[0]

        try:
            return float(value)
        except (TypeError, ValueError):
            return np.nan

    def _extract_numeric(self, df, keys):
        for key in keys:
            value = df.get(key, None)
            if value is None:
                continue
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                value = value[0]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return np.nan

    def log_projection_entries(self, inst_params):
        log_entries = inst_params.get("Logs", [])
        log_entry = inst_params.get("LogEntry", inst_params["GoniometerEntry"])

        projection = []
        for name in log_entries:
            base = ".".join([log_entry, name.lower()])
            projection.append(base + ".average_value")
            projection.append(base + ".average")
            # Backward compatibility: allow direct key lookups if present.
            projection.append(name)

        return list(dict.fromkeys(projection))

    def log_value_keys(self, inst_params, name):
        log_entry = inst_params.get("LogEntry", inst_params["GoniometerEntry"])
        base = ".".join([log_entry, name.lower()])
        return [base + ".average_value", base + ".average", name]

    def default_export_path(self, inst_params, ipts_number):
        facility = str(inst_params.get("Facility", "")).strip()
        instrument = str(inst_params.get("Name", "")).strip()
        ipts = str(ipts_number).strip()

        if facility == "" or instrument == "" or ipts == "":
            return os.getcwd()

        candidates = [
            os.path.join(
                os.sep, facility, instrument, "IPTS-{}".format(ipts), "shared"
            ),
            os.path.join(os.sep, facility, instrument, "IPTS-{}".format(ipts)),
            os.path.join(os.sep, facility, instrument),
        ]

        for path in candidates:
            if os.path.isdir(path):
                return path

        return os.getcwd()

    def experiment_summary_rows(self, data_files, inst_params):
        if not data_files:
            return [], []

        title_entry = inst_params["Title"]
        run_number_entry = inst_params["RunNumber"]
        gonio_axes = list(inst_params["Goniometer"].keys())
        log_entries = inst_params.get("Logs", [])
        log_units = inst_params.get("LogUnits", [])

        if len(log_units) < len(log_entries):
            log_units = list(log_units) + [""] * (
                len(log_entries) - len(log_units)
            )

        unique_entry_units = []
        seen = set()
        for entry, unit in zip(log_entries, log_units):
            if entry not in seen:
                seen.add(entry)
                unique_entry_units.append((entry, unit))

        grouped = {}
        for df in data_files:
            title = str(df.get(title_entry, "")).strip()
            if title == "":
                title = "(untitled)"

            title = title.replace(",", "")

            grouped.setdefault(title, []).append(df)

        rows = []
        for title in sorted(grouped):
            files = grouped[title]
            files = sorted(files, key=lambda x: int(x[run_number_entry]))

            runs = [int(df[run_number_entry]) for df in files]
            run_string = self._runs_to_string_with_step(runs)

            angle_values = []
            step_values = []
            for axis in gonio_axes:
                values = np.array(
                    [
                        self._extract_goniometer_average(df, inst_params, axis)
                        for df in files
                    ],
                    dtype=float,
                )

                valid = values[~np.isnan(values)]
                if len(valid) == 0:
                    angle_values.append("n/a")
                    step_values.append("n/a")
                    continue

                angle_values.append(
                    "{:.1f} -> {:.1f}".format(
                        np.nanmin(valid), np.nanmax(valid)
                    )
                )

                if len(valid) > 1:
                    med_step = np.nanmedian(np.abs(np.diff(valid)))
                    step_values.append("{:.1f}".format(med_step))
                else:
                    step_values.append("n/a")

            proton_charge = np.array(
                [
                    self._extract_numeric(
                        df,
                        [
                            "metadata.entry.proton_charge",
                            "metadata.entry.proton_charge.average",
                        ],
                    )
                    for df in files
                ],
                dtype=float,
            )

            proton_charge = proton_charge[np.isfinite(proton_charge)]
            if len(proton_charge) > 0:
                proton_scale = 1e12
                if inst_params.get("Scale") != "metadata.entry.proton_charge":
                    proton_scale = 1.0

                proton_charge_stat = "{:.6g}".format(
                    np.nansum(proton_charge) / proton_scale
                )
            else:
                proton_charge_stat = "n/a"

            log_medians = []
            for entry, unit in unique_entry_units:
                values = np.array(
                    [
                        self._extract_numeric(
                            df, self.log_value_keys(inst_params, entry)
                        )
                        for df in files
                    ],
                    dtype=float,
                )
                valid = values[np.isfinite(values)]
                if len(valid) > 0:
                    median_value = np.nanmedian(valid)
                    log_medians.append("{:.6g}".format(median_value))
                else:
                    log_medians.append("n/a")

            rows.append(
                [
                    title.replace("|", "/"),
                    run_string,
                    proton_charge_stat,
                    *angle_values,
                    *step_values,
                    *log_medians,
                ]
            )

        headers = ["Title", "Run String"]
        headers += ["Proton Charge (C, sum)"]
        headers += ["{} Range (deg)".format(axis) for axis in gonio_axes]
        headers += ["{} Median Step (deg)".format(axis) for axis in gonio_axes]
        headers += [
            "{} Median".format(
                entry if unit == "" else "{} ({})".format(entry, unit)
            )
            for entry, unit in unique_entry_units
        ]

        return headers, rows

    def prepare_runs_for_multiple_plots(self, run_number_list):
        rs = run_number_list.copy()

        rs.sort(0)

        out_list = []
        breaks = [0]
        for i in range(1, len(rs)):
            if rs[i] - rs[i - 1] > 2:
                breaks.append(i)

        if len(breaks) == 1:
            out_list.append([rs[0], rs[-1]])

        else:
            for i in range(len(breaks) - 1):
                out_list.append([rs[breaks[i]], rs[breaks[i + 1] - 1]])
            out_list.append([rs[breaks[-1]], rs[-1]])

        self.subplot_limits = out_list

    def beamline_info(self, bl):
        inst_params = beamlines[bl]

        return inst_params

    def run_numbers_indices(
        self, name, data_files, run_title_dict, inst_params
    ):
        run_number_entry = inst_params["RunNumber"]
        this_run_numbers = self.run_numbers_list(run_title_dict[name])
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )
        indices = np.arange(len(data_files))

        mask = np.array([i in this_run_numbers for i in run_numbers])

        self.prepare_runs_for_multiple_plots(run_numbers[mask])

        return run_numbers[mask], indices[mask]

    def run_numbers_indices_1(self, data_files, run_number_list, inst_params):
        run_number_entry = inst_params["RunNumber"]
        run_numbers = np.array(
            [int(df[run_number_entry]) for df in data_files]
        )
        indices = np.arange(len(data_files))

        mask = np.array([i in run_number_list for i in run_numbers])

        if sum(mask) != 0:
            self.prepare_runs_for_multiple_plots(run_numbers[mask])

        return run_numbers[mask], indices[mask]

    def goniometer_values(self, data_files, indices, inst_params):
        a = []
        gonio_entry = inst_params["Goniometer"]

        for entry in gonio_entry:
            b = []
            try:
                values = np.array(
                    [
                        float(
                            df[
                                inst_params["GoniometerEntry"]
                                + "."
                                + entry.lower()
                                + ".average_value"
                            ]
                        )
                        for df in data_files
                    ]
                )
            except KeyError:
                values = np.array(
                    [
                        [
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".average"
                                ]
                            ),
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".minimum"
                                ]
                            ),
                            float(
                                df[
                                    inst_params["GoniometerEntry"]
                                    + "."
                                    + entry.lower()
                                    + ".maximum"
                                ]
                            ),
                        ]
                        for df in data_files
                    ]
                )
            except TypeError:
                values = []
                for df in data_files:
                    try:
                        val = df[
                            inst_params["GoniometerEntry"]
                            + "."
                            + entry.lower()
                            + ".average_value"
                        ]
                    except KeyError:
                        val = np.array(
                            [
                                [
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".average"
                                        ]
                                    ),
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".minimum"
                                        ]
                                    ),
                                    float(
                                        df[
                                            inst_params["GoniometerEntry"]
                                            + "."
                                            + entry.lower()
                                            + ".maximum"
                                        ]
                                    ),
                                ]
                                for df in data_files
                            ]
                        )
                    if val is None:
                        val = np.nan
                    if type(val) is list and val[0] is None:
                        val[0] = np.nan
                        val[1] = np.nan
                        val[2] = np.nan
                    values.append(float(val))
                values = np.array(values)

            b.append([values[i] for i in indices])
            b = np.array(b).T
            a.append(b)

        return a, [i.lower() for i in gonio_entry]

    def scale_values(self, data_files, indices, inst_params):
        scale_entry = inst_params["Scale"]
        if inst_params["FancyName"] == "DEMAND":
            scale_entry += ".average"

        values = np.array([float(df[scale_entry]) for df in data_files])
        if inst_params["Scale"] == "metadata.entry.proton_charge":
            a = [values[i] / 1e12 for i in indices]
        else:
            a = [values[i] for i in indices]
        a = np.array(a).T

        return a

    def log_values(self, data_files, indices, inst_params):
        log_entries = inst_params.get("Logs", [])
        log_units = inst_params.get("LogUnits", [])

        if len(log_units) < len(log_entries):
            log_units = list(log_units) + [""] * (
                len(log_entries) - len(log_units)
            )

        # Keep first occurrence order while preserving the entry/unit pairing.
        unique_entry_units = []
        seen = set()
        for entry, unit in zip(log_entries, log_units):
            if entry not in seen:
                seen.add(entry)
                unique_entry_units.append((entry, unit))

        values_out = []
        names_out = []

        for entry, unit in unique_entry_units:
            values = []
            for df in data_files:
                val = self._extract_numeric(
                    df, self.log_value_keys(inst_params, entry)
                )
                values.append(val)

            selected = np.array([values[i] for i in indices], dtype=float)

            if np.any(np.isfinite(selected)):
                values_out.append(selected)
                label = entry if unit == "" else f"{entry} ({unit})"
                names_out.append(label)

        return values_out, names_out


class ExperimentBrowser(QMainWindow):
    __instance = None

    def __new__(cls):
        if ExperimentBrowser.__instance is None:
            ExperimentBrowser.__instance = QMainWindow.__new__(cls)
        return ExperimentBrowser.__instance

    def __init__(self, parent=None):
        super().__init__(parent)

        icon_path = "./icon.png"
        self.setWindowIcon(QIcon(icon_path))
        name = "Experiment Browser"
        self.setWindowTitle(name)
        self.setGeometry(0, 0, 1024, 635)

        main_window = QWidget(self)
        self.setCentralWidget(main_window)
        layout = QVBoxLayout(main_window)

        view = View()
        model = Model()
        self.form = Presenter(view, model)
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
    msg_box.exec_()


if __name__ == "__main__":
    sys.excepthook = handle_exception
    app = QApplication(sys.argv)
    if style:
        app.setStyleSheet(qdarkstyle.load_stylesheet(palette=LightPalette))
    window = ExperimentBrowser()
    window.show()
    sys.exit(app.exec_())
