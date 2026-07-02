import sys
import os
import tempfile

from qtpy.QtWidgets import QApplication, QComboBox, QMainWindow
from qtpy.QtGui import QIcon, QPixmap, QPainter, QColor, QFont, QPalette
from qtpy.QtCore import Qt, QSettings

_local_cfg = os.path.join(
    tempfile.gettempdir(), os.environ.get("USER", "user"), "qt"
)
os.makedirs(_local_cfg, exist_ok=True)
QSettings.setPath(
    QSettings.Format.NativeFormat, QSettings.Scope.UserScope, _local_cfg
)

from mantid import config

config["Q.convention"] = "Crystallography"

from mantid.simpleapi import LoadMD, mtd
from mantidqt.widgets.sliceviewer.presenters.presenter import SliceViewer
from mantidqt.plotting.functions import plot_md_ws_from_names

theme = True
try:
    from qdarkstyle.light.palette import LightPalette
    from qdarkstyle.dark.palette import DarkPalette
    import qdarkstyle
except ImportError:
    theme = False


def _make_icon(char, color="#555555", size=16):
    px = QPixmap(size, size)
    px.fill(Qt.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.Antialiasing)
    font = QFont()
    font.setPixelSize(size - 2)
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor(color))
    p.drawText(px.rect(), Qt.AlignCenter, char)
    p.end()
    return QIcon(px)


def _adjust_combos(widget):
    fm = widget.fontMetrics()
    for combo in widget.findChildren(QComboBox):
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        digit = all(combo.itemText(i).isdigit() for i in range(combo.count()))
        icon = _make_icon("#" if digit else "—")
        max_width = 0
        for i in range(combo.count()):
            combo.setItemIcon(i, icon)
            text = combo.itemText(i)
            max_width = max(
                max_width,
                fm.horizontalAdvance(text) + combo.iconSize().width() + 8,
            )
        if max_width:
            combo.setMinimumWidth(max_width + 40)


class MainWindow(QMainWindow):
    def __init__(self, filename):
        super().__init__()
        name = os.path.splitext(os.path.basename(filename))[0]
        self.setWindowTitle(name)
        LoadMD(Filename=filename, OutputWorkspace=name)
        try:
            viewer = SliceViewer(mtd[name])
            self.setCentralWidget(viewer.view)
            _adjust_combos(viewer.view)
        except:
            plot_md_ws_from_names([name], True, False)
            sys.exit(app.exec())


if __name__ == "__main__":
    app = QApplication(sys.argv)

    if theme:
        bg = app.palette().color(QPalette.Window)
        palette = DarkPalette if bg.lightness() < 128 else LightPalette
        app.setStyleSheet(qdarkstyle.load_stylesheet(palette=palette))

    window = MainWindow(sys.argv[1])
    window.show()
    sys.exit(app.exec())
