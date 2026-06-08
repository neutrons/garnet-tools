import os
import glob
import shutil
import subprocess
import traceback
import faulthandler

faulthandler.enable()

os.environ.setdefault("QT_API", "pyqt6")

from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer
from qtpy.QtTest import QTest

import qdarkstyle
from qdarkstyle.light.palette import LightPalette

"""
QT_API=pyqt6 \
QT_OPENGL=software \
LIBGL_ALWAYS_SOFTWARE=1 \
MESA_GL_VERSION_OVERRIDE=3.3 \
xvfb-run -s "-screen 0 1920x1080x24" \
python tests/applications/garnet.py
"""


def _run_scenario(window_factory, scenario):
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    app.setQuitOnLastWindowClosed(False)
    app.setStyleSheet(
        qdarkstyle.load_stylesheet(
            qt_api=os.environ["QT_API"],
            palette=LightPalette,
        )
    )

    window = window_factory()
    window.show()

    result = {"rc": 0}

    def _wrapped():
        try:
            scenario(app, window)
            result["rc"] = 0
        except BaseException:
            traceback.print_exc()
            result["rc"] = 1
        finally:
            for _ in range(10):
                app.processEvents()
                QTest.qWait(50)
            app.exit(result["rc"])

    QTimer.singleShot(0, _wrapped)
    rc = app.exec()
    os._exit(rc)


def run_garnet_scenario(scenario):
    from garnet.application import Garnet

    _run_scenario(Garnet, scenario)


def run_browser_scenario(scenario):
    from garnet.utilities.ipts import ExperimentBrowser

    _run_scenario(ExperimentBrowser, scenario)


def copy_generated_pngs(directory):
    static = os.path.abspath(os.path.join(directory, "../../../docs/source"))
    os.makedirs(static, exist_ok=True)
    for png in glob.glob(
        os.path.join(directory, "**", "*.png"), recursive=True
    ):
        shutil.copy2(png, static)


def build_docs():
    docs_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../docs")
    )
    result = subprocess.run(
        ["sphinx-build", "-b", "html", ".", "_build/html"],
        cwd=docs_dir,
    )
    if result.returncode != 0:
        print("sphinx-build failed")
    else:
        print(
            f"Docs built: {os.path.join(docs_dir, '_build/html/index.html')}"
        )
