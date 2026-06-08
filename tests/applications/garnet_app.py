import os
import sys
import subprocess

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_garnet_scenario, copy_generated_pngs, build_docs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

YAML = "/SNS/TOPAZ/IPTS-31856/shared/2026A_Si_cal/Si_AG_cal.yaml"


def TOPAZ_Si(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    presenter = window.form
    view = presenter.view

    # Load the YAML plan without triggering the file dialog
    original_dialog = view.load_config_file_dialog
    view.load_config_file_dialog = lambda *a, **kw: YAML
    try:
        presenter.load_config()
    finally:
        view.load_config_file_dialog = original_dialog

    QTest.qWait(1000 * 2)

    # --- Normalization tab ---
    view.plan_widget.setCurrentIndex(0)
    view.norm_run_button.setStyleSheet("background-color: green;")
    QTest.qWait(1000 * 2)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_normalization.png"), "png"
    )
    view.norm_run_button.setStyleSheet("")

    # --- Parametrization tab ---
    view.plan_widget.setCurrentIndex(1)
    view.param_run_button.setStyleSheet("background-color: green;")
    QTest.qWait(1000 * 2)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_parametrization.png"), "png"
    )
    view.param_run_button.setStyleSheet("")

    # --- Integration tab ---
    view.plan_widget.setCurrentIndex(2)
    view.int_run_button.setStyleSheet("background-color: green;")
    QTest.qWait(1000 * 2)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "Si_integration.png"), "png"
    )
    view.int_run_button.setStyleSheet("")

    copy_generated_pngs(directory)
    build_docs()


SCENARIOS = {
    "TOPAZ_Si": TOPAZ_Si,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS)}")
            sys.exit(1)
        run_garnet_scenario(SCENARIOS[name])
    else:
        script = os.path.abspath(__file__)
        failed = []
        for name in SCENARIOS:
            print(f"Running {name} ...")
            rc = subprocess.run([sys.executable, script, name]).returncode
            if rc != 0:
                failed.append(name)
        if failed:
            print(f"Failed: {', '.join(failed)}")
            sys.exit(1)
