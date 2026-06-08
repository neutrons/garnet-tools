import os
import sys
import subprocess
import getpass

from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

from utilities import run_browser_scenario, copy_generated_pngs, build_docs

DIRECTORY = os.path.dirname(os.path.abspath(__file__))

"""
ONCAT_USER=user ONCAT_PASSWORD=password \
  python tests/applications/experiment_browser_app.py TOPAZ_browser
"""


def _get_credentials():
    user = os.environ.get("ONCAT_USER", "")
    pw = os.environ.get("ONCAT_PASSWORD", "")
    if not user:
        user = input("ONCat username: ").strip()
    if not pw:
        pw = getpass.getpass("ONCat password: ")
    return user, pw


def TOPAZ_browser(app, window):
    directory = os.path.join(DIRECTORY, "TOPAZ")

    presenter = window.form
    view = presenter.view

    user, pw = _get_credentials()

    view.user_line.setText(user)
    view.pass_line.setText(pw)
    presenter.sign_in()
    QTest.qWait(1000 * 5)

    index = view.instrument_cbox.findText("TOPAZ")
    view.instrument_cbox.setCurrentIndex(index)
    presenter.switch_instrument()
    QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "browser_instrument.png"), "png"
    )

    ipts_index = view.ipts_field.findText("31856")
    if ipts_index >= 0:
        view.ipts_field.setCurrentIndex(ipts_index)
        presenter.set_ipts()
        QTest.qWait(1000 * 5)

    items = view.name_list.findItems("Blue Garnet 3-*", Qt.MatchExactly)
    if items:
        view.name_list.setCurrentItem(items[0])
        presenter.select_name()
        QTest.qWait(1000 * 5)

    app.primaryScreen().grabWindow(window.winId()).save(
        os.path.join(directory, "browser_ipts.png"), "png"
    )

    copy_generated_pngs(directory)
    build_docs()


SCENARIOS = {
    "TOPAZ_browser": TOPAZ_browser,
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        name = sys.argv[1]
        if name not in SCENARIOS:
            print(f"Unknown scenario: {name}")
            print(f"Available: {', '.join(SCENARIOS)}")
            sys.exit(1)
        run_browser_scenario(SCENARIOS[name])
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
