# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import inspect
import importlib

# -- Project information -----------------------------------------------------

project = "garnet"
copyright = ""
author = "Zachary Morgan"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
    "matplotlib.sphinxext.plot_directive",
    "pyvista.ext.plot_directive",
]

templates_path = ["_templates"]

exclude_patterns = []

root_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True

html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "_static/garnet_logo.svg",
        "image_dark": "_static/garnet_logo.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/neutrons/garnet-tools",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {
            "url": "https://single-crystal.ornl.gov",
            "name": "Single Crystal Diffraction",
        },
    ],
    "show_nav_level": 2,
}

# -- Extension configuration -------------------------------------------------

plot_pre_code = """
import numpy as np
np.random.seed(13)

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.ioff()

"""

plot_include_source = True
plot_html_show_formats = False
plot_html_show_source_link = False
plot_basedir = ""

add_module_names = False


def linkcode_resolve(domain, info):
    baseurl = "https://github.com/neutrons/garnet-tools/blob/main/src/{}.py"
    if "py" not in domain:
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    url = baseurl.format(filename)
    mod = importlib.import_module(info["module"])
    if hasattr(mod, "__pyx_unpickle_Enum"):
        url += "x"
    objname, *attrname = info["fullname"].split(".")
    obj = getattr(mod, objname)
    if attrname:
        for attr in attrname:
            obj = getattr(obj, attr)
    if not hasattr(mod, "__pyx_unpickle_Enum"):
        try:
            lines = inspect.getsourcelines(obj)
            start, stop = lines[1], lines[1] + len(lines[0]) - 1
            return "{}#L{}-L{}".format(url, start, stop)
        except (TypeError, OSError):
            return url
    else:
        return url


def skip_qt_members(app, what, name, obj, skip, options):
    try:
        from PyQt6.QtCore import pyqtSignal, QMetaObject
    except ImportError:
        try:
            from qtpy.QtCore import Signal as pyqtSignal, QMetaObject
        except ImportError:
            pyqtSignal = None
            QMetaObject = None
    if pyqtSignal is not None and isinstance(obj, pyqtSignal):
        return True
    if QMetaObject is not None and isinstance(obj, QMetaObject):
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_qt_members)
