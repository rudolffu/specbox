from __future__ import annotations

import datetime
import os
import sys

from importlib.metadata import PackageNotFoundError, version as dist_version

project = "specbox"
author = "Yuming Fu"
copyright = f"{datetime.datetime.now().year}, {author}"

try:
    release = dist_version(project)
except PackageNotFoundError:
    release = "0.0.0"

version = release.split("+", 1)[0]

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
}

autodoc_member_order = "bysource"

# Importing Qt/pyqtgraph in a headless RTD build can be fragile; mock them.
autodoc_mock_imports = [
    "PySide6",
    "pyqtgraph",
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.qt_compat",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = os.environ.get("SPHINX_HTML_THEME", "sphinx_rtd_theme")

root_doc = "index"

html_context = {
    "display_github": True,
    "github_user": "rudolffu",
    "github_repo": "specbox",
    "github_version": "main",
    "conf_py_path": "/specbox/docs/",
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

myst_enable_extensions = [
    "colon_fence",
]

myst_heading_anchors = 3

nb_execution_mode = "off"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Ensure local builds work even without installing the package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
