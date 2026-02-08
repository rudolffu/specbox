from __future__ import annotations

import datetime

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
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "alabaster"

root_doc = "index"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
]
