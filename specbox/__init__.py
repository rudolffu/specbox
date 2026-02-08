#!/usr/bin/env python
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as dist_version

try:
    __version__ = dist_version("specbox")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = []
from . basemodule import *
__all__ += basemodule.__all__
