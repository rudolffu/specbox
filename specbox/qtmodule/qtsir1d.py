"""Backward compatibility module.

The functionality of the original ``qtsir1d`` viewer has been merged into
``qtmodule``.  This file simply re-exports the public classes so that existing
imports continue to work.
"""

from .qtmodule import PGSpecPlot, PGSpecPlotApp, PGSpecPlotThread

__all__ = ['PGSpecPlot', 'PGSpecPlotApp', 'PGSpecPlotThread']

