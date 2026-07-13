import os
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/specbox-matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _plot_stub(total=12):
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    plot = PGSpecPlotEnhanced.__new__(PGSpecPlotEnhanced)
    plot.len_list = total
    return plot


def test_terminal_spectrum_message_includes_finite_coordinates():
    plot = _plot_stub(total=12)
    spec = SimpleNamespace(ra=1.23456789, dec=-2.34567891)

    message = plot._format_terminal_spectrum_message(3, spec)

    assert message == "Spectrum 3/12. RA, DEC = 1.234568, -2.345679."


def test_terminal_spectrum_message_omits_partial_coordinates():
    plot = _plot_stub(total=12)

    assert plot._format_terminal_spectrum_message(3, SimpleNamespace(ra=1.0)) == "Spectrum 3/12."
    assert plot._format_terminal_spectrum_message(3, SimpleNamespace(dec=2.0)) == "Spectrum 3/12."


def test_terminal_spectrum_message_omits_invalid_coordinates():
    plot = _plot_stub(total=12)

    invalid_specs = [
        SimpleNamespace(ra=None, dec=2.0),
        SimpleNamespace(ra=np.nan, dec=2.0),
        SimpleNamespace(ra=1.0, dec=np.inf),
        SimpleNamespace(ra="not-ra", dec=2.0),
    ]
    for spec in invalid_specs:
        assert plot._format_terminal_spectrum_message(3, spec) == "Spectrum 3/12."
