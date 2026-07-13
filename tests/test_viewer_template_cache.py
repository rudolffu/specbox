import os

import numpy as np
import pandas as pd
import pytest

from specbox.basemodule import SpecEuclid1d, SpecPandasRow, SpecSparcl

os.environ.setdefault("MPLCONFIGDIR", "/tmp/specbox-matplotlib")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_parquet_or_skip(df, path):
    try:
        df.to_parquet(path)
        SpecPandasRow._DATAFRAME_CACHE.pop(str(path), None)
    except ImportError as exc:
        pytest.skip(f"parquet engine unavailable: {exc}")


def test_sparcl_redshift_change_reuses_prepared_plot_data(tmp_path):
    from PySide6.QtWidgets import QApplication
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    path = tmp_path / "sparcl.parquet"
    wave = np.linspace(3600.0, 9800.0, 96)
    df = pd.DataFrame(
        {
            "targetid": [101],
            "redshift": [2.1],
            "ra": [10.0],
            "dec": [20.0],
            "wavelength": [wave],
            "flux": [np.sin(wave / 500.0) + 2.0],
            "ivar": [np.ones_like(wave)],
        }
    )
    _write_parquet_or_skip(df, path)
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(str(path), SpecClass=SpecSparcl)
    generation = plot._prepared_plot_generation
    prepared_key = plot._prepared_plot_key
    wave_before = plot.wave.copy()
    flux_before = plot.flux.copy()

    plot.spin_changed(2.5)

    assert plot._prepared_plot_generation == generation
    assert plot._prepared_plot_key == prepared_key
    assert np.array_equal(plot.wave, wave_before)
    assert np.array_equal(plot.flux, flux_before)
    assert plot.spec.z_vi == pytest.approx(2.5)
    assert "z_vi = 2.5000" in plot.spectrum_info_label.text()
    assert len(plot._template_items) > 0

    plot.change_template(plot.template_manager.current_template)
    assert plot._prepared_plot_generation == generation
    assert np.array_equal(plot.wave, wave_before)
    assert np.array_equal(plot.flux, flux_before)
    app.processEvents()


def test_sparcl_navigation_rebuilds_prepared_plot_data(tmp_path):
    from PySide6.QtWidgets import QApplication
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    path = tmp_path / "sparcl.parquet"
    wave1 = np.linspace(3600.0, 9800.0, 64)
    wave2 = np.linspace(4000.0, 9000.0, 64)
    df = pd.DataFrame(
        {
            "targetid": [101, 102],
            "redshift": [1.1, 1.2],
            "ra": [10.0, 11.0],
            "dec": [20.0, 21.0],
            "wavelength": [wave1, wave2],
            "flux": [np.ones_like(wave1), np.full_like(wave2, 2.0)],
            "ivar": [np.ones_like(wave1), np.ones_like(wave2)],
        }
    )
    _write_parquet_or_skip(df, path)
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(str(path), SpecClass=SpecSparcl)
    generation = plot._prepared_plot_generation
    prepared_key = plot._prepared_plot_key

    plot.jump_to_spectrum(2)

    assert plot._prepared_plot_generation > generation
    assert plot._prepared_plot_key != prepared_key
    assert plot.spec.targetid == 102
    assert np.allclose(plot.wave, wave2)
    assert np.allclose(plot.flux, 2.0e-17)
    app.processEvents()


def test_euclid_redshift_change_reuses_prepared_plot_data(tmp_path):
    from PySide6.QtWidgets import QApplication
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    path = tmp_path / "euclid.parquet"
    wave = np.linspace(11900.0, 19002.0, 531)
    df = pd.DataFrame(
        {
            "object_id": [42],
            "ra": [1.2],
            "dec": [3.4],
            "wavelength": [wave],
            "flux": [np.ones_like(wave)],
            "var": [np.ones_like(wave)],
            "z_vi": [1.0],
        }
    )
    _write_parquet_or_skip(df, path)
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(str(path), SpecClass=SpecEuclid1d)
    generation = plot._prepared_plot_generation
    wave_before = plot.wave.copy()

    plot.spin_changed(1.5)

    assert plot._prepared_plot_generation == generation
    assert np.array_equal(plot.wave, wave_before)
    assert plot.spec.z_vi == pytest.approx(1.5)
    assert "z_vi = 1.5000" in plot.spectrum_info_label.text()
    app.processEvents()


def test_template_emission_lines_include_ne_v_3426():
    from specbox.qtmodule.qtmodule_enhanced import _TEMPLATE_EMISSION_LINES

    assert ("[Ne V]", 3426.84) in _TEMPLATE_EMISSION_LINES
