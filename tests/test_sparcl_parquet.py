import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from specbox.basemodule import SpecPandasRow, SpecSparcl


def _write_parquet_or_skip(df, path):
    try:
        df.to_parquet(path)
        SpecPandasRow._DATAFRAME_CACHE.pop(str(path), None)
    except ImportError as exc:
        pytest.skip(f"parquet engine unavailable: {exc}")


def test_default_sparcl_parquet_redshift_initializes_z_vi(tmp_path):
    path = tmp_path / "desispecs_tierA.parquet"
    wave = np.array([3600.0, 4200.0, 5000.0, 7000.0])
    df = pd.DataFrame(
        {
            "specid": [123],
            "redshift": [2.3456],
            "data_release": ["DESI-DR1"],
            "dec": [2.5],
            "ra": [150.5],
            "specprimary": [True],
            "spectype": ["QSO"],
            "targetid": [987654321],
            "sparcl_id": ["sparcl-abc"],
            "program": ["dark"],
            "survey": ["main"],
            "flux": [np.array([1.0, 2.0, 3.0, 4.0])],
            "model": [np.array([1.1, 2.1, 3.1, 4.1])],
            "mask": [np.array([0, 0, 0, 0])],
            "ivar": [np.array([1.0, 4.0, 0.0, np.nan])],
            "wavelength": [wave],
            "_dr": ["DESI-DR1"],
        }
    )
    _write_parquet_or_skip(df, path)

    sp = SpecSparcl(str(path), ext=1)

    assert sp.redshift == pytest.approx(2.3456)
    assert sp.z_vi == pytest.approx(2.3456)
    assert sp.z_vi_initial == pytest.approx(2.3456)
    assert sp.z_vi_source == "redshift"
    assert sp.targetid == 987654321
    assert sp.objid == "sparcl:sparcl-abc"
    assert np.isinf(sp.err[2])
    assert np.isinf(sp.err[3])


def test_sparcl_parquet_redshift_falls_back_to_z_desi(tmp_path):
    path = tmp_path / "sparcl_missing_redshift.parquet"
    df = pd.DataFrame(
        {
            "targetid": [42],
            "redshift": [np.nan],
            "z_desi": [1.23],
            "ra": [1.0],
            "dec": [2.0],
            "wavelength": [np.array([1.0, 2.0, 3.0])],
            "flux": [np.array([4.0, 5.0, 6.0])],
            "ivar": [np.array([1.0, 1.0, 1.0])],
        }
    )
    _write_parquet_or_skip(df, path)

    sp = SpecSparcl(str(path), ext=1)

    assert sp.redshift == pytest.approx(1.23)
    assert sp.z_vi == pytest.approx(1.23)
    assert sp.redshift_source == "z_desi"
    assert sp.z_vi_source == "z_desi"


def test_sparcl_viewer_starts_from_parquet_redshift(tmp_path):
    from PySide6.QtWidgets import QApplication
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    path = tmp_path / "desispecs_tierA.parquet"
    df = pd.DataFrame(
        {
            "targetid": [99],
            "redshift": [3.21],
            "data_release": ["DESI-DR1"],
            "ra": [10.0],
            "dec": [20.0],
            "wavelength": [np.linspace(3600.0, 9800.0, 64)],
            "flux": [np.ones(64)],
            "ivar": [np.ones(64)],
        }
    )
    _write_parquet_or_skip(df, path)
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(str(path), SpecClass=SpecSparcl)
    text = plot.spectrum_info_label.text()

    assert plot.z_max == 7.0
    assert plot.spec.redshift == pytest.approx(3.21)
    assert plot.spec.z_vi == pytest.approx(3.21)
    assert plot.redshiftSpin.value() == pytest.approx(3.21)
    assert "z_vi = 3.2100" in text
    assert "z_desi = 3.2100" in text
    app.processEvents()


def test_sparcl_parquet_count():
    path = Path("outlier_sparcl_spectra.parquet")
    if not path.exists():
        pytest.skip("outlier_sparcl_spectra.parquet not present in this checkout")
    n = SpecSparcl.count_in_file(str(path))
    assert n == 46


def test_sparcl_parquet_first_row_loads():
    path = Path("outlier_sparcl_spectra.parquet")
    if not path.exists():
        pytest.skip("outlier_sparcl_spectra.parquet not present in this checkout")
    sp = SpecSparcl(str(path), ext=1)
    assert sp.wave.shape[0] == 7781
    assert sp.flux.shape[0] == 7781
    assert np.isfinite(sp.wave.value).all()
    assert isinstance(sp.objid, str)
    assert hasattr(sp, "ra")
    assert hasattr(sp, "dec")
    assert hasattr(sp, "redshift")
