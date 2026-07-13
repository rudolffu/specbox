import numpy as np
import os
import pandas as pd
import pytest

from specbox.basemodule import SpecEuclid1d, SpecPandasRow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _write_parquet_or_skip(df, path):
    try:
        df.to_parquet(path)
        SpecPandasRow._DATAFRAME_CACHE.pop(str(path), None)
    except ImportError as exc:
        pytest.skip(f"parquet engine unavailable: {exc}")


def _base_euclid_row(**extra):
    row = {
        "object_id": 42,
        "ra": 1.23456789,
        "dec": 2.34567891,
        "wavelength": np.array([1.0, 2.0, 3.0, 4.0]),
        "flux": np.array([10.0, 20.0, 30.0, 40.0]),
        "var": np.array([1.0, 4.0, 9.0, 16.0]),
    }
    row.update(extra)
    return pd.DataFrame({key: [value] for key, value in row.items()})


def test_euclid_archive_parquet_uses_var_when_err_missing(tmp_path):
    path = tmp_path / "euclid_archive.parquet"
    variance = np.array([0.0, 4.0, -1.0, np.nan], dtype=float)
    df = pd.DataFrame(
        {
            "object_id": [1301699551666636977],
            "source_id": [1301699551666636977],
            "ra": [130.16995517828],
            "dec": [66.6636977919485],
            "wavelength": [np.array([11900.0, 11913.4, 11926.8, 11940.2])],
            "signal": [np.array([1.0, 2.0, 3.0, 4.0])],
            "mask": [np.array([0, 2, 65, 0])],
            "quality": [np.array([0.1, 0.2, 0.3, 0.4])],
            "var": [variance],
            "ndith": [np.array([1, 1, 0, 1])],
        }
    )
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)

    assert np.allclose(sp.wave.value, [11900.0, 11913.4, 11926.8, 11940.2])
    assert np.allclose(sp.flux.value, np.array([1.0, 2.0, 3.0, 4.0]) * 1e-16)
    assert np.allclose(sp.err[:2], np.array([0.0, 2.0]) * 1e-16)
    assert np.isinf(sp.err[2])
    assert np.isinf(sp.err[3])
    assert sp.flux_scale == pytest.approx(1e-16)
    assert sp.flux_scale_source == "default_euclid_parquet"
    assert sp.objid == 1301699551666636977
    assert sp.object_id == 1301699551666636977
    assert sp.source_id == 1301699551666636977
    assert sp.ra == 130.16995517828
    assert sp.dec == 66.6636977919485
    assert np.array_equal(sp.mask, [0, 2, 65, 0])
    assert np.allclose(sp.quality, [0.1, 0.2, 0.3, 0.4])
    assert np.array_equal(sp.ndith, [1, 1, 0, 1])
    assert SpecEuclid1d.count_in_file(str(path)) == 1


def test_euclid_archive_parquet_good_pixels_filters_side_arrays(tmp_path):
    path = tmp_path / "euclid_archive.parquet"
    df = pd.DataFrame(
        {
            "object_id": [42],
            "ra": [1.0],
            "dec": [2.0],
            "wavelength": [np.array([1.0, 2.0, 3.0, 4.0])],
            "flux": [np.array([10.0, 20.0, 30.0, 40.0])],
            "var": [np.array([1.0, 4.0, 9.0, 16.0])],
            "mask": [np.array([0, 1, 64, 2])],
            "quality": [np.array([0.1, 0.2, 0.3, 0.4])],
            "ndith": [np.array([1, 1, 0, 1])],
        }
    )
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False, good_pixels_only=True)

    assert np.allclose(sp.wave.value, [1.0, 4.0])
    assert np.allclose(sp.flux.value, np.array([10.0, 40.0]) * 1e-16)
    assert np.allclose(sp.err, np.array([1.0, 4.0]) * 1e-16)
    assert np.array_equal(sp.mask, [0, 2])
    assert np.allclose(sp.quality, [0.1, 0.4])
    assert np.array_equal(sp.ndith, [1, 1])


def test_euclid_parquet_flux_scale_override(tmp_path):
    path = tmp_path / "euclid_archive.parquet"
    df = _base_euclid_row(flux_scale=1.0)
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)

    assert np.allclose(sp.flux.value, [10.0, 20.0, 30.0, 40.0])
    assert np.allclose(sp.err, [1.0, 2.0, 3.0, 4.0])
    assert sp.flux_scale == pytest.approx(1.0)
    assert sp.flux_scale_source == "flux_scale"


def test_euclid_processed_parquet_z_vi_wins_over_all_fallbacks(tmp_path):
    path = tmp_path / "euclid_processed.parquet"
    df = _base_euclid_row(
        z_vi=0.31,
        z_sdss=0.42,
        z_desi=0.43,
        z_hybrid=0.44,
        z_fusion=0.45,
        z_temp=0.46,
        z_pcf_best=0.47,
        z_gaia=0.48,
        z_phot=0.49,
    )
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)

    assert sp.z_vi == pytest.approx(0.31)
    assert sp.z_vi_initial == pytest.approx(0.31)
    assert sp.z_vi_source == "z_vi"
    assert sp.redshift == pytest.approx(0.31)


def test_euclid_processed_parquet_sdss_then_desi_priority(tmp_path):
    path = tmp_path / "euclid_processed.parquet"
    df = _base_euclid_row(
        z_vi=0.0,
        z_sdss=0.52,
        z_desi=0.53,
        z_hybrid=0.54,
    )
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)
    assert sp.z_vi == pytest.approx(0.52)
    assert sp.z_vi_source == "z_sdss"

    df = _base_euclid_row(
        z_vi=np.nan,
        z_sdss=np.nan,
        z_desi=0.63,
        z_hybrid=0.64,
    )
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)
    assert sp.z_vi == pytest.approx(0.63)
    assert sp.z_vi_source == "z_desi"


def test_euclid_processed_parquet_uses_pcf_best_when_z_temp_absent(tmp_path):
    path = tmp_path / "euclid_processed.parquet"
    df = _base_euclid_row(z_vi=0.0, z_pcf_best=1.23)
    _write_parquet_or_skip(df, path)

    sp = SpecEuclid1d(str(path), ext=1, clip=False)

    assert sp.z_temp is None
    assert sp.z_pcf_best == pytest.approx(1.23)
    assert sp.z_vi == pytest.approx(1.23)
    assert sp.z_vi_source == "z_pcf_best"

    df = _base_euclid_row(z_vi=0.0, z_temp=1.34, z_pcf_best=1.23)
    _write_parquet_or_skip(df, path)
    sp = SpecEuclid1d(str(path), ext=1, clip=False)
    assert sp.z_vi == pytest.approx(1.34)
    assert sp.z_vi_source == "z_temp"


def test_euclid_processed_parquet_late_priority_order(tmp_path):
    path = tmp_path / "euclid_processed.parquet"
    priority_cases = [
        ({"z_hybrid": 0.71, "z_fusion": 0.72, "z_gaia": 0.73, "z_phot": 0.74}, "z_hybrid", 0.71),
        ({"z_fusion": 0.82, "z_gaia": 0.83, "z_phot": 0.84}, "z_fusion", 0.82),
        ({"z_gaia": 0.93, "z_phot": 0.94}, "z_gaia", 0.93),
        ({"z_phot": 1.04}, "z_phot", 1.04),
    ]
    for redshifts, source, expected in priority_cases:
        df = _base_euclid_row(z_vi=0.0, **redshifts)
        _write_parquet_or_skip(df, path)
        sp = SpecEuclid1d(str(path), ext=1, clip=False)
        assert sp.z_vi == pytest.approx(expected)
        assert sp.z_vi_source == source


def test_euclid_info_label_has_coordinates_and_compact_z_source(tmp_path):
    from PySide6.QtWidgets import QApplication
    from specbox.qtmodule.qtmodule_enhanced import PGSpecPlotEnhanced

    path = tmp_path / "euclid_processed.parquet"
    wave = np.linspace(11900.0, 19002.0, 531)
    df = _base_euclid_row(
        wavelength=wave,
        flux=np.ones_like(wave),
        var=np.ones_like(wave),
        z_vi=0.0,
        z_hybrid=1.23456,
    )
    _write_parquet_or_skip(df, path)
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(str(path), SpecClass=SpecEuclid1d)
    text = plot.spectrum_info_label.text()

    assert plot.z_max == 6.0
    assert "RA: 1.234568" in text
    assert "DEC: 2.345679" in text
    assert "z_vi = 1.2346" in text
    assert "z_source: z_hybrid = 1.2346" in text
    assert "z_fusion" not in text
    app.processEvents()


def test_viewer_default_z_max_by_spec_class():
    from specbox.basemodule import SpecAIMSZReview, SpecSparcl
    from specbox.qtmodule.qtmodule_enhanced import default_z_max_for_spec_class

    assert default_z_max_for_spec_class(SpecEuclid1d) == 6.0
    assert default_z_max_for_spec_class(SpecSparcl) == 7.0
    assert default_z_max_for_spec_class(SpecAIMSZReview) == 7.0
