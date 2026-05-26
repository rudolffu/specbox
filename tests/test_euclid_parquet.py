import numpy as np
import pandas as pd
import pytest

from specbox.basemodule import SpecEuclid1d


def _write_parquet_or_skip(df, path):
    try:
        df.to_parquet(path)
    except ImportError as exc:
        pytest.skip(f"parquet engine unavailable: {exc}")


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
    assert np.allclose(sp.flux.value, [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(sp.err[:2], [0.0, 2.0])
    assert np.isinf(sp.err[2])
    assert np.isinf(sp.err[3])
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
    assert np.allclose(sp.flux.value, [10.0, 40.0])
    assert np.allclose(sp.err, [1.0, 4.0])
    assert np.array_equal(sp.mask, [0, 2])
    assert np.allclose(sp.quality, [0.1, 0.4])
    assert np.array_equal(sp.ndith, [1, 1])
