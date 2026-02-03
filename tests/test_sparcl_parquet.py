import numpy as np
import pytest
from pathlib import Path

from specbox.basemodule import SpecSparcl


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
