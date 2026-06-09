import os

import numpy as np
import pytest
from astropy.io import fits

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from specbox.basemodule import SpecEuclid1d, SpecEuclid1dDual
from specbox.qtmodule.qtmodule_enhanced import (
    PGSpecPlotEnhanced,
    _ordered_dual_pair_selectors,
)


def _euclid_arm_hdu(source_id, *, arm, ra=None, dec=None):
    if arm == "RGS":
        wave = np.linspace(11900.0, 19002.0, 531)
    elif arm == "BGS":
        wave = np.linspace(9200.0, 14600.0, 433)
    else:
        raise ValueError(arm)
    flux = np.full_like(wave, float(str(source_id)[-2:]))
    cols = [
        fits.Column(name="WAVELENGTH", array=wave, format="D"),
        fits.Column(name="SIGNAL", array=flux, format="D"),
        fits.Column(name="MASK", array=np.zeros_like(wave, dtype=np.int16), format="I"),
        fits.Column(name="QUALITY", array=np.ones_like(wave, dtype=float), format="D"),
        fits.Column(name="VAR", array=np.ones_like(wave, dtype=float), format="D"),
        fits.Column(name="NDITH", array=np.ones_like(wave, dtype=np.int16), format="I"),
    ]
    hdu = fits.BinTableHDU.from_columns(cols, name=str(source_id))
    hdu.header["RA"] = float(ra if ra is not None else 10.0 + int(source_id))
    hdu.header["DEC"] = float(dec if dec is not None else -5.0 - int(source_id))
    hdu.header["LRANGE"] = arm
    return hdu


def _write_arm_file(path, source_ids, *, arm):
    hdus = [fits.PrimaryHDU()]
    for source_id in source_ids:
        hdus.append(_euclid_arm_hdu(source_id, arm=arm))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def test_dual_fits_pair_selectors_use_source_id_union(tmp_path):
    rgs_path = tmp_path / "rgs.fits"
    bgs_path = tmp_path / "bgs.fits"
    _write_arm_file(rgs_path, [101, 102], arm="RGS")
    _write_arm_file(bgs_path, [101, 103], arm="BGS")

    pairs, key_column = _ordered_dual_pair_selectors(str(rgs_path), str(bgs_path))

    assert key_column == "source_id"
    assert [pair["key"] for pair in pairs] == ["101", "102", "103"]
    assert pairs[0]["rgs"]["ext"] == 1
    assert pairs[0]["bgs"]["ext"] == 1
    assert pairs[1]["rgs"]["ext"] == 2
    assert pairs[1]["bgs"] is None
    assert pairs[2]["rgs"] is None
    assert pairs[2]["bgs"]["ext"] == 2


def test_euclid_fits_reader_exposes_extname_as_source_id(tmp_path):
    rgs_path = tmp_path / "rgs.fits"
    _write_arm_file(rgs_path, [2666336258651300273], arm="RGS")

    spec = SpecEuclid1d(str(rgs_path), ext=1)

    assert spec.source_id == "2666336258651300273"
    assert spec.object_id == "2666336258651300273"
    assert spec.extname == "2666336258651300273"


def test_dual_loader_accepts_missing_rgs_or_bgs_arm(tmp_path):
    rgs_path = tmp_path / "rgs.fits"
    bgs_path = tmp_path / "bgs.fits"
    _write_arm_file(rgs_path, [101, 102], arm="RGS")
    _write_arm_file(bgs_path, [101, 103], arm="BGS")

    both = SpecEuclid1dDual(
        rgs_file=str(rgs_path),
        bgs_file=str(bgs_path),
        rgs_ext=1,
        bgs_ext=1,
    )
    assert both.rgs.objid == 101
    assert both.bgs.objid == 101

    rgs_only = SpecEuclid1dDual(rgs_file=str(rgs_path), bgs_file=None, rgs_ext=2)
    assert rgs_only.rgs.objid == 102
    assert rgs_only.bgs is None
    assert rgs_only.scale_status == "missing_arm"

    bgs_only = SpecEuclid1dDual(rgs_file=None, bgs_file=str(bgs_path), bgs_ext=2)
    assert bgs_only.rgs is None
    assert bgs_only.bgs.objid == 103
    assert bgs_only.scale_status == "missing_arm"


def test_viewer_dual_fits_loads_by_source_id_not_index(tmp_path):
    from PySide6.QtWidgets import QApplication

    rgs_path = tmp_path / "rgs.fits"
    bgs_path = tmp_path / "bgs.fits"
    _write_arm_file(rgs_path, [101, 102], arm="RGS")
    _write_arm_file(bgs_path, [101, 103], arm="BGS")
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(
        None,
        SpecClass=SpecEuclid1d,
        rgs_file=str(rgs_path),
        bgs_file=str(bgs_path),
    )

    assert plot.len_list == 3
    assert plot._dual_pair_keys == ["101", "102", "103"]

    first_spec, first_dual = plot._load_dual_spec(0)
    assert first_spec.objid == 101
    assert first_dual.rgs.objid == 101
    assert first_dual.bgs.objid == 101

    rgs_spec, rgs_dual = plot._load_dual_spec(1)
    assert rgs_spec.objid == 102
    assert rgs_dual.rgs.objid == 102
    assert rgs_dual.bgs is None
    assert rgs_dual.scale_status == "missing_arm"

    bgs_spec, bgs_dual = plot._load_dual_spec(2)
    assert bgs_spec.objid == 103
    assert bgs_dual.rgs is None
    assert bgs_dual.bgs.objid == 103
    assert bgs_dual.scale_status == "missing_arm"
    app.processEvents()


def test_viewer_dual_extname_can_select_bgs_only_source(tmp_path):
    from PySide6.QtWidgets import QApplication

    rgs_path = tmp_path / "rgs.fits"
    bgs_path = tmp_path / "bgs.fits"
    _write_arm_file(rgs_path, [101], arm="RGS")
    _write_arm_file(bgs_path, [101, 103], arm="BGS")
    app = QApplication.instance() or QApplication([])

    plot = PGSpecPlotEnhanced(
        None,
        SpecClass=SpecEuclid1d,
        rgs_file=str(rgs_path),
        bgs_file=str(bgs_path),
        extname="103",
    )

    assert plot.len_list == 1
    assert plot.spec.objid == 103
    assert plot.spec_dual.rgs is None
    assert plot.spec_dual.bgs.objid == 103
    app.processEvents()
