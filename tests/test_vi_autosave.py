from types import SimpleNamespace

import pandas as pd

from specbox.qtmodule.qtmodule_enhanced import (
    PGSpecPlotEnhanced,
    _vi_history_sample_name,
)


def test_vi_history_sample_name_for_single_input():
    assert (
        _vi_history_sample_name("/data/edfn_qc_tier.parquet")
        == "edfn_qc_tier"
    )
    assert _vi_history_sample_name("/data/sample.list") == "sample"
    assert _vi_history_sample_name("/data/My Sample-é.parquet") == "My_Sample"


def test_vi_history_sample_name_for_dual_arm_inputs():
    assert _vi_history_sample_name(
        None,
        rgs_file="/data/combined_spec_rgs_chunk_001.fits",
        bgs_file="/data/combined_spec_bgs_chunk_001.fits",
    ) == "combined_spec_chunk_001"

    assert _vi_history_sample_name(
        None,
        rgs_file="/data/north_rgs.fits",
        bgs_file="/data/south_bgs.fits",
    ) == "north_rgs__south_bgs"


def test_vi_history_sample_name_for_python_file_list():
    assert _vi_history_sample_name(
        ["/data/sample/spec-1.fits", "/data/sample/spec-2.fits"]
    ) == "sample"
    assert _vi_history_sample_name([]) == "spectra_list"


def _autosave_plot(temp_dir, rows):
    return SimpleNamespace(
        vi_temp_dir=temp_dir,
        _history_rows_for_csv=lambda: list(rows),
    )


def test_autosave_history_creates_sample_folder_and_keeps_snapshots(tmp_path):
    sample_dir = tmp_path / "temp" / "sample"
    rows = [{"objid": 1, "class_vi": "QSO", "z_vi": 1.25}]
    plot = _autosave_plot(sample_dir, rows)

    first = PGSpecPlotEnhanced._autosave_history(plot, 50)
    second = PGSpecPlotEnhanced._autosave_history(plot, 100)

    assert first == sample_dir / "vi_temp_50.csv"
    assert second == sample_dir / "vi_temp_100.csv"
    assert first.exists()
    assert second.exists()
    assert not (tmp_path / "vi_temp_50.csv").exists()
    saved = pd.read_csv(first)
    assert saved.to_dict("records") == rows


def test_autosave_history_failure_does_not_raise(tmp_path, monkeypatch, capsys):
    sample_dir = tmp_path / "temp" / "sample"
    plot = _autosave_plot(sample_dir, [{"objid": 1}])

    def fail_to_csv(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    result = PGSpecPlotEnhanced._autosave_history(plot, 50)

    assert result is None
    assert "Failed to auto-save VI history" in capsys.readouterr().out
    assert not (sample_dir / "vi_temp_50.csv.tmp").exists()
