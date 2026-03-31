from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from specbox.basemodule import SpecAimszReview
from specbox.qtmodule.qtmodule_enhanced import display_review_class_vi, normalize_review_class_vi


def test_aimsz_review_parquet_row_loads(tmp_path: Path):
    path = tmp_path / "review_bundle.parquet"
    pd.DataFrame(
        {
            "object_id": [10],
            "ra": [150.0],
            "dec": [2.0],
            "targetid": [1010],
            "specid": ["10"],
            "spectype": ["QSO"],
            "data_release": ["DESI-DR1"],
            "review_priority_tier": [1],
            "review_score": [0.9],
            "z_ref": [0.1],
            "z_ml_expect": [0.12],
            "z_pcf_best": [0.11],
            "pcf_template_best": ["red"],
            "pcf_score_best": [0.8],
            "wavelength": [np.array([1.0, 2.0, 3.0], dtype=float)],
            "flux": [np.array([10.0, 11.0, 12.0], dtype=float)],
            "ivar": [np.array([4.0, 4.0, 4.0], dtype=float)],
            "mask": [np.array([0, 0, 0], dtype=int)],
        }
    ).to_parquet(path, index=False)

    spec = SpecAimszReview(path, ext=1)
    assert spec.objid == "10"
    assert spec.object_id == 10
    assert spec.targetid == 1010
    assert spec.review_priority_tier == 1
    assert np.isfinite(spec.wave.value).all()
    assert np.isfinite(spec.flux.value).all()


def test_review_class_label_normalization_aliases():
    assert normalize_review_class_vi("QSO(Default)") == "QSO_DEFAULT"
    assert normalize_review_class_vi("QSO(Narrow)") == "QSO_NARROW"
    assert normalize_review_class_vi("LIKELY") == "LIKELY_Q"
    assert normalize_review_class_vi("QSO_BAL") == "QSO_BAL"
    assert display_review_class_vi("QSO_NARROW") == "QSO(Narrow)"
