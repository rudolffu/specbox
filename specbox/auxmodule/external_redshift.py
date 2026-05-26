"""Helpers for external redshift table overlay and parquet enrichment."""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.units import UnitsWarning


_INT_TEXT_RE = re.compile(r"^[+-]?\d+$")
_INT_FLOAT_TEXT_RE = re.compile(r"^[+-]?\d+\.0+$")


def normalize_redshift_lookup_key(value):
    """Normalize a join key without lossy float coercion."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return int(value)
        return str(value).strip()

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if _INT_TEXT_RE.fullmatch(text):
        try:
            return int(text)
        except Exception:
            return text
    if _INT_FLOAT_TEXT_RE.fullmatch(text):
        try:
            return int(text.split(".", 1)[0])
        except Exception:
            return text
    return text


def _coerce_positive_finite_float(value):
    try:
        z = float(value)
    except Exception:
        return None
    if not np.isfinite(z):
        return None
    return z


def _read_tabular_table(table_path):
    path = Path(table_path)
    suffix = path.suffix.lower()
    if suffix in {".parquet"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".fits", ".fit", ".fts"}:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnitsWarning)
            table = Table.read(path)
        return table.to_pandas()
    raise ValueError(f"Unsupported redshift table format: {path}")


def load_external_redshift_lookup(table_path, *, key_col="object_id", redshift_col="Z"):
    """Load an external redshift table into a normalized key -> z map."""
    df = _read_tabular_table(table_path)
    if key_col not in df.columns:
        raise KeyError(f"Join key column '{key_col}' not found in {table_path}")
    if redshift_col not in df.columns:
        raise KeyError(f"Redshift column '{redshift_col}' not found in {table_path}")

    lookup: Dict[object, float] = {}
    for key_raw, z_raw in zip(df[key_col], df[redshift_col]):
        key = normalize_redshift_lookup_key(key_raw)
        if key is None:
            continue
        z_value = _coerce_positive_finite_float(z_raw)
        if z_value is None:
            continue
        lookup[key] = z_value
    return lookup


def merge_external_redshift_into_dataframe(
    df,
    *,
    lookup,
    key_col="object_id",
    target_col="z_ref",
    fill_z_vi=False,
):
    """Return a dataframe copy with external redshift values merged in."""
    if key_col not in df.columns:
        raise KeyError(f"Join key column '{key_col}' not found in spectra dataframe")

    out = df.copy()
    z_ref_values = []
    for raw_key in out[key_col]:
        norm_key = normalize_redshift_lookup_key(raw_key)
        z_ref_values.append(lookup.get(norm_key, np.nan))
    out[target_col] = np.asarray(z_ref_values, dtype=float)

    if fill_z_vi:
        if "z_vi" not in out.columns:
            out["z_vi"] = np.nan
        current = pd.to_numeric(out["z_vi"], errors="coerce")
        z_ref = pd.to_numeric(out[target_col], errors="coerce")
        fill_mask = (~np.isfinite(current)) | (current <= 0)
        valid_ref = np.isfinite(z_ref) & (z_ref > 0)
        out.loc[fill_mask & valid_ref, "z_vi"] = z_ref[fill_mask & valid_ref]

    return out


def merge_external_redshift_into_parquet(
    spectra_path,
    *,
    table_path,
    key_col="object_id",
    redshift_col="Z",
    output_path,
    fill_z_vi=False,
):
    """Write a parquet copy enriched with external redshift values."""
    df = pd.read_parquet(spectra_path)
    lookup = load_external_redshift_lookup(table_path, key_col=key_col, redshift_col=redshift_col)
    out = merge_external_redshift_into_dataframe(
        df,
        lookup=lookup,
        key_col=key_col,
        target_col="z_ref",
        fill_z_vi=fill_z_vi,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    return out
