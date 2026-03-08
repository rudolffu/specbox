#!/usr/bin/env python
"""PCF redshift estimation utilities for specbox.

Provides a clean API for single/dual-template PCF runs and optional writes
back to FITS headers / parquet tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from importlib.resources import files


ArrayLike = Union[np.ndarray, list, tuple]
TemplateInput = Union[str, Path, Tuple[ArrayLike, ArrayLike]]


def default_template_paths() -> Dict[str, Path]:
    root = Path(files("specbox").joinpath("data/templates"))
    return {
        "type1": root / "qso1" / "optical_nir_qso_template_v1.fits",
        "type2": root / "qso2" / "Lusso_2024_compo_SED19.txt",
    }


@dataclass
class PCFConfig:
    z_min: float = 0.0
    z_max: float = 5.5
    base_z_step: float = 0.001
    min_overlap: int = 50
    nan_threshold: float = 0.5


def build_scaled_z_grid(z_min: float, z_max: float, base_z_step: float) -> np.ndarray:
    z_vals = []
    z = float(z_min)
    while z <= z_max:
        z_vals.append(z)
        z += base_z_step * (1.0 + z)
    return np.asarray(z_vals, dtype=np.float64)


def _pick_column(cols: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    low = {str(c).lower(): c for c in cols}
    for cand in candidates:
        cand_l = str(cand).lower()
        if cand_l in low:
            return low[cand_l]
    return None


def load_template(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load a template from FITS table or 2-column text-like file."""
    template_path = Path(path)
    suffix = template_path.suffix.lower()

    if suffix in {".txt", ".dat", ".csv"}:
        data = np.loadtxt(str(template_path), dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Template text file must have >=2 columns: {template_path}")
        wave = data[:, 0]
        flux = data[:, 1]
    else:
        tb = Table.read(template_path)
        cols = list(tb.colnames)
        wave_col = _pick_column(cols, ["wavelength", "wave", "lambda", "lam", "wbin"])
        flux_col = _pick_column(cols, ["flux", "template_flux", "f_lambda", "fnu", "signal"])
        if wave_col is None or flux_col is None:
            if len(cols) < 2:
                raise KeyError(f"Could not infer wave/flux columns in template: {template_path}")
            wave_col, flux_col = cols[0], cols[1]
        wave = np.asarray(tb[wave_col], dtype=np.float64)
        flux = np.asarray(tb[flux_col], dtype=np.float64)

    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    order = np.argsort(wave)
    return wave[order], flux[order]


def _normalize_observed(
    flux: np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if valid_mask is None:
        valid = np.ones_like(flux, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(flux) & (flux != 0.0)

    frac_nan = 1.0 - valid.mean() if valid.size > 0 else 1.0
    obs = np.where(valid, flux, np.nan)
    med = np.nanmedian(obs)
    if (not np.isfinite(med)) or med == 0:
        med = 1.0
    return (flux / med).astype(np.float64), valid.astype(bool), float(frac_nan)


def _normalize_observed_batch(
    flux: np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if valid_mask is None:
        valid = np.ones_like(flux, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool).copy()
    valid &= np.isfinite(flux) & (flux != 0.0)
    frac_nan = 1.0 - valid.mean(axis=1)

    obs = np.where(valid, flux, np.nan)
    med = np.nanmedian(obs, axis=1)
    med = np.where(np.isfinite(med) & (med != 0), med, 1.0)
    flux_norm = flux / med[:, None]
    return flux_norm.astype(np.float64), valid.astype(bool), frac_nan.astype(np.float64)


def _precompute_template_matrix(
    template_wave: np.ndarray,
    template_flux: np.ndarray,
    obs_wave: np.ndarray,
    z_grid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    nz = len(z_grid)
    l = len(obs_wave)
    tpl = np.full((nz, l), np.nan, dtype=np.float64)

    for i, z in enumerate(z_grid):
        shifted = template_wave * (1.0 + z)
        tpl[i] = np.interp(obs_wave, shifted, template_flux, left=np.nan, right=np.nan)

    tpl_valid = np.isfinite(tpl)
    med = np.nanmedian(np.where(tpl_valid, tpl, np.nan), axis=1)
    med = np.where(np.isfinite(med) & (med != 0), med, 1.0)
    tpl_norm = tpl / med[:, None]
    return tpl_norm.astype(np.float64), tpl_valid


def _compute_corr_vector(
    obs_flux_norm: np.ndarray,
    obs_valid: np.ndarray,
    tpl_norm: np.ndarray,
    tpl_valid: np.ndarray,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nz = tpl_norm.shape[0]
    corr = np.full(nz, -np.inf, dtype=np.float64)
    n_valid = np.zeros(nz, dtype=np.int32)
    obs = obs_flux_norm.astype(np.float64)

    for zi in range(nz):
        tv = tpl_norm[zi]
        vm = obs_valid & tpl_valid[zi]
        n = int(vm.sum())
        n_valid[zi] = n
        if n < min_overlap:
            continue

        o = obs[vm]
        t = tv[vm]
        o_mean = np.mean(o)
        t_mean = np.mean(t)
        do = o - o_mean
        dt = t - t_mean
        num = float(np.sum(do * dt))
        den = float(np.sqrt(np.sum(do * do) * np.sum(dt * dt)))
        corr[zi] = num / den if den > 0 else -np.inf

    return corr, n_valid


def _compute_corr_batch(
    obs_flux_norm: np.ndarray,
    obs_valid: np.ndarray,
    tpl_norm: np.ndarray,
    tpl_valid: np.ndarray,
    min_overlap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    b, _ = obs_flux_norm.shape
    nz = tpl_norm.shape[0]
    corr = np.full((b, nz), -np.inf, dtype=np.float64)
    n_valid = np.zeros((b, nz), dtype=np.int32)
    obs = obs_flux_norm.astype(np.float64)
    obs_clean = np.where(np.isfinite(obs), obs, 0.0)

    for zi in range(nz):
        tv = tpl_norm[zi].astype(np.float64)
        tv_clean = np.where(np.isfinite(tv), tv, 0.0)
        vm = obs_valid & tpl_valid[zi][None, :]
        n = vm.sum(axis=1)
        n_valid[:, zi] = n
        ok = n >= min_overlap
        if not np.any(ok):
            continue

        w = vm.astype(np.float64)
        sum_o = (obs_clean * w).sum(axis=1)
        sum_t = (tv_clean[None, :] * w).sum(axis=1)
        mean_o = np.divide(sum_o, n, out=np.zeros_like(sum_o), where=n > 0)
        mean_t = np.divide(sum_t, n, out=np.zeros_like(sum_t), where=n > 0)

        do = (obs_clean - mean_o[:, None]) * w
        dt = (tv_clean[None, :] - mean_t[:, None]) * w
        num = (do * dt).sum(axis=1)
        den = np.sqrt((do * do).sum(axis=1) * (dt * dt).sum(axis=1))
        c = np.divide(num, den, out=np.full_like(num, -np.inf), where=den > 0)
        c[~ok] = -np.inf
        corr[:, zi] = c

    return corr, n_valid


def _second_largest(x: np.ndarray) -> float:
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return float("nan")
    idx = np.argpartition(finite, -2)[-2:]
    vals = np.sort(finite[idx])
    return float(vals[-2])


def _extract_stats(corr: np.ndarray, n_valid: np.ndarray, z_grid: np.ndarray) -> Dict[str, float]:
    if not np.any(np.isfinite(corr)):
        return {
            "z_best": np.nan,
            "peak1": np.nan,
            "peak2": np.nan,
            "peak_ratio": np.nan,
            "peak_width": np.nan,
            "n_valid_best": 0,
            "valid": 0,
        }

    bi = int(np.nanargmax(corr))
    p1 = float(corr[bi])
    p2 = _second_largest(corr)
    z_best = float(z_grid[bi])
    nvb = int(n_valid[bi])

    if np.isfinite(p1):
        thr = 0.8 * p1
        left = bi
        while left > 0 and corr[left - 1] >= thr:
            left -= 1
        right = bi
        while right < len(corr) - 1 and corr[right + 1] >= thr:
            right += 1
        pwidth = float(z_grid[right] - z_grid[left])
    else:
        pwidth = np.nan

    pratio = float(p1 / p2) if np.isfinite(p2) and p2 != 0 else np.nan
    return {
        "z_best": z_best,
        "peak1": p1,
        "peak2": p2,
        "peak_ratio": pratio,
        "peak_width": pwidth,
        "n_valid_best": nvb,
        "valid": 1,
    }


def _extract_stats_batch(corr: np.ndarray, n_valid: np.ndarray, z_grid: np.ndarray) -> Dict[str, np.ndarray]:
    b, nz = corr.shape
    z_best = np.full(b, np.nan, dtype=np.float64)
    peak1 = np.full(b, np.nan, dtype=np.float64)
    peak2 = np.full(b, np.nan, dtype=np.float64)
    peak_ratio = np.full(b, np.nan, dtype=np.float64)
    peak_width = np.full(b, np.nan, dtype=np.float64)
    n_valid_best = np.zeros(b, dtype=np.int32)
    valid = np.zeros(b, dtype=np.int8)

    for i in range(b):
        row = corr[i]
        if not np.any(np.isfinite(row)):
            continue
        bi = int(np.nanargmax(row))
        p1 = float(row[bi])
        if not np.isfinite(p1):
            continue
        p2 = _second_largest(row)
        z_best[i] = float(z_grid[bi])
        peak1[i] = p1
        peak2[i] = p2
        peak_ratio[i] = float(p1 / p2) if np.isfinite(p2) and p2 != 0 else np.nan
        n_valid_best[i] = int(n_valid[i, bi])
        valid[i] = 1

        thr = 0.8 * p1
        left = bi
        while left > 0 and row[left - 1] >= thr:
            left -= 1
        right = bi
        while right < nz - 1 and row[right + 1] >= thr:
            right += 1
        peak_width[i] = float(z_grid[right] - z_grid[left])

    return {
        "z_best": z_best,
        "peak1": peak1,
        "peak2": peak2,
        "peak_ratio": peak_ratio,
        "peak_width": peak_width,
        "n_valid_best": n_valid_best,
        "valid": valid,
    }


def run_pcf_single(
    flux: ArrayLike,
    wave: ArrayLike,
    valid_mask: Optional[ArrayLike],
    template_wave: ArrayLike,
    template_flux: ArrayLike,
    config: Optional[PCFConfig] = None,
) -> Dict[str, Union[np.ndarray, float, int]]:
    cfg = config or PCFConfig()
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    template_wave = np.asarray(template_wave, dtype=np.float64)
    template_flux = np.asarray(template_flux, dtype=np.float64)

    z_grid = build_scaled_z_grid(cfg.z_min, cfg.z_max, cfg.base_z_step)
    obs_norm, obs_valid, frac_nan = _normalize_observed(flux, None if valid_mask is None else np.asarray(valid_mask, dtype=bool))

    if frac_nan > cfg.nan_threshold:
        corr = np.full(len(z_grid), -np.inf, dtype=np.float64)
        n_valid = np.zeros(len(z_grid), dtype=np.int32)
        stats = {
            "z_best": np.nan,
            "peak1": np.nan,
            "peak2": np.nan,
            "peak_ratio": np.nan,
            "peak_width": np.nan,
            "n_valid_best": 0,
            "valid": 0,
        }
    else:
        tpl_norm, tpl_valid = _precompute_template_matrix(template_wave, template_flux, wave, z_grid)
        corr, n_valid = _compute_corr_vector(obs_norm, obs_valid, tpl_norm, tpl_valid, cfg.min_overlap)
        stats = _extract_stats(corr, n_valid, z_grid)

    return {
        "z_grid": z_grid,
        "corr": corr,
        "n_valid": n_valid,
        "frac_nan": frac_nan,
        **stats,
    }


def select_best_template(result_by_template: Mapping[str, Mapping[str, Union[float, int]]]) -> Dict[str, Union[str, float]]:
    """Select best template by peak1, then overlap, then peak_ratio."""
    candidates = []
    for name, result in result_by_template.items():
        peak1 = float(result.get("peak1", np.nan))
        if not np.isfinite(peak1):
            continue
        n_valid = int(result.get("n_valid_best", 0))
        peak_ratio = float(result.get("peak_ratio", np.nan))
        if not np.isfinite(peak_ratio):
            peak_ratio = -np.inf
        z_best = float(result.get("z_best", np.nan))
        candidates.append((peak1, n_valid, peak_ratio, name, z_best))

    if not candidates:
        return {
            "z_temp": np.nan,
            "template_best": "none",
            "score_best": np.nan,
            "selection_reason": "no_valid_template",
        }

    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    peak1, n_valid, peak_ratio, name, z_best = candidates[0]

    if len(candidates) > 1 and np.isclose(peak1, candidates[1][0], rtol=0.0, atol=1e-8):
        if n_valid > candidates[1][1]:
            reason = "tie_peak1_break_n_valid"
        elif np.isclose(float(n_valid), float(candidates[1][1]), rtol=0.0, atol=0):
            reason = "tie_peak1_n_valid_break_peak_ratio"
        else:
            reason = "peak1"
    else:
        reason = "peak1"

    return {
        "z_temp": z_best,
        "template_best": name,
        "score_best": peak1,
        "selection_reason": reason,
    }


def _resolve_template_map(
    template_map: Optional[Mapping[str, TemplateInput]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if template_map is None:
        defaults = default_template_paths()
        template_map = {
            "type1": defaults["type1"],
            "type2": defaults["type2"],
        }

    resolved: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for name, item in template_map.items():
        if isinstance(item, tuple) and len(item) == 2:
            wave = np.asarray(item[0], dtype=np.float64)
            flux = np.asarray(item[1], dtype=np.float64)
            resolved[name] = (wave, flux)
        else:
            resolved[name] = load_template(Path(item))
    return resolved


def run_pcf_dual_templates(
    flux: ArrayLike,
    wave: ArrayLike,
    valid_mask: Optional[ArrayLike],
    template_map: Optional[Mapping[str, TemplateInput]] = None,
    config: Optional[PCFConfig] = None,
) -> Dict[str, Union[Dict, float, str]]:
    resolved = _resolve_template_map(template_map)
    per_template: Dict[str, Dict[str, Union[np.ndarray, float, int]]] = {}

    for name, (tw, tf) in resolved.items():
        per_template[name] = run_pcf_single(
            flux=flux,
            wave=wave,
            valid_mask=valid_mask,
            template_wave=tw,
            template_flux=tf,
            config=config,
        )

    selection = select_best_template(per_template)
    return {
        "templates": per_template,
        **selection,
    }


def run_pcf_dual_templates_batch(
    flux_batch: np.ndarray,
    wave: ArrayLike,
    valid_mask_batch: Optional[np.ndarray] = None,
    template_map: Optional[Mapping[str, TemplateInput]] = None,
    config: Optional[PCFConfig] = None,
) -> Dict[str, Union[Dict[str, Dict[str, np.ndarray]], np.ndarray]]:
    """Batch-level vectorized PCF for multiple spectra on a common wavelength grid."""
    cfg = config or PCFConfig()
    wave = np.asarray(wave, dtype=np.float64)
    flux_batch = np.asarray(flux_batch, dtype=np.float64)
    if flux_batch.ndim != 2:
        raise ValueError("flux_batch must be 2D (n_spectra, n_wave)")
    if flux_batch.shape[1] != wave.shape[0]:
        raise ValueError("flux_batch second dimension must match wave length")

    obs_norm, obs_valid, frac_nan = _normalize_observed_batch(flux_batch, valid_mask_batch)
    z_grid = build_scaled_z_grid(cfg.z_min, cfg.z_max, cfg.base_z_step)
    resolved = _resolve_template_map(template_map)

    per_template = {}
    for name, (tw, tf) in resolved.items():
        tpl_norm, tpl_valid = _precompute_template_matrix(
            np.asarray(tw, dtype=np.float64),
            np.asarray(tf, dtype=np.float64),
            wave,
            z_grid,
        )
        corr, n_valid = _compute_corr_batch(
            obs_flux_norm=obs_norm,
            obs_valid=obs_valid,
            tpl_norm=tpl_norm,
            tpl_valid=tpl_valid,
            min_overlap=cfg.min_overlap,
        )
        bad = frac_nan > cfg.nan_threshold
        if np.any(bad):
            corr[bad, :] = -np.inf
            n_valid[bad, :] = 0
        stats = _extract_stats_batch(corr, n_valid, z_grid)
        per_template[name] = {
            "z_grid": z_grid,
            "corr": corr,
            "n_valid": n_valid,
            "frac_nan": frac_nan,
            **stats,
        }

    n_obj = flux_batch.shape[0]
    z_temp = np.full(n_obj, np.nan, dtype=np.float64)
    template_best = np.full(n_obj, "none", dtype=object)
    score_best = np.full(n_obj, np.nan, dtype=np.float64)
    selection_reason = np.full(n_obj, "no_valid_template", dtype=object)

    names = list(per_template.keys())
    for i in range(n_obj):
        single = {
            name: {
                "z_best": float(per_template[name]["z_best"][i]),
                "peak1": float(per_template[name]["peak1"][i]),
                "peak2": float(per_template[name]["peak2"][i]),
                "peak_ratio": float(per_template[name]["peak_ratio"][i]),
                "peak_width": float(per_template[name]["peak_width"][i]),
                "n_valid_best": int(per_template[name]["n_valid_best"][i]),
                "valid": int(per_template[name]["valid"][i]),
            }
            for name in names
        }
        sel = select_best_template(single)
        z_temp[i] = float(sel["z_temp"]) if np.isfinite(sel["z_temp"]) else np.nan
        template_best[i] = sel["template_best"]
        score_best[i] = float(sel["score_best"]) if np.isfinite(sel["score_best"]) else np.nan
        selection_reason[i] = sel["selection_reason"]

    return {
        "templates": per_template,
        "z_temp": z_temp,
        "template_best": template_best,
        "score_best": score_best,
        "selection_reason": selection_reason,
    }


def update_fits_z_temp(
    fits_path: Union[str, Path],
    rows: Iterable[Mapping[str, Union[str, float, int]]],
    *,
    by: str = "ext",
    z_key: str = "Z_TEMP",
    backend: str = "fitsio",
) -> int:
    """Update FITS HDU headers with z-temp values.

    rows: items should include ``z_temp`` and either ``ext`` or ``extname``.
    Returns count of updated HDUs.
    """
    rows = list(rows)
    mode = str(backend).lower()
    if mode not in {"fitsio", "astropy"}:
        raise ValueError("backend must be 'fitsio' or 'astropy'")

    if mode == "fitsio":
        try:
            import fitsio  # type: ignore
        except Exception:
            mode = "astropy"

    def _resolve_mode(row):
        if by != "auto":
            return by
        ext_val = row.get("ext", None)
        if ext_val is not None:
            try:
                if np.isfinite(float(ext_val)):
                    return "ext"
            except Exception:
                pass
        return "extname"

    if mode == "fitsio":
        updated = 0
        import fitsio  # type: ignore
        with fitsio.FITS(str(fits_path), "rw") as ff:
            for row in rows:
                if "z_temp" not in row:
                    continue
                z_val = row["z_temp"]
                if z_val is None or not np.isfinite(float(z_val)):
                    continue

                this_by = _resolve_mode(row)
                target_ext = None
                if this_by == "ext":
                    ext_val = row.get("ext", None)
                    if ext_val is None:
                        continue
                    try:
                        ext_i = int(ext_val)
                    except Exception:
                        continue
                    if ext_i < 1 or ext_i >= len(ff):
                        continue
                    target_ext = ext_i
                elif this_by == "extname":
                    key = row.get("extname", row.get("objid"))
                    if key is None:
                        continue
                    key_s = str(key).strip()
                    for ext_i in range(1, len(ff)):
                        h = ff[ext_i].read_header()
                        if str(h.get("EXTNAME", "")).strip() == key_s:
                            target_ext = ext_i
                            break
                    if target_ext is None:
                        continue
                else:
                    raise ValueError("by must be 'ext', 'extname', or 'auto'")

                ff[target_ext].write_key(z_key, float(z_val))
                if "template_best" in row and row["template_best"] is not None:
                    ff[target_ext].write_key("ZTEMPPL", str(row["template_best"]))
                if "score_best" in row and row["score_best"] is not None and np.isfinite(float(row["score_best"])):
                    ff[target_ext].write_key("ZTMPSCR", float(row["score_best"]))
                updated += 1
        return updated

    updated = 0
    with fits.open(fits_path, mode="update") as hdul:
        for row in rows:
            if "z_temp" not in row:
                continue
            z_val = row["z_temp"]
            if z_val is None or not np.isfinite(float(z_val)):
                continue

            this_by = _resolve_mode(row)
            if this_by == "ext":
                if "ext" not in row:
                    continue
                ext = int(row["ext"])
                if ext < 1 or ext >= len(hdul):
                    continue
                hdu = hdul[ext]
            elif this_by == "extname":
                key = row.get("extname", row.get("objid"))
                if key is None:
                    continue
                try:
                    hdu = hdul[str(key)]
                except Exception:
                    continue
            else:
                raise ValueError("by must be 'ext', 'extname', or 'auto'")

            hdu.header[z_key] = float(z_val)
            if "template_best" in row and row["template_best"] is not None:
                hdu.header["ZTEMPPL"] = str(row["template_best"])
            if "score_best" in row and row["score_best"] is not None and np.isfinite(float(row["score_best"])):
                hdu.header["ZTMPSCR"] = float(row["score_best"])
            updated += 1
        hdul.flush()
    return updated


def update_parquet_z_temp(
    parquet_path: Union[str, Path],
    rows: Iterable[Mapping[str, Union[str, float, int]]],
    *,
    id_col: str = "objid",
    z_col: str = "z_temp_pcf",
) -> int:
    """Update parquet table with PCF z-temp columns (in-place write)."""
    path = Path(parquet_path)
    df = pd.read_parquet(path)

    if id_col not in df.columns:
        raise KeyError(f"id_col '{id_col}' not found in parquet columns")

    update_df = pd.DataFrame(list(rows))
    if update_df.empty:
        return 0
    if id_col not in update_df.columns:
        # allow extname fallback mapping
        if "extname" in update_df.columns and id_col == "objid":
            update_df[id_col] = update_df["extname"]
        else:
            raise KeyError(f"id_col '{id_col}' missing in rows")

    cols = [id_col]
    if "z_temp" in update_df.columns:
        update_df = update_df.rename(columns={"z_temp": z_col})
        cols.append(z_col)
    for extra in ["template_best", "score_best", "selection_reason"]:
        if extra in update_df.columns:
            cols.append(extra)

    update_df = update_df[cols].drop_duplicates(subset=[id_col], keep="last")

    merged = df.merge(update_df, on=id_col, how="left", suffixes=("", "__pcf_new"))

    updated_rows = 0
    if z_col in update_df.columns:
        if z_col in df.columns:
            newcol = f"{z_col}__pcf_new"
            mask = merged[newcol].notna()
            merged.loc[mask, z_col] = merged.loc[mask, newcol]
            merged.drop(columns=[newcol], inplace=True)
            updated_rows = int(mask.sum())
        else:
            newcol = f"{z_col}__pcf_new"
            merged[z_col] = merged[newcol]
            merged.drop(columns=[newcol], inplace=True)
            updated_rows = int(merged[z_col].notna().sum())

    for extra in ["template_best", "score_best", "selection_reason"]:
        newcol = f"{extra}__pcf_new"
        if newcol not in merged.columns:
            continue
        if extra in df.columns:
            mask = merged[newcol].notna()
            merged.loc[mask, extra] = merged.loc[mask, newcol]
        else:
            merged[extra] = merged[newcol]
        merged.drop(columns=[newcol], inplace=True)

    merged.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return updated_rows
