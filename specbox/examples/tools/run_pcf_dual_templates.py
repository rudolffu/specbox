#!/usr/bin/env python
"""Run dual-template PCF redshift on Euclid FITS/coadd and write Z_TEMP."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from specbox.auxmodule.pcf_redshift import (
    PCFConfig,
    default_template_paths,
    run_pcf_dual_templates,
    update_fits_z_temp,
    update_parquet_z_temp,
)
from specbox.basemodule import SpecEuclid1d


def _iter_hdus(path: Path):
    with fits.open(path) as hdul:
        for ext in range(1, len(hdul)):
            yield ext, hdul[ext].name


def main():
    parser = argparse.ArgumentParser(description="Run dual-template PCF and update Z_TEMP.")
    parser.add_argument("--fits", required=True, help="Input Euclid FITS/coadd file.")
    parser.add_argument("--output-csv", default="pcf_results.csv", help="Output CSV path.")
    parser.add_argument("--parquet", default=None, help="Optional parquet file to update.")
    parser.add_argument("--id-col", default="objid", help="Parquet ID column.")

    parser.add_argument("--template-type1", default=None, help="Override Type 1 template path.")
    parser.add_argument("--template-type2", default=None, help="Override Type 2 template path.")

    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=5.5)
    parser.add_argument("--base-z-step", type=float, default=0.001)
    parser.add_argument("--min-overlap", type=int, default=50)
    parser.add_argument("--nan-threshold", type=float, default=0.5)

    parser.add_argument("--by", choices=["ext", "extname"], default="ext", help="Key used to update FITS headers.")
    parser.add_argument("--z-key", default="Z_TEMP", help="Header keyword to write.")
    args = parser.parse_args()

    fits_path = Path(args.fits)
    templates = default_template_paths()
    if args.template_type1:
        templates["type1"] = Path(args.template_type1)
    if args.template_type2:
        templates["type2"] = Path(args.template_type2)

    cfg = PCFConfig(
        z_min=args.z_min,
        z_max=args.z_max,
        base_z_step=args.base_z_step,
        min_overlap=args.min_overlap,
        nan_threshold=args.nan_threshold,
    )

    rows = []
    for ext, extname in _iter_hdus(fits_path):
        try:
            sp = SpecEuclid1d(str(fits_path), ext=ext, clip=False, good_pixels_only=False)
            wave = np.asarray(sp.wave.value if hasattr(sp.wave, "value") else sp.wave, dtype=float)
            flux = np.asarray(sp.flux.value if hasattr(sp.flux, "value") else sp.flux, dtype=float)

            if getattr(sp, "mask", None) is not None:
                mask = np.asarray(sp.mask, dtype=np.int64)
                valid_mask = np.isfinite(wave) & np.isfinite(flux) & (flux != 0.0) & ((mask & 1) == 0) & ((mask & 64) == 0)
            else:
                valid_mask = np.isfinite(wave) & np.isfinite(flux) & (flux != 0.0)

            result = run_pcf_dual_templates(
                flux=flux,
                wave=wave,
                valid_mask=valid_mask,
                template_map=templates,
                config=cfg,
            )

            row = {
                "ext": ext,
                "extname": extname,
                "objid": getattr(sp, "objid", extname),
                "z_temp": float(result.get("z_temp", np.nan)),
                "template_best": result.get("template_best", "none"),
                "score_best": float(result.get("score_best", np.nan)),
                "selection_reason": result.get("selection_reason", ""),
            }
            for tname, tres in result.get("templates", {}).items():
                row[f"{tname}_z_best"] = float(tres.get("z_best", np.nan))
                row[f"{tname}_peak1"] = float(tres.get("peak1", np.nan))
                row[f"{tname}_peak2"] = float(tres.get("peak2", np.nan))
                row[f"{tname}_peak_ratio"] = float(tres.get("peak_ratio", np.nan))
                row[f"{tname}_peak_width"] = float(tres.get("peak_width", np.nan))
                row[f"{tname}_n_valid_best"] = int(tres.get("n_valid_best", 0))
            rows.append(row)
        except Exception as exc:
            rows.append({
                "ext": ext,
                "extname": extname,
                "objid": extname,
                "z_temp": np.nan,
                "template_best": "none",
                "score_best": np.nan,
                "selection_reason": "error",
                "error": str(exc),
            })

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved PCF summary: {out_csv.resolve()}")

    rows_ok = [r for r in rows if np.isfinite(r.get("z_temp", np.nan))]
    n_updated = update_fits_z_temp(fits_path, rows_ok, by=args.by, z_key=args.z_key)
    print(f"Updated FITS headers: {n_updated} HDUs in {fits_path}")

    if args.parquet:
        n_pq = update_parquet_z_temp(args.parquet, rows_ok, id_col=args.id_col, z_col="z_temp_pcf")
        print(f"Updated parquet rows: {n_pq} in {args.parquet}")


if __name__ == "__main__":
    main()
