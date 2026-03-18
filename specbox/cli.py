#!/usr/bin/env python
"""Console entry points for specbox."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from astropy.io import fits

from .auxmodule.pcf_redshift import (
    PCFConfig,
    default_template_paths,
    run_pcf_dual_templates_batch,
    run_pcf_dual_templates,
    update_fits_z_temp,
    update_parquet_z_temp,
)
from .basemodule import SpecEuclid1d, SpecIRAF, SpecLAMOST, SpecSDSS, SpecSparcl
from .examples.tools.build_euclid_bgs_rgs_coadd import build_coadds
from .examples.tools.build_euclid_raw_parquet import build_raw_euclid_parquet
from .basemodule import SpecEuclidCoaddRow


def _iter_hdus(path: Path):
    with fits.open(path) as hdul:
        for ext in range(1, len(hdul)):
            yield ext, hdul[ext].name


def _guess_parquet_parts_from_fits(fits_path: Path):
    base = fits_path.with_suffix("")
    candidates = sorted(base.parent.glob(f"{base.name}_part*.parquet"))
    return candidates


def _build_rows_from_batch_result(df_in: pd.DataFrame, result: Dict) -> list:
    rows = []
    template_names = list(result.get("templates", {}).keys())
    for i, (_, rec) in enumerate(df_in.iterrows()):
        ext_val = rec.get("ext", None)
        try:
            if ext_val is not None and np.isfinite(float(ext_val)):
                ext_val = int(ext_val)
        except Exception:
            ext_val = None
        extname_val = rec.get("extname", rec.get("objid", None))
        objid_val = rec.get("objid", extname_val)
        row = {
            "ext": ext_val,
            "extname": extname_val,
            "objid": objid_val,
            "z_temp": float(result["z_temp"][i]) if np.isfinite(result["z_temp"][i]) else np.nan,
            "template_best": str(result["template_best"][i]),
            "score_best": float(result["score_best"][i]) if np.isfinite(result["score_best"][i]) else np.nan,
            "selection_reason": str(result["selection_reason"][i]),
        }
        for tname in template_names:
            tres = result["templates"][tname]
            row[f"{tname}_z_best"] = float(tres["z_best"][i]) if np.isfinite(tres["z_best"][i]) else np.nan
            row[f"{tname}_peak1"] = float(tres["peak1"][i]) if np.isfinite(tres["peak1"][i]) else np.nan
            row[f"{tname}_peak2"] = float(tres["peak2"][i]) if np.isfinite(tres["peak2"][i]) else np.nan
            row[f"{tname}_peak_ratio"] = float(tres["peak_ratio"][i]) if np.isfinite(tres["peak_ratio"][i]) else np.nan
            row[f"{tname}_peak_width"] = float(tres["peak_width"][i]) if np.isfinite(tres["peak_width"][i]) else np.nan
            row[f"{tname}_n_valid_best"] = int(tres["n_valid_best"][i])
        rows.append(row)
    return rows


def _spec_class_map() -> Dict[str, type]:
    return {
        "euclid": SpecEuclid1d,
        "euclid-coadd": SpecEuclidCoaddRow,
        "sparcl": SpecSparcl,
        "lamost": SpecLAMOST,
        "sdss": SpecSDSS,
        "iraf": SpecIRAF,
    }


def viewer_cli() -> None:
    parser = argparse.ArgumentParser(description="Launch specbox enhanced viewer.")
    parser.add_argument("--spectra", default=None, help="Input FITS/parquet/list file path.")
    parser.add_argument("--spec-class", default="euclid", choices=sorted(_spec_class_map().keys()))
    parser.add_argument("--output-file", default=None, help="Output CSV for labels/history.")
    parser.add_argument("--z-max", type=float, default=5.0)
    parser.add_argument(
        "--load-history",
        dest="load_history",
        action="store_true",
        default=None,
        help="Force loading existing output-file as history.",
    )
    parser.add_argument(
        "--no-load-history",
        dest="load_history",
        action="store_false",
        help="Disable history loading even if output-file exists.",
    )
    parser.add_argument("--euclid-fits", default=None, help="Optional Euclid file for SPARCL overlay.")
    parser.add_argument("--cutout-buffer-dir", default=None, help="Optional cutout buffer directory.")
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Disable the image panel and all cutout downloading.",
    )
    parser.add_argument("--disable-background-prefetch", action="store_true")
    parser.add_argument("--rgs-file", default=None, help="Dual-arm mode: RGS FITS file.")
    parser.add_argument("--bgs-file", default=None, help="Dual-arm mode: BGS FITS file.")
    parser.add_argument("--ext", type=int, default=None, help="Dual-arm shared extension index.")
    parser.add_argument("--extname", default=None, help="Dual-arm shared extension name.")
    parser.add_argument("--dual-good-pixels-only", action="store_true")
    args = parser.parse_args()

    from .qtmodule import PGSpecPlotThreadEnhanced

    spec_class = _spec_class_map()[args.spec_class]
    spectra = args.spectra
    if spectra is None:
        spectra = args.rgs_file if args.rgs_file else args.bgs_file
    if spectra is None:
        raise ValueError("Provide --spectra or dual-arm inputs --rgs-file/--bgs-file.")
    if args.output_file is None:
        input_path = Path(str(spectra))
        input_file_name = input_path.stem
        output_file = str(input_path.parent / f"vi_{input_file_name}_results.csv")
    else:
        output_file = args.output_file
    should_load_history = args.load_history
    if should_load_history is None:
        should_load_history = os.path.exists(output_file)

    viewer = PGSpecPlotThreadEnhanced(
        spectra=spectra,
        SpecClass=spec_class,
        output_file=output_file,
        z_max=args.z_max,
        load_history=should_load_history,
        euclid_fits=args.euclid_fits,
        cutout_buffer_dir=args.cutout_buffer_dir,
        enable_image_panel=not args.no_images,
        enable_background_prefetch=not args.disable_background_prefetch,
        rgs_file=args.rgs_file,
        bgs_file=args.bgs_file,
        ext=args.ext,
        extname=args.extname,
        dual_good_pixels_only=args.dual_good_pixels_only,
    )
    viewer.run()


def pcf_cli() -> None:
    parser = argparse.ArgumentParser(description="Run dual-template PCF and update Z_TEMP.")
    parser.add_argument("--fits", required=True, help="Input Euclid FITS/coadd file.")
    parser.add_argument("--output-csv", default="pcf_results.csv", help="Output CSV path.")
    parser.add_argument("--parquet-input", default=None, help="Optional parquet input for faster PCF compute (default: auto-detect from FITS prefix).")
    parser.add_argument("--parquet", default=None, help="Optional parquet file to update (defaults to parquet input when provided).")
    parser.add_argument("--id-col", default="objid", help="Parquet ID column.")
    parser.add_argument("--by", choices=["auto", "ext", "extname"], default="auto", help="Key for FITS header update.")
    parser.add_argument("--write-backend", choices=["fitsio", "astropy"], default="fitsio", help="FITS header write backend.")
    parser.add_argument("--z-key", default="Z_TEMP", help="Header key for best PCF redshift.")
    parser.add_argument("--template-type1", default=None, help="Override Type 1 template path.")
    parser.add_argument("--template-ragn-dr1", default=None, help="Override ragn_dr1 template path.")
    parser.add_argument(
        "--ragn-dr1-only",
        action="store_true",
        help="Use ragn_dr1 as the only active template (mapped to type1).",
    )
    parser.add_argument("--template-type2", default=None, help="Override Type 2 template path.")
    parser.add_argument(
        "--enable-ragn-dr1",
        action="store_true",
        help="Enable ragn_dr1 template in PCF.",
    )
    parser.add_argument(
        "--enable-type2",
        action="store_true",
        help="Enable Type 2 template in PCF (disabled by default).",
    )
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=5.5)
    parser.add_argument("--base-z-step", type=float, default=0.001)
    parser.add_argument("--min-overlap", type=int, default=50)
    parser.add_argument("--nan-threshold", type=float, default=0.5)
    args = parser.parse_args()

    fits_path = Path(args.fits)
    default_templates = default_template_paths()
    if args.ragn_dr1_only:
        templates = {
            "type1": Path(args.template_ragn_dr1) if args.template_ragn_dr1 else default_templates["ragn_dr1"]
        }
    else:
        templates = {"type1": default_templates["type1"]}
        if args.template_type1:
            templates["type1"] = Path(args.template_type1)

        use_ragn_dr1 = args.enable_ragn_dr1 or args.template_ragn_dr1 is not None
        if use_ragn_dr1:
            templates["ragn_dr1"] = (
                Path(args.template_ragn_dr1) if args.template_ragn_dr1 else default_templates["ragn_dr1"]
            )

    use_type2 = (not args.ragn_dr1_only) and (args.enable_type2 or args.template_type2 is not None)
    if use_type2:
        templates["type2"] = (
            Path(args.template_type2) if args.template_type2 else default_templates["type2"]
        )

    cfg = PCFConfig(
        z_min=args.z_min,
        z_max=args.z_max,
        base_z_step=args.base_z_step,
        min_overlap=args.min_overlap,
        nan_threshold=args.nan_threshold,
    )

    parquet_inputs = []
    if args.parquet_input:
        parquet_inputs = [Path(args.parquet_input)]
    else:
        parquet_inputs = _guess_parquet_parts_from_fits(fits_path)

    rows = []
    if parquet_inputs:
        print(f"Using parquet input for PCF: {[str(p) for p in parquet_inputs]}")
        for pq in parquet_inputs:
            df_in = pd.read_parquet(pq)
            if df_in.empty:
                continue
            try:
                wave0 = np.asarray(df_in.iloc[0]["wavelength"], dtype=float)
                same_grid = True
                flux_batch = []
                valid_batch = []
                for _, rec in df_in.iterrows():
                    wave = np.asarray(rec["wavelength"], dtype=float)
                    flux = np.asarray(rec["flux"], dtype=float)
                    if wave.shape != wave0.shape or not np.allclose(wave, wave0, rtol=0.0, atol=1e-8, equal_nan=True):
                        same_grid = False
                        break
                    flux_batch.append(flux)
                    if "mask" in rec and rec["mask"] is not None:
                        mask = np.asarray(rec["mask"], dtype=np.int64)
                        valid = np.isfinite(wave) & np.isfinite(flux) & (flux != 0.0) & ((mask & 1) == 0) & ((mask & 64) == 0)
                    else:
                        valid = np.isfinite(wave) & np.isfinite(flux) & (flux != 0.0)
                    valid_batch.append(valid)

                if same_grid and len(flux_batch) > 0:
                    result_batch = run_pcf_dual_templates_batch(
                        flux_batch=np.asarray(flux_batch, dtype=float),
                        wave=wave0,
                        valid_mask_batch=np.asarray(valid_batch, dtype=bool),
                        template_map=templates,
                        config=cfg,
                    )
                    rows.extend(_build_rows_from_batch_result(df_in, result_batch))
                    continue
            except Exception:
                # Fallback to single-spectrum flow below.
                pass

            for _, rec in df_in.iterrows():
                ext_val = rec.get("ext", None)
                try:
                    if ext_val is not None and np.isfinite(float(ext_val)):
                        ext_val = int(ext_val)
                except Exception:
                    ext_val = None
                extname_val = rec.get("extname", rec.get("objid", None))
                objid_val = rec.get("objid", extname_val)
                try:
                    wave = np.asarray(rec["wavelength"], dtype=float)
                    flux = np.asarray(rec["flux"], dtype=float)
                    if "mask" in rec and rec["mask"] is not None:
                        mask = np.asarray(rec["mask"], dtype=np.int64)
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
                        "ext": ext_val,
                        "extname": extname_val,
                        "objid": objid_val,
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
                    rows.append(
                        {
                            "ext": ext_val,
                            "extname": extname_val,
                            "objid": objid_val,
                            "z_temp": np.nan,
                            "template_best": "none",
                            "score_best": np.nan,
                            "selection_reason": "error",
                            "error": str(exc),
                        }
                    )
    else:
        print("No parquet input detected; using FITS spectra for PCF.")
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
                rows.append(
                    {
                        "ext": ext,
                        "extname": extname,
                        "objid": extname,
                        "z_temp": np.nan,
                        "template_best": "none",
                        "score_best": np.nan,
                        "selection_reason": "error",
                        "error": str(exc),
                    }
                )

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved PCF summary: {output_csv.resolve()}")

    rows_ok = [row for row in rows if np.isfinite(row.get("z_temp", np.nan))]
    n_updated = update_fits_z_temp(
        fits_path,
        rows_ok,
        by=args.by,
        z_key=args.z_key,
        backend=args.write_backend,
    )
    print(f"Updated FITS headers: {n_updated} HDUs in {fits_path}")

    parquet_update_path = args.parquet
    if parquet_update_path is None and args.parquet_input:
        parquet_update_path = args.parquet_input
    if parquet_update_path:
        n_pq = update_parquet_z_temp(parquet_update_path, rows_ok, id_col=args.id_col, z_col="z_temp_pcf")
        print(f"Updated parquet rows: {n_pq} in {parquet_update_path}")


def coadd_cli() -> None:
    parser = argparse.ArgumentParser(description="Build Euclid BGS+RGS coadds (FITS + parquet).")
    parser.add_argument("--rgs-file", required=True, help="Input RGS FITS file")
    parser.add_argument("--bgs-file", required=True, help="Input BGS FITS file")
    parser.add_argument("--output-prefix", required=True, help="Output prefix (no extension)")
    parser.add_argument("--ext", type=int, default=None, help="Single extension index (1-based)")
    parser.add_argument("--extnames", default=None, help="Comma-separated extnames to process")
    parser.add_argument(
        "--pair-by",
        choices=["extname_intersection", "ext_index"],
        default="extname_intersection",
        help="Pairing mode when --ext/--extnames are not provided.",
    )
    parser.add_argument("--include-arms", action="store_true", help="Include per-arm arrays in parquet")
    parser.add_argument("--parquet-chunk-size", type=int, default=2000, help="Rows per parquet chunk")
    args = parser.parse_args()

    build_coadds(
        rgs_file=args.rgs_file,
        bgs_file=args.bgs_file,
        output_prefix=args.output_prefix,
        ext=args.ext,
        extnames=args.extnames,
        include_arms=args.include_arms,
        parquet_chunk_size=args.parquet_chunk_size,
        pair_by=args.pair_by,
    )


def euclid_parquet_cli() -> None:
    parser = argparse.ArgumentParser(description="Build parquet partitions from raw single-arm Euclid FITS.")
    parser.add_argument("--fits", required=True, help="Input raw Euclid combined FITS file")
    parser.add_argument("--output-prefix", required=True, help="Output prefix (no extension)")
    parser.add_argument("--parquet-chunk-size", type=int, default=2000, help="Rows per parquet chunk")
    args = parser.parse_args()

    build_raw_euclid_parquet(
        fits_file=args.fits,
        output_prefix=args.output_prefix,
        parquet_chunk_size=args.parquet_chunk_size,
    )
