#!/usr/bin/env python
"""Build Euclid BGS+RGS coadds (FITS + parquet) using SpecEuclid1dDual.

Coadds are produced on the RGS wavelength grid; BGS is resampled to RGS.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits

from specbox.basemodule import SpecEuclid1dDual


def _count_hdus(filename: str) -> int:
    with fits.open(filename) as hdul:
        return max(0, len(hdul) - 1)


def _parse_extname_list(extnames: Optional[str]) -> List[str]:
    if not extnames:
        return []
    values = []
    for item in extnames.split(","):
        value = item.strip()
        if value:
            values.append(value)
    return values


def _open_extnames(path: str) -> List[str]:
    with fits.open(path) as hdul:
        return [str(h.name).strip() for h in hdul[1:] if str(h.name).strip()]


def _iter_targets(
    rgs_file: str,
    bgs_file: str,
    ext: Optional[int],
    extnames: Optional[str],
    pair_by: str = "extname_intersection",
) -> Iterable[Tuple[Optional[int], Optional[str]]]:
    extname_values = _parse_extname_list(extnames)
    if extname_values:
        for extname in extname_values:
            yield None, extname
        return

    if ext is not None:
        yield int(ext), None
        return

    if pair_by == "extname_intersection":
        rgs_ext = set(_open_extnames(rgs_file))
        bgs_ext = set(_open_extnames(bgs_file))
        shared = sorted(rgs_ext.intersection(bgs_ext))
        for extname in shared:
            yield None, extname
        return

    n = min(_count_hdus(rgs_file), _count_hdus(bgs_file))
    for index in range(1, n + 1):
        yield index, None


def _to_table_hdu(row: Dict, index: int) -> fits.BinTableHDU:
    wave = np.asarray(row["wavelength"], dtype=float)
    flux = np.asarray(row["flux"], dtype=float)
    err = np.asarray(row["err"], dtype=float)
    mask = np.asarray(row["mask"], dtype=np.int64)
    arm = np.asarray(row["arm"], dtype="U12")

    columns = [
        fits.Column(name="WAVELENGTH", format="D", unit="Angstrom", array=wave),
        fits.Column(name="FLUX", format="D", unit="10**-16 erg/s/cm2/Angstrom", array=flux),
        fits.Column(name="ERR", format="D", unit="10**-16 erg/s/cm2/Angstrom", array=err),
        fits.Column(name="MASK", format="K", unit="Number", array=mask),
        fits.Column(name="ARM", format="12A", array=arm),
    ]
    hdu = fits.BinTableHDU.from_columns(columns)
    hdu.header["EXTNAME"] = str(row.get("objid", f"OBJ_{index:05d}"))
    hdu.header["LRANGE"] = "COADD"
    hdu.header["WMIN"] = float(np.nanmin(wave)) if wave.size > 0 else np.nan
    hdu.header["BINWIDTH"] = (
        float(np.nanmedian(np.diff(wave))) if wave.size > 1 else np.nan
    )
    hdu.header["BINCOUNT"] = int(len(wave))
    hdu.header["OBJID"] = str(row.get("objid", ""))
    hdu.header["RA"] = float(row.get("ra", np.nan))
    hdu.header["DEC"] = float(row.get("dec", np.nan))
    hdu.header["SCALB2R"] = float(row.get("scale_bgs_to_rgs", np.nan))
    hdu.header["SCLSTAT"] = str(row.get("scale_status", "unknown"))
    hdu.header["OVWMIN"] = float(row.get("overlap_wmin", np.nan))
    hdu.header["OVWMAX"] = float(row.get("overlap_wmax", np.nan))
    hdu.header["OVNBGS"] = int(row.get("overlap_n_bgs", 0))
    hdu.header["OVNRGS"] = int(row.get("overlap_n_rgs", 0))
    hdu.header["STATUS"] = str(row.get("status", "ok"))
    hdu.header["EXT"] = int(row.get("ext", 0)) if row.get("ext") is not None else -1
    hdu.header["EXTNAMEI"] = str(row.get("extname", ""))
    return hdu


def _parquet_chunks(rows: List[Dict], prefix: Path, chunk_size: int):
    if not rows:
        return []
    if chunk_size <= 0:
        chunk_size = len(rows)
    files = []
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        df = pd.DataFrame(chunk)
        filename = prefix.with_name(f"{prefix.name}_part{i // chunk_size + 1:03d}.parquet")
        df.to_parquet(filename, compression="snappy", engine="pyarrow", index=False)
        files.append(filename)
    return files


def build_coadds(
    rgs_file: str,
    bgs_file: str,
    output_prefix: str,
    ext: Optional[int] = None,
    extnames: Optional[str] = None,
    include_arms: bool = False,
    parquet_chunk_size: int = 2000,
    pair_by: str = "extname_intersection",
):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    success_rows: List[Dict] = []
    log_rows: List[Dict] = []

    for idx, (ext_value, extname_value) in enumerate(
        _iter_targets(rgs_file, bgs_file, ext, extnames, pair_by=pair_by), start=1
    ):
        status = "ok"
        message = ""
        try:
            sp = SpecEuclid1dDual(
                rgs_file=rgs_file,
                bgs_file=bgs_file,
                ext=ext_value,
                extname=extname_value,
                good_pixels_only=False,
            )
            payload = sp.for_redshift()
            merged = payload["merged"]
            if len(merged["wavelength"]) == 0:
                status = "empty"
            elif payload.get("scale_status") in ("missing_arm", "no_overlap"):
                status = payload.get("scale_status")

            row = {
                "ext": ext_value,
                "extname": extname_value,
                "objid": payload.get("objid"),
                "ra": payload.get("ra", np.nan),
                "dec": payload.get("dec", np.nan),
                "scale_bgs_to_rgs": payload.get("scale_bgs_to_rgs", np.nan),
                "scale_status": payload.get("scale_status", "unknown"),
                "overlap_wmin": merged.get("overlap_wmin", np.nan),
                "overlap_wmax": merged.get("overlap_wmax", np.nan),
                "overlap_n_bgs": merged.get("overlap_n_bgs", 0),
                "overlap_n_rgs": merged.get("overlap_n_rgs", 0),
                "wavelength": np.asarray(merged.get("wavelength", []), dtype=float).tolist(),
                "flux": np.asarray(merged.get("flux", []), dtype=float).tolist(),
                "err": np.asarray(merged.get("err", []), dtype=float).tolist(),
                "mask": np.asarray(merged.get("mask", []), dtype=np.int64).tolist(),
                "arm": np.asarray(merged.get("arm", []), dtype="U12").tolist(),
                "status": status,
            }
            if include_arms:
                row["rgs_wavelength"] = np.asarray(payload["rgs"]["wavelength"], dtype=float).tolist() if payload["rgs"]["wavelength"] is not None else []
                row["rgs_flux"] = np.asarray(payload["rgs"]["flux"], dtype=float).tolist() if payload["rgs"]["flux"] is not None else []
                row["rgs_err"] = np.asarray(payload["rgs"]["err"], dtype=float).tolist() if payload["rgs"]["err"] is not None else []
                row["bgs_wavelength"] = np.asarray(payload["bgs"]["wavelength"], dtype=float).tolist() if payload["bgs"]["wavelength"] is not None else []
                row["bgs_flux"] = np.asarray(payload["bgs"]["flux"], dtype=float).tolist() if payload["bgs"]["flux"] is not None else []
                row["bgs_err"] = np.asarray(payload["bgs"]["err"], dtype=float).tolist() if payload["bgs"]["err"] is not None else []
                row["bgs_raw_wavelength"] = np.asarray(payload["bgs_raw"]["wavelength"], dtype=float).tolist() if payload["bgs_raw"]["wavelength"] is not None else []
                row["bgs_raw_flux"] = np.asarray(payload["bgs_raw"]["flux"], dtype=float).tolist() if payload["bgs_raw"]["flux"] is not None else []
                row["bgs_raw_err"] = np.asarray(payload["bgs_raw"]["err"], dtype=float).tolist() if payload["bgs_raw"]["err"] is not None else []

            success_rows.append(row)
        except Exception as exc:
            status = "failed"
            message = str(exc)

        log_rows.append(
            {
                "index": idx,
                "ext": ext_value,
                "extname": extname_value,
                "status": status,
                "message": message,
            }
        )

    fits_path = prefix.with_suffix(".fits")
    hdus = [fits.PrimaryHDU()]
    for idx, row in enumerate(success_rows, start=1):
        hdus.append(_to_table_hdu(row, idx))
    fits.HDUList(hdus).writeto(fits_path, overwrite=True)

    parquet_files = _parquet_chunks(success_rows, prefix, parquet_chunk_size)

    log_df = pd.DataFrame(log_rows)
    log_path = prefix.with_name(f"{prefix.name}_log.csv")
    log_df.to_csv(log_path, index=False)

    print(f"Wrote FITS coadd file: {fits_path}")
    for p in parquet_files:
        print(f"Wrote parquet chunk: {p}")
    print(f"Wrote processing log: {log_path}")
    ok_count = int((log_df["status"] == "ok").sum()) if not log_df.empty else 0
    fail_count = int((log_df["status"] == "failed").sum()) if not log_df.empty else 0
    print(f"Summary: total={len(log_rows)} ok={ok_count} failed={fail_count}")


def main():
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


if __name__ == "__main__":
    main()
