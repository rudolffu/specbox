#!/usr/bin/env python
"""Build parquet partitions from raw single-arm Euclid combined FITS files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

from specbox.basemodule import SpecEuclid1d


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


def build_raw_euclid_parquet(
    fits_file: str,
    output_prefix: str,
    parquet_chunk_size: int = 2000,
):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    log_rows: List[Dict] = []

    with fits.open(fits_file) as hdul:
        total = max(0, len(hdul) - 1)

    for ext in range(1, total + 1):
        status = "ok"
        message = ""
        try:
            sp = SpecEuclid1d(fits_file, ext=ext, clip=False, good_pixels_only=False)
            quality = None
            ndith = None
            if getattr(sp, "data", None) is not None and getattr(sp.data, "dtype", None) is not None:
                names = sp.data.dtype.names or ()
                if "QUALITY" in names:
                    quality = np.asarray(sp.data["QUALITY"], dtype=float)
                if "NDITH" in names:
                    ndith = np.asarray(sp.data["NDITH"], dtype=np.int16)

            row = {
                "ext": ext,
                "extname": str(getattr(sp, "objname", ext)),
                "objid": getattr(sp, "objid", ext),
                "objname": getattr(sp, "objname", "Unknown"),
                "lrange": getattr(sp, "lrange", None),
                "ra": getattr(sp, "ra", np.nan),
                "dec": getattr(sp, "dec", np.nan),
                "z_vi": getattr(sp, "z_vi", 0.0),
                "z_temp": getattr(sp, "z_temp", np.nan),
                "z_ph": getattr(sp, "z_ph", 0.0),
                "z_gaia": getattr(sp, "z_gaia", 0.0),
                "wavelength": np.asarray(sp.wave.value if hasattr(sp.wave, "value") else sp.wave, dtype=float).tolist(),
                "flux": np.asarray(sp.flux.value if hasattr(sp.flux, "value") else sp.flux, dtype=float).tolist(),
                "err": np.asarray(sp.err.value if hasattr(sp.err, "value") else sp.err, dtype=float).tolist(),
                "mask": np.asarray(getattr(sp, "mask", []), dtype=np.int64).tolist() if getattr(sp, "mask", None) is not None else [],
                "quality": quality.tolist() if quality is not None else [],
                "ndith": ndith.tolist() if ndith is not None else [],
            }
            rows.append(row)
        except Exception as exc:
            status = "failed"
            message = str(exc)
        log_rows.append({"ext": ext, "status": status, "message": message})

    parquet_files = _parquet_chunks(rows, prefix, parquet_chunk_size)
    log_path = prefix.with_name(f"{prefix.name}_log.csv")
    pd.DataFrame(log_rows).to_csv(log_path, index=False)

    for filename in parquet_files:
        print(f"Wrote parquet chunk: {filename}")
    print(f"Wrote processing log: {log_path}")
    print(f"Summary: total={len(log_rows)} ok={sum(r['status'] == 'ok' for r in log_rows)} failed={sum(r['status'] == 'failed' for r in log_rows)}")


def main():
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


if __name__ == "__main__":
    main()
