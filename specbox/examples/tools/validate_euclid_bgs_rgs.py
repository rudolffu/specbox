#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from specbox.basemodule import SpecEuclid1d


def _median_abs(values):
    array = np.asarray(values)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.nan
    return float(np.nanmedian(np.abs(array[finite])))


def _quantile(values, q):
    array = np.asarray(values)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.nan
    return float(np.nanquantile(array[finite], q))


def _values(array_like):
    if hasattr(array_like, "value"):
        return np.asarray(array_like.value)
    return np.asarray(array_like)


def _mask_stats(spec):
    if getattr(spec, "mask", None) is None:
        return np.nan, np.nan, 0, 0
    mask = np.asarray(spec.mask)
    bad = (mask % 2 == 1) | (mask >= 64)
    good = ~bad
    return float(np.mean(good)), float(np.mean(bad)), int(np.sum(good)), int(np.sum(bad))


def _open_extnames(path):
    with fits.open(path) as hdul:
        return [h.name for h in hdul[1:]]


def _overlap_metrics(spec_bgs, spec_rgs):
    wb = _values(spec_bgs.wave)
    wr = _values(spec_rgs.wave)
    fb = _values(spec_bgs.flux)
    fr = _values(spec_rgs.flux)
    wmin = max(np.nanmin(wb), np.nanmin(wr))
    wmax = min(np.nanmax(wb), np.nanmax(wr))
    if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
        return np.nan, np.nan, 0, 0, np.nan
    mb = (wb >= wmin) & (wb <= wmax) & np.isfinite(fb)
    mr = (wr >= wmin) & (wr <= wmax) & np.isfinite(fr)
    nb = int(np.sum(mb))
    nr = int(np.sum(mr))
    if nb == 0 or nr == 0:
        ratio = np.nan
    else:
        db = _median_abs(fb[mb])
        dr = _median_abs(fr[mr])
        ratio = db / dr if np.isfinite(db) and np.isfinite(dr) and dr != 0 else np.nan
    return float(wmin), float(wmax), nb, nr, ratio


def _spec_row(path_bgs, path_rgs, extname):
    bgs = SpecEuclid1d(path_bgs, extname=extname, clip=False, good_pixels_only=False)
    rgs = SpecEuclid1d(path_rgs, extname=extname, clip=False, good_pixels_only=False)

    bgs_good_frac, bgs_bad_frac, bgs_good_n, bgs_bad_n = _mask_stats(bgs)
    rgs_good_frac, rgs_bad_frac, rgs_good_n, rgs_bad_n = _mask_stats(rgs)
    overlap_wmin, overlap_wmax, overlap_bgs_n, overlap_rgs_n, overlap_norm_ratio = _overlap_metrics(bgs, rgs)

    return {
        "objid": extname,
        "ra_bgs": float(getattr(bgs, "ra", np.nan)),
        "dec_bgs": float(getattr(bgs, "dec", np.nan)),
        "ra_rgs": float(getattr(rgs, "ra", np.nan)),
        "dec_rgs": float(getattr(rgs, "dec", np.nan)),
        "n_bins_bgs": int(len(bgs.wave)),
        "n_bins_rgs": int(len(rgs.wave)),
        "wave_min_bgs": float(np.nanmin(_values(bgs.wave))),
        "wave_max_bgs": float(np.nanmax(_values(bgs.wave))),
        "wave_min_rgs": float(np.nanmin(_values(rgs.wave))),
        "wave_max_rgs": float(np.nanmax(_values(rgs.wave))),
        "mask_good_frac_bgs": bgs_good_frac,
        "mask_bad_frac_bgs": bgs_bad_frac,
        "mask_good_n_bgs": bgs_good_n,
        "mask_bad_n_bgs": bgs_bad_n,
        "mask_good_frac_rgs": rgs_good_frac,
        "mask_bad_frac_rgs": rgs_bad_frac,
        "mask_good_n_rgs": rgs_good_n,
        "mask_bad_n_rgs": rgs_bad_n,
        "flux_p16_bgs": _quantile(_values(bgs.flux), 0.16),
        "flux_p50_bgs": _quantile(_values(bgs.flux), 0.50),
        "flux_p84_bgs": _quantile(_values(bgs.flux), 0.84),
        "flux_p16_rgs": _quantile(_values(rgs.flux), 0.16),
        "flux_p50_rgs": _quantile(_values(rgs.flux), 0.50),
        "flux_p84_rgs": _quantile(_values(rgs.flux), 0.84),
        "err_p50_bgs": _quantile(_values(bgs.err), 0.50),
        "err_p50_rgs": _quantile(_values(rgs.err), 0.50),
        "overlap_wmin": overlap_wmin,
        "overlap_wmax": overlap_wmax,
        "overlap_bins_bgs": overlap_bgs_n,
        "overlap_bins_rgs": overlap_rgs_n,
        "overlap_norm_ratio_bgs_over_rgs": overlap_norm_ratio,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Euclid DR1 BGS/RGS spectra consistency.")
    parser.add_argument("--bgs", required=True, help="Path to BGS combined spectrum FITS.")
    parser.add_argument("--rgs", required=True, help="Path to RGS combined spectrum FITS.")
    parser.add_argument("--output", default="euclid_bgs_rgs_validation.csv", help="Output CSV path.")
    args = parser.parse_args()

    bgs_path = Path(args.bgs)
    rgs_path = Path(args.rgs)
    bgs_ext = set(_open_extnames(bgs_path))
    rgs_ext = set(_open_extnames(rgs_path))
    shared = sorted(bgs_ext.intersection(rgs_ext))

    rows = []
    for extname in shared:
        try:
            rows.append(_spec_row(str(bgs_path), str(rgs_path), extname))
        except Exception as exc:
            rows.append({"objid": extname, "error": str(exc)})

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"BGS objects: {len(bgs_ext)}")
    print(f"RGS objects: {len(rgs_ext)}")
    print(f"Shared objects analyzed: {len(shared)}")
    print(f"Validation CSV: {Path(args.output).resolve()}")
    if "overlap_norm_ratio_bgs_over_rgs" in df.columns:
        ratio = pd.to_numeric(df["overlap_norm_ratio_bgs_over_rgs"], errors="coerce")
        ratio = ratio[np.isfinite(ratio)]
        if len(ratio) > 0:
            print(
                "Overlap normalization ratio (BGS/RGS) median="
                f"{np.nanmedian(ratio):.4g}, p16={np.nanquantile(ratio, 0.16):.4g}, "
                f"p84={np.nanquantile(ratio, 0.84):.4g}"
            )


if __name__ == "__main__":
    main()
