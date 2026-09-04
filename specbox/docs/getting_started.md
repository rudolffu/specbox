# Getting started

## Installation

The package declares Python >=3.9; Python 3.12 is recommended for WP9.
Dependencies, including `pyarrow` for parquet from 1.0.3 onward, install with pip.

It is recommended to set up an isolated environment before installing (choose either option A or B):

```bash
# Option A: Python venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
# Option B: conda
conda create -n specbox python=3.12 -y
conda activate specbox
python -m pip install --upgrade pip
```

### Install the stable release from PyPI (recommended)

```bash
python -m pip install specbox
```

### Update an existing installation

Activate the same environment and close any running viewer before upgrading:

```bash
python -m pip install --upgrade specbox
python -m pip show specbox
specbox-viewer --help
```

The campaign workflow requires 1.0.3 or later once published. Restart the viewer
after upgrading. For an unexpectedly old version, check `python -m pip --version`
and `python -c "import sys; print(sys.executable)"`, and locate the launcher with
`command -v specbox-viewer` (macOS/Linux) or `where specbox-viewer` (Windows).

### Run an assigned batch

Confirm your claimed batch with Yuming and extract the whole folder, keeping
scripts and inputs together. In the activated environment:

```bash
cd /path/to/your/batch
bash alias_001_review.sh
```

Finalize each redshift/classification and press `Q` to commit and advance.
Use `Save` regularly and `Save & Quit` when finished. Send the results CSV
configured in the script to Yuming on Slack or <yfu@strw.leidenuniv.nl>, with
batch ID and complete/partial status. See [the VI tutorial](VITUTORIAL.md) for
line markers, history recovery, and platform troubleshooting.

### Install a pre-release/development version from source

```bash
git clone https://github.com/rudolffu/specbox.git
cd specbox
python -m pip install .
```

### Editable install (development)

```bash
python -m pip install -e .
```

If you already have the repository cloned, run the install command from the repo root.

## Quick examples

### Plot a LAMOST spectrum

```python
from specbox import SpecLAMOST

spec = SpecLAMOST("input_file.fits")
spec.plot()
```

### View a multi-row parquet table (SPARCL)

```python
from specbox.basemodule import SpecSparcl
from specbox.qtmodule import PGSpecPlotThreadEnhanced

viewer = PGSpecPlotThreadEnhanced(
    spectra="sparcl_spectra.parquet",
    SpecClass=SpecSparcl,
    output_file="sparcl_vi_results.csv",
    load_history=True,
)
viewer.run()
```

For default SPARCL parquet tables, the scalar `redshift` column initializes
both `SpecSparcl.redshift` and the viewer startup `z_vi`. If `redshift` is
missing or non-finite, positive finite `z_desi`, `z_sdss`, `z_ref`, then `z`
are used as fallbacks.

## CLI quick start

### Viewer

```bash
specbox-viewer --spectra COMBINED_SPECS.fits --spec-class euclid
```

Notes:
- If `--output-file` is omitted, viewer writes to `vi_{input_file_name}_results.csv`.
- History is auto-loaded when that CSV already exists.
- Images and cutout downloads are off by default; add `--images` to enable them.
- Add `--redshift-table PATH --redshift-key object_id --redshift-column Z` to overlay a reference redshift table at startup without rewriting the source spectra.

For direct dual-arm Euclid review:

```bash
specbox-viewer \
  --rgs-file dual_001_rgs.parquet \
  --bgs-file dual_001_bgs.parquet \
  --spec-class euclid \
  --z-max 6.5 \
  --no-images \
  --output-file dual_001_vi_results.csv
```

This pairs and displays the BGS and RGS arms without requiring a coadd. Rows
are matched using the first shared column from `source_id`, `object_id`,
`extname`, `objid`, including objects found in only one arm. Keep the output CSV next
to the input files and rerun the same command to resume from its auto-loaded
history. Use `--spectra` for a single FITS or parquet input; use
`--rgs-file` together with `--bgs-file` for dual-arm mode.

Resume starts from the saved CSV row count, not an exhaustive search for
unreviewed objects. Work sequentially and verify the index after resuming.

For AIMS-z review bundles:

```bash
specbox-viewer --spectra review_bundle_specbox.parquet --spec-class aimsz-review
```

Notes:
- `aimsz-review` reads parquet rows directly using `wavelength`, `flux`, `ivar`, and `mask`.
- Session CSVs use canonical string IDs like `aimsz:{object_id}` to make history loading stable.
- `sparcl` and `aimsz-review` plot raw spectra by default; use the `Downsample` toolbar toggle for native pyqtgraph downsampling.
- Dual-arm Euclid viewer mode pairs BGS/RGS rows by source ID union, not by row index; rows with only one arm still load with the missing arm marked unavailable.
- Processed Euclid parquet startup uses `z_vi > z_sdss > z_desi > z_hybrid > z_fusion > z_temp > z_pcf_best > z_gaia > z_phot`; external `z_ref` values from `--redshift-table` remain an overlay.

### Euclid coadd (BGS+RGS)

```bash
specbox-coadd \
  --rgs-file sz_ragn_dr1_rgs_chunk_001.fits \
  --bgs-file sz_ragn_dr1_bgs_chunk_001.fits \
  --output-prefix coadd/sz_ragn_dr1_coadd_chunk_001 \
  --pair-by extname_intersection
```

### Raw Euclid FITS to parquet

```bash
specbox-euclid-parquet \
  --fits sz_ragn_dr1_rgs_chunk_001.fits \
  --output-prefix parquet/sz_ragn_dr1_rgs_chunk_001
```

### View Euclid parquet

```bash
# Raw single-arm Euclid parquet
specbox-viewer --spectra parquet/sz_ragn_dr1_rgs_chunk_001_part001.parquet --spec-class euclid

# Coadd parquet
specbox-viewer --spectra coadd/sz_ragn_dr1_coadd_chunk_001_part001.parquet --spec-class euclid-coadd
```

Euclid parquet flux and uncertainty default to the raw archive scale of
`1e-16 erg/s/cm^2/Angstrom`. Variance and inverse variance use squared and
inverse-squared flux units; uncertainty is derived before scaling. Add a scalar
`flux_scale` column to override this (1 for arrays already in physical units).

### Merge an external redshift table into parquet

```bash
specbox-merge-redshift-table \
  --spectra parquet/sz_ragn_dr1_rgs_chunk_001_part001.parquet \
  --redshift-table catalog.fits \
  --redshift-key object_id \
  --redshift-column Z \
  --output parquet/sz_ragn_dr1_rgs_chunk_001_part001_with_zref.parquet \
  --fill-z-vi
```

### PCF redshift

```bash
# Default: Type 1 template only
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits

# Type 1 + Type 2 (ragn_na; internally constrained to 0 < z < 3)
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits --enable-type2

# ragn_dr1 only (as type1)
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits --ragn-dr1-only
```

## Development and releases

Package versions are derived from Git tags with `setuptools-scm`. Do not edit
`specbox.__version__` or hard-code a package version in `pyproject.toml`; the
runtime `__version__` comes from installed package metadata.

Release flow:

```bash
git tag v1.0.3
git push origin v1.0.3
```

Then publish a GitHub Release for that tag. PyPI upload is intentionally tied
to the GitHub Release publication event, not to tag pushes.

Only tag the validated release commit; never move `v1.0.2`. Follow the
[release checklist](releases.md) before publishing.
