# Getting started

## Installation

It is recommended to set up an isolated environment before installing (choose either option A or B):

```bash
# Option A: Python venv
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

```bash
# Option B: conda
conda create -n specbox python=3.13 -y
conda activate specbox
python -m pip install --upgrade pip
```

### Install the stable release from PyPI (recommended)

```bash
python -m pip install specbox
```

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
    z_max=6.0,
    load_history=True,
)
viewer.run()
```

## CLI quick start

### Viewer

```bash
specbox-viewer --spectra COMBINED_SPECS.fits --spec-class euclid
```

Notes:
- If `--output-file` is omitted, viewer writes to `vi_{input_file_name}_results.csv`.
- History is auto-loaded when that CSV already exists.
- Add `--no-images` to disable the image panel and all cutout downloading.

For AIMS-z review bundles:

```bash
specbox-viewer --spectra review_bundle_specbox.parquet --spec-class aimsz-review
```

Notes:
- `aimsz-review` reads parquet rows directly using `wavelength`, `flux`, `ivar`, and `mask`.
- Session CSVs use canonical string IDs like `aimsz:{object_id}` to make history loading stable.
- `aimsz-review` disables images and cutout downloading by default; add `--images` to opt in.

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

### PCF redshift

```bash
# Default: Type 1 template only
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits

# Type 1 + Type 2 (ragn_na; internally constrained to 0 < z < 3)
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits --enable-type2

# ragn_dr1 only (as type1)
specbox-pcf --fits coadd/sz_ragn_dr1_coadd_chunk_001.fits --ragn-dr1-only
```
