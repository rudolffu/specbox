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
