# specbox documentation

[![GitHub](https://img.shields.io/badge/GitHub-rudolffu%2Fspecbox-181717?logo=github)](https://github.com/rudolffu/specbox)
[![PyPI](https://img.shields.io/pypi/v/specbox)](https://pypi.org/project/specbox/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18642758.svg)](https://doi.org/10.5281/zenodo.18642758)

```{toctree}
:maxdepth: 2
:caption: User guide

getting_started
spectrum_classes
notebooks
VITUTORIAL
api/index
```

## Overview

`specbox` is a small toolkit to read, manipulate, and visually inspect UV/optical/NIR spectra
from multiple surveys and file formats.

GitHub repository: https://github.com/rudolffu/specbox

See {doc}`getting_started` for a quick start, and {doc}`api/index` for the full API
reference generated from the source code.

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

Install the stable release from PyPI (recommended):

```bash
python -m pip install specbox
```

To install a pre-release/development version from source:

```bash
git clone https://github.com/rudolffu/specbox.git
cd specbox
python -m pip install .
```

## Citation

If you use `specbox`, please cite:

- DOI: https://doi.org/10.5281/zenodo.18642758
- Repository citation metadata: `CITATION.cff`
- BibTeX record: see the `Citation` section in `README.md`
