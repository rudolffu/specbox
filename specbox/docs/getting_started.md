# Getting started

## Installation

```bash
python -m pip install .
```

For an editable install (development):

```bash
python -m pip install -e .
```

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

