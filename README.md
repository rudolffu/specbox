# specbox
A simple tool to manipulate and visualize optical spectra for astronomical research.

## Documentation
Online documentation is hosted on Read the Docs: https://specbox.readthedocs.io/en/latest/index.html

## Installation
### Dependencies
- `numpy`
- `scipy`
- `astropy`
- [`pyqtgraph`](https://www.pyqtgraph.org/)
- [`PySide6`](https://doc.qt.io/qtforpython-6/gettingstarted.html#getting-started)
- [`specutils`](https://specutils.readthedocs.io/en/stable/installation.html)
- `matplotlib`
- `pandas`
- `pyarrow` or `fastparquet` (optional, for reading parquet spectra tables)
- `requests`
- `pillow` (PIL)
- `astroquery`

To install the latest version of specbox, run the following command in your terminal:

```bash
git clone https://github.com/rudolffu/specbox.git 
# or git clone https://gitee.com/rudolffu/specbox.git 
cd specbox
python -m pip install .
```

## Usage
### Main classes and functions
The main classes and functions of specbox are:
#### `basemodule.py`:
- `SpecLAMOST` and `SpecSDSS`: classes to read and manipulate spectra from the LAMOST and SDSS surveys, respectively.
- `SpecIRAF`: class to read and manipulate spectra from the IRAF format.
- `SpecPandasRow`: generic reader for "table-of-spectra" files readable by pandas (parquet/csv/feather/...), where each row stores arrays (e.g. wavelength/flux/ivar).
- `SpecSparcl`: SPARCL parquet/table reader (e.g., for file `sparcl_spectra.parquet`). Common metadata columns include `data_release`, `targetid`, and (optional) `euclid_object_id` for Euclid overlay.
#### `qtmodule.py`:
- `PGSpecPlot`: class to plot spectra in a `pyqtgraph` plot.
- `PGSpecPlotApp`: class to create a `pyqtgraph` plot with a `QApplication` instance.
- `PGSpecPlotThread`: class to create a `pyqtgraph` plot in a thread.

### Examples
#### Plotting a spectrum from the LAMOST survey
```python
from specbox import SpecLAMOST

spec = SpecLAMOST('input_file.fits')
spec.plot()
# Smooth the spectrum
spec.smooth(5, 3, inplace=False)
```

#### Reading a SPARCL parquet spectra table (one row per spectrum)
```python
from specbox.basemodule import SpecSparcl

# ext is a 1-based row index (ext=1 -> first row)
sp1 = SpecSparcl("sparcl_spectra.parquet", ext=1)
sp1.plot()
```

#### Run a `PGSpecPlotThread` for visual inspection of a list of spectra
```python
from specbox import SpecLAMOST
from specbox.qtmodule import PGSpecPlotThread
from glob import glob

basepath = 'lamost_spec/fits_files/'
flist = glob(basepath+'*fits.gz')
flist.sort()
flist = flist[0:60]

a = PGSpecPlotThread(speclist=flist, SpecClass=SpecLAMOST, output_file='vi_output_test60.csv')
a.run()
```

#### Run a viewer over a multi-row parquet file (SPARCL/table-of-spectra)
```python
from specbox.basemodule import SpecSparcl
from specbox.qtmodule import PGSpecPlotThreadEnhanced

viewer = PGSpecPlotThreadEnhanced(
    spectra="sparcl_spectra.parquet",
    SpecClass=SpecSparcl,
    # Optional: overlay Euclid spectrum when the parquet has `euclid_object_id`
    # and the Euclid combined FITS uses that ID as `EXTNAME`.
    euclid_fits="COMBINED_EUCLID_SPECS.fits",
    output_file="sparcl_vi_results.csv",
    z_max=6.0,
    load_history=True,
)
viewer.run()
```

Notes:
- Results CSV includes `targetid` and `data_release` (when available from the input table).
- The enhanced viewer has a `Save PNG` button that writes screenshots to `./saved_pngs/`.
<img src="specbox/docs/figs/PGSpecPlotThread_example.jpg" width="600">
