# Visual Inspection Tool for Quasar Spectra

This repository provides a visual inspection tool for quasar spectra. The tool enables you to load FITS spectra, interactively adjust the redshift using a non‐linear slider and spin box, and classify each spectrum using simple keyboard shortcuts. It also supports loading previous inspection results from a CSV history file so that already inspected spectra are skipped.

Online documentation (Read the Docs): https://specbox.readthedocs.io/en/latest/index.html

Citation:
```bibtex
@software{fu_2026_18642758,
  author       = {Fu, Yuming},
  title        = {specbox: A simple tool to manipulate and visualize
                   UV/optical/NIR spectra for astronomical research.
                  },
  month        = feb,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.18642758},
  url          = {https://doi.org/10.5281/zenodo.18642758},
}
```

---

## Table of Contents

- [Prerequisites and Installation of specbox](#prerequisites-and-installation-of-specbox)
- [Running the Tool](#running-the-tool)
- [User Interface Overview](#user-interface-overview)
- [Keyboard Shortcuts and Actions](#keyboard-shortcuts-and-actions)
- [History and Resuming Inspections](#history-and-resuming-inspections)
- [Tips for Effective Use](#tips-for-effective-use)

---

## Prerequisites and Installation of `specbox`

- **Python Version:** Python 3  
- **Dependencies:**  
  Ensure that you have the following Python packages installed:
  - PySide6
  - pyqtgraph
  - matplotlib
  - numpy
  - pandas
  - astropy
  - specutils
  - requests
  - pillow (PIL)
  - astroquery

You can create a new environment (e.g. `euclid`) and install the required packages using `conda`:

```bash
conda create -n euclid python=3.13
conda activate euclid
pip install PySide6 specutils pyqtgraph astropy pandas matplotlib requests pillow astroquery setuptools
```


<!-- ```bash
conda create -n euclid pyqtgraph pyside6 specutils astropy -c conda-forge
conda activate euclid
```

Using the command above, `pandas`, `matplotlib`, and `numpy` will be installed automatically. 

Alternatively, you can install the packages using `pip`:
```bash
pip install PySide6 specutils pyqtgraph # install only the missing package(s)
``` -->

- **Installation:**  

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

- **Project Structure:**  
  The visual inspection tool is part of the package `specbox` which contains:
  - `qtmodule/qtmodule_enhanced.py` – Main GUI code.
  - `basemodule.py` – Contains classes (such as `SpecEuclid1d`) to read the FITS spectra.

---

## Running the Tool

### Test installation of `specbox`

To test the installation, you can run the following code snippet in a Python shell or script:

```python
import matplotlib.pyplot as plt
from specbox.basemodule import SpecEuclid1d

sp1 = SpecEuclid1d(
    'COMBINED_SPECS.fits',
    ext=1,
    good_pixels_only=True,  # Keep only recommended bins from Euclid MASK flags
)  # example path to the FITS file containing spectra

sp1.plot()
plt.show()
```

If the installation is successful, you should see a plot of the spectrum.

Notes:
- `SpecEuclid1d` exposes `mask`, `good_mask`, and `bad_mask` (when the `MASK` column is present).
- `good_pixels_only=True` applies the Euclid recommendation to discard bins with odd `MASK` or `MASK >= 64`.

### Reading SPARCL parquet spectra (dataframe-backed)

If your spectra are stored in a table file (e.g. parquet) where each row is a spectrum and the row contains array columns like ``wavelength``, ``flux``, and ``ivar``, you can use ``SpecSparcl``:

```python
from specbox.basemodule import SpecSparcl

sp1 = SpecSparcl('outlier_sparcl_spectra.parquet', ext=1)  # ext is 1-based row index
sp1.plot()
```

Parquet input requires either ``pyarrow`` or ``fastparquet`` to be installed.

To run the visual inspection GUI directly on such a multi-row parquet file:

```python
from specbox.basemodule import SpecSparcl
from specbox.qtmodule import PGSpecPlotThreadEnhanced

viewer = PGSpecPlotThreadEnhanced(
    spectra='outlier_sparcl_spectra.parquet',
    SpecClass=SpecSparcl,
    # Optional: overlay Euclid spectrum when the parquet has `euclid_object_id`
    # and the Euclid combined FITS uses that ID as `EXTNAME`.
    euclid_fits='COMBINED_EUCLID_SPECS.fits',
    output_file='sparcl_vi_results.csv',
    z_max=6.0,
    load_history=True,
)
viewer.run()
```

Notes:
- The results CSV includes `targetid` and `data_release` when present in the input table.
- Use the `Save PNG` button to save a screenshot to `./saved_pngs/`.

### Running the Visual Inspection Tool

Use the CLI:

```bash
specbox-viewer --spectra COMBINED_SPECS.fits --spec-class euclid
```

Add `--no-images` to disable the image panel and skip all cutout downloads.

The first time you run the tool in a new Python environment, `matplotlib` will take some time to build the font cache. Subsequent runs will be faster.

### Parameter Explanation

- **spectra:**  
  The path to the FITS file containing the spectra.
- **output_file:**  
  The CSV file where inspection results (object classification and redshift) are saved. If omitted, viewer uses `vi_{input_file_name}_results.csv`.
- **z_max:**  
  The maximum redshift to be considered for the slider. The default is 5.0.
- **load_history:**  
  Optional CLI flag to force history loading. By default, history is auto-loaded when the output CSV exists.
- **no-images:**  
  Optional CLI flag to disable the image panel and all cutout downloading when remote cutouts are not needed or unavailable.

---

## User Interface Overview

### Layout

- **Plot Area:**  
  The main window displays the current quasar spectrum.
- **Slider:**  
  A horizontal slider at the bottom controls the visually inspected redshift (`z_vi`). It uses a non-linear (1+z) mapping.
- **Spin Box:**  
  Next to the slider is a QDoubleSpinBox that shows the current redshift value. You can type a custom redshift here. Both controls are synchronized.

### How the Slider Works

The slider’s mapping is given by:

\[
z = \exp(\text{base\_z\_step} \times \text{slider\_value}) \times (1 + z_{\min}) - 1
\]

This mapping allows the step size to increase with redshift, matching the natural (1+z) scaling of spectral features.

---

## Keyboard Shortcuts and Actions

When the tool is active, use the following keys:

- **Q:**  
  Loads the next spectrum. If only **Q** is pressed, the default classification **QSO(Default)** will be adopted. If the user chooses other classifications (keys below), using **Q** is also needed to load the next spectrum. 
  
- **S:**  
  Classifies the spectrum as **STAR** (sets redshift to 0).

- **G:**  
  Classifies the spectrum as **GALAXY**.

- **A:**  
  Classifies the spectrum as **QSO (AGN)**.

- **N:**  
  Classifies the spectrum as **QSO(Narrow)**.

- **U:**  
  Classifies the spectrum as **UNKNOWN**.

- **L:**  
  Classifies the spectrum as **LIKELY/Unusual QSO**.

In `aimsz-review` mode, additional review shortcuts are enabled and the stored labels are normalized:

- **B:**  
  Classifies the spectrum as **QSO(BAL)** and stores `QSO_BAL`.

- **D:**  
  Classifies the spectrum as **BAD** and stores `BAD`.

- **L:**  
  Stores `LIKELY_Q` in review mode, while still accepting legacy `LIKELY` when loading older CSVs.

- **M:**  
  Prints the current mouse position in the plot (useful for measurements).

- **Spacebar:**  
  Prints the wavelength and flux at the mouse location and annotates the plot.

- **R:**  
  Resets the plot to its original state (undo zooming/panning). 

- **Ctrl+R:** 
  Resets the redshift (`z_vi`) to the original value in the current plot.

- **Left Arrow:**  
  Loads the previous spectrum (useful for reviewing).

- **Right Arrow:**  
  Loads the next spectrum. Only for reviewing because this action does not save the classification and redshift of the current spectrum.

- **Ctrl+Left Arrow:**  
  Goes back to the first spectrum in the list.

- **Ctrl+Right Arrow:**  
  Goes to the last spectrum in the list.

- **Ctrl+B:**  
  Goes back to the last labeled spectrum.
---

## History and Resuming Inspections

- **Saving Results:**  
  The tool saves classifications to the specified CSV file (with columns for object ID, object name, RA, DEC, assigned class, and visually inspected redshift `z_vi`) periodically and when exiting.

For `aimsz-review`, the saved CSV also carries:
- `object_id`
- `targetid`
- `data_release`
- `qa_flag`
- `notes`

The canonical review labels written in this mode are:
- `QSO_DEFAULT`
- `QSO`
- `QSO_NARROW`
- `QSO_BAL`
- `LIKELY_Q`
- `GALAXY`
- `STAR`
- `UNKNOWN`
- `BAD`

- **Loading History:**  
  The tool reads the output CSV when it exists and loads object IDs into a dictionary. It then skips spectra that already exist in history, so you can resume where you left off.

---

## Tips for Effective Use

- **Adjust the Redshift:**  
  Use the slider or spin box to fine-tune the redshift until the template (plotted in a contrasting color) aligns well with the observed spectrum.
  
- **Keyboard Shortcuts:**  
  Familiarize yourself with the key commands to quickly classify and navigate spectra without needing to use the mouse extensively.

- **Review History:**  
  Check the CSV file if you need to confirm that classifications are being saved correctly and that object IDs match.

- **Customization:**  
  You can modify parameter `z_max` in the script if your spectral redshift range differs.
