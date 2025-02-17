# Visual Inspection Tool for Quasar Spectra

This repository provides a visual inspection tool for quasar spectra. The tool enables you to load FITS spectra, interactively adjust the redshift using a non‐linear slider and spin box, and classify each spectrum using simple keyboard shortcuts. It also supports loading previous inspection results from a CSV history file so that already inspected spectra are skipped.

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

Anaconda includes `matplotlib`, `numpy`, `pandas`, and `astropy` by default. `PySide6`, `pyqtgraph`, `specutils` can be installed using `conda`:

```bash
conda install -c conda-forge pyqtgraph pyside6 specutils # or install them individually
```

In case `conda` does not provide a valid solution for any of the packages, you can install them using `pip`:

```bash
pip install PySide6 specutils pyqtgraph # install only the missing package(s)
```

- **Installation:**  

To install `specbox`, clone the repository and run the following command in the terminal:

```bash
git clone https://github.com/rudolffu/specbox.git
cd specbox
python -m pip install .
```

- **Project Structure:**  
  The visual inspection tool is part of the package `specbox` which contains:
  - `qtmodule/qtsir1d.py` – Main GUI code.
  - `basemodule.py` – Contains classes (such as `SpecEuclid1d`) to read the FITS spectra.

---

## Running the Tool

You can run the tool by creating an instance of the inspection thread. For example, from a Python shell or script:

```python
from specbox.basemodule import SpecEuclid1d
from specbox.qtmodule.qtsir1d import PGSpecPlotThread

a = PGSpecPlotThread(
    specfile='../COMBINED_SPECS.fits',
    SpecClass=SpecEuclid1d,
    output_file='vi_results.csv',
    load_history=True
)
a.run()
```

### Parameter Explanation

- **specfile:**  
  The path to the FITS file containing the spectra.
- **SpecClass:**  
  The class used to read and process each spectrum (e.g., `SpecEuclid1d`).
- **output_file:**  
  The CSV file where inspection results (object classification and redshift) are saved.
- **load_history:**  
  If set to `True` and the CSV exists, the tool loads previous classifications and skips those spectra.

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

- **U:**  
  Classifies the spectrum as **UNKNOWN**.

- **L:**  
  Classifies the spectrum as **LIKELY/Unusual QSO**.

- **M:**  
  Prints the current mouse position in the plot (useful for measurements).

- **Spacebar:**  
  Prints the wavelength and flux at the mouse location and annotates the plot.

- **R:**  
  Resets the plot to its original state (undo zooming/panning).

- **Ctrl+Left Arrow:**  
  Loads the previous spectrum for re-inspection.

---

## History and Resuming Inspections

- **Saving Results:**  
  The tool saves classifications to the specified CSV file (with columns for object ID, object name, RA, DEC, assigned class, and visually inspected redshift `z_vi`) periodically and when exiting.

- **Loading History:**  
  When `load_history=True` is provided, the tool reads the CSV file and loads the object IDs into a dictionary. It then skips any spectrum whose ID already exists in the history, allowing you to resume from where you left off.

---

## Tips for Effective Use

- **Adjust the Redshift:**  
  Use the slider or spin box to fine-tune the redshift until the template (plotted in a contrasting color) aligns well with the observed spectrum.
  
- **Keyboard Shortcuts:**  
  Familiarize yourself with the key commands to quickly classify and navigate spectra without needing to use the mouse extensively.

- **Review History:**  
  Check the CSV file if you need to confirm that classifications are being saved correctly and that object IDs match.

- **Customization:**  
  You can modify parameters such as `z_max` and `base_z_step` in the code if your spectral redshift range differs.
