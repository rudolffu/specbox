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

You can create a new environment (e.g. `euclid`) and install the required packages using `conda`:

```bash
conda create -n euclid python=3.13
conda activate euclid
pip install PySide6 specutils pyqtgraph astropy pandas specutils matplotlib setuptools
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

### Test installation of `specbox`

To test the installation, you can run the following code snippet in a Python shell or script:

```python
import matplotlib.pyplot as plt
from specbox.basemodule import SpecEuclid1d

sp1 = SpecEuclid1d('COMBINED_SPECS.fits', ext=1) # example path to the FITS file containing the spectra, and the extension number

sp1.plot()
plt.show()
```

If the installation is successful, you should see a plot of the spectrum.

### Running the Visual Inspection Tool

You can run the tool by creating an instance of the inspection thread. For example, create a Python script (`my_vi_script.py`) with the following code:

```python
#!/usr/bin/env python
from specbox.qtmodule.qtsir1d import PGSpecPlotThread

a = PGSpecPlotThread(
    specfile='COMBINED_SPECS.fits', # example path to the FITS file containing the spectra
    output_file='vi_results.csv', # path to the output CSV file
    z_max=5.0,
    load_history=True
)
a.run()
```

Run the script in a terminal (ensure that the correct environment is activated):

```bash
python my_vi_script.py
```

The first time you run the tool in a new Python environment, `matplotlib` will take some time to build the font cache. Subsequent runs will be faster.

### Parameter Explanation

- **specfile:**  
  The path to the FITS file containing the spectra.
- **output_file:**  
  The CSV file where inspection results (object classification and redshift) are saved.
- **z_max:**
  The maximum redshift to be considered for the slider. The default is 5.0.
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
  You can modify parameter `z_max` in the script if your spectral redshift range differs.
