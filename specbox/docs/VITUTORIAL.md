# Visual Inspection Tool for Quasar Spectra

This guide focuses on Euclid DR1 spectra prepared in parquet files for better loading performance. Use `--spec-class euclid` for these inspection batches. SpecBox lets you adjust redshift with templates and line markers and record visual classifications in a CSV file. See the resume limitations below before continuing an existing session.

## Table of Contents

- [Teamwork Quick Start](#teamwork-quick-start)
- [Prerequisites and Installation of specbox](#prerequisites-and-installation-of-specbox)
- [Running the Tool](#running-the-tool)
- [User Interface Overview](#user-interface-overview)
- [Keyboard Shortcuts and Actions](#keyboard-shortcuts-and-actions)
- [History and Resuming Inspections](#history-and-resuming-inspections)
- [Troubleshooting](#troubleshooting)
- [Tips for Effective Use](#tips-for-effective-use)
- [Non-Euclid](#non-euclid)

---

## Teamwork Quick Start

Use SpecBox **1.0.3 or later** for this workflow once that release is published.
Earlier PyPI versions do not include all the controls described here.

1. Claim a batch in the assignment spreadsheet supplied in the invitation and confirm the assignment with the coordinator before starting.
2. Download and extract the entire batch folder. Keep its `.sh` script and all parquet inputs together, with their original filenames.
3. Activate the Python environment where SpecBox is installed, upgrade it as below, then open a terminal in the batch folder:

   ```bash
   cd /path/to/your/batch
   bash alias_001_review.sh
   ```

4. Inspect spectra sequentially. Adjust the redshift, choose a classification, and press `Q` to commit the current result and advance. Press `Save` regularly; finish with `Save & Quit`.
5. Send the results CSV to the coordinator using the contact channel supplied in the invitation. Include your batch ID and whether it is complete or partial. Partial batches are welcome.

The output filename is set by `--output-file` in the batch script. Return that
CSV, rather than the input parquet or a screenshot. Keep a local backup and
report completion in the assignment spreadsheet. Do not edit object IDs or
reorder input rows during a review session.

## Prerequisites and Installation of `specbox`

The package declares Python >=3.9; Python **3.12** is recommended for the
campaign. Pip installs the dependencies, including `pyarrow` for parquet in
version 1.0.3 onward. A desktop graphical session is required for the viewer.

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

Install the stable release from PyPI (recommended):

```bash
python -m pip install specbox
```

### Upgrade an existing installation

Activate the same environment used by the batch script, close any running
viewer, then run:

```bash
python -m pip install --upgrade specbox
python -m pip show specbox
specbox-viewer --help
```

Check that the reported version is at least 1.0.3 once published, then restart
the viewer. Upgrading does not update a running process. If an older version
persists, check `python -c "import sys; print(sys.executable)"` and
`python -m pip --version`; on macOS/Linux use `command -v specbox-viewer`, or
`where specbox-viewer` on Windows, to locate the launcher in that environment.

To install a pre-release/development version from source:

```bash
git clone https://github.com/rudolffu/specbox.git
cd specbox
python -m pip install .
```

Package versions are derived from Git tags with `setuptools-scm`. Do not edit
`specbox.__version__` or hard-code a package version in `pyproject.toml`; the
runtime `__version__` comes from installed package metadata. PyPI upload runs
only when a GitHub Release is published for a version tag, not when the tag is
pushed.

- **Project Structure:**  
  The visual inspection tool is part of the package `specbox` which contains:
  - `qtmodule/qtmodule_enhanced.py` – Main GUI code.
  - `basemodule.py` – Contains classes (such as `SpecEuclid1d`) to read the parquet spectra.

---

## Running the Tool

### Euclid DR1 parquet inspection

The coordinator has prepared the Euclid DR1 spectra as parquet tables for better
loading performance. Prefer the supplied batch script; to launch directly, use:

```bash
specbox-viewer --spectra alias_001.parquet --spec-class euclid
```

For direct dual-arm Euclid inspection, pass the RGS and BGS tables separately:

```bash
specbox-viewer \
  --rgs-file dual_001_rgs.parquet \
  --bgs-file dual_001_bgs.parquet \
  --spec-class euclid \
  --z-max 6.5 \
  --no-images \
  --output-file dual_001_vi_results.csv
```

Dual-arm mode displays both arms without a coadd. For parquet, the first column
present in both tables from `source_id`, `object_id`, `extname`, `objid` is the
matching key. The viewer uses the union of objects from both arms; an object
with only one arm remains inspectable, with the other arm unavailable.
Rerunning the command loads the output CSV; see the resume limitations below.

Images and cutout downloads are off by default. Add `--images` to enable the image panel when needed, or `--no-images` for an explicit image-off CLI.
For processed Euclid parquet files, viewer startup uses the first positive finite redshift in `z_vi > z_sdss > z_desi > z_hybrid > z_fusion > z_temp > z_pcf_best > z_gaia > z_phot`. `z_temp` and `z_pcf_best` are aliases, with `z_temp` preferred when both are present.
For Euclid parquet, the default flux scale is `1e-16 erg/s/cm^2/Angstrom`.
`flux`/`signal` and `err` use this scale; `var` has squared flux units and
`ivar` inverse-squared units. The reader derives uncertainty before applying
the flux scale. A positive scalar `flux_scale` (`signal_scale`, `fscale`, or
`FSCALE`) overrides the default; use 1 for arrays already in physical units.
With `--redshift-table`, the viewer loads the external table once at startup and stores the matched value as `z_ref`; this remains an external overlay and is not part of the processed Euclid parquet priority list.

The first time you run the tool in a new Python environment, `matplotlib` will take some time to build the font cache. Subsequent runs will be faster.

### Parameter Explanation

- **spectra:**  
  The path to the multi-row Euclid parquet spectra file.
- **rgs_file / bgs_file:**
  The paired Euclid RGS and BGS parquet inputs for direct dual-arm
  inspection. Supply both instead of `spectra`; a coadd is not needed.
- **output_file:**  
  The CSV file where inspection results (object classification and redshift) are saved. If omitted, viewer uses `vi_{input_file_name}_results.csv`.
- **z_max:**  
  The maximum redshift to be considered for the slider. Defaults are 6.0 for `euclid`, 7.0 for `sparcl`/`aimsz-review`, and 5.0 for other spectrum classes.
- **load_history:**  
  Optional CLI flag to force history loading. By default, history is auto-loaded when the output CSV exists.
- **no-images:**  
  Optional CLI flag to disable the image panel and all cutout downloading when remote cutouts are not needed or unavailable.

- **images:**  
  Optional CLI flag to enable the image panel and cutout downloading.

---

## User Interface Overview

### Layout

- **Plot Area:**  
  The main window displays the current quasar spectrum.
- **External Redshift Overlay:**  
  Add `--redshift-table`, `--redshift-key`, and `--redshift-column` to inject `z_ref` values from an external FITS/parquet/CSV catalog without modifying the original spectra files.
- **Euclid processed parquet redshifts:**  
  `SpecEuclid1d` reads optional scalar columns `z_vi`, `z_sdss`, `z_desi`, `z_hybrid`, `z_fusion`, `z_temp`, `z_pcf_best`, `z_gaia`, and `z_phot`. If `z_vi` is missing or not usable, the slider starts from the highest-priority available fallback and the message panel reports the selected source.
- **Slider:**  
  A horizontal slider at the bottom controls the visually inspected redshift (`z_vi`). It uses a non-linear (1+z) mapping.
- **Spin Box:**  
  Next to the slider is a QDoubleSpinBox that shows the current redshift value. You can type a custom redshift here. Both controls are synchronized.

### Line markers

Choose one of `Hα`, `[O III] 5008`, `[O II] 3728`, `Mg II`, or `[S III] 9533`
in the six-button row (including `Off`). A single left-click in the plot sets
`z_vi = observed wavelength / rest wavelength - 1` and aligns the template.
The sulfur marker uses rest wavelength 9533.2 Angstrom. Verify the identification
against other features before committing a redshift.

Mode defaults to `Off` and stays active for repeated clicks. Press `Esc`, select
`Off`, or adjust the redshift spin box to exit. Slider changes do not exit the
mode. Clicks implying redshifts outside the configured range are rejected.

### Slider mapping

The slider’s mapping is given by:

\[
z = \exp(\text{base\_z\_step} \times \text{slider\_value}) \times (1 + z_{\min}) - 1
\]

This mapping allows the step size to increase with redshift, matching the natural (1+z) scaling of spectral features.

---

## Keyboard Shortcuts and Actions

When the tool is active, use the following keys:

- **Q:**  
  Loads the next spectrum. If only **Q** is pressed, the default classification **QSO(Default)** will be adopted. If the user chooses other classifications (keys below), using **Q** is also needed to load the next spectrum. Saved history uses the canonical token `QSO_DEFAULT`.
  
- **S:**  
  Classifies the spectrum as **STAR**.

- **G:**  
  Classifies the spectrum as **GALAXY**.

- **A:**  
  Classifies the spectrum as **QSO**.

- **N:**  
  Classifies the spectrum as **QSO(Narrow)**.

- **B:**  
  Classifies the spectrum as **QSO(BAL)**.

- **F:**  
  Classifies the spectrum as **QSO(FeLoBAL)**.

- **U:**  
  Classifies the spectrum as **UNKNOWN**.

- **D:** Classifies the spectrum as **BAD**.

- **L:**  
  Classifies the spectrum as **LIKELY_Q**.

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

`Q` commits the active `z_vi` to in-memory history and advances. An unclassified
object receives `QSO_DEFAULT`; this is a default label, not an explicit QSO
assessment. Choose the appropriate classification rather than accepting the
default for uncertain spectra. `Save` and `Save & Quit` write committed history
to the configured CSV. In Euclid mode, changing the slider or spin box alone
does not update the saved record; press `Q` after finalizing the result.

For the WP9 inspection, redshifts recorded for **STAR**, **UNKNOWN**, and **BAD**
are not adopted in the analysis. Press `S`, `U`, or `D`, then `Q` to advance;
there is no need to adjust or zero the redshift for these classifications.
Arrow navigation does not commit subsequent redshift edits.

- **Saving Results:**  
  Use `Save` regularly and `Save & Quit` when finished. Normal exit writes committed history; a crash may lose edits since the last explicit save or recovery snapshot.

- **Temporary Recovery Snapshots:**
  Every 50 completed spectra, the viewer writes a cumulative snapshot to
  `./temp/<sample-name>/vi_temp_<count>.csv`. The sample name comes from the
  input filename; paired RGS/BGS files share a folder after the arm token is
  removed. These snapshots do not replace the configured output CSV.

- **Loading History:**  
  The CLI auto-loads an existing output CSV. The starting position is based on
  its row count, not a search for every unreviewed object. Inspect sequentially
  and check the displayed index after resuming, especially after reviewing out
  of order. When the history count reaches the input length, the viewer starts
  at the first spectrum again; that does not mean the saved history was lost.

To recover after a crash, close the viewer and back up any existing output CSV.
Copy the latest appropriate `temp/<sample-name>/vi_temp_<count>.csv` to the
output filename specified by the batch script, then rerun that script. Check
the restored index and results. Snapshots live under the directory from which
the viewer was launched, and are not loaded automatically.

## Troubleshooting

- **Command not found or old version:** activate the installation environment
  and check the Python/pip and launcher locations as described above.
- **Missing parquet engine:** run `python -m pip install --upgrade specbox pyarrow`
  in that environment and restart the viewer.
- **Script will not execute:** use `bash alias_001_review.sh` from the extracted
  batch directory; this does not require making the script executable. Confirm
  all input files are present. `.sh` scripts require Bash. On Windows, run the
  contained `specbox-viewer` command in your activated environment, replacing
  Bash variables with actual quoted paths and adapting line continuations.
- **Qt/display error:** run from a local desktop terminal, not a headless SSH
  session. `--no-images` disables cutouts, not the GUI. Send the coordinator the error
  text, operating system, batch ID, and `python -m pip show specbox` output if
  the problem persists.

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

## Non-Euclid

Euclid inspectors can skip this section. These readers are for SPARCL tables
and AIMS-z review bundles and use different spectrum classes.

### Reading SPARCL parquet spectra (dataframe-backed)

If your spectra are stored in a table file (e.g. parquet) where each row is a spectrum and the row contains array columns like ``wavelength``, ``flux``, and ``ivar``, you can use ``SpecSparcl``:

```python
from specbox.basemodule import SpecSparcl

sp1 = SpecSparcl('outlier_sparcl_spectra.parquet', ext=1)  # ext is 1-based row index
sp1.plot()
```

Default SPARCL parquet files with scalar columns like `specid`, `redshift`,
`ra`, `dec`, `targetid`, `flux`, `ivar`, and `wavelength` are read directly.
The `redshift` value initializes both `SpecSparcl.redshift` and the viewer's
startup `z_vi`; if it is missing or non-finite, positive finite `z_desi`,
`z_sdss`, `z_ref`, then `z` are used as fallbacks.

Parquet input uses `pyarrow`, installed automatically with SpecBox 1.0.3 onward.

To run the visual inspection GUI directly on such a multi-row parquet file:

```python
from specbox.basemodule import SpecSparcl
from specbox.qtmodule import PGSpecPlotThreadEnhanced

viewer = PGSpecPlotThreadEnhanced(
    spectra='outlier_sparcl_spectra.parquet',
    SpecClass=SpecSparcl,
    output_file='sparcl_vi_results.csv',
    load_history=True,
)
viewer.run()
```

Notes:
- The results CSV includes `targetid` and `data_release` when present in the input table.
- Use the `Save PNG` button to save a screenshot to `./saved_pngs/`.

### AIMS-z review parquet workflow

Use `SpecAIMSZReview` for AIMS-z review bundles that include review metadata alongside the spectra:

```python
from specbox.basemodule import SpecAIMSZReview

sp1 = SpecAIMSZReview('review_bundle_specbox.parquet', ext=1)
sp1.plot()
```

To launch the reviewer UI directly:

```bash
specbox-viewer --spectra review_bundle_specbox.parquet --spec-class aimsz-review
```

Notes:
- `aimsz-review` uses canonical string history keys: `aimsz:{object_id}`.
- Saved session CSV columns are: `objid,targetid,ra,dec,data_release,class_vi,z_vi,qa_flag,notes,reviewer,reviewed_at`.
- Legacy labels such as `QSO(Default)` and `LIKELY` are normalized on load; saved output always uses canonical uppercase tokens.
- `sparcl` and `aimsz-review` now show raw spectra by default; use the `Downsample` toolbar toggle to turn on native pyqtgraph downsampling.
- Add `--redshift-table PATH --redshift-key object_id --redshift-column Z` to overlay an external reference-redshift catalog.
