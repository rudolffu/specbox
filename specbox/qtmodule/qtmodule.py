from PySide6.QtGui import QCursor, QFont
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import QApplication, QFrame, QWidget, QSlider, QHBoxLayout, QVBoxLayout, QLabel, QDoubleSpinBox, QSizePolicy
import sys
from ..basemodule import *
import pyqtgraph as pg
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip
import pandas as pd
from importlib.resources import files
from importlib.metadata import PackageNotFoundError, version as dist_version
from pathlib import Path
import os

# locate the data file in the package
data_path = Path(files("specbox").joinpath("data/templates"))
fits_file = data_path / "qso1" / "optical_nir_qso_template_v1.fits"
tb_temp = Table.read(str(fits_file))
tb_temp.rename_columns(['wavelength', 'flux'], ['Wave', 'Flux'])
try:
    viewer_version = dist_version("specbox")
except PackageNotFoundError:
    viewer_version = "0.0.0"


class PGSpecPlot(pg.PlotWidget):
    """Interactive spectrum viewer based on :mod:`pyqtgraph`.

    This widget was originally developed for Euclid SIR 1D spectra but has
    been generalized to handle any spectrum class derived from
    :class:`~specbox.basemodule.ConvenientSpecMixin` (e.g. ``SpecEuclid1d``,
    ``SpecLAMOST``).
    """
    
    coordinate_changed = Signal(float, float)  # Signal for coordinate updates

    def __init__(self, spectra, SpecClass=SpecEuclid1d, initial_counter=0,
                 z_max=5.0, history_dict=None):
        super().__init__()
        self.SpecClass = SpecClass

        # ``spectra`` can either be a FITS file containing multiple extensions
        # or a list of individual spectrum files.
        if isinstance(spectra, (list, tuple)):
            self.speclist = list(spectra)
            self.specfile = None
            self.len_list = len(self.speclist)
        else:
            self.specfile = spectra
            if hasattr(SpecClass, "count_in_file"):
                self.len_list = int(SpecClass.count_in_file(spectra))
            else:
                with fits.open(spectra) as hdul:
                    self.len_list = len(hdul) - 1
            self.speclist = None

        if initial_counter >= self.len_list:
            print("No more spectra to plot.\n\t Plotting the first spectrum.")
            initial_counter = 0

        self.history = history_dict if history_dict is not None else {}

        self.setWindowTitle("Spectrum")
        self.resize(1200, 800)
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=True, y=True)
        self.setLogMode(x=False, y=False)
        self.setAspectLocked(False)
        left_axis = self.getAxis('left')
        left_axis.enableAutoSIPrefix(False)
        self.enableAutoRange()
        self.vb = self.getViewBox()
        self.vb.setMouseMode(self.vb.RectMode)
        self.z_min = 0.0
        self.z_max = z_max
        self.base_z_step = 0.001
        self.slider_min = 0
        self.slider_max = int((1/self.base_z_step) * np.log((1+self.z_max)/(1+self.z_min)))
        self.message = ''
        self.counter = initial_counter

        # Create slider and spin box for redshift control
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.slider_min)
        self.slider.setMaximum(self.slider_max)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(max(1, int((self.slider_max - self.slider_min) / 10)))
        self.slider.valueChanged.connect(self.slider_changed)

        # Create spectrum info title box
        self.spectrum_info_label = QLabel()
        self.spectrum_info_label.setFont(QFont("Arial", 14))
        self.spectrum_info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px solid #666666;
                padding: 8px;
                color: black;
                border-radius: 5px;
            }
        """)
        self.spectrum_info_label.setAlignment(Qt.AlignCenter)
        self.spectrum_info_label.setText("Loading spectrum info...")

        self.redshiftSpin = QDoubleSpinBox()
        self.redshiftSpin.setMinimum(self.z_min)
        self.redshiftSpin.setMaximum(self.z_max)
        self.redshiftSpin.setDecimals(4)
        self.redshiftSpin.setSingleStep(0.001)
        self.redshiftSpin.valueChanged.connect(self.spin_changed)

        self.slider.setStyleSheet("""
            QSlider {
                background-color: white;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #b0c4de;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #6495ed;
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)

        self.plot_next()

    # ------------------------------------------------------------------
    # Utility methods
    def _load_spec(self, index):
        """Load a spectrum by index from ``spectra``."""
        if self.speclist is not None:
            filename = self.speclist[index]
            spec = self.SpecClass(filename)
        else:
            spec = self.SpecClass(self.specfile, ext=index + 1)
        self._ensure_spec_defaults(spec)
        return spec

    def _ensure_spec_defaults(self, spec):
        """Ensure common attributes exist on ``spec``."""
        if not hasattr(spec, 'z_vi'):
            spec.z_vi = getattr(spec, 'redshift', 0.0)
        if not hasattr(spec, 'z_ph'):
            spec.z_ph = getattr(spec, 'redshift', 0.0)
        if not hasattr(spec, 'z_gaia'):
            spec.z_gaia = getattr(spec, 'redshift', 0.0)
        if not hasattr(spec, 'objid'):
            spec.objid = self.counter
        if not hasattr(spec, 'objname'):
            spec.objname = 'Unknown'

    def update_slider_and_spin(self):
        spec = self.spec
        initial_z = spec.z_vi if spec.z_vi > 0 else self.z_min
        initial_slider_value = int((1/self.base_z_step) * np.log((1+initial_z)/(1+self.z_min)))
        self.slider.blockSignals(True)
        self.slider.setValue(initial_slider_value)
        self.slider.blockSignals(False)

        self.redshiftSpin.blockSignals(True)
        self.redshiftSpin.setValue(initial_z)
        self.redshiftSpin.blockSignals(False)

    # ------------------------------------------------------------------
    # Slots
    def slider_changed(self, slider_value):
        z = np.exp(self.base_z_step * slider_value) * (1+self.z_min) - 1
        self.spec.z_vi = z
        self.redshiftSpin.blockSignals(True)
        self.redshiftSpin.setValue(z)
        self.redshiftSpin.blockSignals(False)
        self.clear()
        self.plot_single()

    def spin_changed(self, z):
        slider_value = int((1/self.base_z_step) * np.log((1+z)/(1+self.z_min)))
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.spec.z_vi = z
        self.clear()
        self.plot_single()

    # ------------------------------------------------------------------
    # Plotting helpers
    def plot_single(self):
        spec = self.spec
        if spec.z_vi == 0 and spec.z_ph > 0:
            spec.z_vi = spec.z_ph
        z_vi = spec.z_vi
        z_gaia = spec.z_gaia
        objname = spec.objname
        flux = np.ma.masked_invalid(spec.flux.value)
        flux_sigclip = sigma_clip(flux, sigma=10, maxiters=3)
        wave = spec.wave.value[~flux_sigclip.mask]
        flux = flux_sigclip.data[~flux_sigclip.mask]
        err = spec.err[~flux_sigclip.mask]
        self.plot(wave, flux, pen='b',
                  symbol='o', symbolSize=4, symbolPen=None, connect='finite',
                  symbolBrush='k', antialias=True)
        self.wave = wave
        self.flux = flux

        if getattr(spec, 'telescope', '').lower() == 'euclid':
            wave_temp = tb_temp['Wave'].data * (1+z_vi)
            idx = np.where((wave_temp >= 12047.4) & (wave_temp <= 18734))
            flux_temp = tb_temp['Flux'].data
            wave_temp = wave_temp[idx]
            flux_temp = flux_temp[idx] / np.mean(flux_temp[idx]) * np.abs(flux.mean()) * 1.5
            self.plot(wave_temp, flux_temp, pen=(240, 128, 128), symbol='+',
                      symbolSize=2, symbolPen=None)

        self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
        self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())

        # Update info label above plot
        self.update_spectrum_info_label()
        
        self.autoRange()

    def update_spectrum_info_label(self):
        """Update the spectrum info label above the plot."""
        if not hasattr(self, 'spec'):
            return
            
        spec = self.spec
        z_vi = getattr(spec, 'z_vi', 0.0)
        z_gaia = getattr(spec, 'z_gaia', 0.0)
        objname = getattr(spec, 'objname', 'Unknown')
        objid = getattr(spec, 'objid', 'Unknown')
        
        # Calculate the display number based on which spectrum we're actually showing
        if hasattr(self, '_displaying_spectrum_number'):
            current_spectrum_number = self._displaying_spectrum_number
        else:
            # Fallback to counter (this handles template updates and other cases)
            current_spectrum_number = self.counter if hasattr(self, 'len_list') else 1

        message = f"Spectrum {current_spectrum_number}/{self.len_list}" if hasattr(self, 'len_list') else ""
        text_content = f"{message}  ID: {objid}  z_vi = {z_vi:.4f}, z_gaia = {z_gaia:.4f}"
        
        if hasattr(self, 'spectrum_info_label'):
            self.spectrum_info_label.setText(text_content)

    # ------------------------------------------------------------------
    # Navigation
    def plot_next(self):
        if self.counter >= self.len_list:
            print("No more spectra to plot.")
            return
            
        self.clear()
        spec = self._load_spec(self.counter)
        self.spec = spec
        if spec.objid in self.history:
            spec.z_vi = self.history[spec.objid][4]
            class_vi = self.history[spec.objid][3]
            print(f"\tVisual class from history: {class_vi}.")
            
        # Set the display number before plotting (counter + 1 because we haven't incremented yet)
        self._displaying_spectrum_number = self.counter + 1
        print(f"Spectrum {self._displaying_spectrum_number}/{self.len_list}.")
        
        self.update_slider_and_spin()
        self.plot_single()
        
        # Emit coordinate change signal for future extensions
        if hasattr(self.spec, 'ra') and hasattr(self.spec, 'dec'):
            self.coordinate_changed.emit(self.spec.ra, self.spec.dec)
            
        self.counter += 1

    def plot_previous(self):
        if self.counter > 1:
            self.clear()
            spec = self._load_spec(self.counter - 2)
            self.spec = spec
            if spec.objid in self.history:
                spec.z_vi = self.history[spec.objid][4]
                class_vi = self.history[spec.objid][3]
                print(f"\tVisual class from history: {class_vi}.")
                
            self.counter -= 1
            # Set the display number before plotting (counter is correct after decrement)
            self._displaying_spectrum_number = self.counter
            print(f"Spectrum {self._displaying_spectrum_number}/{self.len_list}.")
            
            self.update_slider_and_spin()
            self.plot_single()
            
            # Emit coordinate change signal for future extensions
            if hasattr(self.spec, 'ra') and hasattr(self.spec, 'dec'):
                self.coordinate_changed.emit(self.spec.ra, self.spec.dec)
        else:
            print("No previous spectrum to plot.")

    # ------------------------------------------------------------------
    # Event handlers
    def keyPressEvent(self, event):
        spec = self.spec
        if event.key() == Qt.Key_Q:
            if spec.objid not in self.history:
                self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                            'QSO(Default)', spec.z_vi]
            else:
                # Update existing entry with current z_vi (preserves classification but updates redshift)
                self.history[spec.objid][4] = spec.z_vi
            if self.counter < self.len_list:
                self.clear()
                self.plot_next()
            else:
                print("No more spectra to plot.")
            if (self.counter-1) % 50 == 0:
                print("Saving temp file to csv (n={})...".format(self.counter))
                temp_filename = f"vi_temp_{self.counter-1}.csv"
                df_new = pd.DataFrame.from_dict(self.history, orient='index')
                df_new.reset_index(inplace=True)
                df_new.rename(columns={'index': 'objid', 0: 'objname', 1: 'ra',
                                        2: 'dec', 3: 'class_vi', 4: 'z_vi'},
                               inplace=True)
                df_new = df_new[['objid', 'objname', 'ra', 'dec', 'class_vi', 'z_vi']]
                df_new.to_csv(temp_filename, index=False)
        if event.key() == Qt.Key_M:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            mouse_pos = self.vb.mapSceneToView(mouse_pos)
            print(f"Mouse position - Wavelength: {mouse_pos.x():.2f}, Flux: {mouse_pos.y():.2e}")
        if event.key() == Qt.Key_Space:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            wave = self.vb.mapSceneToView(mouse_pos).x()
            idx = np.abs(self.wave - wave).argmin()
            wave = self.wave[idx]
            flux = self.flux[idx]
            annotation_text = pg.TextItem(
                text="Wavelength: {0:.2f} Flux: {1:.2e}".format(wave, flux), 
                anchor=(0, 0), color='r', border='w',
                fill=(255, 255, 255, 200))
            annotation_text.setFont(QFont("Arial", 18, QFont.Bold))
            annotation_text.setPos(wave, flux)
            self.addItem(annotation_text)
            print("Wavelength: {0:.2f} Flux: {1:.2e}".format(wave, flux))
        if event.key() == Qt.Key_S:
            print("\tClass: STAR.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'STAR', 0.0]
        if event.key() == Qt.Key_G:
            print("\tClass: GALAXY.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'GALAXY', spec.z_vi]
        if event.key() == Qt.Key_A:
            print("\tClass: QSO(AGN).")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'QSO', spec.z_vi]
        if event.key() == Qt.Key_U:
            print("\tClass: UNKNOWN.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'UNKNOWN', 0.0]
        if event.key() == Qt.Key_L:
            print("\tClass: LIKELY/Unusual QSO.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'LIKELY', spec.z_vi]
        if event.key() == Qt.Key_R:
            self.clear()
            self.plot_single()
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_R:
                self.clear()
                self.spec = self._load_spec(self.counter - 1)
                # For reload, display number is current counter
                self._displaying_spectrum_number = self.counter
                self.update_slider_and_spin()
                self.plot_single()
            if event.key() == Qt.Key_Right:
                self.clear()
                self.counter = self.len_list - 1
                self.plot_next()
            if event.key() == Qt.Key_Left:
                self.clear()
                self.counter = 0
                self.plot_next()
            if event.key() == Qt.Key_B:
                self.clear()
                self.counter = len(self.history) - 1
                self.plot_next()
        if event.key() == Qt.Key_Left:
            self.plot_previous()
        if event.key() == Qt.Key_Right:
            self.plot_next()


class PGSpecPlotApp(QApplication):
    """Standalone application running :class:`PGSpecPlot`."""

    @staticmethod
    def _normalize_objid(value):
        """Normalize objid loaded from CSV.

        Keeps integers as int (for legacy FITS workflows) and keeps non-numeric
        IDs (e.g. SPARCL UUIDs) as str.
        """
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return value
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            if float(value).is_integer():
                return int(value)
            return str(value)
        s = str(value)
        try:
            return int(s)
        except Exception:
            return s

    def __init__(self, spectra, SpecClass=SpecEuclid1d,
                 output_file='vi_output.csv', z_max=5.0, load_history=False):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.spectra = spectra
        self.SpecClass = SpecClass

        if load_history and os.path.exists(self.output_file):
            print(f"Loading history from {self.output_file} ...")
            df = pd.read_csv(self.output_file)
            if 'vi_class' in df.columns:
                df.rename(columns={'vi_class': 'class_vi'}, inplace=True)
            history_dict = {}
            for _, row in df.iterrows():
                objid = self._normalize_objid(row['objid'])
                history_dict[objid] = [row['objname'], row['ra'],
                                                  row['dec'], row['class_vi'],
                                                  row['z_vi']]
            initial_counter = df.shape[0]
        else:
            history_dict = {}
            initial_counter = 0

        self.plot = PGSpecPlot(
            self.spectra, self.SpecClass,
            initial_counter=initial_counter,
            z_max=z_max,
            history_dict=history_dict)
        self.len_list = self.plot.len_list
        self.make_layout()
        self.aboutToQuit.connect(self.save_dict_todf)

    def make_layout(self):
        layout = pg.LayoutWidget()
        layout.resize(1200, 800)
        layout.setWindowTitle(f"PGSpecPlot - Spectra Viewer (v{viewer_version})")
        if self.plot.counter < self.len_list + 1:
            # Instructions with comprehensive keyboard shortcuts
            instruction_text = (
                "Navigation: 'Q' next spectrum, Left/Right arrows previous/next | "
                "Classification: 'A' QSO(AGN), 'S' STAR, 'G' GALAXY, 'U' UNKNOWN, 'L' LIKELY | "
                "Tools: 'Space' wavelength info, 'M' mouse position, 'R' reset zoom | "
                "Advanced: Ctrl+R reload, Ctrl+Left first, Ctrl+Right last, Ctrl+B resume from history"
            )
            toplabel = layout.addLabel(instruction_text, row=0, col=0, colspan=2)
            toplabel.setFont(QFont("Arial", 13))
            toplabel.setMinimumHeight(60)
            toplabel.setMaximumHeight(80)
            toplabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            toplabel.setStyleSheet("background-color: white;color: black;")
            toplabel.setFrameStyle(QFrame.Panel | QFrame.Raised)
            toplabel.setWordWrap(True)
            toplabel.setMidLineWidth(2)
            toplabel.setFrameShadow(QFrame.Sunken)
            toplabel.setMargin(5)
            toplabel.setIndent(5)
            toplabel.setWordWrap(True)
            
            # Add spectrum info label above plot
            layout.addWidget(self.plot.spectrum_info_label, row=1, col=0, colspan=2)
            layout.addWidget(self.plot, row=2, col=0, colspan=2)
            slider_container = QWidget()
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.plot.slider)
            slider_layout.addWidget(self.plot.redshiftSpin)
            slider_container.setLayout(slider_layout)
            layout.addWidget(slider_container, row=3, col=0, colspan=2)
            self.layout = layout
            self.layout.show()

    def keyPressEvent(self, event):
        self.plot.keyPressEvent(event)

    def mousePressEvent(self, event):
        self.plot.mousePressEvent(event)

    def save_dict_todf(self):
        df_new = pd.DataFrame.from_dict(self.plot.history, orient='index')
        df_new.reset_index(inplace=True)
        df_new.rename(columns={'index': 'objid', 0: 'objname', 1: 'ra',
                               2: 'dec', 3: 'class_vi', 4: 'z_vi'},
                      inplace=True)
        df_new = df_new[['objid', 'objname', 'ra', 'dec', 'class_vi', 'z_vi']]
        df_new.to_csv(self.output_file, index=False)


class PGSpecPlotThread(QThread):
    """Run :class:`PGSpecPlotApp` in a separate thread."""

    def __init__(self, spectra=None, SpecClass=SpecEuclid1d, specfile=None, **kwargs):
        super().__init__()
        # Handle backward compatibility: if specfile is provided but spectra is not
        if spectra is None and specfile is not None:
            self.spectra = specfile
        elif spectra is not None:
            self.spectra = spectra
        else:
            raise ValueError("Either 'spectra' or 'specfile' must be provided")
            
        self.SpecClass = SpecClass
        self.app = PGSpecPlotApp(self.spectra, self.SpecClass, **kwargs)

    def run(self):
        exit_code = self.app.exec_()
        sys.exit(exit_code)
