from PySide6.QtGui import QCursor, QFont, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QThreadPool, QRunnable
from PySide6.QtWidgets import (QApplication, QFrame, QWidget, QSlider, QHBoxLayout, 
                               QVBoxLayout, QGridLayout, QDoubleSpinBox, QPushButton, 
                               QLabel, QButtonGroup, QRadioButton, QGroupBox, 
                               QFileDialog, QMessageBox, QComboBox, QProgressBar, QSpinBox,
                               QScrollArea, QCheckBox, QDialog, QTextEdit, QSplitter)
import sys
from ..basemodule import *
import pyqtgraph as pg
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy import units as u
import pandas as pd
from importlib.resources import files
from pathlib import Path
import os
import requests
from urllib.parse import urlencode
import tempfile
from PIL import Image
import io
import json
import hashlib
import time
from scipy.signal import savgol_filter
import re
import concurrent.futures
import threading
from specutils import Spectrum
from importlib.metadata import PackageNotFoundError, version as dist_version
from ..auxmodule.cutout_download import (
    EUCLID_CUTOUT_SURVEYS,
    fetch_cutout,
    get_cache_filename,
    is_no_data_error,
    is_valid_cutout_target,
    load_cutout_from_cache,
    predownload_cutouts,
    print_cli_progress,
    save_cutout_to_cache,
)

# locate the data files in the package
data_path = Path(files("specbox").joinpath("data/templates"))
fits_file = data_path / "qso1" / "optical_nir_qso_template_v1.fits"
tb_temp = Table.read(str(fits_file))
tb_temp.rename_columns(['wavelength', 'flux'], ['Wave', 'Flux'])
try:
    viewer_version = dist_version("specbox")
except PackageNotFoundError:
    viewer_version = "0.0.0"

# Rest-frame emission lines used for template annotations.
# All values are in Angstrom.
_TEMPLATE_EMISSION_LINES = [
    ("Ly α", 1215.67),
    ("C IV", 1549.06),
    ("C III]", 1908.73),
    ("Mg II", 2798.75),
    ("[O II]", 3728.48),
    ("Hβ", 4862.68),
    ("[O III]", 4960.30),
    ("[O III]", 5008.24),
    ("Hα", 6564.61),
    ("O I", 8448.7),
    ("[S III]", 9071.1),
    ("[S III]", 9533.2),
    ("Pa δ", 10052.1),
    ("He I", 10833.2),
    ("Pa γ", 10941.1),
    ("O I", 11290.0),
    ("Pa β", 12821.6),
]


class ImageCutoutWidget(QWidget):
    """Widget for displaying astronomical image cutouts."""
    QA_CONTAMINATION_BIT = 1
    QA_UNUSABLE_BIT = 2
    
    def __init__(self, buffer_dir=None):
        super().__init__()
        self.setFixedWidth(300)
        self._fetch_in_progress = False
        self.buffer_dir = Path(buffer_dir) if buffer_dir else None
        if self.buffer_dir:
            self.buffer_dir.mkdir(exist_ok=True)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Coordinate display
        self.coord_label = QLabel("RA, DEC = -, -")
        self.coord_label.setFont(QFont("Arial", 14))
        self.coord_label.setStyleSheet("color: blue; background-color: #f0f0f0; padding: 2px;")
        self.coord_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        layout.addWidget(self.coord_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMaximumHeight(400)  # Limit height to make layout more compact
        
        self.images_widget = QWidget()
        self.images_layout = QVBoxLayout()
        self.images_layout.setSpacing(5)  # Reduce spacing between images
        self.images_widget.setLayout(self.images_layout)
        # scroll.setWidget(self.images_widget)
        
        # layout.addWidget(scroll)
        layout.addWidget(self.images_widget)

        # Title + QA + toggle
        header_layout = QVBoxLayout()
        qa_group = QGroupBox("Spec-image QA:")
        qa_group.setFont(QFont("Arial", 14, QFont.Bold))
        qa_layout = QVBoxLayout()
        self.qa_contamination_cb = QCheckBox("Contamination from nearby\nsource(s)")
        self.qa_unusable_cb = QCheckBox("Unusable spectrum due to\ndominating artifacts")
        self.qa_contamination_cb.setFont(QFont("Arial", 13))
        self.qa_unusable_cb.setFont(QFont("Arial", 13))
        qa_checkbox_style = "QCheckBox::indicator { width: 24px; height: 24px; }"
        self.qa_contamination_cb.setStyleSheet(qa_checkbox_style)
        self.qa_unusable_cb.setStyleSheet(qa_checkbox_style)
        qa_layout.addWidget(self.qa_contamination_cb)
        qa_layout.addWidget(self.qa_unusable_cb)
        qa_group.setLayout(qa_layout)
        header_layout.addWidget(qa_group)
        
        # Toggle for auto-fetch
        auto_fetch_row = QHBoxLayout()
        self.auto_fetch_cb = QCheckBox("Auto-fetch image cutouts")
        self.auto_fetch_cb.setChecked(True)
        self.auto_fetch_cb.setToolTip("Automatically fetch images when coordinates change")
        auto_fetch_row.addWidget(self.auto_fetch_cb)
        auto_fetch_row.addStretch()
        header_layout.addLayout(auto_fetch_row)
        
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        layout.addWidget(header_widget)
        
        # Controls
        controls_group = QGroupBox("Cutout Settings")
        controls_layout = QGridLayout()
        
        # Size control  
        controls_layout.addWidget(QLabel("Size (arcsec):"), 0, 0)
        self.size_combo = QComboBox()
        self.size_combo.addItems(["5", "10", "15", "30", "60"])
        self.size_combo.setCurrentText("10")
        controls_layout.addWidget(self.size_combo, 0, 1)
        
        # Removed Load Local button as we now use buffer folder
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Progress bar for downloads (moved below images)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.setLayout(layout)

    def get_qa_flag(self):
        flag = 0
        if self.qa_contamination_cb.isChecked():
            flag |= self.QA_CONTAMINATION_BIT
        if self.qa_unusable_cb.isChecked():
            flag |= self.QA_UNUSABLE_BIT
        return int(flag)

    def set_qa_flag(self, qa_flag):
        try:
            flag = int(qa_flag)
        except Exception:
            flag = 0
        self.qa_contamination_cb.blockSignals(True)
        self.qa_unusable_cb.blockSignals(True)
        self.qa_contamination_cb.setChecked((flag & self.QA_CONTAMINATION_BIT) != 0)
        self.qa_unusable_cb.setChecked((flag & self.QA_UNUSABLE_BIT) != 0)
        self.qa_contamination_cb.blockSignals(False)
        self.qa_unusable_cb.blockSignals(False)
        
    # Removed load_local_cutouts method since we removed the Load Local button
    
    def load_online_cutouts(self, ra, dec, objid=None):
        """Load cutouts from HiPS2FITS service for given coordinates."""
        if not self.auto_fetch_cb.isChecked():
            return
            
        if ra is None or dec is None or np.isnan(ra) or np.isnan(dec):
            self.coord_label.setText("RA, DEC = -, -")
            self.add_status_message("No valid coordinates available")
            return
            
        # Update coordinate display
        self.coord_label.setText(f"RA, DEC = {ra:.6f}, {dec:.6f}")
            
        # Prevent multiple simultaneous fetches
        if self._fetch_in_progress:
            return
            
        self._fetch_in_progress = True
        self.clear_images()
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # Indeterminate progress

        # Only fetch color for now to avoid loading issues
        euclid_surveys = list(EUCLID_CUTOUT_SURVEYS)
        
        size_arcsec = float(self.size_combo.currentText())
        
        # Fetch bands sequentially to avoid threading issues
        try:
            for survey, band_name in euclid_surveys:
                try:
                    # Check cache first
                    cached_data = None
                    if objid:
                        cached_data = self.load_cutout_from_cache(objid, survey)
                    
                    if cached_data is not None:
                        self.add_status_message(f"Loading {band_name} from cache...")
                        self.add_image_from_rgb_array(cached_data, band_name)
                    else:
                        self.add_status_message(f"Fetching {band_name}...")
                        # Use JPEG format for faster loading
                        result = fetch_cutout(
                            ra=ra,
                            dec=dec,
                            survey_name=survey,
                            size_arcsec=size_arcsec,
                            width=150,
                            height=150,
                        )
                        if result is not None:
                            # Save to cache
                            if objid:
                                self.save_cutout_to_cache(objid, survey, result)
                            # result is a numpy array for JPEG format (RGB)
                            self.add_image_from_rgb_array(result, band_name)
                        else:
                            self.add_status_message(f"No data returned for {band_name}")
                except Exception as e:
                    self.add_status_message(f"Error fetching {band_name}: {str(e)}")
                    
        except Exception as e:
            self.add_status_message(f"General error in image fetching: {str(e)}")
        finally:
            self.progress.setVisible(False)
            self._fetch_in_progress = False
    
    def add_image_from_pil(self, pil_image, label=""):
        """Add PIL image directly to display (for JPEG format)."""
        try:
            # Convert PIL image to QPixmap
            with io.BytesIO() as buffer:
                pil_image.save(buffer, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue())
            
            self.add_image_widget(pixmap, label)
                
        except Exception as e:
            self.add_status_message(f"Error processing image {label}: {str(e)}")
    
    def add_image_from_array(self, data, label=""):
        """Add image from numpy array to display."""
        try:
            # Simple scaling - could be improved
            data_scaled = np.nan_to_num(data)
            if np.all(data_scaled == 0):
                self.add_status_message(f"Empty image data for {label}")
                return
                
            # Use arcsinh scaling like in notebook for VIS band
            if "VIS" in label or "$I_" in label:
                data_scaled = np.arcsinh(data_scaled * 500)
            else:
                data_scaled = np.arcsinh(data_scaled)
                
            vmin, vmax = np.percentile(data_scaled[data_scaled > 0], [1, 99])
            if vmax > vmin:
                data_scaled = np.clip((data_scaled - vmin) / (vmax - vmin) * 255, 0, 255)
            else:
                data_scaled = np.zeros_like(data_scaled)
            
            # Convert to QPixmap
            height, width = data_scaled.shape
            image = Image.fromarray(data_scaled.astype(np.uint8), mode='L')
            
            # Convert PIL to QPixmap
            with io.BytesIO() as buffer:
                image.save(buffer, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue())
            
            self.add_image_widget(pixmap, label)
                
        except Exception as e:
            self.add_status_message(f"Error processing image {label}: {str(e)}")
    
    def add_image_from_rgb_array(self, rgb_data, label=""):
        """Add RGB image from numpy array (for JPEG format)."""
        try:
            # rgb_data is (height, width, 3) uint8 array
            if len(rgb_data.shape) != 3 or rgb_data.shape[2] != 3:
                self.add_status_message(f"Expected RGB array for {label}, got shape {rgb_data.shape}")
                return
                
            # Convert numpy array directly to PIL Image (RGB mode)
            image = Image.fromarray(rgb_data, mode='RGB')
            
            # Convert PIL to QPixmap
            with io.BytesIO() as buffer:
                image.save(buffer, format='PNG')
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.getvalue())
            
            self.add_image_widget(pixmap, label)
                
        except Exception as e:
            self.add_status_message(f"Error processing RGB image {label}: {str(e)}")
    
    def create_composite_image(self, ra, dec, size_arcsec):
        """Create RGB composite from multiple bands."""
        try:
            # For now, just add a placeholder for composite
            self.add_status_message("Composite image creation not yet implemented")
        except Exception as e:
            self.add_status_message(f"Error creating composite: {str(e)}")
    
    def add_image_from_fits(self, fits_path, label=""):
        """Add FITS image to display."""
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                if data is not None:
                    # Convert to displayable image
                    # Simple scaling - could be improved
                    data_scaled = np.nan_to_num(data)
                    vmin, vmax = np.percentile(data_scaled, [1, 99])
                    data_scaled = np.clip((data_scaled - vmin) / (vmax - vmin) * 255, 0, 255)
                    
                    # Convert to QPixmap
                    height, width = data_scaled.shape
                    image = Image.fromarray(data_scaled.astype(np.uint8), mode='L')
                    
                    # Convert PIL to QPixmap
                    with io.BytesIO() as buffer:
                        image.save(buffer, format='PNG')
                        pixmap = QPixmap()
                        pixmap.loadFromData(buffer.getvalue())
                    
                    self.add_image_widget(pixmap, label)
                    
        except Exception as e:
            self.add_status_message(f"Error loading FITS {fits_path}: {str(e)}")
    
    def add_image_from_file(self, file_path):
        """Add image from file (FITS or regular image formats)."""
        path = Path(file_path)
        
        if path.suffix.lower() in ['.fits', '.fit']:
            self.add_image_from_fits(file_path, path.stem)
        else:
            # Regular image file
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                self.add_image_widget(pixmap, path.stem)
            else:
                self.add_status_message(f"Could not load image: {file_path}")
    
    def add_image_widget(self, pixmap, label=""):
        """Add image widget to layout."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(3)  # Reduce spacing between label and image
        layout.setContentsMargins(2, 2, 2, 2)  # Reduce margins
        
        if label:
            label_widget = QLabel(label)
            label_widget.setAlignment(Qt.AlignCenter)
            label_widget.setFont(QFont("Arial", 16))  # Smaller font
            label_widget.setStyleSheet("margin-bottom: 2px;")  # Reduce bottom margin
            layout.addWidget(label_widget)
        
        image_label = QLabel()
        # Scale image to fit width while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(250, 250, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        image_label.setPixmap(scaled_pixmap)
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(image_label)
        
        container.setLayout(layout)
        self.images_layout.addWidget(container)
    
    def add_status_message(self, message):
        """Add status message to display."""
        label = QLabel(message)
        label.setWordWrap(True)
        label.setStyleSheet("color: green; font-style: italic;")
        self.images_layout.addWidget(label)
        
        # Auto-remove after 3 seconds with weak reference
        import weakref
        weak_label = weakref.ref(label)
        QTimer.singleShot(3000, lambda: self.remove_widget(weak_label()) if weak_label() else None)
    
    def remove_widget(self, widget):
        """Remove widget from layout."""
        try:
            if widget and hasattr(widget, 'parent') and widget.parent():
                self.images_layout.removeWidget(widget)
                widget.deleteLater()
        except RuntimeError:
            # Widget already deleted, ignore
            pass
    
    def clear_images(self):
        """Clear all displayed images."""
        while self.images_layout.count():
            child = self.images_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
    
    def get_cache_filename(self, objid, survey_name):
        """Get cache filename for an object and survey."""
        return get_cache_filename(self.buffer_dir, objid, survey_name)
    
    def save_cutout_to_cache(self, objid, survey_name, image_data):
        """Save cutout image to cache."""
        if not self.buffer_dir:
            return
        try:
            save_cutout_to_cache(self.buffer_dir, objid, survey_name, image_data)
        except Exception as e:
            cache_file = self.get_cache_filename(objid, survey_name)
            print(f"Error saving cache file {cache_file}: {e}")
    
    def load_cutout_from_cache(self, objid, survey_name):
        """Load cutout from cache if exists."""
        if not self.buffer_dir:
            return None
        try:
            return load_cutout_from_cache(self.buffer_dir, objid, survey_name)
        except Exception as e:
            cache_file = self.get_cache_filename(objid, survey_name)
            print(f"Error loading cache file {cache_file}: {e}")
        return None
    
    def prefetch_cutouts_background(self, spectrum_list, current_index, num_prefetch=None):
        """Prefetch cutouts for upcoming spectra in the background."""
        if not self.buffer_dir or current_index >= len(spectrum_list) - 1:
            return
            
        def prefetch_worker():
            # Download all remaining cutouts if num_prefetch is None
            if num_prefetch is None:
                end_index = len(spectrum_list)
            else:
                end_index = min(current_index + num_prefetch + 1, len(spectrum_list))
            surveys = list(EUCLID_CUTOUT_SURVEYS)
            
            for i in range(current_index + 1, end_index):
                try:
                    # Load spectrum to get coordinates and objid
                    spec_data = spectrum_list[i]
                    if hasattr(spec_data, 'ra') and hasattr(spec_data, 'dec') and hasattr(spec_data, 'objid'):
                        ra, dec, objid = spec_data.ra, spec_data.dec, spec_data.objid
                    else:
                        continue
                        
                    # Check if cutout already exists
                    for survey, _ in surveys:
                        if not is_valid_cutout_target(objid, ra, dec):
                            continue
                        if self.load_cutout_from_cache(objid, survey) is not None:
                            continue  # Already cached
                            
                        # Download cutout
                        try:
                            result = fetch_cutout(
                                ra=ra,
                                dec=dec,
                                survey_name=survey,
                                size_arcsec=10,
                                width=150,
                                height=150,
                            )
                            if result is not None:
                                self.save_cutout_to_cache(objid, survey, result)
                            time.sleep(0.1)
                                
                        except Exception as e:
                            if not is_no_data_error(e):
                                print(f"Error prefetching cutout for object {objid}: {e}")
                            time.sleep(0.1)
                            
                except Exception as e:
                    print(f"Error processing spectrum {i} for prefetch: {e}")
                    
        # Run prefetch in background thread
        threading.Thread(target=prefetch_worker, daemon=True).start()


class TemplateManager:
    """Manages spectrum templates."""
    
    def __init__(self):
        self.templates = {}
        self.current_template = "Type 1"
        self.load_default_templates()
    
    def load_default_templates(self):
        """Load default templates."""
        # Load Type 1 template (existing)
        self.templates["Type 1"] = {
            'wave': tb_temp['Wave'].data,
            'flux': tb_temp['Flux'].data,
            'description': 'Type 1 AGN/QSO Template'
        }
        
        # Placeholder for Type 2 template
        self.templates["Type 2"] = None  # Will be loaded if available
    
    def get_template(self, template_name):
        """Get template by name."""
        return self.templates.get(template_name)
    
    def get_available_templates(self):
        """Get list of available template names."""
        return [name for name, template in self.templates.items() if template is not None]
    
    def add_template(self, name, wave, flux, description=""):
        """Add new template."""
        self.templates[name] = {
            'wave': wave,
            'flux': flux, 
            'description': description
        }


class PGSpecPlotEnhanced(pg.PlotWidget):
    """Enhanced version of PGSpecPlot with image cutouts and template switching."""
    
    coordinate_changed = Signal(float, float)  # Signal for coordinate updates
    
    def __init__(self, spectra, SpecClass=SpecEuclid1d, initial_counter=0,
                 z_max=5.0, history_dict=None, euclid_fits=None):
        super().__init__()
        self.SpecClass = SpecClass
        self.template_manager = TemplateManager()
        self.euclid_fits = euclid_fits
        self._euclid_overlay_cache = {}
        self._observed_wmin = None
        self._observed_wmax = None
        self._annotation_wave = None
        self._annotation_flux = None

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

    # Copy all existing methods from original PGSpecPlot
    def _load_spec(self, index_zero_based):
        """Load a spectrum by 0-based index from ``spectra``.
        
        Args:
            index_zero_based: 0-based spectrum index (0 = first spectrum)
                                  For FITS files: automatically converted to 1-based extension number
        """
        if self.speclist is not None:
            # For list of files: use 0-based index directly
            filename = self.speclist[index_zero_based]
            spec = self.SpecClass(filename)
        else:
            # For multi-extension FITS: convert 0-based index to 1-based extension number
            spec = self.SpecClass(self.specfile, ext=index_zero_based + 1)
        self._ensure_spec_defaults(spec)
        return spec

    def _ensure_spec_defaults(self, spec):
        """Ensure common attributes exist on ``spec``."""
        if not hasattr(spec, 'z_vi'):
            spec.z_vi = getattr(spec, 'redshift', 0.0)
        if not hasattr(spec, 'z_ph'):
            spec.z_ph = getattr(spec, 'redshift', 0.0)
        if not hasattr(spec, 'z_gaia'):
            spec.z_gaia = None
        if not hasattr(spec, 'objid'):
            spec.objid = self.counter
        if not hasattr(spec, 'objname'):
            spec.objname = 'Unknown'
        if not hasattr(spec, 'qa_flag'):
            spec.qa_flag = 0

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

    def slider_changed(self, slider_value):
        z = np.exp(self.base_z_step * slider_value) * (1+self.z_min) - 1
        self.spec.z_vi = z
        self.redshiftSpin.blockSignals(True)
        self.redshiftSpin.setValue(z)
        self.redshiftSpin.blockSignals(False)
        self.clear()
        self.plot_single()

    def spin_changed(self, z_value):
        self.spec.z_vi = z_value
        slider_value = int((1/self.base_z_step) * np.log((1+z_value)/(1+self.z_min)))
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.clear()
        self.plot_single()

    def plot_single(self):
        """Plot the spectrum without template."""
        spec = self.spec
        is_sparcl = 'SpecSparcl' in globals() and isinstance(spec, SpecSparcl)
        is_euclid = getattr(spec, 'telescope', '').lower() == 'euclid'
        self._observed_wmin = None
        self._observed_wmax = None
        self._annotation_wave = None
        self._annotation_flux = None
        
        # Follow original code pattern with sigma clipping
        wave_full = spec.wave.value if hasattr(spec.wave, 'value') else spec.wave
        flux_full = spec.flux.value if hasattr(spec.flux, 'value') else spec.flux

        if is_sparcl:
            idx = (wave_full >= 3800) & (wave_full <= 9300)
            wave_full = wave_full[idx]
            flux_full = flux_full[idx]
        
        if is_euclid:
            finite = np.isfinite(wave_full) & np.isfinite(flux_full)
            wave = wave_full[finite]
            flux = flux_full[finite]
        else:
            # Apply sigma clipping like original code
            flux_masked = np.ma.masked_invalid(flux_full)
            flux_sigclip = sigma_clip(flux_masked, sigma=10, maxiters=3)
            wave = wave_full[~flux_sigclip.mask]
            flux = flux_sigclip.data[~flux_sigclip.mask]
        
        # Store cleaned data for template scaling
        self.wave = wave
        self.flux = flux
        annotation_wave_parts = [np.asarray(wave)]
        annotation_flux_parts = [np.asarray(flux)]
        
        if is_sparcl:
            # Make the raw SPARCL spectrum semi-transparent to emphasize smoothing.
            self.plot(wave, flux, pen=pg.mkPen((0, 0, 255, 80), width=1), antialias=True)
            n = int(len(flux))
            if n >= 7:
                window_length = min(31, n if n % 2 == 1 else n - 1)
                if window_length >= 7:
                    polyorder = 3 if window_length > 3 else 2
                    flux_sm = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
                    self.plot(wave, flux_sm, pen=pg.mkPen('k', width=3), antialias=True)

            # Optional: overlay matching Euclid spectrum (extname == euclid_object_id).
            euclid_object_id = getattr(spec, "euclid_object_id", None)
            if self.euclid_fits is not None and euclid_object_id not in (None, "", 0):
                euclid_spec = self._load_euclid_overlay(euclid_object_id)
                if euclid_spec is not None:
                    euclid_wave = euclid_spec.wave.value
                    euclid_flux = euclid_spec.flux.value
                    euclid_good_mask = getattr(euclid_spec, 'good_mask', None)
                    scale = 1.0
                    denom_flux = euclid_flux
                    if euclid_good_mask is not None and len(euclid_good_mask) == len(euclid_flux):
                        good = np.asarray(euclid_good_mask, dtype=bool)
                        good = good & np.isfinite(euclid_wave) & np.isfinite(euclid_flux)
                        if np.any(good):
                            denom_flux = euclid_flux[good]
                    denom = np.nanmedian(np.abs(denom_flux))
                    numer = np.nanmedian(np.abs(flux))
                    if np.isfinite(denom) and denom > 0 and np.isfinite(numer) and numer > 0:
                        scale = numer / denom
                    euclid_flux_scaled = euclid_flux * scale
                    # Plot unmasked Euclid overlay in grey with alpha.
                    self.plot(
                        euclid_wave,
                        euclid_flux_scaled,
                        pen=pg.mkPen((95, 95, 95, 150), width=2),
                        antialias=True,
                    )
                    # Overlay good Euclid pixels.
                    if euclid_good_mask is not None and len(euclid_good_mask) == len(euclid_flux):
                        good = np.asarray(euclid_good_mask, dtype=bool)
                        good = good & np.isfinite(euclid_wave) & np.isfinite(euclid_flux_scaled)
                        if np.any(good):
                            self.plot(
                                euclid_wave[good],
                                euclid_flux_scaled[good],
                                pen=pg.mkPen((0, 150, 0, 220), width=2),
                                antialias=True,
                            )
                    annotation_wave_parts.append(np.asarray(euclid_wave))
                    annotation_flux_parts.append(np.asarray(euclid_flux_scaled))

                    try:
                        self._observed_wmin = float(min(np.nanmin(wave), np.nanmin(euclid_wave)))
                        self._observed_wmax = float(max(np.nanmax(wave), np.nanmax(euclid_wave)))
                    except Exception:
                        self._observed_wmin = float(np.nanmin(wave))
                        self._observed_wmax = float(np.nanmax(wave))
            else:
                self._observed_wmin = float(np.nanmin(wave))
                self._observed_wmax = float(np.nanmax(wave))
        else:
            if is_euclid:
                # Plot unmasked Euclid spectrum in grey with alpha.
                self.plot(wave, flux, pen=pg.mkPen((95, 95, 95, 150), width=2), antialias=True)
                # Overlay good pixels when mask info is available.
                good_mask = getattr(spec, 'good_mask', None)
                if good_mask is not None and len(good_mask) == len(wave_full):
                    good_mask = np.asarray(good_mask, dtype=bool)
                    good_mask = good_mask & np.isfinite(wave_full) & np.isfinite(flux_full)
                    if np.any(good_mask):
                        wave_good = wave_full[good_mask]
                        flux_good = flux_full[good_mask]
                        self.plot(wave_good, flux_good, pen=pg.mkPen((0, 0, 180, 220), width=2), antialias=True)
                        self.wave = wave_good
                        self.flux = flux_good
                        annotation_wave_parts.append(np.asarray(wave))
                        annotation_flux_parts.append(np.asarray(flux))
                self._observed_wmin = float(np.nanmin(wave))
                self._observed_wmax = float(np.nanmax(wave))
            else:
                # Plot as dots connected by lines like original
                self.plot(wave, flux, pen='b', symbol='o', symbolSize=4, 
                         symbolPen=None, connect='finite', symbolBrush='k', antialias=True)
                self._observed_wmin = float(np.nanmin(wave))
                self._observed_wmax = float(np.nanmax(wave))

        if annotation_wave_parts and annotation_flux_parts:
            self._annotation_wave = np.concatenate(annotation_wave_parts)
            self._annotation_flux = np.concatenate(annotation_flux_parts)

        # Update labels with proper units
        if hasattr(spec, 'flux_unit') and spec.flux_unit is not None:
            flux_unit_str = f'Flux ({spec.flux_unit})'

        wave_unit_str = 'Wavelength (Å)'
        if hasattr(spec, 'wave_unit') and spec.wave_unit is not None:
            wave_unit_str = f'Wavelength ({spec.wave_unit})'

        
        # Plot template and always clip it to the observed wavelength span.
        template = self.template_manager.get_template(self.template_manager.current_template)
        if template is not None:
            z_vi = getattr(spec, 'z_vi', getattr(spec, 'redshift', 0.0))
            if z_vi is None:
                z_vi = 0.0
            wave_temp = template['wave'] * (1 + z_vi)
            flux_temp = template['flux']

            if self._observed_wmin is not None and self._observed_wmax is not None:
                wmin = float(self._observed_wmin)
                wmax = float(self._observed_wmax)
            else:
                finite = np.isfinite(wave) & np.isfinite(flux)
                if np.any(finite):
                    wmin = float(np.nanmin(wave[finite]))
                    wmax = float(np.nanmax(wave[finite]))
                else:
                    wmin = float(np.nanmin(wave))
                    wmax = float(np.nanmax(wave))
            idx = (wave_temp >= wmin) & (wave_temp <= wmax)
            wave_temp = wave_temp[idx]
            flux_temp = flux_temp[idx]

            if wave_temp.size > 0 and np.isfinite(np.mean(flux_temp)) and np.mean(flux_temp) != 0:
                flux_temp_scaled = flux_temp / np.mean(flux_temp) * np.abs(flux.mean()) * 1.5
                self.plot(wave_temp, flux_temp_scaled, pen=pg.mkPen('r', width=2), antialias=True)
                self._label_template_emission_lines(
                    wmin=wmin,
                    wmax=wmax,
                    z=z_vi,
                )

        self.setLabel('left', flux_unit_str)
        self.setLabel('bottom', wave_unit_str)

        # Update info label above plot
        self.update_spectrum_info_label()
        
        # Keep x-limits fixed to the clipped observed wavelength span.
        if self._observed_wmin is not None and self._observed_wmax is not None:
            self.setXRange(float(self._observed_wmin), float(self._observed_wmax), padding=0.0)
        # Keep y dynamic.
        self.enableAutoRange(axis='y', enable=True)
        
        # Coordinate signal emission is now handled in navigation methods

    def update_spectrum_info_label(self):
        """Update the spectrum info label above the plot."""
        if not hasattr(self, 'spec'):
            return
            
        spec = self.spec
        z_vi = getattr(spec, 'z_vi', 0.0)
        z_gaia = getattr(spec, 'z_gaia', None)
        objname = getattr(spec, 'objname', 'Unknown')
        objid = getattr(spec, 'objid', 'Unknown')
        class_vi = None
        if objid in self.history and len(self.history[objid]) > 3:
            class_vi = self.history[objid][3]
        if class_vi in (None, ""):
            class_vi = getattr(spec, "class_vi", None)

        def _fmt_z(label, value, *, hide_zero=True):
            if value is None:
                return None
            try:
                v = float(value)
            except Exception:
                return None
            if not np.isfinite(v):
                return None
            if hide_zero and v == 0.0:
                return None
            return f"{label} = {v:.4f}"
        
        # Calculate the display number based on which spectrum we're actually showing
        # In plot_next: counter gets incremented AFTER plotting, so counter+1 is the display number
        # In plot_previous: counter gets decremented BEFORE plotting, so counter is the display number
        # We need to determine what spectrum we're actually displaying
        if hasattr(self, '_displaying_spectrum_number'):
            current_spectrum_number = self._displaying_spectrum_number
        else:
            # Fallback to counter (this handles template updates and other cases)
            current_spectrum_number = self.counter if hasattr(self, 'len_list') else 1

        message = f"Spectrum {current_spectrum_number}/{self.len_list}" if hasattr(self, 'len_list') else ""
        parts = []
        if message:
            parts.append(message)
        parts.append(f"ID: {objid}")
        if 'SpecSparcl' in globals() and isinstance(spec, SpecSparcl):
            targetid = getattr(spec, 'targetid', None)
            if targetid not in (None, "", 0):
                parts.append(f"targetid: {targetid}")
        parts.append(_fmt_z("z_vi", z_vi, hide_zero=False) or "z_vi = -")
        if class_vi not in (None, ""):
            parts.append(f"class_vi: {class_vi}")

        if 'SpecSparcl' in globals() and isinstance(spec, SpecSparcl):
            dr = str(getattr(spec, 'data_release', '') or '')
            if 'desi' in dr.lower():
                parts.append(_fmt_z("z_desi", getattr(spec, 'redshift', None), hide_zero=False) or "z_desi = -")

        z_gaia_str = _fmt_z("z_gaia", z_gaia, hide_zero=True)
        if z_gaia_str is not None:
            parts.append(z_gaia_str)

        text_content = "  ".join(parts)
        
        if hasattr(self, 'spectrum_info_label'):
            self.spectrum_info_label.setText(text_content)

    def plot_template(self):
        """Plot template with current redshift."""
        template = self.template_manager.get_template(self.template_manager.current_template)
        if template is None:
            return
            
        z = getattr(self.spec, 'z_vi', getattr(self.spec, 'redshift', 0.0))
        if z is None:
            z = 0.0
        wave_shifted = template['wave'] * (1 + z)
        flux_template = template['flux']

        if self._observed_wmin is not None and self._observed_wmax is not None:
            wmin = float(self._observed_wmin)
            wmax = float(self._observed_wmax)
        else:
            wave_full = self.spec.wave.value if hasattr(self.spec.wave, 'value') else self.spec.wave
            flux_full = self.spec.flux.value if hasattr(self.spec.flux, 'value') else self.spec.flux
            finite = np.isfinite(wave_full) & np.isfinite(flux_full)
            if np.any(finite):
                wmin = float(np.nanmin(wave_full[finite]))
                wmax = float(np.nanmax(wave_full[finite]))
            else:
                wmin = float(np.nanmin(wave_full))
                wmax = float(np.nanmax(wave_full))
        idx = (wave_shifted >= wmin) & (wave_shifted <= wmax)
        wave_shifted = wave_shifted[idx]
        flux_template = flux_template[idx]

        if wave_shifted.size == 0 or not np.isfinite(np.mean(flux_template)) or np.mean(flux_template) == 0:
            return
        
        # Scale template like in original code (unclipped on both sides as requested)
        if hasattr(self, 'flux') and len(self.flux) > 0:
            # Use the cleaned flux from plot_single for scaling
            flux_scaled = flux_template / np.mean(flux_template) * np.abs(self.flux.mean()) * 1.5
        else:
            # Fallback if cleaned flux not available
            spec_flux = self.spec.flux.value if hasattr(self.spec.flux, 'value') else self.spec.flux
            flux_scaled = flux_template / np.mean(flux_template) * np.abs(np.nanmean(spec_flux)) * 1.5
        
        # Plot template clipped to the observed wavelength range.
        self.plot(wave_shifted, flux_scaled, pen=pg.mkPen('r', width=2), antialias=True)
        if self._observed_wmin is not None and self._observed_wmax is not None:
            self._label_template_emission_lines(wmin=self._observed_wmin, wmax=self._observed_wmax, z=z)
        elif hasattr(self, 'wave') and self.wave is not None and len(self.wave) > 0:
            self._label_template_emission_lines(wmin=float(np.nanmin(self.wave)), wmax=float(np.nanmax(self.wave)), z=z)
        
        # Update info label to reflect new redshift
        self.update_spectrum_info_label()
        
        # Do NOT auto-range during template updates - preserves x-axis range

    def _load_euclid_overlay(self, euclid_object_id):
        if euclid_object_id is None:
            return None
        try:
            if isinstance(euclid_object_id, float) and np.isnan(euclid_object_id):
                return None
        except Exception:
            pass

        key = euclid_object_id
        if isinstance(key, (np.integer, int)):
            key = int(key)
        elif isinstance(key, (np.floating, float)):
            if float(key).is_integer():
                key = int(key)
        key = str(key).strip()
        if not key or key.lower() == "nan":
            return None

        if key in self._euclid_overlay_cache:
            return self._euclid_overlay_cache[key]
        try:
            sp = SpecEuclid1d(self.euclid_fits, extname=key)
        except Exception as e:
            print(f"Euclid overlay load failed for extname={key}: {e}")
            self._euclid_overlay_cache[key] = None
            return None
        self._euclid_overlay_cache[key] = sp
        return sp

    def _label_template_emission_lines(self, *, wmin, wmax, z):
        """Overlay emission-line markers for the template within [wmin, wmax]."""
        if not np.isfinite(wmin) or not np.isfinite(wmax) or wmax <= wmin:
            return
        try:
            z = float(z)
        except Exception:
            return
        if not np.isfinite(z) or z < 0:
            return

        # Use current (cleaned) spectrum flux to place labels near the top.
        y_ref = None
        if hasattr(self, 'flux') and self.flux is not None and len(self.flux) > 0:
            try:
                y_ref = float(np.nanmax(self.flux))
            except Exception:
                y_ref = None
        if y_ref is None or not np.isfinite(y_ref):
            y_ref = 0.0
        y_base = y_ref * 0.92 if y_ref != 0 else 0.0

        pen = pg.mkPen((140, 140, 140), width=2, style=Qt.DashLine)
        text_color = (80, 80, 80)

        lines = _TEMPLATE_EMISSION_LINES

        # Stagger labels to reduce overlap.
        y_offsets = [0.0, 0.06, 0.12]
        k = 0
        for name, rest_aa in lines:
            x = rest_aa * (1.0 + z)
            if x < wmin or x > wmax:
                continue
            self.addItem(pg.InfiniteLine(pos=x, angle=90, pen=pen, movable=False))
            y = y_base * (1.0 - y_offsets[k % len(y_offsets)]) if y_base != 0 else 0.0
            label = pg.TextItem(text=str(name), color=text_color, anchor=(0.5, 1.0))
            label.setPos(x, y)
            self.addItem(label)
            k += 1

    def plot_next(self):
        """Plot next spectrum."""
        if self.counter >= self.len_list:
            print("No more spectra to plot.")
            return
        
        self.clear()
        
        # Load spectrum using original logic
        if self.speclist is not None:
            filename = self.speclist[self.counter]
            spec = self.SpecClass(filename)
        else:
            spec = self.SpecClass(self.specfile, ext=self.counter + 1)
            
        self.spec = spec
        self._ensure_spec_defaults(spec)
        
        if spec.objid in self.history:
            spec.z_vi = self.history[spec.objid][4]
            if len(self.history[spec.objid]) > 7:
                spec.qa_flag = self.history[spec.objid][7]
            class_vi = self.history[spec.objid][3]
            print(f"\tVisual class from history: {class_vi}.")
            
        self.update_slider_and_spin()
        
        # Set the display number before plotting (counter + 1 because we haven't incremented yet)
        self._displaying_spectrum_number = self.counter + 1
        self.plot_single()
        
        # Emit coordinate change signal for cutout loading
        if hasattr(self.spec, 'ra') and hasattr(self.spec, 'dec'):
            self.coordinate_changed.emit(self.spec.ra, self.spec.dec)
            
        self.counter += 1

    def plot_previous(self):
        """Plot previous spectrum."""
        if self.counter > 1:
            self.clear()
            
            # Load spectrum using original logic
            if self.speclist is not None:
                filename = self.speclist[self.counter - 2]
                spec = self.SpecClass(filename)
            else:
                spec = self.SpecClass(self.specfile, ext=self.counter - 1)
                
            self.spec = spec
            self._ensure_spec_defaults(spec)
            
            if spec.objid in self.history:
                spec.z_vi = self.history[spec.objid][4]
                if len(self.history[spec.objid]) > 7:
                    spec.qa_flag = self.history[spec.objid][7]
                class_vi = self.history[spec.objid][3]
                print(f"\tVisual class from history: {class_vi}.")
                
            self.counter -= 1
            self.update_slider_and_spin()
            
            # Set the display number before plotting (counter is correct after decrement)
            self._displaying_spectrum_number = self.counter
            self.plot_single()
            
            # Emit coordinate change signal for cutout loading
            if hasattr(self.spec, 'ra') and hasattr(self.spec, 'dec'):
                self.coordinate_changed.emit(self.spec.ra, self.spec.dec)
        else:
            print("No previous spectrum to plot.")

    def jump_to_spectrum(self, index_one_based):
        """Jump directly to a spectrum by 1-based index."""
        try:
            index_one_based = int(index_one_based)
        except Exception:
            print(f"Invalid index: {index_one_based}")
            return

        if index_one_based < 1 or index_one_based > self.len_list:
            print(f"Index out of range: {index_one_based}. Valid range is 1..{self.len_list}.")
            return

        self.clear()
        target_zero_based = index_one_based - 1
        if self.speclist is not None:
            filename = self.speclist[target_zero_based]
            spec = self.SpecClass(filename)
        else:
            spec = self.SpecClass(self.specfile, ext=target_zero_based + 1)

        self.spec = spec
        self._ensure_spec_defaults(spec)

        if spec.objid in self.history:
            spec.z_vi = self.history[spec.objid][4]
            if len(self.history[spec.objid]) > 7:
                spec.qa_flag = self.history[spec.objid][7]
            class_vi = self.history[spec.objid][3]
            print(f"\tVisual class from history: {class_vi}.")

        self.counter = index_one_based
        self.update_slider_and_spin()
        self._displaying_spectrum_number = index_one_based
        self.plot_single()

        if hasattr(self.spec, 'ra') and hasattr(self.spec, 'dec'):
            self.coordinate_changed.emit(self.spec.ra, self.spec.dec)

    def change_template(self, template_name):
        """Change current template."""
        if template_name in self.template_manager.get_available_templates():
            self.template_manager.current_template = template_name
            self.clear()
            self.plot_single()

    def _annotate_at_wave(self, wave_pos):
        """Annotate the nearest plotted point to ``wave_pos``."""
        wave_arr = self._annotation_wave if self._annotation_wave is not None else getattr(self, 'wave', None)
        flux_arr = self._annotation_flux if self._annotation_flux is not None else getattr(self, 'flux', None)
        if wave_arr is None or flux_arr is None:
            return
        finite = np.isfinite(wave_arr) & np.isfinite(flux_arr)
        if not np.any(finite):
            return
        wave_fin = wave_arr[finite]
        flux_fin = flux_arr[finite]
        idx = np.abs(wave_fin - wave_pos).argmin()
        wave_val = wave_fin[idx]
        flux_val = flux_fin[idx]
        annotation_text = pg.TextItem(
            text="Wavelength: {0:.2f} Flux: {1:.2e}".format(wave_val, flux_val),
            anchor=(0, 0), color='r', border='w',
            fill=(255, 255, 255, 200))
        annotation_text.setFont(QFont("Arial", 18, QFont.Bold))
        annotation_text.setPos(wave_val, flux_val)
        self.addItem(annotation_text)
        print("Wavelength: {0:.2f} Flux: {1:.2e}".format(wave_val, flux_val))

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        spec = self.spec

        def _history_payload(class_vi, z_vi):
            targetid = getattr(spec, 'targetid', None)
            data_release = getattr(spec, 'data_release', None)
            qa_flag = getattr(spec, 'qa_flag', 0)
            return [spec.objname, spec.ra, spec.dec, class_vi, z_vi, targetid, data_release, qa_flag]

        if event.key() == Qt.Key_Q:
            if spec.objid not in self.history:
                self.history[spec.objid] = _history_payload('QSO(Default)', spec.z_vi)
            else:
                # Update existing entry with current z_vi (preserves classification but updates redshift)
                self.history[spec.objid][4] = spec.z_vi
                if len(self.history[spec.objid]) < 8:
                    self.history[spec.objid].extend([None] * (8 - len(self.history[spec.objid])))
                self.history[spec.objid][5] = self.history[spec.objid][5] if self.history[spec.objid][5] is not None else getattr(spec, 'targetid', None)
                self.history[spec.objid][6] = self.history[spec.objid][6] if self.history[spec.objid][6] is not None else getattr(spec, 'data_release', None)
                self.history[spec.objid][7] = getattr(spec, 'qa_flag', 0)
            if self.counter < self.len_list:
                self.clear()
                self.plot_next()
            else:
                print("No more spectra to plot.")
            # Temp save every 50 spectra like original
            if (self.counter-1) % 50 == 0:
                print("Saving temp file to csv (n={})...".format(self.counter))
                temp_filename = f"vi_temp_{self.counter-1}.csv"
                rows = []
                for objid_key, v in self.history.items():
                    if len(v) < 5:
                        continue
                    targetid = v[5] if len(v) > 5 else None
                    data_release = v[6] if len(v) > 6 else None
                    qa_flag = v[7] if len(v) > 7 else 0
                    rows.append(
                        {
                            "objid": objid_key,
                            "objname": v[0],
                            "ra": v[1],
                            "dec": v[2],
                            "class_vi": v[3],
                            "z_vi": v[4],
                            "targetid": targetid,
                            "data_release": data_release,
                            "qa_flag": qa_flag,
                        }
                    )
                df_new = pd.DataFrame(rows)
                df_new.to_csv(temp_filename, index=False)
                
        elif event.key() == Qt.Key_S:
            print("\tClass: STAR.")
            self.history[spec.objid] = _history_payload('STAR', 0.0)
            self.update_spectrum_info_label()
        elif event.key() == Qt.Key_G:
            print("\tClass: GALAXY.")
            self.history[spec.objid] = _history_payload('GALAXY', spec.z_vi)
            self.update_spectrum_info_label()
        elif event.key() == Qt.Key_A:
            print("\tClass: QSO(AGN).")
            self.history[spec.objid] = _history_payload('QSO', spec.z_vi)
            self.update_spectrum_info_label()
        elif event.key() == Qt.Key_U:
            print("\tClass: UNKNOWN.")
            self.history[spec.objid] = _history_payload('UNKNOWN', 0.0)
            self.update_spectrum_info_label()
        elif event.key() == Qt.Key_L:
            print("\tClass: LIKELY/Unusual QSO.")
            self.history[spec.objid] = _history_payload('LIKELY', spec.z_vi)
            self.update_spectrum_info_label()
        if event.key() == Qt.Key_R:
            self.clear()
            self.plot_single()
        if event.modifiers() & Qt.ControlModifier:
            # Mouse position like original
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            vb = self.getViewBox()
            mouse_pos = vb.mapSceneToView(mouse_pos)
            print(mouse_pos)
        elif event.key() == Qt.Key_Space:
            # Annotate spectrum at mouse position like original
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            vb = self.getViewBox()
            wave_pos = vb.mapSceneToView(mouse_pos).x()
            self._annotate_at_wave(wave_pos)
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_R:
                self.clear()
                # Reload current spectrum using original logic
                if self.speclist is not None:
                    filename = self.speclist[self.counter - 1]
                    spec = self.SpecClass(filename)
                else:
                    spec = self.SpecClass(self.specfile, ext=self.counter)
                self.spec = spec
                self._ensure_spec_defaults(spec)
                # For reload, display number is current counter
                self._displaying_spectrum_number = self.counter
                self.update_slider_and_spin()
                self.plot_single()
            elif event.key() == Qt.Key_Right:
                self.clear()
                self.counter = self.len_list - 1
                self.plot_next()
            elif event.key() == Qt.Key_Left:
                self.clear()
                self.counter = 0
                self.plot_next()
            elif event.key() == Qt.Key_B:
                self.clear()
                self.counter = len(self.history) - 1
                self.plot_next()
        elif event.key() == Qt.Key_Left:
            self.plot_previous()
        elif event.key() == Qt.Key_Right:
            self.plot_next()
        elif event.key() == Qt.Key_M:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            mouse_pos = self.vb.mapSceneToView(mouse_pos)
            print(f"Mouse position - Wavelength: {mouse_pos.x():.2f}, Flux: {mouse_pos.y():.2e}")
        elif event.key() == Qt.Key_Space:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            wave = self.vb.mapSceneToView(mouse_pos).x()
            self._annotate_at_wave(wave)


class PGSpecPlotAppEnhanced(QApplication):
    """Enhanced standalone application with image cutouts and template switching."""

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

    @staticmethod
    def _normalize_qa_flag(value):
        """Normalize qa_flag loaded from CSV/history."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0
        try:
            return int(value)
        except Exception:
            return 0

    def __init__(self, spectra, SpecClass=SpecEuclid1d,
                 output_file='vi_output.csv', z_max=5.0, load_history=False,
                 euclid_fits=None, cutout_buffer_dir=None, enable_background_prefetch=True):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.spectra = spectra
        self.SpecClass = SpecClass
        self.euclid_fits = euclid_fits
        self.enable_background_prefetch = enable_background_prefetch

        if load_history and os.path.exists(self.output_file):
            print(f"Loading history from {self.output_file} ...")
            df = pd.read_csv(self.output_file)
            if 'vi_class' in df.columns:
                df.rename(columns={'vi_class': 'class_vi'}, inplace=True)
            if 'qa_flag' not in df.columns:
                df['qa_flag'] = 0
                try:
                    df.to_csv(self.output_file, index=False)
                    print("Added missing 'qa_flag' column to history file with default 0.")
                except Exception as e:
                    print(f"Could not persist new 'qa_flag' column to history file: {e}")
            history_dict = {}
            for _, row in df.iterrows():
                objid = self._normalize_objid(row['objid'])
                targetid = row['targetid'] if 'targetid' in df.columns else None
                data_release = row['data_release'] if 'data_release' in df.columns else None
                qa_flag = self._normalize_qa_flag(row['qa_flag']) if 'qa_flag' in df.columns else 0
                history_dict[objid] = [
                    row.get('objname', 'Unknown'),
                    row.get('ra', np.nan),
                    row.get('dec', np.nan),
                    row.get('class_vi', ''),
                    row.get('z_vi', np.nan),
                    targetid,
                    data_release,
                    qa_flag,
                ]
            initial_counter = df.shape[0]
        else:
            history_dict = {}
            initial_counter = 0

        self.plot = PGSpecPlotEnhanced(
            self.spectra, self.SpecClass,
            initial_counter=initial_counter,
            z_max=z_max,
            history_dict=history_dict,
            euclid_fits=self.euclid_fits)
        self.len_list = self.plot.len_list
        
        # Create buffer directory for cutouts
        if cutout_buffer_dir is not None:
            buffer_dir = Path(cutout_buffer_dir)
        elif isinstance(spectra, str):  # Single FITS file
            buffer_dir = Path(spectra).parent / "cutout_buffer"
        else:  # List of files
            buffer_dir = Path(spectra[0]).parent / "cutout_buffer"
        
        # Create image cutout widget with buffer directory
        self.cutout_widget = ImageCutoutWidget(buffer_dir=buffer_dir)
        self.cutout_widget.qa_contamination_cb.stateChanged.connect(self.on_qa_flag_changed)
        self.cutout_widget.qa_unusable_cb.stateChanged.connect(self.on_qa_flag_changed)
        
        # Connect signals - need a wrapper to pass object ID
        self.plot.coordinate_changed.connect(self.on_coordinate_changed)
        
        # Trigger initial image load for first spectrum
        if hasattr(self.plot, 'spec') and hasattr(self.plot.spec, 'ra') and hasattr(self.plot.spec, 'dec'):
            objid = getattr(self.plot.spec, 'objid', None)
            self.cutout_widget.load_online_cutouts(self.plot.spec.ra, self.plot.spec.dec, objid)
        self.sync_qa_checkbox_from_current_spec()
            
        # Start background prefetching for next objects
        if self.enable_background_prefetch:
            self.start_background_prefetch()
        
        self.make_layout()
        self.aboutToQuit.connect(self.save_dict_todf)
    
    def on_coordinate_changed(self, ra, dec):
        """Handle coordinate changes and pass object ID to cutout widget."""
        objid = getattr(self.plot.spec, 'objid', None) if hasattr(self.plot, 'spec') else None
        self.cutout_widget.load_online_cutouts(ra, dec, objid)
        self.sync_qa_checkbox_from_current_spec()

    def sync_qa_checkbox_from_current_spec(self):
        """Restore QA checkboxes from current spec/history."""
        if not hasattr(self.plot, 'spec'):
            return
        spec = self.plot.spec
        objid = getattr(spec, 'objid', None)
        qa_flag = 0
        if objid in self.plot.history and len(self.plot.history[objid]) > 7:
            qa_flag = self._normalize_qa_flag(self.plot.history[objid][7])
        else:
            qa_flag = self._normalize_qa_flag(getattr(spec, 'qa_flag', 0))
        spec.qa_flag = qa_flag
        self.cutout_widget.set_qa_flag(qa_flag)

    def on_qa_flag_changed(self, _state):
        """Persist QA flag from checkbox selections into current spec/history."""
        if not hasattr(self.plot, 'spec'):
            return
        spec = self.plot.spec
        qa_flag = self.cutout_widget.get_qa_flag()
        spec.qa_flag = qa_flag

        objid = getattr(spec, 'objid', None)
        if objid is None:
            return

        if objid not in self.plot.history:
            targetid = getattr(spec, 'targetid', None)
            data_release = getattr(spec, 'data_release', None)
            self.plot.history[objid] = [
                getattr(spec, 'objname', 'Unknown'),
                getattr(spec, 'ra', np.nan),
                getattr(spec, 'dec', np.nan),
                'QSO(Default)',
                getattr(spec, 'z_vi', 0.0),
                targetid,
                data_release,
                qa_flag,
            ]
            return

        row = self.plot.history[objid]
        if len(row) < 8:
            row.extend([None] * (8 - len(row)))
        row[7] = qa_flag
    
    def start_background_prefetch(self):
        """Start background prefetching of cutouts for upcoming spectra."""
        def prefetch_worker():
            try:
                # Load upcoming spectra data for prefetching
                current_index = getattr(self.plot, 'counter', 0)
                spectra_data = []
                
                # Get ALL remaining spectra data
                for i in range(current_index, self.len_list):
                    try:
                        if self.plot.speclist is not None:
                            # Load from individual files
                            filename = self.plot.speclist[i]
                            spec = self.SpecClass(filename)
                        else:
                            # Load from multi-extension FITS
                            spec = self.SpecClass(self.spectra, ext=i + 1)
                        
                        self.plot._ensure_spec_defaults(spec)
                        
                        # Only prefetch if has coordinates
                        if hasattr(spec, 'ra') and hasattr(spec, 'dec') and hasattr(spec, 'objid'):
                            spectra_data.append(spec)
                            
                    except Exception as e:
                        print(f"Error loading spectrum {i} for prefetch: {e}")
                        continue
                
                # Trigger prefetching if we have data
                if spectra_data:
                    self.cutout_widget.prefetch_cutouts_background(spectra_data, 0, None)
                    
            except Exception as e:
                print(f"Error in background prefetch setup: {e}")
                
        # Run in background thread
        threading.Thread(target=prefetch_worker, daemon=True).start()

    def make_layout(self):
        """Create the enhanced layout with image cutouts and controls."""
        layout = pg.LayoutWidget()
        layout.resize(1300, 900)  # Reasonable width with text wrapping
        layout.setWindowTitle(f"PGSpecPlot Enhanced - Spectra Viewer (v{viewer_version})")
        
        if self.plot.counter < self.len_list + 1:
            # Create toolbar with template controls and save buttons
            toolbar = QWidget()
            toolbar_layout = QHBoxLayout()
            
            # Template selection
            template_group = QGroupBox("Template")
            template_group.setMaximumHeight(45)  # Make template group more compact
            template_layout = QHBoxLayout()
            template_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins
            
            self.template_buttons = QButtonGroup()
            template_names = self.plot.template_manager.get_available_templates()
            
            for i, template_name in enumerate(template_names):
                btn = QRadioButton(template_name)
                if template_name == "Type 1":
                    btn.setChecked(True)
                btn.clicked.connect(lambda checked, name=template_name: 
                                   self.plot.change_template(name) if checked else None)
                self.template_buttons.addButton(btn)
                template_layout.addWidget(btn)
            
            template_group.setLayout(template_layout)
            toolbar_layout.addWidget(template_group)
            
            # Image panel toggle
            self.image_toggle_btn = QPushButton("Hide Images")
            self.image_toggle_btn.clicked.connect(self.toggle_image_panel)
            self.image_toggle_btn.setCheckable(True)
            self.image_toggle_btn.setMaximumHeight(35)  # Make button more compact
            toolbar_layout.addWidget(self.image_toggle_btn)

            toolbar_layout.addWidget(QLabel("Go to index:"))
            self.goto_index_spin = QSpinBox()
            self.goto_index_spin.setMinimum(1)
            self.goto_index_spin.setMaximum(self.len_list)
            self.goto_index_spin.setValue(1)
            self.goto_index_spin.setMaximumHeight(35)
            toolbar_layout.addWidget(self.goto_index_spin)

            self.goto_index_btn = QPushButton("Go")
            self.goto_index_btn.setMaximumHeight(35)
            self.goto_index_btn.clicked.connect(self.go_to_index)
            toolbar_layout.addWidget(self.goto_index_btn)
            
            # Add spacer
            toolbar_layout.addStretch()
            
            # Save buttons
            self.save_png_btn = QPushButton("Save PNG")
            self.save_png_btn.clicked.connect(self.save_png)
            self.save_png_btn.setMaximumHeight(35)
            toolbar_layout.addWidget(self.save_png_btn)

            self.save_btn = QPushButton("Save")
            self.save_btn.clicked.connect(self.save_data)
            self.save_btn.setMaximumHeight(35)  # Make button more compact
            toolbar_layout.addWidget(self.save_btn)
            
            self.save_quit_btn = QPushButton("Save & Quit")
            self.save_quit_btn.clicked.connect(self.save_and_quit)
            self.save_quit_btn.setMaximumHeight(35)  # Make button more compact
            toolbar_layout.addWidget(self.save_quit_btn)
            
            toolbar.setLayout(toolbar_layout)
            toolbar.setMaximumHeight(50)  # Limit toolbar height
            toolbar.setMinimumHeight(40)  # Set minimum height
            layout.addWidget(toolbar, row=0, col=0, colspan=2)
            
            # Instructions with comprehensive keyboard shortcuts
            instruction_text = (
                "Navigation: 'Q' next spectrum, Left/Right arrows previous/next | "
                "Classification: 'A' QSO(AGN), 'S' STAR, 'G' GALAXY, 'U' UNKNOWN, 'L' LIKELY | "
                "Tools: 'Space' wavelength info, 'M' mouse position, 'R' reset zoom | "
                "Advanced: Ctrl+R reload, Ctrl+Left first, Ctrl+Right last, Ctrl+B resume from history"
            )
            toplabel = layout.addLabel(instruction_text, row=1, col=0, colspan=2)
            toplabel.setFont(QFont("Arial", 13))
            toplabel.setMinimumHeight(60)
            toplabel.setMaximumHeight(80)
            toplabel.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            toplabel.setStyleSheet("background-color: white;color: black;")
            toplabel.setFrameStyle(QFrame.Panel | QFrame.Raised)
            toplabel.setWordWrap(True)
            
            # Main content area with splitter
            main_splitter = QSplitter(Qt.Horizontal)
            
            # Left side: spectrum plot and controls
            left_widget = QWidget()
            left_layout = QVBoxLayout()
            
            # Add spectrum info label above plot
            left_layout.addWidget(self.plot.spectrum_info_label)
            left_layout.addWidget(self.plot)
            
            # Redshift slider
            slider_container = QWidget()
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(QLabel("Redshift:"))
            slider_layout.addWidget(self.plot.slider)
            slider_layout.addWidget(self.plot.redshiftSpin)
            slider_container.setLayout(slider_layout)
            left_layout.addWidget(slider_container)
            
            left_widget.setLayout(left_layout)
            main_splitter.addWidget(left_widget)
            
            # Right side: image cutouts (initially visible)
            main_splitter.addWidget(self.cutout_widget)
            
            # Set initial splitter sizes (80% spectrum, 20% cutouts)
            main_splitter.setSizes([800, 200])
            main_splitter.setCollapsible(1, True)  # Make cutout panel collapsible
            
            layout.addWidget(main_splitter, row=2, col=0, colspan=2)
            
            self.main_splitter = main_splitter  # Store reference for toggle
            self._update_go_to_controls()
            
        self.layout = layout
        self.layout.show()

    def keyPressEvent(self, event):
        """Forward keyboard events to plot widget."""
        self.plot.keyPressEvent(event)
        self.sync_qa_checkbox_from_current_spec()
        self._update_go_to_controls()

    def _current_index_one_based(self):
        if hasattr(self.plot, "_displaying_spectrum_number"):
            return int(self.plot._displaying_spectrum_number)
        counter = int(getattr(self.plot, "counter", 1))
        if counter < 1:
            return 1
        if counter > self.len_list:
            return self.len_list
        return counter

    def _update_go_to_controls(self):
        if not hasattr(self, "goto_index_spin"):
            return
        current_idx = self._current_index_one_based()
        self.goto_index_spin.blockSignals(True)
        self.goto_index_spin.setValue(current_idx)
        self.goto_index_spin.blockSignals(False)

    def go_to_index(self):
        if not hasattr(self, "goto_index_spin"):
            return
        index_one_based = int(self.goto_index_spin.value())
        self.plot.jump_to_spectrum(index_one_based)
        self.sync_qa_checkbox_from_current_spec()
        self._update_go_to_controls()

    def mousePressEvent(self, event):
        """Forward mouse events to plot widget."""
        self.plot.mousePressEvent(event)

    def save_data(self):
        """Save current data to CSV."""
        self.save_dict_todf()
        QMessageBox.information(self.layout, "Saved", f"Data saved to {self.output_file}")

    def save_png(self):
        """Save the entire application window as a PNG image."""
        def _sanitize_component(value):
            s = str(value).strip()
            s = s.replace(os.sep, "_").replace(" ", "_")
            s = re.sub(r"[^0-9A-Za-z_.-]+", "_", s)
            return s.strip("_") or "unknown"

        def _infer_survey(spec):
            dr = str(getattr(spec, "data_release", "") or getattr(spec, "_dr", "") or "")
            dr_l = dr.lower()
            if "desi" in dr_l:
                return "desi"
            if "sdss" in dr_l or "boss" in dr_l or "eboss" in dr_l:
                return "sdss"
            if "lamost" in dr_l:
                return "lamost"
            if "gaia" in dr_l:
                return "gaia"
            if dr_l:
                # keep it short and filesystem-friendly
                return _sanitize_component(dr_l)[:24]
            return "sparcl"

        spec = self.plot.spec if hasattr(self.plot, "spec") else None
        objid = getattr(spec, "objid", "spectrum") if spec is not None else "spectrum"
        objid_str = _sanitize_component(objid)

        if spec is not None and getattr(spec, "telescope", "").lower() == "euclid":
            base_name = f"euclid_{objid_str}_vi.png"
        elif "SpecSparcl" in globals() and spec is not None and isinstance(spec, SpecSparcl):
            survey = _infer_survey(spec)
            targetid = getattr(spec, "targetid", None)
            if targetid not in (None, "", 0):
                base_name = f"{survey}_{_sanitize_component(targetid)}_vi.png"
            else:
                base_name = f"{survey}_{objid_str}_vi.png"
        else:
            base_name = f"{objid_str}_vi.png"

        out_dir = Path.cwd() / "saved_pngs"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self.layout, "Save PNG failed", f"Could not create {out_dir}: {e}")
            return

        filename = out_dir / base_name
        if filename.exists():
            i = 2
            while True:
                stem = filename.stem
                candidate = out_dir / f"{stem}_{i}.png"
                if not candidate.exists():
                    filename = candidate
                    break
                i += 1

        try:
            # Ensure latest visuals are painted before grabbing.
            self.layout.repaint()
            QApplication.processEvents()
            pixmap = self.layout.grab()
            ok = pixmap.save(str(filename), "PNG")
        except Exception as e:
            QMessageBox.warning(self.layout, "Save PNG failed", str(e))
            return

        if not ok:
            QMessageBox.warning(self.layout, "Save PNG failed", "Qt failed to write the PNG file.")
            return

        QMessageBox.information(self.layout, "Saved", f"Saved PNG to {filename}")

    def save_and_quit(self):
        """Save data and quit application."""
        self.save_dict_todf()
        QMessageBox.information(self.layout, "Saved", f"Data saved to {self.output_file}")
        self.quit()
    
    def toggle_image_panel(self):
        """Toggle visibility of the image cutout panel."""
        if self.image_toggle_btn.isChecked():
            # Hide images
            self.main_splitter.setSizes([1000, 0])
            self.image_toggle_btn.setText("Show Images")
            # Also disable auto-fetch
            self.cutout_widget.auto_fetch_cb.setChecked(False)
        else:
            # Show images  
            self.main_splitter.setSizes([800, 200])
            self.image_toggle_btn.setText("Hide Images")
            # Re-enable auto-fetch
            self.cutout_widget.auto_fetch_cb.setChecked(True)

    def run_cross_correlation(self):
        """Run cross-correlation analysis - placeholder."""
        QMessageBox.information(self, "Cross-Correlation", 
                              "Cross-correlation feature is coming soon!\n\n"
                              "This will perform automatic redshift measurement\n"
                              "using template cross-correlation.")

    def save_dict_todf(self):
        """Save classification results to CSV."""
        if not self.plot.history:
            return

        rows = []
        for objid_key, v in self.plot.history.items():
            if len(v) < 5:
                continue
            targetid = v[5] if len(v) > 5 else None
            data_release = v[6] if len(v) > 6 else None
            qa_flag = self._normalize_qa_flag(v[7]) if len(v) > 7 else 0
            rows.append(
                {
                    "objid": objid_key,
                    "objname": v[0],
                    "ra": v[1],
                    "dec": v[2],
                    "class_vi": v[3],
                    "z_vi": v[4],
                    "targetid": targetid,
                    "data_release": data_release,
                    "qa_flag": qa_flag,
                }
            )
        df_new = pd.DataFrame(rows)
        df_new.to_csv(self.output_file, index=False)
        print(f"Results saved to {self.output_file}")


class PGSpecPlotThreadEnhanced(QThread):
    """Enhanced thread wrapper for the enhanced application."""

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
        self.app = None
        self._skip_window = False
        self._disable_background_prefetch = False
        self.buffer_dir = self._resolve_buffer_dir(self.spectra)

        if self._should_offer_predownload(self.buffer_dir):
            if self._prompt_for_bulk_download():
                self._skip_window = self._run_bulk_predownload()

        if not self._skip_window:
            if "enable_background_prefetch" not in kwargs:
                kwargs["enable_background_prefetch"] = not self._disable_background_prefetch
            self.app = PGSpecPlotAppEnhanced(
                self.spectra,
                self.SpecClass,
                cutout_buffer_dir=self.buffer_dir,
                **kwargs,
            )

    @staticmethod
    def _resolve_buffer_dir(spectra):
        """Resolve cutout buffer path from input spectra."""
        if isinstance(spectra, str):
            return Path(spectra).parent / "cutout_buffer"
        return Path(spectra[0]).parent / "cutout_buffer"

    @staticmethod
    def _should_offer_predownload(buffer_dir):
        """Offer predownload only when cutout buffer does not exist yet."""
        return buffer_dir is not None and not Path(buffer_dir).exists()

    @staticmethod
    def _prompt_for_bulk_download():
        """Prompt user in CLI to decide whether to pre-download all cutouts."""
        if not sys.stdin or not sys.stdin.isatty():
            print("No interactive terminal detected; continuing with on-the-fly cutout download.")
            return False

        while True:
            answer = input(
                "No 'cutout_buffer' folder found. Download all cutouts before launching the Qt window? [y/N]: "
            ).strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("", "n", "no"):
                return False
            print("Please answer 'y' or 'n'.")

    def _collect_cutout_records(self):
        """Collect objid/ra/dec records for all spectra."""
        records = []
        if isinstance(self.spectra, (list, tuple, np.ndarray)):
            input_list = list(self.spectra)
            for filename in input_list:
                try:
                    spec = self.SpecClass(filename)
                    objid = getattr(spec, "objid", None)
                    ra = getattr(spec, "ra", None)
                    dec = getattr(spec, "dec", None)
                    records.append({"objid": objid, "ra": ra, "dec": dec})
                except Exception as exc:
                    print(f"Failed to load spectrum '{filename}' for predownload: {exc}")
            return records

        try:
            if hasattr(self.SpecClass, "count_in_file"):
                total = int(self.SpecClass.count_in_file(self.spectra))
            else:
                with fits.open(self.spectra) as hdul:
                    total = len(hdul) - 1
        except Exception as exc:
            print(f"Failed to determine spectrum count for predownload: {exc}")
            return records

        for ext in range(1, total + 1):
            try:
                spec = self.SpecClass(self.spectra, ext=ext)
                objid = getattr(spec, "objid", None)
                ra = getattr(spec, "ra", None)
                dec = getattr(spec, "dec", None)
                records.append({"objid": objid, "ra": ra, "dec": dec})
            except Exception as exc:
                print(f"Failed to load spectrum ext={ext} for predownload: {exc}")
        return records

    def _run_bulk_predownload(self):
        """Run one-time bulk predownload and ask user to restart."""
        print(f"Preparing bulk cutout download into '{self.buffer_dir}' ...")
        records = self._collect_cutout_records()
        if not records:
            print("No valid spectra records found for bulk predownload. Continuing with on-the-fly downloads.")
            return False

        summary = predownload_cutouts(
            records=records,
            buffer_dir=self.buffer_dir,
            surveys=EUCLID_CUTOUT_SURVEYS,
            size_arcsec=10,
            progress_callback=print_cli_progress,
        )
        attempted = max(summary["total"] - summary["skipped"], 0)
        failed_total = summary["failed"] + summary["no_data"]
        fail_rate = failed_total / attempted if attempted > 0 else 0.0
        print(
            "Bulk cutout download finished. "
            f"downloaded={summary['downloaded']}, skipped={summary['skipped']}, "
            f"no_data={summary['no_data']}, failed={summary['failed']}."
        )
        if fail_rate >= 0.5:
            print(
                "Bulk predownload failure rate is high; continuing now with on-the-fly downloads "
                "and disabling background prefetch for this run."
            )
            self._disable_background_prefetch = True
            return False
        print("Please run the program again to launch the Qt window with the prebuilt cutout buffer.")
        return True

    def run(self):
        if self._skip_window:
            return
        exit_code = self.app.exec_()
        sys.exit(exit_code)
