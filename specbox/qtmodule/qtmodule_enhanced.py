from PySide6.QtGui import QCursor, QFont, QPixmap
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QThreadPool, QRunnable
from PySide6.QtWidgets import (QApplication, QFrame, QWidget, QSlider, QHBoxLayout, 
                               QVBoxLayout, QGridLayout, QDoubleSpinBox, QPushButton, 
                               QLabel, QButtonGroup, QRadioButton, QGroupBox, 
                               QFileDialog, QMessageBox, QComboBox, QProgressBar,
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
from scipy.signal import savgol_filter
from astroquery.hips2fits import hips2fits
from astropy.coordinates import Longitude, Latitude, Angle
import concurrent.futures
import threading
from specutils import Spectrum

# locate the data files in the package
data_path = Path(files("specbox").joinpath("data"))
fits_file = data_path / "optical_nir_qso_template_v1.fits"
tb_temp = Table.read(str(fits_file))
tb_temp.rename_columns(['wavelength', 'flux'], ['Wave', 'Flux'])
viewer_version = '1.3.0-pre'

# Rest-frame emission lines used for template annotations.
# All values are in Angstrom.
_TEMPLATE_EMISSION_LINES = [
    (r"Ly $\alpha$", 1215.67),
    ("C IV", 1549.48),
    ("C III]", 1908.73),
    ("Mg II", 2798.0),
    ("[O II]", 3728.48),
    (r"H $\beta$", 4861.33),
    ("[O III] 4959", 4958.91),
    ("[O III] 5007", 5006.84),
    ("O I", 8448.7),
    ("[S III]", 9071.1),
    ("[S III]", 9533.2),
    (r"Pa $\delta$", 10052.1),
    ("He I", 10833.2),
    (r"Pa $\gamma$", 10941.1),
    ("O I", 11290.0),
    (r"Pa $\beta$", 12821.6),
]


class ImageCutoutWidget(QWidget):
    """Widget for displaying astronomical image cutouts."""
    
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

        # Title with toggle
        header_layout = QHBoxLayout()
        title = QLabel("Image Cutouts")
        title.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title)
        
        # Toggle for auto-fetch
        self.auto_fetch_cb = QCheckBox("Auto-fetch")
        self.auto_fetch_cb.setChecked(True)
        self.auto_fetch_cb.setToolTip("Automatically fetch images when coordinates change")
        header_layout.addWidget(self.auto_fetch_cb)
        
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
        euclid_surveys = [
            ("CDS/P/Euclid/Q1/color", "Euclid composite")
        ]
        
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
                        result = hips2fits.query(
                            hips=survey,
                            width=150, height=150,
                            ra=Longitude(ra * u.deg),
                            dec=Latitude(dec * u.deg), 
                            fov=Angle(size_arcsec * u.arcsec),
                            projection="CAR",
                            format='jpg'
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
        if not self.buffer_dir:
            return None
        # Create a safe filename from survey name
        safe_survey = survey_name.replace('/', '_').replace(' ', '_')
        return self.buffer_dir / f"{objid}_{safe_survey}.jpg"
    
    def save_cutout_to_cache(self, objid, survey_name, image_data):
        """Save cutout image to cache."""
        if not self.buffer_dir:
            return
        cache_file = self.get_cache_filename(objid, survey_name)
        if cache_file:
            try:
                # Convert numpy array to PIL and save
                if len(image_data.shape) == 3:  # RGB
                    image = Image.fromarray(image_data, mode='RGB')
                else:  # Grayscale
                    image = Image.fromarray(image_data, mode='L')
                image.save(cache_file, 'JPEG')
            except Exception as e:
                print(f"Error saving cache file {cache_file}: {e}")
    
    def load_cutout_from_cache(self, objid, survey_name):
        """Load cutout from cache if exists."""
        if not self.buffer_dir:
            return None
        cache_file = self.get_cache_filename(objid, survey_name)
        if cache_file and cache_file.exists():
            try:
                image = Image.open(cache_file)
                return np.array(image)
            except Exception as e:
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
            surveys = [("CDS/P/Euclid/Q1/color", "Euclid composite")]
            
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
                        if self.load_cutout_from_cache(objid, survey) is not None:
                            continue  # Already cached
                            
                        # Download cutout
                        try:
                            result = hips2fits.query(
                                hips=survey,
                                width=150, height=150,
                                ra=Longitude(ra * u.deg),
                                dec=Latitude(dec * u.deg), 
                                fov=Angle(10 * u.arcsec),  # Default size
                                projection="CAR",
                                format='jpg'
                            )
                            if result is not None:
                                self.save_cutout_to_cache(objid, survey, result)
                                
                        except Exception as e:
                            print(f"Error prefetching cutout for object {objid}: {e}")
                            
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
                 z_max=5.0, history_dict=None):
        super().__init__()
        self.SpecClass = SpecClass
        self.template_manager = TemplateManager()

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
        
        # Follow original code pattern with sigma clipping
        wave_full = spec.wave.value if hasattr(spec.wave, 'value') else spec.wave
        flux_full = spec.flux.value if hasattr(spec.flux, 'value') else spec.flux

        if is_sparcl:
            idx = (wave_full >= 3800) & (wave_full <= 9300)
            wave_full = wave_full[idx]
            flux_full = flux_full[idx]
        
        # Apply sigma clipping like original code
        flux_masked = np.ma.masked_invalid(flux_full)
        flux_sigclip = sigma_clip(flux_masked, sigma=10, maxiters=3)
        wave = wave_full[~flux_sigclip.mask]
        flux = flux_sigclip.data[~flux_sigclip.mask]
        
        # Store cleaned data for template scaling
        self.wave = wave
        self.flux = flux
        
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
        else:
            # Plot as dots connected by lines like original
            self.plot(wave, flux, pen='b', symbol='o', symbolSize=4, 
                     symbolPen=None, connect='finite', symbolBrush='k', antialias=True)

        # Update labels with proper units
        if hasattr(spec, 'flux_unit') and spec.flux_unit is not None:
            flux_unit_str = f'Flux ({spec.flux_unit})'

        wave_unit_str = 'Wavelength (Å)'
        if hasattr(spec, 'wave_unit') and spec.wave_unit is not None:
            wave_unit_str = f'Wavelength ({spec.wave_unit})'

        
        # Plot template integrated like original code
        if getattr(spec, 'telescope', '').lower() == 'euclid':
            template = self.template_manager.get_template(self.template_manager.current_template)
            if template is not None:
                z_vi = spec.z_vi
                wave_temp = template['wave'] * (1 + z_vi)
                # Use original wavelength clipping for Euclid
                idx = np.where((wave_temp >= 12047.4) & (wave_temp <= 18734))
                flux_temp = template['flux']
                wave_temp = wave_temp[idx]
                flux_temp = flux_temp[idx]
                
                # Use your original scaling with the sigma-clipped flux
                flux_temp_scaled = flux_temp / np.mean(flux_temp) * np.abs(flux.mean()) * 1.5
                
                self.plot(wave_temp, flux_temp_scaled, pen=pg.mkPen('r', width=2), antialias=True)
                self._label_template_emission_lines(wmin=float(np.nanmin(wave)), wmax=float(np.nanmax(wave)), z=z_vi)
        else:
            # For non-Euclid data, plot template unclipped
            template = self.template_manager.get_template(self.template_manager.current_template)
            if template is not None:
                z_vi = spec.z_vi
                wave_temp = template['wave'] * (1 + z_vi)
                flux_temp = template['flux']
                
                # For dataframe-backed SPARCL spectra (and similar), only plot the
                # template over the observed wavelength range.
                if is_sparcl:
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

                # Use original scaling with the sigma-clipped flux
                flux_temp_scaled = flux_temp / np.mean(flux_temp) * np.abs(flux.mean()) * 1.5
                
                self.plot(wave_temp, flux_temp_scaled, pen=pg.mkPen('r', width=2), antialias=True)
                self._label_template_emission_lines(wmin=float(np.nanmin(wave)), wmax=float(np.nanmax(wave)), z=z_vi)

        self.setLabel('left', flux_unit_str)
        self.setLabel('bottom', wave_unit_str)

        # Update info label above plot
        self.update_spectrum_info_label()
        
        # Auto-range like original code - ONLY after initial plotting
        self.autoRange()
        
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
        parts.append(_fmt_z("z_vi", z_vi, hide_zero=False) or "z_vi = -")

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
            
        is_sparcl = 'SpecSparcl' in globals() and isinstance(self.spec, SpecSparcl)
        z = self.spec.z_vi
        wave_shifted = template['wave'] * (1 + z)
        flux_template = template['flux']

        if is_sparcl:
            wave_full = self.spec.wave.value if hasattr(self.spec.wave, 'value') else self.spec.wave
            flux_full = self.spec.flux.value if hasattr(self.spec.flux, 'value') else self.spec.flux
            finite = np.isfinite(wave_full) & np.isfinite(flux_full)
            in_range = (wave_full >= 3800) & (wave_full <= 9300)
            finite_in_range = finite & in_range
            if np.any(finite_in_range):
                wmin = float(np.nanmin(wave_full[finite_in_range]))
                wmax = float(np.nanmax(wave_full[finite_in_range]))
            elif np.any(finite):
                wmin = float(np.nanmin(wave_full[finite]))
                wmax = float(np.nanmax(wave_full[finite]))
            else:
                wmin = float(np.nanmin(wave_full))
                wmax = float(np.nanmax(wave_full))
            idx = (wave_shifted >= wmin) & (wave_shifted <= wmax)
            wave_shifted = wave_shifted[idx]
            flux_template = flux_template[idx]
        
        # Scale template like in original code (unclipped on both sides as requested)
        if hasattr(self, 'flux') and len(self.flux) > 0:
            # Use the cleaned flux from plot_single for scaling
            flux_scaled = flux_template / np.mean(flux_template) * np.abs(self.flux.mean()) * 1.5
        else:
            # Fallback if cleaned flux not available
            spec_flux = self.spec.flux.value if hasattr(self.spec.flux, 'value') else self.spec.flux
            flux_scaled = flux_template / np.mean(flux_template) * np.abs(np.nanmean(spec_flux)) * 1.5
        
        # Plot template unclipped (full wavelength range)
        self.plot(wave_shifted, flux_scaled, pen=pg.mkPen('r', width=2), antialias=True)
        if hasattr(self, 'wave') and self.wave is not None and len(self.wave) > 0:
            self._label_template_emission_lines(
                wmin=float(np.nanmin(self.wave)),
                wmax=float(np.nanmax(self.wave)),
                z=z,
            )
        
        # Update info label to reflect new redshift
        self.update_spectrum_info_label()
        
        # Do NOT auto-range during template updates - preserves x-axis range

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

    def change_template(self, template_name):
        """Change current template."""
        if template_name in self.template_manager.get_available_templates():
            self.template_manager.current_template = template_name
            self.clear()
            self.plot_single()

    def keyPressEvent(self, event):
        """Handle keyboard events."""
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
            # Temp save every 50 spectra like original
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
                
        elif event.key() == Qt.Key_S:
            print("\tClass: STAR.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'STAR', 0.0]
        elif event.key() == Qt.Key_G:
            print("\tClass: GALAXY.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'GALAXY', spec.z_vi]
        elif event.key() == Qt.Key_A:
            print("\tClass: QSO(AGN).")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'QSO', spec.z_vi]
        elif event.key() == Qt.Key_U:
            print("\tClass: UNKNOWN.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'UNKNOWN', 0.0]
        elif event.key() == Qt.Key_L:
            print("\tClass: LIKELY/Unusual QSO.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec,
                                        'LIKELY', spec.z_vi]
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
            if hasattr(self, 'wave') and hasattr(self, 'flux'):
                idx = np.abs(self.wave - wave_pos).argmin()
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
            if hasattr(self, 'wave') and hasattr(self, 'flux'):
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

        self.plot = PGSpecPlotEnhanced(
            self.spectra, self.SpecClass,
            initial_counter=initial_counter,
            z_max=z_max,
            history_dict=history_dict)
        self.len_list = self.plot.len_list
        
        # Create buffer directory for cutouts
        if isinstance(spectra, str):  # Single FITS file
            buffer_dir = Path(spectra).parent / "cutout_buffer"
        else:  # List of files
            buffer_dir = Path(spectra[0]).parent / "cutout_buffer"
        
        # Create image cutout widget with buffer directory
        self.cutout_widget = ImageCutoutWidget(buffer_dir=buffer_dir)
        
        # Connect signals - need a wrapper to pass object ID
        self.plot.coordinate_changed.connect(self.on_coordinate_changed)
        
        # Trigger initial image load for first spectrum
        if hasattr(self.plot, 'spec') and hasattr(self.plot.spec, 'ra') and hasattr(self.plot.spec, 'dec'):
            objid = getattr(self.plot.spec, 'objid', None)
            self.cutout_widget.load_online_cutouts(self.plot.spec.ra, self.plot.spec.dec, objid)
            
        # Start background prefetching for next 50 objects
        self.start_background_prefetch()
        
        self.make_layout()
        self.aboutToQuit.connect(self.save_dict_todf)
    
    def on_coordinate_changed(self, ra, dec):
        """Handle coordinate changes and pass object ID to cutout widget."""
        objid = getattr(self.plot.spec, 'objid', None) if hasattr(self.plot, 'spec') else None
        self.cutout_widget.load_online_cutouts(ra, dec, objid)
    
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
            
            # Add spacer
            toolbar_layout.addStretch()
            
            # Save buttons
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
            
        self.layout = layout
        self.layout.show()

    def keyPressEvent(self, event):
        """Forward keyboard events to plot widget."""
        self.plot.keyPressEvent(event)

    def mousePressEvent(self, event):
        """Forward mouse events to plot widget."""
        self.plot.mousePressEvent(event)

    def save_data(self):
        """Save current data to CSV."""
        self.save_dict_todf()
        QMessageBox.information(self.layout, "Saved", f"Data saved to {self.output_file}")

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
            
        df_new = pd.DataFrame.from_dict(self.plot.history, orient='index')
        df_new.reset_index(inplace=True)
        df_new.rename(columns={'index': 'objid', 0: 'objname', 1: 'ra',
                               2: 'dec', 3: 'class_vi', 4: 'z_vi'},
                      inplace=True)
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
        self.app = PGSpecPlotAppEnhanced(self.spectra, self.SpecClass, **kwargs)

    def run(self):
        exit_code = self.app.exec_()
        sys.exit(exit_code)
