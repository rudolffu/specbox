from PySide6.QtGui import QCursor, QFont 
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication, QFrame, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QSpacerItem, QLayout, QSizePolicy, QDoubleSpinBox, QMessageBox
import sys
from ..basemodule import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import pandas as pd
# locate the data file in the package
import pkg_resources
import os

data_path = pkg_resources.resource_filename('specbox', 'data/')

tb_temp = Table.read(data_path + 'optical_nir_qso_template.fits')
tb_temp.rename_columns(['wavelength', 'flux'], ['Wave', 'Flux'])
viewer_version = '1.2'

class PGSpecPlot(pg.PlotWidget):
    def __init__(self, specfile, SpecClass=SpecEuclid1d, initial_counter=0, z_max=5.0, history_dict=None):
        super().__init__()
        self.specfile = specfile
        with fits.open(specfile) as hdul:
            self.len_list = len(hdul) - 1
        if initial_counter >= self.len_list:
            print("No more spectra to plot.\n\t Plotting the first spectrum.")
            initial_counter = 0
        self.SpecClass = SpecClass
        self.history = history_dict if history_dict is not None else {}
        self.setWindowTitle("Spectrum")
        self.resize(1200, 800)
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=True, y=True)
        self.setLogMode(x=False, y=False)
        self.setAspectLocked(False)
        self.enableAutoRange()
        self.vb = self.getViewBox()
        self.vb.setMouseMode(self.vb.RectMode)
        self.z_min = 0.0                # minimum redshift
        self.z_max = z_max                # maximum redshift
        self.base_z_step = 0.001        # base step for mapping
        self.slider_min = 0
        self.slider_max = int((1/self.base_z_step) * np.log((1+self.z_max)/(1+self.z_min)))
        self.message = ''
        self.counter = initial_counter  # set counter based on history
        
        # Create slider and spin box only once
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.slider_min)
        self.slider.setMaximum(self.slider_max)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(max(1, int((self.slider_max - self.slider_min) / 10)))
        self.slider.valueChanged.connect(self.slider_changed)
        
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
        
        self.plot_next()  # load the first un-inspected spectrum
        
    def update_slider_and_spin(self):
        # Update slider and spin values based on the current spectrum's z_vi.
        initial_z = self.spec.z_vi if self.spec.z_vi > 0 else self.z_min
        initial_slider_value = int((1/self.base_z_step)*np.log((1+initial_z)/(1+self.z_min)))
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
    
    def spin_changed(self, z):
        slider_value = int((1/self.base_z_step) * np.log((1+z)/(1+self.z_min)))
        self.slider.blockSignals(True)
        self.slider.setValue(slider_value)
        self.slider.blockSignals(False)
        self.spec.z_vi = z
        self.clear()
        self.plot_single()

    def plot_single(self):
        spec = self.spec
        # If history contains a saved z_vi for this spectrum, use it.
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
        wave_temp = tb_temp['Wave'].data * (1+z_vi)
        idx = np.where((wave_temp>=12047.4) & (wave_temp<=18734))
        flux_temp = tb_temp['Flux'].data
        wave_temp = wave_temp[idx]
        flux_temp = flux_temp[idx] / np.mean(flux_temp[idx])  * np.abs(flux.mean()) * 1.5
        c1 = self.plot(wave_temp, flux_temp, pen=(240,128,128), symbol='+', symbolSize=2, symbolPen=None)
        self.legend = self.addLegend(labelTextSize='16pt')  
        self.text = pg.TextItem(text="{0}  {1}  z_vi = {2:.4f}, z_gaia = {3:.4f}".format(
            self.message, objname, z_vi, z_gaia), anchor=(0,0), color='k', border='w', fill=(255, 255, 255, 200))
        self.text.setPos(wave[0] * 1.3, flux.max() * 1.3)
        self.text.setFont(QFont("Arial", 14))
        self.addItem(self.text)
        self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
        self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())
        self.autoRange()

    def plot_next(self):
        specfile = self.specfile
        if self.counter >= self.len_list:
            print("No more spectra to plot.")
            return
        self.message = "Spectrum {0}/{1}.".format(self.counter + 1, self.len_list)
        print(self.message)
        self.clear()
        spec = self.SpecClass(specfile, ext=self.counter + 1)
        self.spec = spec
        if spec.objid in self.history:
            spec.z_vi = self.history[spec.objid][4]
        self.update_slider_and_spin()  
        self.plot_single()
        self.counter += 1

    def plot_previous(self):
        specfile = self.specfile
        if self.counter > 1:
            self.message = "Spectrum {0}/{1}.".format(self.counter-1, self.len_list)
            print(self.message)
            self.clear()
            spec = self.SpecClass(specfile, ext=self.counter - 1)
            self.spec = spec
            if spec.objid in self.history:
                spec.z_vi = self.history[spec.objid][4]
            self.counter -= 1
            self.update_slider_and_spin()
            self.plot_single()
        else:
            print("No previous spectrum to plot.")

    def keyPressEvent(self, event):
        spec = self.spec
        if event.key() == Qt.Key_Q:
            # Update history with the current classification (using self.history)
            if spec.objid not in self.history:
                self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO(Default)', spec.z_vi]
            if self.counter < self.len_list:
                self.clear()
                self.plot_next()
            else:
                print("No more spectra to plot.")
            # Every 50 spectra, save a temporary file that includes all history
            if (self.counter-1) % 50 == 0:
                print("Saving temp file to csv (n={})...".format(self.counter))
                temp_filename = f"vi_temp_{self.counter-1}.csv"
                # Create a DataFrame from the current history
                df_new = pd.DataFrame.from_dict(self.history, orient='index')
                df_new.reset_index(inplace=True)
                df_new.rename(columns={'index': 'objid', 0: 'objname', 1: 'ra', 2: 'dec', 3: 'vi_class', 4: 'z_vi'}, inplace=True)
                df_new = df_new[['objid', 'objname', 'ra', 'dec', 'vi_class', 'z_vi']]
                df_new.to_csv(temp_filename, index=False)
        if event.key() == Qt.Key_M:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            mouse_pos = self.vb.mapSceneToView(mouse_pos)
            print(mouse_pos)
        if event.key() == Qt.Key_Space:
            mouse_pos = self.mapFromGlobal(QCursor.pos())
            self.vb = self.getViewBox()
            wave = self.vb.mapSceneToView(mouse_pos).x()
            idx = np.abs(self.wave - wave).argmin()
            wave = self.wave[idx]
            flux = self.flux[idx]
            #Add text to the plot at the mouse position
            self.text = pg.TextItem(
                text="Wavelength: {0:.2f} Flux: {1:.2f} x 1e-17".format(wave, flux*1e17), anchor=(0,0), color='r', border='w', fill=(255, 255, 255, 200))
            self.text.setFont(QFont("Arial", 18, QFont.Bold))
            self.text.setPos(wave, flux)
            self.addItem(self.text)
            print("Wavelength: {0:.2f} Flux: {1:.2f}".format(wave, flux))
        if event.key() == Qt.Key_S:
            print("Class: STAR.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'STAR', 0.0]
        if event.key() == Qt.Key_G:
            print("Class: GALAXY.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'GALAXY', self.spec.z_vi]
        if event.key() == Qt.Key_A:
            print("Class: QSO(AGN).")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO', self.spec.z_vi]
        if event.key() == Qt.Key_U:
            print("Class: UNKNOWN.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'UNKNOWN', 0.0]
        if event.key() == Qt.Key_L:
            print("Class: LIKELY/Unusual QSO.")
            self.history[spec.objid] = [spec.objname, spec.ra, spec.dec, 'LIKELY', self.spec.z_vi]
        if event.key() == Qt.Key_R:
            self.clear()
            self.plot_single()   
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_R:
                # Reset the plot with the initial z_vi
                self.clear()
                self.spec = self.SpecClass(self.specfile, ext=self.counter)
                self.update_slider_and_spin()
                self.plot_single() 
            if event.key() == Qt.Key_Right:
                # Plot the last spectrum
                self.clear()
                self.counter = self.len_list - 1
                self.plot_next()
            if event.key() == Qt.Key_Left:
                # Plot the first spectrum
                self.clear()
                self.counter = 2
                self.plot_previous()  
            if event.key() == Qt.Key_B:
                # Plot the last labeled spectrum
                self.clear()
                self.counter = len(self.history) - 1
                self.plot_next()
        if event.key() == Qt.Key_Left:
            self.plot_previous()
        if event.key() == Qt.Key_Right:
            self.plot_next()

class PGSpecPlotApp(QApplication):
    def __init__(self, specfile, SpecClass=SpecEuclid1d, output_file='vi_output.csv', z_max=5.0, load_history=False):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.specfile = specfile
        self.SpecClass = SpecClass
        if load_history and os.path.exists(self.output_file):
            print(f"Loading history from {self.output_file} ...")
            df = pd.read_csv(self.output_file)
            # Build a history dictionary from the CSV
            history_dict = {}
            for idx, row in df.iterrows():
                history_dict[int(row['objid'])] = [row['objname'], row['ra'], row['dec'], row['vi_class'], row['z_vi']]
            initial_counter = df.shape[0]  # assuming rows are in order
        else:
            history_dict = {}
            initial_counter = 0
        self.plot = PGSpecPlot(
            self.specfile, self.SpecClass, 
            initial_counter=initial_counter, 
            z_max=z_max,
            history_dict=history_dict)
        self.len_list = self.plot.len_list
        self.make_layout()
        self.aboutToQuit.connect(self.save_dict_todf)
    
    def make_layout(self):
        layout = pg.LayoutWidget()
        layout.resize(1200, 800)
        layout.setWindowTitle(f"PGSpecPlot - Euclid Spectra Viewer (v{viewer_version})")
        if self.plot.counter < self.len_list + 1:
            toplabel = layout.addLabel(
                f"Press 'Q' for next spectrum, \t press no key or 'A' to set class as QSO(AGN),\n"
                f"'U' to set class as UNKNOWN,\t 'L' for LIKELY/Unusual QSO,\t 'S' for STAR, and 'G' for GALAXY,\n"
                f"'M' to get mouse position, \t\t 'Space' to get spectrum value at current wavelength.\n"
                f"Use mouse scroll to zoom in/out,\t use mouse select to zoom in.\n" 
                f"Press 'R' to reset the zoom scale.\t"
                f"Press 'Ctrl+R' to reset the plot with the initial z_vi.\n"
                f"Press 'Left' to plot the previous spectrum,\t press 'Right' to plot the next spectrum.\n", 
                row=0, col=0, colspan=2)
            toplabel.setFont(QFont("Arial", 16))
            toplabel.setFixedHeight(140)
            toplabel.setAlignment(Qt.AlignLeft)
            toplabel.setStyleSheet("background-color: white;color: black;")
            toplabel.setFrameStyle(QFrame.Panel | QFrame.Raised)
            toplabel.setLineWidth(2)
            toplabel.setMidLineWidth(2)
            toplabel.setFrameShadow(QFrame.Sunken)
            toplabel.setMargin(5)
            toplabel.setIndent(5)
            toplabel.setWordWrap(True)
            layout.addWidget(self.plot, row=1, col=0, colspan=2) 
            slider_container = QWidget()
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.plot.slider)
            slider_layout.addWidget(self.plot.redshiftSpin)
            slider_container.setLayout(slider_layout)
            layout.addWidget(slider_container, row=2, col=0, colspan=2)
            self.layout = layout
            self.layout.show()

    def keyPressEvent(self, event):
        self.plot.keyPressEvent(event)

    def mousePressEvent(self, event):
        self.plot.mousePressEvent(event)
    
    def save_dict_todf(self):
        # Merge history from PGSpecPlot with any new entries and save
        df_new = pd.DataFrame.from_dict(self.plot.history, orient='index')
        df_new.reset_index(inplace=True)
        df_new.rename(columns={'index': 'objid', 0: 'objname', 1: 'ra', 2: 'dec', 3: 'vi_class', 4: 'z_vi'}, inplace=True)
        df_new = df_new[['objid', 'objname', 'ra', 'dec', 'vi_class', 'z_vi']]
        # Save the merged history
        df_new.to_csv(self.output_file, index=False)

class PGSpecPlotThread(QThread):
    def __init__(self, specfile, SpecClass=SpecEuclid1d, **kwargs):
        super().__init__()
        self.specfile = specfile
        self.SpecClass = SpecClass
        # Run the PGSpecPlotApp in a thread
        self.app = PGSpecPlotApp(self.specfile, self.SpecClass, **kwargs)

    def run(self):
        exit_code = self.app.exec_()
        sys.exit(exit_code)
