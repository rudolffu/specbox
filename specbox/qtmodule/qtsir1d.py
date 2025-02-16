from PySide6.QtGui import QCursor, QFont 
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication, QFrame, QWidget, QSlider, QVBoxLayout, QHBoxLayout, QLabel, QSpacerItem, QLayout, QSizePolicy, QDoubleSpinBox
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

my_dict = {}
tb_temp = Table.read(data_path + 'optical_nir_qso_template.fits')
tb_temp.rename_columns(['wavelength', 'flux'], ['Wave', 'Flux'])

class PGSpecPlot(pg.PlotWidget):
    """
    Plot widget for plotting spectra using pyqtgraph.
    """
    def __init__(self, specfile, SpecClass=SpecEuclid1d):
        super().__init__()
        self.specfile = specfile
        with fits.open(specfile) as hdul:
            self.len_list = len(hdul) - 1
        self.SpecClass = SpecClass
        self.setWindowTitle("Spectrum")
        self.resize(1200, 800)
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.setMouseEnabled(x=True, y=True)
        self.setLogMode(x=False, y=False)
        self.setAspectLocked(False)
        # Show auto-range button
        self.enableAutoRange()
        self.vb = self.getViewBox()
        # Enable Mouse selection for zooming
        self.vb.setMouseMode(self.vb.RectMode)
        self.z_min = 0.0                # NEW: minimum redshift
        self.z_max = 5.0                # NEW: maximum redshift (adjust as needed)
        self.base_z_step = 0.001        # NEW: base step for mapping
        self.slider_min = 0
        self.slider_max = int((1/self.base_z_step)*np.log((1+self.z_max)/(1+self.z_min)))  # NEW: compute slider max
        self.message = ''
        self.counter = 0

        # --- Create slider and spin box once ---
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.slider_min)
        self.slider.setMaximum(self.slider_max)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(max(1, int((self.slider_max - self.slider_min)/10)))
        self.slider.valueChanged.connect(self.slider_changed)
        
        self.redshiftSpin = QDoubleSpinBox()
        self.redshiftSpin.setMinimum(self.z_min)
        self.redshiftSpin.setMaximum(self.z_max)
        self.redshiftSpin.setDecimals(4)
        self.redshiftSpin.setSingleStep(0.001)
        self.redshiftSpin.valueChanged.connect(self.spin_changed)
        
        # Apply any desired style
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
        
        self.plot_next()  # load the first spectrum
        
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
        # try:
        #     idx_poor = np.where(spec.flux.value/spec.err < 2)
        #     line_poor = self.plot(spec.wave.value[idx_poor], spec.flux.value[idx_poor], pen='r', 
        #                           symbol='x', symbolSize=4, symbolPen=None, connect='finite',
        #                           symbolBrush=(200,0,0,80), antialias=True, name='SNR lower than 2')
        #     self.line_poor = line_poor
        # except:
        #     pass
        
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
        while self.counter < self.len_list:
            ext = self.counter + 1
            temp_spec = self.SpecClass(specfile, ext=ext)
            if temp_spec.objid in my_dict:
                self.counter += 1
                continue
            else:
                break
        if self.counter >= self.len_list:
            print("No more spectra to plot.")
            return
        self.message = "Spectrum {0}/{1}.".format(self.counter+1, self.len_list)
        print(self.message)
        self.clear()
        ext = self.counter + 1
        spec = self.SpecClass(specfile, ext=ext)
        self.spec = spec
        self.update_slider_and_spin()  # update the controls for the new spectrum
        self.plot_single()
        self.counter += 1

    def plot_previous(self):
        specfile = self.specfile
        if self.counter > 1:
            # Decrement counter and load the previous spectrum
            self.counter -= 1
            ext = self.counter
            spec = self.SpecClass(specfile, ext=ext)
            self.spec = spec
            self.clear()
            # Update the slider and spin box values for the new spectrum
            self.update_slider_and_spin()
            self.plot_single()
        else:
            print("No previous spectrum to plot.")

    def keyPressEvent(self, event):
        spec = self.spec
        if event.key() == Qt.Key_Q:
            if spec.objid not in my_dict:
                my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO(Default)', self.spec.z_vi]
            if self.counter < self.len_list:
                self.clear()
                self.plot_next()
            else:
                print("No more spectra to plot.")
            if self.counter % 50 == 0:
                print("Saving temp file to csv (n={})...".format(self.counter))
                df = pd.DataFrame.from_dict(my_dict, orient='index')
                df.rename(columns={0:'objname', 1:'ra', 2:'dec', 3:'vi_class', 4:'z_vi'}, inplace=True)
                df['objid'] = df.index.values
                df.set_index(np.arange(len(df)), inplace=True)
                df.to_csv('vi_temp_{}.csv'.format(self.counter), 
                          index=False)
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
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'STAR', 0.0]
        if event.key() == Qt.Key_G:
            print("Class: GALAXY.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'GALAXY', self.spec.z_vi]
        if event.key() == Qt.Key_A:
            print("Class: QSO(AGN).")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO', self.spec.z_vi]
        if event.key() == Qt.Key_U:
            print("Class: UNKNOWN.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'UNKNOWN', 0.0]
        if event.key() == Qt.Key_L:
            print("Class: LIKELY/Unusual QSO.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'LIKELY', self.spec.z_vi]
        if event.key() == Qt.Key_R:
            # Reset the plot to the original state
            self.clear()
            self.plot_single()   
        # if the user presses the key combination Ctrl+Left, plot the previous spectrum
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Left:
                self.plot_previous()

class PGSpecPlotApp(QApplication):
    def __init__(self, specfile, SpecClass=SpecEuclid1d, 
                 output_file='vi_output.csv', load_history=False):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.specfile = specfile
        self.SpecClass = SpecClass
        # NEW: Load history if available
        if load_history and os.path.exists(self.output_file):
            print(f"Loading history from {self.output_file} ...")
            df = pd.read_csv(self.output_file)
            for idx, row in df.iterrows():
                my_dict[row['objid']] = [row['objname'], row['ra'], row['dec'], row['vi_class'], row['z_vi']]
        self.plot = PGSpecPlot(self.specfile, self.SpecClass)
        self.len_list = self.plot.len_list
        self.make_layout()
        self.exec_()
        self.my_dict = my_dict
        self.save_dict_todf()
        self.exit() 
        sys.exit()    
    
    def make_layout(self):
        layout = pg.LayoutWidget()
        layout.resize(1200, 800)
        layout.setWindowTitle("PGSpecPlot - Euclid Spectra Viewer (v1.0)")
        if self.plot.counter < self.len_list + 1:
            toplabel = layout.addLabel(
                f"Press 'Q' for next spectrum, \t press no key or 'A' to set class as QSO(AGN),\n"
                f"'S' to set class as STAR, \t\t 'G' to set class as GALAXY,\n"
                f"'U' to set class as UNKNOWN, \t 'L' to set class as LIKELY/Unusual QSO,\n"
                f"'M' to get mouse position, \t\t 'Space' to get spectrum value at current wavelength.\n"
                f"Use mouse scroll to zoom in/out, \t use mouse select to zoom in. \t" 
                f"Press 'R' to reset the plot to the original state.\n"
                f"Press 'Ctrl+Left' (MacOS: 'Command+Left') to plot the previous spectrum.", row=0, col=0, colspan=2)
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
            # toplabel = QLabel("Just a test")
            # layout.addWidget(toplabel, row=0, col=0, colspan=2)
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
        if event.key() == Qt.Key_Q:
            if self.plot.counter < self.len_list:
                self.plot.keyPressEvent(event)
            else:
                self.plot.close()
                self.layout.close()
                self.exit()
                sys.exit()
        else:
            self.plot.keyPressEvent(event)

    def mousePressEvent(self, event):
        self.plot.mousePressEvent(event)
    
    def save_dict_todf(self):
        self.my_dict = my_dict
        df = pd.DataFrame.from_dict(self.my_dict, orient='index')
        df.reset_index(inplace=True)  # preserve original keys in a column named "index"
        df.rename(columns={'index':'objid', 0:'objname', 1:'ra', 2:'dec', 3:'vi_class', 4:'z_vi'}, inplace=True)
        df.to_csv(self.output_file, index=False)


class PGSpecPlotThread(QThread):
    def __init__(self, specfile, SpecClass=SpecEuclid1d, **kwargs):
        super().__init__()
        self.specfile = specfile
        self.SpecClass = SpecClass
        # Run the PGSpecPlotApp in a thread
        self.app = PGSpecPlotApp(self.specfile, self.SpecClass, **kwargs)
        # Exit the thread when the app is closed
        self.app.aboutToQuit.connect(self.exit)

    def run(self):
        self.app.exec_()
        self.exit()
        sys.exit()
