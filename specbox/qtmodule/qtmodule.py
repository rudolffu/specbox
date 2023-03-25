from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator, QRegExpValidator, QPalette, QColor, QBrush, QCursor, QFont, QKeySequence
from PyQt5.QtCore import Qt, QRegExp, QThread, pyqtSignal, pyqtSlot, QCoreApplication, QSettings, QTranslator, QLocale, QLibraryInfo, QEvent, QEventLoop, QTimer, QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QGridLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QSizePolicy, QGroupBox, QRadioButton, QButtonGroup, QPlainTextEdit, QInputDialog, QLineEdit, QTabWidget, QScrollArea, QFrame, QStackedWidget, QStackedLayout, QFormLayout, QLayout, QLayoutItem, QSpacerItem, QSizePolicy, QMainWindow, QListWidget
import sys
from ..basemodule import SpecLAMOST
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
import pandas as pd
# locate the data file in the package
import pkg_resources

data_path = pkg_resources.resource_filename('specbox', 'data/')

def iter_by_step(speclist, step=1):
    for i in range(0, len(speclist), step):
        yield speclist[i]

my_dict = {}
tb_temp = Table.read(data_path + 'qso_temp_vandenberk2001.mrt.txt', format='ascii')

class PGSpecPlot(pg.PlotWidget):
    """
    Plot widget for plotting spectra using pyqtgraph.
    """
    def __init__(self, speclist, SpecClass=SpecLAMOST):
        super().__init__()
        self.speclist = speclist
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
        self.iter = None
        self.message = ''
        self.counter = 0
        self.go_iter()

    def go_iter(self):
        self.iter = iter_by_step(self.speclist)
        self.plot_next()

    def plot_single(self):
        spec = self.spec
        z_pipe = spec.redshift
        objname = spec.objname
        self.plot(spec.wave.value, spec.flux.value, pen='b', 
                      symbol='o', symbolSize=4, symbolPen=None, connect='finite',
                      symbolBrush='k', antialias=True)
        if z_pipe >= 0.0:
            wave_temp = tb_temp['Wave'].data * (1+z_pipe)
            idx = np.where((wave_temp>=3800) & (wave_temp<=9020))
            flux_temp = tb_temp['FluxD'].data
            wave_temp = wave_temp[idx]
            flux_temp = flux_temp[idx] / np.mean(flux_temp[idx])  * spec.flux.value.mean() * 1.5
            self.plot(wave_temp, flux_temp, pen=(240,128,128), symbol='+', symbolSize=2, symbolPen=None, connect='finite', symbolBrush=(240,128,128), antialias=True)        
        # Label the objname and redshift on the top center of the plot
        self.text = pg.TextItem(text="{0}  {1}  Z_pipe = {2:.2f}".format(
            self.message, objname, z_pipe), anchor=(0,0), color='k', border='w', fill=(255, 255, 255, 200))
        self.text.setPos(spec.wave.value[0] * 1.3, spec.flux.value.max() * 1.3)
        self.text.setFont(QFont("Arial", 14))
        self.addItem(self.text)
        self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
        self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())
        self.autoRange()
        
    def plot_next(self):
        try:
            specfile = next(self.iter)
            self.counter += 1
            self.message = "Spectrum {0}/{1}.".format(self.counter, len(self.speclist))
            print(self.message)
            spec = self.SpecClass(specfile)
            spec.trim([3800, 9020])
            spec.smooth(5, 3, inplace=True, plot=False, sigclip=True)
            self.spec = spec
            self.plot_single()
        except StopIteration:
            self.iter = None
            self.close()

    def plot_previous(self):
        if self.iter is not None and self.counter > 1:
            print("Plotting previous spectrum...")
            self.counter -= 1
            self.message = "Spectrum {0}/{1}.".format(self.counter, len(self.speclist))
            print(self.message)
            self.clear()
            specfile = self.speclist[self.counter-1]
            spec = self.SpecClass(specfile)
            spec.trim([3800, 9020])
            spec.smooth(5, 3, inplace=True, plot=False, sigclip=True)
            self.spec = spec
            self.plot_single()
        elif self.counter == 1:
            print("No previous spectrum to plot.")

    def keyPressEvent(self, event):
        spec = self.spec
        if event.key() == Qt.Key_Q:
            if spec.objid not in my_dict:
                my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO(Default)']
            if self.iter is not None and self.counter < len(self.speclist):
                self.clear()
                self.plot_next()
            if self.counter % 50 == 0:
                print("Saving temp file to csv (n={})...".format(self.counter))
                df = pd.DataFrame.from_dict(my_dict, orient='index')
                df.rename(columns={0:'objname', 1:'ra', 2:'dec', 3:'vi_class'}, inplace=True)
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
            idx = np.abs(self.spec.wave.value - wave).argmin()
            wave = self.spec.wave.value[idx]
            flux = self.spec.flux.value[idx]
            #Add text to the plot at the mouse position
            self.text = pg.TextItem(text="Wavelength: {0:.2f} Flux: {1:.2f}".format(wave, flux), anchor=(0,0), color='r', border='w', fill=(255, 255, 255, 200))
            self.text.setFont(QFont("Arial", 18, QFont.Bold))
            self.text.setPos(wave, flux)
            self.addItem(self.text)
            print("Wavelength: {0:.2f} Flux: {1:.2f}".format(wave, flux))
        if event.key() == Qt.Key_S:
            print("Class: STAR.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'STAR']
        if event.key() == Qt.Key_G:
            print("Class: GALAXY.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'GALAXY']
        if event.key() == Qt.Key_A:
            print("Class: QSO(AGN).")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO']
        if event.key() == Qt.Key_U:
            print("Class: UNKNOWN.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'UNKNOWN']
        if event.key() == Qt.Key_L:
            print("Class: LIKELY/Unusual QSO.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'LIKELY']
        if event.key() == Qt.Key_R:
            # Reset the plot to the original state
            self.clear()
            self.plot_single()   
        # if the user presses the key combination Ctrl+Left, plot the previous spectrum
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Left:
                self.plot_previous()
            

class PGSpecPlotApp(QApplication):
    def __init__(self, speclist, SpecClass=SpecLAMOST, 
                 output_file='vi_output.csv'):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.speclist = speclist
        self.SpecClass = SpecClass
        self.plot = PGSpecPlot(self.speclist, self.SpecClass)
        self.make_layout()
        # self.layout.show()
        self.exec_()
        self.my_dict = my_dict
        self.save_dict_todf()
        self.exit() 
        sys.exit()    
    
    def make_layout(self):
        layout = pg.LayoutWidget()
        layout.resize(1200, 800)
        layout.setWindowTitle("PGSpecPlot - LAMOST Spectra Viewer (v1.0)")
        if self.plot.iter is not None and self.plot.counter < len(self.speclist):
            toplabel = layout.addLabel("Press 'Q' for next spectrum, \t press no key or 'A' to set class as QSO(AGN), \n\
'S' to set class as STAR, \t\t 'G' to set class as GALAXY, \n'U' to set class as UNKNOWN, \t 'L' to set class as LIKELY/Unusual QSO, \n\
'M' to get mouse position, \t\t 'Space' to get spectrum value at current wavelength.\n\
Use mouse scroll to zoom in/out, \t use mouse select to zoom in. \t Press 'R' to reset the plot to the original state. \nPress 'Ctrl+Left' (MacOS: 'Command+Left') to plot the previous spectrum.", row=0, col=0, colspan=2)
            toplabel.setFont(QFont("Arial", 16))
            toplabel.setFixedHeight(140)
            toplabel.setAlignment(Qt.AlignLeft)
            toplabel.setStyleSheet("background-color: white")
            toplabel.setFrameStyle(QFrame.Panel | QFrame.Raised)
            toplabel.setLineWidth(2)
            toplabel.setMidLineWidth(2)
            toplabel.setFrameShadow(QFrame.Sunken)
            toplabel.setMargin(5)
            toplabel.setIndent(5)
            toplabel.setWordWrap(True)
            layout.addWidget(self.plot, row=1, col=0, colspan=2) 
            self.layout = layout
            self.layout.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            if self.plot.iter is not None and self.plot.counter < len(self.speclist):
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
        df.rename(columns={0:'objname', 1:'ra', 2:'dec', 3:'vi_class'}, inplace=True)
        df['objid'] = df.index.values
        df.set_index(np.arange(len(df)), inplace=True)
        df.to_csv(self.output_file, index=False)


class PGSpecPlotThread(QThread):
    def __init__(self, speclist, SpecClass=SpecLAMOST, **kwargs):
        super().__init__()
        self.speclist = speclist
        self.SpecClass = SpecClass
        # Run the PGSpecPlotApp in a thread
        self.app = PGSpecPlotApp(self.speclist, self.SpecClass, **kwargs)
        # Exit the thread when the app is closed
        self.app.aboutToQuit.connect(self.exit)

    def run(self):
        self.app.exec_()
        self.exit()
        sys.exit()
        

class MatSpecPlot(FigureCanvas):
    """
    Plot widget for plotting spectra using matplotlib.
    """
    def __init__(self, spec):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(spec.wave.value, spec.flux.value, 'r.')
        self.ax.set_xlabel('Wavelength (A)')
        self.ax.set_ylabel('Flux (erg/s/cm^2/A)')
        self.ax.set_title('Spectrum')
        super().__init__(self.fig)
        self.show()

    def closeEvent(self, event):
        self.close()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            self.close()
        else:
            super().keyPressEvent(event)

    def close(self):
        self.close()
        self.quit()


