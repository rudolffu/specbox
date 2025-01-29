from PySide6.QtGui import QCursor, QFont 
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import QApplication, QFrame, QWidget
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

my_dict = {}
tb_temp = Table.read(data_path + 'qso_temp_vandenberk2001.mrt.txt', format='ascii')

class PGSpecPlotFeLo(pg.PlotWidget):
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
        self.message = ''
        self.counter = 0
        self.plot_next()


    def plot_single(self):
        spec = self.spec
        z_pipe = spec.redshift
        objname = spec.objname
        self.plot(spec.wave.value, spec.flux.value, pen='b', 
                      symbol='o', symbolSize=4, symbolPen=None, connect='finite',
                      symbolBrush='k', antialias=True)
        try:
            idx_poor = np.where(spec.flux.value/spec.err < 2)
            line_poor = self.plot(spec.wave.value[idx_poor], spec.flux.value[idx_poor], pen='r', 
                                  symbol='x', symbolSize=4, symbolPen=None, connect='finite',
                                  symbolBrush=(200,0,0,80), antialias=True, name='SNR lower than 2')
            self.line_poor = line_poor
        except:
            pass
        if z_pipe >= 0.0:
            wave_temp = tb_temp['Wave'].data * (1+z_pipe)
            idx = np.where((wave_temp>=3800) & (wave_temp<=9020))
            flux_temp = tb_temp['FluxD'].data
            wave_temp = wave_temp[idx]
            flux_temp = flux_temp[idx] / np.mean(flux_temp[idx])  * spec.flux.value.mean() * 1.5
            self.plot(wave_temp, flux_temp, pen=(240,128,128), symbol='+', symbolSize=2, symbolPen=(135, 190, 135), connect='finite', symbolBrush=(135, 190, 135), antialias=False)     
        self.legend = self.addLegend(labelTextSize='16pt')  
        self.text = pg.TextItem(text="{0}  {1}  Z_pipe = {2:.2f}".format(
            self.message, objname, z_pipe), anchor=(0,0), color='k', border='w', fill=(255, 255, 255, 200))
        self.text.setPos(spec.wave.value[0] * 1.3, spec.flux.value.max() * 1.3)
        self.text.setFont(QFont("Arial", 14))
        self.addItem(self.text)
        self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
        self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())
        self.autoRange()
        
    def plot_next(self):
        specfile = self.speclist[self.counter]
        self.message = "Spectrum {0}/{1}.".format(self.counter+1, len(self.speclist))
        print(self.message)
        spec = self.SpecClass(specfile)
        spec.trim([3800, 9020])
        spec.smooth(5, 3, inplace=True, plot=False, sigclip=True)
        self.spec = spec
        if hasattr(self, 'line_poor'):
            try:
                self.legend.removeItem(self.line_poor)
            except:
                pass
        self.plot_single()
        self.counter += 1

    def plot_previous(self):
        if self.counter >= 1:
            print("Plotting previous spectrum...")
            self.message = "Spectrum {0}/{1}.".format(self.counter, len(self.speclist))
            print(self.message)
            self.clear()
            specfile = self.speclist[self.counter-1]
            spec = self.SpecClass(specfile)
            spec.trim([3800, 9020])
            spec.smooth(5, 3, inplace=True, plot=False, sigclip=True)
            self.spec = spec
            if hasattr(self, 'line_poor'):
                try:
                    self.legend.removeItem(self.line_poor)
                except:
                    pass
            self.counter -= 1
            self.plot_single()
        elif self.counter == 0:
            print("No previous spectrum to plot.")

    def keyPressEvent(self, event):
        spec = self.spec
        if event.key() == Qt.Key_Q:
            if spec.objid not in my_dict:
                my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'QSO(Default)']
            if self.counter < len(self.speclist):
                self.clear()
                self.plot_next()
            else:
                print("No more spectra to plot.")
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
        if event.key() == Qt.Key_F:
            print("Class: FeLoBAL QSO.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'FeLoBAL']
        if event.key() == Qt.Key_N:
            print("Class: Non-FeLoBAL QSO.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'Non-FeLoBAL']
        if event.key() == Qt.Key_U:
            print("Class: UNKNOWN.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'UNKNOWN']
        if event.key() == Qt.Key_L:
            print("Class: LIKELY FeLoBAL QSO.")
            my_dict[spec.objid] = [spec.objname, spec.ra, spec.dec, 'LIKELY']
        if event.key() == Qt.Key_R:
            # Reset the plot to the original state
            self.clear()
            self.plot_single()   
        # if the user presses the key combination Ctrl+Left, plot the previous spectrum
        if event.modifiers() & Qt.ControlModifier:
            if event.key() == Qt.Key_Left:
                self.plot_previous()
            

class PGSpecPlotAppFeLo(QApplication):
    def __init__(self, speclist, SpecClass=SpecSDSS, 
                 output_file='vi_output.csv'):
        super().__init__(sys.argv)
        self.output_file = output_file
        self.speclist = speclist
        self.SpecClass = SpecClass
        self.plot = PGSpecPlotFeLo(self.speclist, self.SpecClass)
        self.make_layout()
        # self.layout.show()
        self.exec_()
        self.my_dict = my_dict
        self.save_dict_todf()
        self.exit() 
        sys.exit()    
    

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            if self.plot.counter < len(self.speclist)-1:
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
 
    
    def make_layout(self):
        layout = pg.LayoutWidget()
        layout.resize(1200, 800)
        layout.setWindowTitle("PGSpecPlotFeLo - FeLoBAL Spectra Labeler (v1.0)")
        if self.plot.counter < len(self.speclist):
            toplabel = layout.addLabel("Press 'Q' for next spectrum, \t\t 'F' to set class as FeLoBAL QSO, \n\
'N' to set class as Non-FeLoBAL QSO, \t 'U' to set class as UNKNOWN, \n\
'L' to set class as LIKELY FeLoBAL QSO, \t 'M' to get mouse position, \t 'Space' to get spectrum value at current wavelength.\n\
Use mouse scroll to zoom in/out, \t use mouse select to zoom in. \nPress 'R' to reset the plot to the original state. \nPress 'Ctrl+Left' (MacOS: 'Command+Left') to plot the previous spectrum.", row=0, col=0, colspan=2)
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
            if self.plot.counter < len(self.speclist)-1:
                self.plot.keyPressEvent(event)
            else:
                self.plot.close()
                self.layout.close()
                self.exit()
                sys.exit()
        else:
            self.plot.keyPressEvent(event)

class PGSpecPlotThreadFeLo(QThread):
    def __init__(self, speclist, SpecClass=SpecSDSS, **kwargs):
        super().__init__()
        self.speclist = speclist
        self.SpecClass = SpecClass
        self.app = PGSpecPlotAppFeLo(self.speclist, self.SpecClass, **kwargs)
        # Exit the thread when the app is closed
        self.app.aboutToQuit.connect(self.exit)

    def run(self):
        self.app.exec_()
        self.exit()
        sys.exit()
