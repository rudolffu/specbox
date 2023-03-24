from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator, QRegExpValidator, QPalette, QColor, QBrush, QCursor, QFont, QKeySequence
from PyQt5.QtCore import Qt, QRegExp, QThread, pyqtSignal, pyqtSlot, QCoreApplication, QSettings, QTranslator, QLocale, QLibraryInfo, QEvent, QEventLoop, QTimer, QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QGridLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QSizePolicy, QGroupBox, QRadioButton, QButtonGroup, QPlainTextEdit, QInputDialog, QLineEdit, QTabWidget, QScrollArea, QFrame, QStackedWidget, QStackedLayout, QFormLayout, QLayout, QLayoutItem, QSpacerItem, QSizePolicy, QMainWindow, QListWidget
import sys
from specbox.base import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

def iter_by_step(speclist, step=1):
    for i in range(0, len(speclist), step):
        yield speclist[i]

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
        self.vb = self.getViewBox()
        self.iter = None
        self.go_iter()
        self.show()

    def go_iter(self):
        self.iter = iter_by_step(self.speclist)
        self.plot_processed()
        
    def plot_processed(self):
        try:
            specfile = next(self.iter)
        except StopIteration:
            self.iter = None
            pass
        else:
            spec = self.SpecClass(specfile)
            self.spec = spec
            z_pipe = spec.redshift
            objname = spec.objname
            self.plot(spec.wave.value, spec.flux.value, pen='b', 
                      symbol='o', symbolSize=4, symbolPen=None, connect='finite',
                      symbolBrush='k', antialias=True)
            self.text = pg.TextItem(text="Object: {0} \t Z_pipe: {1:.2f}".format(objname, z_pipe), anchor=(0,0), color='r', border='w', fill=(255, 255, 255, 255))
            # Set the text position to the top left corner of the plot
            self.text.setPos(spec.wave.value[0]*1.4, spec.flux.value.max()*0.8)
            self.text.setFont(QFont("Arial", 20, QFont.Bold))
            self.addItem(self.text)
            self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
            self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
            if self.iter is not None and self.iter is not iter_by_step(self.speclist):
                self.clear()
                self.plot_processed()
            else:
                self.close()
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
            self.text.setPos(wave, flux)
            self.addItem(self.text)
            print("Wavelength: {0:.2f} Flux: {1:.2f}".format(wave, flux))
        if event.key() == Qt.Key_S:
            print("Class: STAR.")


class PGSpecPlotApp(QApplication):
    def __init__(self, speclist, SpecClass=SpecLAMOST):
        super().__init__(sys.argv)
        self.speclist = speclist
        self.SpecClass = SpecClass
        self.plot = PGSpecPlot(self.speclist, self.SpecClass)
        self.plot.show()
        self.exec_()
        self.exit() 
        sys.exit()    

class PGSpecPlotThread(QThread):
    def __init__(self, speclist, SpecClass=SpecLAMOST):
        super().__init__()
        self.speclist = speclist
        self.SpecClass = SpecClass
        # Run the PGSpecPlotApp in a thread
        self.app = PGSpecPlotApp(self.speclist, self.SpecClass)
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