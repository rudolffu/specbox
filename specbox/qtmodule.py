from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator, QRegExpValidator, QPalette, QColor, QBrush, QCursor, QFont, QKeySequence
from PyQt5.QtCore import Qt, QRegExp, QThread, pyqtSignal, pyqtSlot, QCoreApplication, QSettings, QTranslator, QLocale, QLibraryInfo, QEvent, QEventLoop, QTimer, QUrl
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QGridLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QSizePolicy, QGroupBox, QRadioButton, QButtonGroup, QPlainTextEdit, QInputDialog, QLineEdit, QTabWidget, QScrollArea, QFrame, QStackedWidget, QStackedLayout, QFormLayout, QLayout, QLayoutItem, QSpacerItem, QSizePolicy, QMainWindow, QListWidget
import sys
from specbox.base import *
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class PGSpecPlot(pg.PlotWidget):
    """
    Plot widget for plotting spectra using pyqtgraph.
    """
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.plot(spec.wave.value, spec.flux.value, pen='r', 
                  symbol='o', symbolSize=4, symbolPen=None, connect='finite',
                  symbolBrush=(255, 0, 0, 255))
        self.showGrid(x=True, y=True)
        self.setLabel('left', "Flux", units=spec.flux.unit.to_string())
        self.setLabel('bottom', "Wavelength", units=spec.wave.unit.to_string())
        self.setMouseEnabled(x=True, y=True)
        self.setLogMode(x=False, y=False)
        self.setAspectLocked(False)
        self.setRange(xRange=(spec.wave.value.min(), spec.wave.value.max()), yRange=(spec.flux.value.min(), spec.flux.value.max()))
        self.vb = self.getViewBox()
        self.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q:
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
            print("Wavelength: {0:.2f} Flux: {1:.2f}".format(wave, flux))


class PGSpecPlotApp(QApplication):
    """
    Application for plotting spectra using pyqtgraph.
    """
    def __init__(self, spec):
        super().__init__(sys.argv)
        self.w = PGSpecPlot(spec)
        self.w.show()
        self.exec_()

class PGSpecPlotThread(QThread):
    """
    Thread for plotting spectra using pyqtgraph.
    """
    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def run(self):
        self.app = PGSpecPlotApp(self.spec)


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