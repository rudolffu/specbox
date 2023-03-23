from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QGridLayout, QLabel, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox, QProgressBar, QSizePolicy, QGroupBox, QRadioButton, QButtonGroup, QPlainTextEdit, QInputDialog, QLineEdit, QTabWidget, QScrollArea, QFrame, QStackedWidget, QStackedLayout, QFormLayout, QLayout, QLayoutItem, QSpacerItem, QSizePolicy, QMainWindow
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from specbox.base import *

class SpecPlot(FigureCanvas):
    def __init__(self, spec, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.spec = spec
        self.plot()
        super(SpecPlot, self).__init__(self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def plot(self):
        self.axes.plot(self.spec.wave, self.spec.flux)
        self.axes.set_xlabel("Wavelength ({})".format(self.spec.wave.unit))
        self.axes.set_ylabel("Flux ({})".format(self.spec.flux.unit))
        self.draw()

class SpecPlotWindow(QMainWindow):
    def __init__(self, spec):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle("SpecPlot")
        self.show()
        self.specplot = SpecPlot(spec)
        self.setCentralWidget(self.specplot)
        self.specplot.keyPressEvent = self.keypress_action

    def keypress_action(self, event):   
        if event.key() == 'q':
            self.close()
        if event.key() == 'm':
            self.xdata = self.specplot.axes.lines[0].get_xdata()
            self.ydata = self.specplot.axes.lines[0].get_ydata()
            print(self.xdata, self.ydata)
        