import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

matplotlib.use('Qt5Agg')


class FrameCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(dpi=dpi)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axes = fig.add_subplot(111)
        self.axes.patch.set_facecolor((0.5, 0.5, 0.5, 1))
        super(FrameCanvas, self).__init__(fig)
