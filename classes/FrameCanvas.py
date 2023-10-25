import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from classes import Scan

matplotlib.use('Qt5Agg')


class FrameCanvas(FigureCanvasQTAgg):

    def __init__(self, updateDisplay, showPointsBox, dragButton):
        """Canvas for drawing frame and related point data."""
        # Related Scan object.
        self.linkedScan: Scan = None
        # Method to update display from calling class.
        self.updateDisplay = updateDisplay
        # Show Points Box on MainWindow.
        self.showPointsBox = showPointsBox
        # Drag all points button on MainWindow.
        self.dragButton = dragButton
        # Enable click and drag of points.
        self.enableDrag = False
        # Check if drag was attempted.
        self.dragAttempted = False

        # Figure to draw frames on.
        fig = Figure(dpi=100)
        fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axes = fig.add_subplot(111)
        self.axes.patch.set_facecolor((0.5, 0.5, 0.5, 1))
        # Canvas functionality.
        self.canvas = fig.canvas
        self.canvas.mpl_connect('button_press_event', lambda x: self._axisPressEvent(x))
        self.canvas.mpl_connect('motion_notify_event', lambda x: self._axisMotionEvent(x))
        self.canvas.mpl_connect('button_release_event', lambda x: self._axisReleaseEvent(x))
        self.canvas.mpl_connect('scroll_event', lambda x: self._axisScrollEvent(x))

        super(FrameCanvas, self).__init__(fig)

    def _axisPressEvent(self, event):
        """Handle left presses on axis 1 and 2 (canvas displaying image)."""

        if self.linkedScan is not None:
            if self.dragButton.isChecked():
                self.enableDrag = True

        # displayPoint = [event.x, event.y]
        # # Left click.
        # if event.button == 1:
        #     if scan == 1 and self.s1 and self.leftBoxes.itemAt(0).widget().isChecked():
        #         self.s1.addOrRemovePoint(displayPoint)
        #         self._updateDisplay(1)
        #         return
        #     elif scan == 2 and self.s2 and self.rightBoxes.itemAt(0).widget().isChecked():
        #         self.s2.addOrRemovePoint(displayPoint)
        #         self._updateDisplay(2)
        #         return

    def _axisMotionEvent(self, event):
        """Handle drag events on axis 1 and 2."""
        print(self.dragButton.isChecked())
        if self.linkedScan is not None:
            if self.enableDrag:
                self.dragAttempted = True
                print("Drag")

    def _axisReleaseEvent(self, event):
        """Handle left releases on axis 1 and 2. AddRemove if no drag took place"""
        if self.linkedScan is not None:
            if not self.dragAttempted:
                if event.button == 1:
                    displayPoint = [event.x, event.y]
                    if self.showPointsBox.isChecked():
                        self.linkedScan.addOrRemovePoint(displayPoint)
                        self.updateDisplay()
        self.enableDrag = False
        self.dragAttempted = False

    def _axisScrollEvent(self, event):
        """Handle scroll events on axis (canvas displaying image)."""
        if self.linkedScan is not None:
            if event.button == 'up':
                self.linkedScan.navigate(Scan.NAVIGATION['w'])
            else:
                self.linkedScan.navigate(Scan.NAVIGATION['s'])
            self.updateDisplay()
            return
