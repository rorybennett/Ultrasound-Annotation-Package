import matplotlib
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

from classes import Scan
from classes import ScanUtil as su, Utils

matplotlib.use('Qt5Agg')


class FrameCanvas(FigureCanvasQTAgg):

    def __init__(self, updateDisplay, showPointsBox, showIPVBox, showMaskBox):
        """Canvas for drawing frame and related point data."""
        # Related Scan object.
        self.linkedScan: Scan = None
        # Method to update display from calling class.
        self.updateDisplay = updateDisplay
        # Show Points Box and Show IPV Box on MainWindow.
        self.showPointsBox = showPointsBox
        self.showIPVBox = showIPVBox
        # Show mask Box on MainWindow.
        self.showMaskBox = showMaskBox

        # Figure to draw frames on.
        self.fig = Figure(dpi=100)
        self.fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        self.axis = self.fig.add_subplot(111)
        self.axis.patch.set_facecolor((0.5, 0.5, 0.5, 1))
        # Canvas functionality.
        self.canvas = self.fig.canvas
        self.canvas.mpl_connect('button_press_event', lambda x: self._axisPressEvent(x))
        self.canvas.mpl_connect('motion_notify_event', lambda x: self._axisMotionEvent(x))
        self.canvas.mpl_connect('button_release_event', lambda x: self._axisReleaseEvent(x))
        self.canvas.mpl_connect('scroll_event', lambda x: self._axisScrollEvent(x))

        super(FrameCanvas, self).__init__(self.fig)

    def _axisPressEvent(self, event):
        """Handle left presses on axis 1 and 2 (canvas displaying image)."""
        if self.linkedScan is not None:
            self.fd = self.linkedScan.frames[self.linkedScan.currentFrame - 1].shape
            self.dd = self.linkedScan.displayDimensions

    def _axisMotionEvent(self, event):
        """Handle drag events on axis 1 and 2."""
        if self.linkedScan is not None:
            pass

    def _axisReleaseEvent(self, event):
        """Handle left releases on axis 1 and 2. AddRemove if no drag took place"""
        if self.linkedScan is not None:
            if event.button == 1:
                displayPoint = [event.x, event.y]
                if self.showPointsBox.isChecked():
                    self.linkedScan.addOrRemovePoint(displayPoint)
                    self.updateDisplay()

    def _axisScrollEvent(self, event):
        """Handle scroll events on axis (canvas displaying image)."""
        if self.linkedScan is not None:
            if event.button == 'up':
                self.linkedScan.navigate(Scan.NAVIGATION['w'])
            else:
                self.linkedScan.navigate(Scan.NAVIGATION['s'])
            self.updateDisplay()
            return

    def updateAxis(self):
        """Update axis with frame and points."""
        self.linkedScan.drawFrameOnAxis(self)

        # self.background = self.copy_from_bbox(self.axis.bbox)
        cfi = self.linkedScan.currentFrame - 1
        fd = self.linkedScan.frames[cfi].shape
        dd = self.linkedScan.displayDimensions
        # Draw points on canvas if box ticked.
        if self.showPointsBox.isChecked():
            su.drawPointDataOnAxis(self.axis, self.linkedScan.getPointsOnFrame(), fd, dd)
        # Draw IPV data on canvas if box ticked.
        if self.showIPVBox.isChecked():
            su.drawIPVDataOnAxis(self.axis, self.linkedScan.ipvData, self.linkedScan.frameNames[cfi], fd, dd)
        # Draw mask on canvas if box ticked.
        if self.showMaskBox.isChecked():
            su.drawMaskOnAxis(self.axis, self.linkedScan.getPointsOnFrame(), fd, dd)
        # Draw Bullet data on canvas if box is ticked.
        su.drawBulletDataOnAxis(self.axis, self.linkedScan.frameNames[cfi], self.linkedScan.bulletData, fd, dd)

        self.draw()

    def distributeFramePoints(self, count: int):
        """
        Distribute the points on the current frame evenly along a generated spline.

        Args:
            count: Number of points for distribution.
        """
        # Points on current frame.
        pointsPix = self.linkedScan.getPointsOnFrame()
        if len(pointsPix) == 0:
            return
        # Organise points in a clockwise manner.
        pointsPix = np.asfarray(Utils.organiseClockwise(pointsPix))
        # Add extra point on end to complete spline.
        pointsPix = np.append(pointsPix, [pointsPix[0, :]], axis=0)
        # Polygon, acting as spline.

        poly = Polygon(np.column_stack([pointsPix[:, 0], pointsPix[:, 1]]))
        # Extract points from polygon.
        xs, ys = poly.xy.T
        # Evenly space points along spline line.
        xn, yn = Utils.interpolate(xs, ys, len(xs) if len(xs) > count else count + 1)
        if xn is None:
            return
        # Get all points except the last one, which is a repeat.
        endPointsPix = np.column_stack([xn, yn])[:-1]
        # Clear current points from frame.
        self.linkedScan.clearFramePoints()
        fd = self.linkedScan.frames[self.linkedScan.currentFrame - 1].shape
        # Save points.
        for pointPix in endPointsPix:
            pointDisplay = su.pixelsToDisplay([pointPix[0], fd[0] - pointPix[1]], fd, self.linkedScan.displayDimensions)
            self.linkedScan.addOrRemovePoint(pointDisplay)
