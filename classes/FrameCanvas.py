import math

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from classes import Scan
from classes import ScanUtil as su, Utils

matplotlib.use('Qt5Agg')


class FrameCanvas(FigureCanvasQTAgg):

    def __init__(self, updateDisplay, showProstatePointsCB, showBladderPointsCB, showProstateMaskCB, showBladderMaskCB,
                 showProstateBoxCB, showBladderBoxCB, prostatePointsCB, bladderPointsCB, prostateBoxCB, bladderBoxCB):
        """Canvas for drawing frame and related point data."""
        # Related Scan object.
        self.linkedScan: Scan = None
        # Testing for drag.
        self.downClick = False
        self.startDrag = False
        self.downClickXY = []
        # Method to update display from calling class.
        self.updateDisplay = updateDisplay
        # Show Points/Masks/Boxes on MainWindow.
        self.showProstatePoints = showProstatePointsCB
        self.showBladderPoints = showBladderPointsCB
        self.showProstateMask = showProstateMaskCB
        self.showBladderMask = showBladderMaskCB
        self.showProstateBox = showProstateBoxCB
        self.showBladderBox = showBladderBoxCB
        # Create bounding boxes.
        self.prostateBoundingBox = prostateBoxCB
        self.bladderBoundingBox = bladderBoxCB
        # Drop prostate/bladder points.
        self.prostatePoints = prostatePointsCB
        self.bladderPoints = bladderPointsCB

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
        self.canvas.mpl_connect('figure_leave_event', lambda x: self._axisReleaseEvent(x))

        super(FrameCanvas, self).__init__(self.fig)

    def _axisPressEvent(self, event):
        """Handle left down clicks in preparation for drag."""
        if self.linkedScan is not None and event.button == 1:
            self.downClick = True
            dp = [event.xdata, event.ydata]
            self.downClickXY = dp
            if self.prostateBoundingBox.isChecked():
                self.linkedScan.updateBoxPoints(Scan.PROSTATE, Scan.BOX_START, dp)
            elif self.bladderBoundingBox.isChecked():
                self.linkedScan.updateBoxPoints(Scan.BLADDER, Scan.BOX_START, dp)

    def _axisMotionEvent(self, event):
        """Handle drag if down clicked and motion detected."""
        dp = [event.xdata, event.ydata]
        if self.linkedScan is not None and self.downClick and dp[0] is not None and math.dist(dp, self.downClickXY) > 5:
            self.startDrag = True

            if self.prostateBoundingBox.isChecked():
                self.linkedScan.updateBoxPoints(Scan.PROSTATE, Scan.BOX_DRAW, dp)
            elif self.bladderBoundingBox.isChecked():
                self.linkedScan.updateBoxPoints(Scan.BLADDER, Scan.BOX_DRAW, dp)
            self.updateDisplay()

    def _axisReleaseEvent(self, event):
        """Handle left releases on axis 1 and 2."""
        dp = [event.xdata, event.ydata]
        if self.linkedScan is not None and dp[0] is not None and self.downClick:
            if not self.startDrag:
                if event.button == 1:
                    if self.prostatePoints.isChecked() and dp[0] is not None:
                        self.linkedScan.addOrRemovePoint(dp, Scan.PROSTATE)
                    elif self.bladderPoints.isChecked() and dp[0] is not None:
                        self.linkedScan.addOrRemovePoint(dp, Scan.BLADDER)
            elif self.prostateBoundingBox.isChecked():
                if event.button == 1:
                    self.linkedScan.updateBoxPoints(Scan.PROSTATE, Scan.BOX_END, dp)
            elif self.bladderBoundingBox.isChecked():
                if event.button == 1:
                    self.linkedScan.updateBoxPoints(Scan.BLADDER, Scan.BOX_END, dp)
            self.updateDisplay()

        self.startDrag = False
        self.downClick = False

    def _axisScrollEvent(self, event):
        """Handle scroll events on axis (canvas displaying image)."""
        if self.linkedScan is not None and not self.downClick:
            if event.button == 'up':
                self.linkedScan.navigate(Scan.NAVIGATION['w'])
            else:
                self.linkedScan.navigate(Scan.NAVIGATION['s'])
            self.updateDisplay()

    def updateAxis(self, new):
        """Update axis with frame and points."""
        if not new:
            xLimits = self.axis.get_xlim()
            yLimits = self.axis.get_ylim()
        self.linkedScan.drawFrameOnAxis(self)

        # self.background = self.copy_from_bbox(self.axis.bbox)
        cfi = self.linkedScan.currentFrame - 1
        fd = self.linkedScan.frames[cfi].shape
        dd = self.linkedScan.displayDimensions
        # Draw prostate points on canvas if box ticked.
        if self.showProstatePoints.isChecked():
            su.drawPointDataOnAxis(self.axis, self.linkedScan.getPointsOnFrame(Scan.PROSTATE), fd, dd, 'lime')
        # Draw bladder points on canvas if box ticked.
        if self.showBladderPoints.isChecked():
            su.drawPointDataOnAxis(self.axis, self.linkedScan.getPointsOnFrame(Scan.BLADDER), fd, dd, 'dodgerblue')
        # Draw prostate mask on canvas if box ticked.
        if self.showProstateMask.isChecked():
            su.drawMaskOnAxis(self.axis, self.linkedScan.getPointsOnFrame(Scan.PROSTATE), fd, dd, 'lime')
        # Draw bladder mask on canvas if box ticked.
        if self.showBladderMask.isChecked():
            su.drawMaskOnAxis(self.axis, self.linkedScan.getPointsOnFrame(Scan.BLADDER), fd, dd, 'dodgerblue')
        # Draw prostate box on canvas if box ticked.
        if self.showProstateBox.isChecked():
            su.drawBoxOnAxis(self.axis, self.linkedScan.getBoxPointsOnFrame(Scan.PROSTATE), fd, dd, 'lime')
        # Draw bladder box on canvas if box ticked.
        if self.showBladderBox.isChecked():
            su.drawBoxOnAxis(self.axis, self.linkedScan.getBoxPointsOnFrame(Scan.BLADDER), fd, dd, 'dodgerblue')
        # Draw Bullet data on canvas if box is ticked.
        su.drawBulletDataOnAxis(self.axis, self.linkedScan.frameNames[cfi], self.linkedScan.bulletData, fd, dd)

        if not new:
            self.axis.set_xlim(xLimits)
            self.axis.set_ylim(yLimits)

        self.draw()

    def distributeFramePoints(self, count: int, prostateBladder):
        """
        Distribute the points on the current frame evenly along a generated spline, point order will not be changed.

        Args:
            count: Number of points for distribution.
            prostateBladder: Distribute either prostate or bladder points.
        """
        # Points on current frame.
        pointsPix = self.linkedScan.getPointsOnFrame(prostateBladder)
        if len(pointsPix) == 0 or prostateBladder not in [Scan.PROSTATE, Scan.BLADDER]:
            return

        # Clear current points from frame.
        self.linkedScan.clearFramePoints(prostateBladder)
        fd = self.linkedScan.frames[self.linkedScan.currentFrame - 1].shape

        endPointsPix = Utils.distributePoints(pointsPix, count)
        # Save points.
        for pointPix in endPointsPix:
            pointDisplay = su.pixelsToDisplay([pointPix[0], pointPix[1]], fd, self.linkedScan.displayDimensions)
            self.linkedScan.addOrRemovePoint(pointDisplay, prostateBladder)
