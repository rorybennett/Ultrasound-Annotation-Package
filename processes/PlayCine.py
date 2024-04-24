# PlayCine.py
"""Play a Cine of the given Scan."""
import multiprocessing
import sys
import time

import numpy as np
import pyqtgraph as pg
import qdarktheme
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication, QSlider, QLabel

from classes.ErrorDialog import ErrorDialog


class Window(QMainWindow):
    def __init__(self, frames: np.ndarray, dimensions: list, patient: str, scanType: str, scanPlane: str):
        super().__init__()
        # Flip frames to match MainWindow axis display.
        self.frames = []
        for frame in frames:
            self.frames.append(np.flipud(frame))

        self.dimensions = dimensions
        # Create heading above Cine.
        self.title = f'Patient: {patient}    Scan Type: {scanType}    Scan Plane: {scanPlane}'

        self.setWindowTitle('Cine Playback')
        self._createUI()

    def _createUI(self):
        widget = QWidget(self)

        titleLabel = QLabel(self.title)
        titleLabel.setFont(QFont('Arial', 16))
        titleLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        win = pg.GraphicsLayoutWidget()
        view = win.addViewBox()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(100)
        slider.setSliderPosition(50)

        sliderLabel = QLabel('Adjust Cine Speed')
        sliderLabel.setFont(QFont('Arial', 10))
        sliderLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        # Create imageItem view.
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)

        self.i = 0

        def updateFrame():
            # Display the data
            self.img.setImage(self.frames[self.i].T)
            # Creating a qtimer to call this updateFrame again.
            QTimer.singleShot(1, updateFrame)
            # Automatically cycle through all frames.
            self.i += 1
            if self.i >= len(self.frames):
                self.i = 0
            # Delay function based on slider value.
            sleepTime = slider.value()
            time.sleep(1 / sleepTime)

        # Call the update method
        updateFrame()

        # Creating and fill the layout.
        layout = QVBoxLayout(self)

        widget.setLayout(layout)
        layout.addWidget(titleLabel)
        layout.addWidget(win)
        layout.addWidget(slider)
        layout.addWidget(sliderLabel)

        self.setCentralWidget(widget)


class PlayCine:
    def __init__(self, frames: np.ndarray, patient: str, scanType: str, scanPlane: str):
        """
        Initialise a PlayCine object.
        """
        self.async_process = None
        self.queue = None
        self.pool = None
        self.dimensions = [frames[0].shape[1], frames[0].shape[0]]
        self.frames = frames
        self.patient = patient
        self.scanType = scanType
        self.scanPlane = scanPlane

        self.startProcess()

    def startProcess(self):
        """
        Start the process that will display the frames on a loop.
        """
        self.pool = multiprocessing.Pool(1)
        self.async_process = self.pool.apply_async(process, args=(self.frames, self.dimensions,
                                                                  self.patient, self.scanType,
                                                                  self.scanPlane))


def process(frames: np.ndarray, dimensions: list, patient: str, scanType: str, scanPlane: str):
    try:
        App = QApplication(sys.argv)

        qdarktheme.setup_theme()

        window = Window(frames, dimensions, patient, scanType, scanPlane)
        window.show()

        sys.exit(App.exec())
    except Exception as e:
        ErrorDialog(None, f'Error playing cine.',
                    f'{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}.')
