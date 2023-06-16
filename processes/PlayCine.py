# PlayCine.py
"""Play a Cine of the given Scan."""
import multiprocessing
import sys

import cv2
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, QTimer
from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QApplication


class Window(QMainWindow):

    def __init__(self, frames, dimensions, title):
        super().__init__()
        self.frames = frames
        self.dimensions = dimensions
        self.setWindowTitle(f'{title}')

        self.setGeometry(100, 100, self.dimensions[1], self.dimensions[0])

        self.UiComponents()

    def UiComponents(self):
        widget = QWidget(self)

        pg.setConfigOptions(antialias=True)

        win = pg.GraphicsLayoutWidget()

        view = win.addViewBox()

        # Create image item
        self.img = pg.ImageItem(border='w')
        view.addItem(self.img)

        # Set initial view bounds
        view.setRange(QRectF(0, 0, self.dimensions[1], self.dimensions[0]))

        self.i = 0

        def updateFrame():
            # Display the data
            self.img.setImage(self.frames[self.i])

            # creating a qtimer
            QTimer.singleShot(1, updateFrame)

            self.i += 1

            if self.i >= len(self.frames):
                self.i = 0

        # call the update method
        updateFrame()

        # Creating a grid layout
        layout = QVBoxLayout(self)

        # setting this layout to the widget
        widget.setLayout(layout)

        # plot window goes on right side, spanning 3 rows
        layout.addWidget(win)

        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)


class PlayCine:
    def __init__(self, frames, title):
        """
        Initialise a ProcessCine object.
        """
        self.async_process = None
        self.queue = None
        self.pool = None
        self.dimensions = [frames[0].shape[1], frames[0].shape[0]]
        self.frames = frames
        self.title = title

    def startProcess(self):
        """
        Start the process that will display the frames on a loop. Convert all frames to byte representation - for Graph.
        """
        self.pool = multiprocessing.Pool(1)
        self.async_process = self.pool.apply_async(process, args=(self.frames, self.dimensions, self.title))


def process(framesAsBytes, dimensions, title):
    try:
        # create pyqt5 app
        App = QApplication(sys.argv)

        # create the instance of our Window
        window = Window(framesAsBytes, dimensions, title)
        window.show()

        # start the app
        sys.exit(App.exec())
    except Exception as e:
        print(e)
