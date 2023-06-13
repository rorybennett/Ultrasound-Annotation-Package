# Scan.py

"""Scan class with variables and methods for working with a single scan."""
import subprocess

import cv2
from screeninfo import get_monitors

import ScanUtil as su
from classes import FrameCanvas

# Scan types.
TYPE_TRANSVERSE = 'TRANSVERSE'
TYPE_SAGITTAL = 'SAGITTAL'
# Navigation commands.
NAVIGATION = {
    'w': 'UP',
    's': 'DOWN',
}


class Scan:
    def __init__(self, path: str):
        """
        Initialise a Scan object using the given path string.

        :param path: Path to Scan directory as a String.
        """
        # Path to Recording directory.
        self.path = path
        # Recording frames, stored in working memory.
        self.frames = su.loadFrames(self.path)
        self.frameCount = len(self.frames)
        # Current frame being displayed
        self.currentFrame = 1
        # IMU data.txt file information.
        self.accelerations, self.quaternions, self.depths, self.duration = su.getIMUDataFromFile(self.path)
        # Type of scan.
        self.scanType = su.getScanType(self.path)
        # Display dimensions.
        self.displayDimensions = self.getDisplayDimensions()

    def drawFrameOnAxis(self, canvas: FrameCanvas):
        """
        Draw the current frame on the provided canvas with all supplementary available data.

        :param canvas: Canvas to drawn frame on.
        """
        frame = self.frames[self.currentFrame - 1].copy()
        dd = self.displayDimensions
        # Resize frame to fit display dimensions.
        frame = cv2.resize(frame, dd, cv2.INTER_CUBIC)
        # Corner markers.
        frame[-1][-1], frame[-1][0], frame[0][-1], frame[0][0] = 255, 255, 255, 255
        # Prepare axis and draw frame.
        su.drawFrameOnAxis(canvas.axes, frame)
        # Draw scan details on axis.
        su.drawScanDataOnAxis()
        # todo Get imu offset, and imu position from file.
        canvas.draw()

    def navigate(self, navCommand):
        """
        Navigate through the frames according to the navCommand parameter.

        :param navCommand: Navigation command (NAVIGATION).
        """
        # Navigate.
        if navCommand == NAVIGATION['w']:
            self.currentFrame += 1
        elif navCommand in NAVIGATION['s']:
            self.currentFrame -= 1

        # If the frame position goes beyond max or min, cycle around.
        if self.currentFrame <= 0:
            self.currentFrame = self.frameCount + self.currentFrame
        elif self.currentFrame > self.frameCount:
            self.currentFrame = self.currentFrame - self.frameCount

    def getDisplayDimensions(self):
        """
        Return the dimensions of the canvas used to display the frame. The size of the canvas is a percentage of the
        screen dimension with an aspect ratio that matches the frame dimensions given. The frames will then
        need to be resized to fit the canvas when being displayed.

        :return: displayDimensions : Size of the canvas that will be used to display the frame.
        """

        screenDimensions = [get_monitors()[0].width, get_monitors()[0].height]

        width = screenDimensions[0] * 0.48  # Width of frame is 48% of total screen width.

        ratio = width / self.frames[0].shape[1]

        height = self.frames[0].shape[0] * ratio  # Maintain aspect ratio of frame.

        displayDimensions = [int(width), int(height)]

        return displayDimensions

    def getScanDetails(self):
        """
        Return information about the scan, including patient number, scan type, and total frames.

        :return: patient, scanType, frameCount.
        """
        patient = self.path.split('/')[-3]
        scanType = self.path.split('/')[-2].lower().capitalize()

        return patient, scanType, self.frameCount

    def openDirectory(self):
        """
        Open Windows Explorer at the Scan directory.
        """
        path = self.path.replace('/', '\\')
        try:
            subprocess.Popen(f'explorer "{path}"')
        except Exception as e:
            print(f'Error opening Windows explorer: {e}')
