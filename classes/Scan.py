# Scan.py

"""Scan class with variables and methods for working with a single scan."""
import subprocess

import cv2
from natsort import natsorted
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
# Add or remove points.
ADD_POINT = '-ADD-POINT-'
REMOVE_POINT = '-REMOVE-POINT-'
# Save data to disk.
SAVE_EDITING_DATA = '-SAVE-EDITING-DATA-'
SAVE_POINT_DATA = '-SAVE-POINT-DATA-'
SAVE_PLANE_DATA = '-SAVE-PLANE-DATA-'
SAVE_IPV_DATA = '-SAVE-IPV-DATA-'
SAVE_ALL = '-SAVE-ALL-'


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
        self.frameNames, self.accelerations, self.quaternions, self.depths, self.duration = su.getIMUDataFromFile(
            self.path)
        # EditingData.txt file information.
        self.editPath, self.imuOffset, self.imuPosition = su.getEditDataFromFile(self.path)
        # Type of scan.
        self.scanType = su.getScanType(self.path)
        # Display dimensions.
        self.displayDimensions = self.getDisplayDimensions()
        # Point data from PointData.txt.
        self.pointPath, self.pointsMm = su.getPointDataFromFile(self.path)

    def drawFrameOnAxis(self, canvas: FrameCanvas, showPoints: bool):
        """
        Draw the current frame on the provided canvas with all supplementary available data.

        :param canvas: Canvas to drawn frame on.
        :param showPoints: Show points on frame?
        """
        axis = canvas.axes
        frame = self.frames[self.currentFrame - 1].copy()
        framePosition = self.currentFrame
        count = self.frameCount
        depths = self.depths[self.currentFrame - 1]
        imuOffset = self.imuOffset
        imuPosition = self.imuPosition
        dd = self.displayDimensions
        # Resize frame to fit display dimensions.
        frame = cv2.resize(frame, dd, cv2.INTER_CUBIC)
        # Corner markers.
        frame[-1][-1], frame[-1][0], frame[0][-1], frame[0][0] = 255, 255, 255, 255
        # Prepare axis and draw frame.
        su.drawFrameOnAxis(axis, frame)
        # Draw scan details on axis.
        su.drawScanDataOnAxis(axis, frame, framePosition, count, depths, imuOffset, imuPosition, dd)
        # Show points on frame.
        if showPoints:
            su.drawPointDataOnAxis(axis, self.getPointsOnFrame(), depths, imuOffset, imuPosition, dd)
        # Finalise canvas with draw.
        canvas.draw()

    def getPointsOnFrame(self, position=None):
        """
        Return a list of points on the frame at 'position'. If index is None, use the current frame.

        :param position: Index of frame.
        :return: List of points on frame.
        """
        if position:
            points = [[p[1], p[2]] for p in self.pointsMm if p[0] == self.frameNames[position]]
        else:
            points = [[p[1], p[2]] for p in self.pointsMm if p[0] == self.frameNames[self.currentFrame - 1]]

        return points

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

    def addOrRemovePoint(self, pointDisplay: list):
        """
        Add or remove a point to/from self.points. Point data is saved as real world x and y coordinates on the
        frame, taking the top offset and left offset as zero (ignoring imuOffset). If the new point is within
        a radius of an old point, the old point is removed.

        Args:
            pointDisplay: x/width-, and y/height-coordinates returned by the Graph elements' event.

        Returns:
            Nothing, adds points directly to self.pointsMm.
        """
        pointMm = su.displayToMm(pointDisplay, self.depths[self.currentFrame - 1], self.imuOffset,
                                 self.imuPosition, self.displayDimensions)

        pointRemoved = False

        for point in self.pointsMm:
            if self.frameNames[self.currentFrame - 1] == point[0]:
                # If within radius of another point, remove that point.
                if su.pointInRadius(point[1:], pointMm, 2):
                    self.pointsMm.remove(point)
                    pointRemoved = True
                    break
        # If no point was removed, add the new point.
        if not pointRemoved:
            self.pointsMm.append([self.frameNames[self.currentFrame - 1], pointMm[0], pointMm[1]])
        # Save point data to disk.
        self.__saveToDisk(SAVE_POINT_DATA)

    def __saveToDisk(self, saveType: str):
        """
        Save all in memory data to relevant .txt files. This should be called whenever a value is changed. All previous
        values are overwritten and the current values stored.

        Args:
            saveType: Which data to save to disk.
        """
        try:
            if saveType in [SAVE_EDITING_DATA, SAVE_ALL]:
                with open(self.editPath, 'w') as editingFile:
                    editingFile.write(f'imuOffset:{self.imuOffset}\n')
                    editingFile.write(f'imuPosition:{self.imuPosition}\n')

            if saveType in [SAVE_POINT_DATA, SAVE_ALL]:
                with open(self.pointPath, 'w') as pointFile:
                    self.pointsMm = natsorted(self.pointsMm, key=lambda l: l[0])
                    for point in self.pointsMm:
                        pointFile.write(f'{point[0]},{point[1]},{point[2]}\n')

            # if saveType in [SAVE_PLANE_DATA, SAVE_ALL]:
            #     with open(self.plane_path, 'w') as plane_file:
            #         json.dump(self.plane_mm, plane_file, indent=4)
            #
            # if saveType in [SAVE_IPV_DATA, SAVE_ALL]:
            #     with open(self.ipv_path, 'w') as ipv_file:
            #         json.dump(self.ipv_data, ipv_file, indent=4)

        except Exception as e:
            print(f'\tError saving details to file: {e}')

    def clearFramePoints(self):
        """
        Clear points on the currently displayed frame, then save to disk.
        """
        self.pointsMm = [p for p in self.pointsMm if not p[0] == self.frameNames[self.currentFrame - 1]]

        self.__saveToDisk(SAVE_POINT_DATA)
