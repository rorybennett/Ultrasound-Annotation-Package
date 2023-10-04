# Scan.py

"""Scan class with variables and methods for working with a single scan."""
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow
from matplotlib import pyplot as plt
from natsort import natsorted
from pyquaternion import Quaternion

import ScanUtil as su
from classes import FrameCanvas
from classes.ErrorDialog import ErrorDialog

# Scan Types.
TYPE_AUS = 'AUS'  # Abdominal Ultrasound Scan.
TYPE_PUS = 'PUS'  # Perineal Ultrasound Scan.
# Scan Planes.
PLANE_TRANSVERSE = 'Transverse'
PLANE_SAGITTAL = 'Sagittal'
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
    def __init__(self, path: str, startingFrame=1, window: QMainWindow = None):
        """
        Initialise a Scan object using the given path string.

        Args:
            path: Path to Scan directory as a String.
            startingFrame: Starting frame position.
            window: QMainWindow, for sizing of frames.
        """
        # Path to Recording directory.
        self.path = path
        # Recording frames, stored in working memory.
        self.frames = su.loadFrames(self.path)
        # Shape of frames, assumed equal for all frames.
        self.frameShape = self.frames[0].shape
        # Total number of frames.
        self.frameCount = len(self.frames)
        # Current frame being displayed.
        self.currentFrame = startingFrame if startingFrame < self.frameCount else 1
        # IMU data.txt file information.
        self.frameNames, self.accelerations, self.quaternions, self.depths, self.duration = su.getIMUDataFromFile(
            self.path)
        # EditingData.txt file information.
        self.editPath, self.imuOffset, self.imuPosition = su.getEditDataFromFile(self.path)
        # Type of scan and plane.
        _, self.scanType, self.scanPlane, _, _ = self.getScanDetails()
        # Main Window (used for dimension calculations)
        self.window = window
        # Display dimensions.
        self.displayDimensions = self.getDisplayDimensions()
        # Point data from PointData.txt.
        self.pointPath, self.pointsPix = su.getPointDataFromFile(self.path)
        # IPV data from IPV.JSON.
        self.ipvPath, self.ipvData = su.getIPVDataFromFile(self.path)

    def drawFrameOnAxis(self, canvas: FrameCanvas, showPoints=False, showIPV=False):
        """
        Draw the current frame on the provided canvas with all supplementary available data.

        Args:
            canvas: Canvas to drawn frame on.
            showPoints: Show points on frame?
            showIPV: Show IPV data on frame?
        """
        axis = canvas.axes
        cfi = self.currentFrame - 1
        frame = self.frames[cfi].copy()
        framePosition = self.currentFrame
        name = self.frameNames[cfi]
        count = self.frameCount
        depths = self.depths[cfi]
        imuOffset = self.imuOffset
        imuPosition = self.imuPosition
        dd = self.displayDimensions
        fd = frame.shape
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
            su.drawPointDataOnAxis(axis, self.getPointsOnFrame(), fd, dd)
        # Show IPV data.
        if showIPV:
            su.drawIPVDataOnAxis(axis, self.ipvData, name, dd, self.frames[0].shape)
        # Finalise canvas with draw.
        canvas.draw()

    def getPointsOnFrame(self, position=None):
        """
        Return a list of points on the frame at 'position'. If index is None, use the current frame.

        :param position: Index of frame.
        :return: List of points on frame.
        """
        if position:
            points = [[p[1], p[2]] for p in self.pointsPix if p[0] == self.frameNames[position]]
        else:
            points = [[p[1], p[2]] for p in self.pointsPix if p[0] == self.frameNames[self.currentFrame - 1]]

        return points

    def navigate(self, navCommand):
        """
        Navigate through the frames according to the navCommand parameter.

        :param navCommand: Navigation command (NAVIGATION) or index value.
        """
        # Navigate.
        try:
            goToFrame = int(navCommand)
            if self.frameCount >= goToFrame > 0:
                self.currentFrame = goToFrame
            elif goToFrame > self.frameCount:
                self.currentFrame = self.frameCount
            elif goToFrame < 1:
                self.currentFrame = 1
        except ValueError:
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
        Main Window dimension with an aspect ratio that matches the frame dimensions given. The frames will then
        need to be resized to fit the canvas when being displayed.

        :return: displayDimensions : Size of the canvas that will be used to display the frame.
        """
        windowDimensions = [self.window.size().width(), self.window.size().height]

        width = windowDimensions[0] * 0.48  # Width of frame is 48% of total screen width.

        ratio = width / self.frames[0].shape[1]

        height = self.frames[0].shape[0] * ratio  # Maintain aspect ratio of frame.

        displayDimensions = [int(width), int(height)]

        return displayDimensions

    def getScanDetails(self):
        """
        Return information about the scan, including patient number, scan type, scan plane, and total frames.

        :return: patient, scanType, scanPlane, frameCount.
        """
        path = self.path.split('/')
        patient = path[-4]
        scanType = path[-3]
        scanPlane = path[-2].lower().capitalize()
        scanNumber = path[-1]

        return patient, scanType, scanPlane, scanNumber, self.frameCount

    def openDirectory(self):
        """
        Open Windows Explorer at the Scan directory.
        """
        path = self.path.replace('/', '\\')
        try:
            subprocess.Popen(f'explorer "{path}"')
        except Exception as e:
            ErrorDialog(None, f'Error opening Windows explorer', e)

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
        pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape, self.displayDimensions)

        pointRemoved = False

        for point in self.pointsPix:
            if self.frameNames[self.currentFrame - 1] == point[0]:
                # If within radius of another point, remove that point.
                if su.pointInRadius(point[1:], pointPixel, 2):
                    self.pointsPix.remove(point)
                    pointRemoved = True
                    break
        # If no point was removed, add the new point.
        if not pointRemoved:
            self.pointsPix.append([self.frameNames[self.currentFrame - 1], pointPixel[0], pointPixel[1]])
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
                    self.pointsPix = natsorted(self.pointsPix, key=lambda l: l[0])
                    for point in self.pointsPix:
                        pointFile.write(f'{point[0]},{point[1]},{point[2]}\n')

            # if saveType in [SAVE_PLANE_DATA, SAVE_ALL]:
            #     with open(self.plane_path, 'w') as plane_file:
            #         json.dump(self.plane_mm, plane_file, indent=4)
            #
            if saveType in [SAVE_IPV_DATA, SAVE_ALL]:
                with open(self.ipvPath, 'w') as ipvFile:
                    json.dump(self.ipvData, ipvFile, indent=4)

        except Exception as e:
            print(f'\tError saving details to file: {e}')

    def clearFramePoints(self):
        """
        Clear points on the currently displayed frame, then save to disk.
        """
        self.pointsPix = [p for p in self.pointsPix if not p[0] == self.frameNames[self.currentFrame - 1]]

        self.__saveToDisk(SAVE_POINT_DATA)

    def clearScanPoints(self):
        """
        Clear all points in the Scan, then save to disk.
        """
        self.pointsPix = []

        self.__saveToDisk(SAVE_POINT_DATA)

    def loadSaveData(self, saveName: str):
        """
        Load the saved PointData.txt, BulletData.JSON, Editing.txt, and IPV.JSON  files from the directory selected.
        This will overwrite the current files in the recording directory.

        Args:
            saveName (str): Directory containing files to be loaded.

        Return:
            successFlags (bool): True if the load was successful, else False.
        """
        successFlags = [True, True, True, True]
        # try:
        #     shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.plane_path.name), self.plane_path)
        # except Exception as e:
        #     print(f'\tError loading plane data: {e}.')
        #     successFlags[0] = False

        print(f'Loading Data: {saveName}')

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.pointPath.name), self.pointPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading point data', e)
            successFlags[1] = False

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.editPath.name), self.editPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading editing data', e)
            successFlags[2] = False
        #
        # try:
        #     shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.ipv_path.name), self.ipv_path)
        # except Exception as e:
        #     print(f'\tError loading ipv data: {e}.')
        #     successFlags[3] = False

        return successFlags

    def getSaveData(self):
        """
        Return a list of all the sub folders stored in the Save Data directory.

        Returns:
            folders (list): List of sub folders as strings.
        """
        folders = [vd.stem for vd in Path(f'{self.path}/Save Data').iterdir() if vd.is_dir()]

        return folders

    def saveUserData(self, username: str):
        """
        Save the current PointData.txt, BulletData.JSON, Editing.txt, and IPV.JSON files to the Save Data folder
        under the entered userName + time.

        Args:
            username (str): Username entered by user, will have time appended.
        """
        try:
            # Create Save Data directory if not present.
            saveDataPath = su.checkSaveDataDirectory(self.path)

            userPath = Path(saveDataPath, f'{username}_{int(time.time() * 1000)}')
            # Create directory with username and current time in milliseconds.
            userPath.mkdir(parents=True, exist_ok=True)
            # Copy current files to new user directory.
            # shutil.copy(self.plane_path, Path(userPath, self.plane_path.name))
            shutil.copy(self.pointPath, Path(userPath, self.pointPath.name))
            shutil.copy(self.editPath, Path(userPath, self.editPath.name))
            shutil.copy(self.ipvPath, Path(userPath, self.ipvPath.name))
            print(f'\tUser data save to {userPath.name}')

        except Exception as e:
            print(f'\tError saving user data: {e}.')

    def frameAtScanPercent(self, percentage: int):
        """
        Find the frame at a percentage of the scan, using the axisAngles/quaternion of the frames. NB: The result is
        based on indexing, and must be incremented by 1 to match frames stored from 0. The current frame is changed
        to the found index.

        Args:
            percentage (int): Percentage of scan to return as index.

        """
        index_at_percentage = 0
        # Find index.
        try:
            axisAngles = su.quaternionsToAxisAngles(self.quaternions)

            index_start, index_end = su.estimateSlopeStartAndEnd(axisAngles)

            index_from_start = int((index_end - index_start) * (percentage / 100))

            index_at_percentage = index_start + index_from_start
        except Exception as e:
            ErrorDialog(None, f'Error finding axis angle centre', e)

        return index_at_percentage

    def updateIPVCentre(self, pointDisplay: list, addOrRemove: str):
        """
        Add or remove the IPV centre circle. This circle is used to reduce inference time by limiting the total patch
        window to within the circle.

        Args:
            pointDisplay: Centre of the circle.
            addOrRemove: Either add or remove the currently placed circle.
        """
        if addOrRemove == ADD_POINT:
            pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape,
                                            self.displayDimensions)
            self.ipvData['centre'] = [self.frameNames[self.currentFrame - 1], pointPixel[0], pointPixel[1]]
        else:
            self.ipvData['centre'] = ['', 0, 0]
            self.ipvData['radius'] = 0

        self.__saveToDisk(SAVE_IPV_DATA)

    def updateIPVInferredPoints(self, inferredPoints: list, frameName: str):
        """
        Update inferred points of IPV data.

        Args:
            inferredPoints: Either 4 points (transverse) or  2 points (sagittal).
            frameName: Frame name that the points are inferred on.
        """
        points = []
        pass
        if self.scanPlane == PLANE_TRANSVERSE:
            for i in range(0, 7, 2):
                points.append([inferredPoints[i], inferredPoints[i + 1]])
        else:
            for i in range(0, 3, 2):
                points.append([inferredPoints[i], inferredPoints[i + 1]])
        self.ipvData['inferred_points'] = [frameName, points]

        self.__saveToDisk(SAVE_IPV_DATA)

    def updateIPVRadius(self, radius: int):
        """
        Update the IPV radius for the region of interest.

        Args:
            radius: Size of radius (in pixels).
        """
        self.ipvData['radius'] = radius

        self.__saveToDisk(SAVE_IPV_DATA)

    def removeIPVData(self):
        """
        Remove all saved IPV data, including centre frame, radius, and any inferred points.
        """
        self.ipvData = {
            'centre': ['', 0, 0],
            'radius': 100,
            'inferred_points': ['', []]
        }

        self.__saveToDisk(SAVE_IPV_DATA)

    def axisAnglePlot(self):
        """
        Plot the axis angle representation of the quaternions for the Scan.
        """
        axis_angles = self.quaternionsToAxisAngles()

        fig, ax = plt.subplots(1)
        fig.canvas.manager.set_window_title('Axis Angle Plot')

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Probe Axis Angle (degrees)')

        ax.plot(range(1, len(axis_angles) + 1), axis_angles, c='blue')

        ax.set_xlim([0, len(axis_angles) + 1])
        ax.set_ylim([min(axis_angles) - 2, max(axis_angles) + 2])

        plt.show()

    def quaternionsToAxisAngles(self) -> list:
        """
        Convert quaternions to a list of axis angles (in degrees) in the following manner:
            1. Get the initial quaternion, to be used as the reference quaternion.
            2. Calculate the difference between all subsequent quaternions and the initial quaternion using:
                    r = p * conj(q)
               where r is the difference quaternion, p is the initial quaternion, and conj(q) is the conjugate of the
               current quaternion.
            3. Calculate the axis angle of r (the difference quaternion).

        Converting the raw quaternions to their axis angle representation for rotation comparisons is not the correct way
        to do it, the axis angle has to be calculated from the quaternion difference.

        Returns:
            axisAngles (list): List of axis angles (in degrees) relative to the first rotation (taken as 0 degrees).
        """
        initialQ = Quaternion(self.quaternions[0])
        axis_angles = []
        # Get angle differences (as quaternion rotations).
        for row in self.quaternions:
            q = Quaternion(row)
            r = initialQ * q.conjugate

            axis_angles.append(180 / np.pi * 2 * np.arctan2(np.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2), r[0]))

        return axis_angles
