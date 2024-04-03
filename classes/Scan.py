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
from pyquaternion import Quaternion

from classes import FrameCanvas, Utils
from classes import ScanUtil as su
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
NAV_TYPE_IMU = '-NAV-IMU-'
NAV_TYPE_TS1 = '-NAV-TS1-'
# Add or remove points.
ADD_POINT = '-ADD-POINT-'
REMOVE_POINT = '-REMOVE-POINT-'
# Save data to disk.
SAVE_EDITING_DATA = '-SAVE-EDITING-DATA-'
SAVE_POINT_DATA = '-SAVE-POINT-DATA-'
SAVE_BULLET_DATA = '-SAVE-PLANE-DATA-'
SAVE_IPV_DATA = '-SAVE-IPV-DATA-'
SAVE_ALL = '-SAVE-ALL-'
# Copy points from previous or next frame.
NEXT = '-NEXT-'
PREVIOUS = '-PREVIOUS-'
# Shrink or expand points around centre of mass.
SHRINK = '-SHRINK-'
EXPAND = '-EXPAND-'


class Scan:

    def __init__(self, window: QMainWindow = None):
        """
        Creates all attributes, mostly set as None values, to be properly instantiated in self.load.

        Args:
            window: QMainWindow, for sizing of frames.
        """
        # Main Window (used for dimension calculations)
        self.window = window
        # Path to Recording directory.
        self.path = None
        # Recording frames, stored in working memory.
        self.frames = None
        # Shape of frames, assumed equal for all frames.
        self.frameShape = None
        # Total number of frames.
        self.frameCount = None
        # Current frame being displayed.
        self.currentFrame = None
        # IMU data.txt file information.
        self.frameNames, self.accelerations, self.quaternions, self.depths, self.duration = None, None, None, None, None
        # EditingData.txt file information.
        self.editPath, self.imuOffset, self.imuPosition = None, None, None
        # Type of scan and plane.
        self.scanType, self.scanPlane = None, None
        # Display dimensions.
        self.displayDimensions = None
        # Point data from PointData.json.
        self.pointPath, self.pointsProstate, self.pointsBladder = None, None, None
        # IPV data from IPV.JSON.
        self.ipvPath, self.ipvData = None, None
        # Bullet data from Bullet.json
        self.bulletPath, self.bulletData = None, None
        # Has a Scan been loaded?
        self.loaded = False

    def load(self, path: str, startingFrame=1):
        """
        Load a Scan object using the given path string. See __init__ for attribute details.

        Args:
            path: Path to Scan directory as a String.
            startingFrame: Starting frame position.
        """
        self.frames = su.loadFrames(path)
        self.path = path
        self.frameShape = self.frames[0].shape
        self.frameCount = len(self.frames)
        self.currentFrame = startingFrame if startingFrame < self.frameCount else 1
        self.frameNames, self.accelerations, self.quaternions, self.depths, self.duration = su.getIMUDataFromFile(
            self.path)
        self.editPath, self.imuOffset, self.imuPosition = su.getEditDataFromFile(self.path)
        _, self.scanType, self.scanPlane, _, _ = self.getScanDetails()
        self.displayDimensions = self.getDisplayDimensions()
        self.pointPath, self.pointsProstate, self.pointsBladder = su.getPointDataFromFile(self.path)
        self.ipvPath, self.ipvData = su.getIPVDataFromFile(self.path)
        self.bulletPath, self.bulletData = su.getBulletDataFromFile(self.path)
        self.loaded = True

    def drawFrameOnAxis(self, canvas: FrameCanvas):
        """
        Draw the current frame on the provided canvas with all Scan details drawn on.

        Args:
            canvas: Canvas to draw frame on.
        """
        axis = canvas.axis
        cfi = self.currentFrame - 1
        frame = self.frames[cfi].copy()
        framePosition = self.currentFrame
        count = self.frameCount
        depths = self.depths[cfi]
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
        su.drawScanDataOnAxis(axis, frame, framePosition, count, depths, imuOffset, imuPosition,
                              self.countFrameProstatePoints() + self.countFrameBladderPoints(),
                              len(self.pointsProstate) + len(self.pointsBladder), dd)

    def getProstatePointsOnFrame(self, position=None):
        """
        Return a list of prostate points on the frame at 'position'. If position is None, use the current frame.

        :param position: Index of frame.
        :return: List of points on frame.
        """
        if position is not None:
            points = [[p[1], p[2]] for p in self.pointsProstate if p[0] == self.frameNames[position]]
        else:
            points = [[p[1], p[2]] for p in self.pointsProstate if p[0] == self.frameNames[self.currentFrame - 1]]

        return points

    def getBladderPointsOnFrame(self, position=None):
        """
        Return a list of bladder points on the frame at 'position'. If position is None, use the current frame.

        :param position: Index of frame.
        :return: List of points on frame.
        """
        if position is not None:
            points = [[p[1], p[2]] for p in self.pointsBladder if p[0] == self.frameNames[position]]
        else:
            points = [[p[1], p[2]] for p in self.pointsBladder if p[0] == self.frameNames[self.currentFrame - 1]]

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
        if self.window:
            windowDimensions = [self.window.centralWidget().size().width(), self.window.centralWidget().size().height]

            width = windowDimensions[0] * 0.48  # Width of frame is 48% of total screen width.

            ratio = width / self.frames[0].shape[1]

            height = self.frames[0].shape[0] * ratio  # Maintain aspect ratio of frame.

            displayDimensions = [int(width), int(height)]

            return displayDimensions
        return None

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

    def addOrRemoveProstatePoint(self, pointDisplay: list):
        """
        Add or remove a point to/from self.pointsProstate. Point data is saved in pixel values. If the new point is
        within a radius of an old point, the old point is removed.

        Args:
            pointDisplay: x/width-, and y/height-coordinates returned by the canvas elements' event.

        Returns:
            Nothing, adds points directly to self.pointsProstate.
        """
        pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape, self.displayDimensions)

        pointRemoved = False

        for point in self.pointsProstate:
            if self.frameNames[self.currentFrame - 1] == point[0]:
                # If within radius of another point, remove that point.
                if su.pointInRadius(point[1:], pointPixel, 5):
                    self.pointsProstate.remove(point)
                    pointRemoved = True
                    break
        # If no point was removed, add the new point.
        if not pointRemoved:
            self.pointsProstate.append([self.frameNames[self.currentFrame - 1], pointPixel[0], pointPixel[1]])
        # Save point data to disk.
        self.__saveToDisk(SAVE_POINT_DATA)

    def addOrRemoveBladderPoint(self, pointDisplay: list):
        """
        Add or remove a point to/from self.pointsBladder. Point data is saved in pixel values. If the new point is
        within a radius of an old point, the old point is removed.

        Args:
            pointDisplay: x/width-, and y/height-coordinates returned by the canvas elements' event.

        Returns:
            Nothing, adds points directly to self.pointsBladder.
        """
        pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape, self.displayDimensions)

        pointRemoved = False

        for point in self.pointsBladder:
            if self.frameNames[self.currentFrame - 1] == point[0]:
                # If within radius of another point, remove that point.
                if su.pointInRadius(point[1:], pointPixel, 5):
                    self.pointsBladder.remove(point)
                    pointRemoved = True
                    break
        # If no point was removed, add the new point.
        if not pointRemoved:
            self.pointsBladder.append([self.frameNames[self.currentFrame - 1], pointPixel[0], pointPixel[1]])
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
                saveData = {'Prostate': self.pointsProstate,
                            'Bladder': self.pointsBladder}
                with open(self.pointPath, 'w') as pointFile:
                    json.dump(saveData, pointFile, indent=4)

            if saveType in [SAVE_BULLET_DATA, SAVE_ALL]:
                with open(self.bulletPath, 'w') as bulletFile:
                    json.dump(self.bulletData, bulletFile, indent=4)

            if saveType in [SAVE_IPV_DATA, SAVE_ALL]:
                with open(self.ipvPath, 'w') as ipvFile:
                    json.dump(self.ipvData, ipvFile, indent=4)

        except Exception as e:
            print(f'\tError saving details to file: {e}')

    def clearFrameProstatePoints(self):
        """
        Clear points on the currently displayed frame, then save to disk.
        """
        self.pointsProstate = [p for p in self.pointsProstate if not p[0] == self.frameNames[self.currentFrame - 1]]

        self.__saveToDisk(SAVE_POINT_DATA)

    def clearFrameBladderPoints(self):
        """
        Clear points on the currently displayed frame, then save to disk.
        """
        self.pointsBladder = [p for p in self.pointsBladder if not p[0] == self.frameNames[self.currentFrame - 1]]

        self.__saveToDisk(SAVE_POINT_DATA)

    def clearScanPoints(self):
        """
        Clear all points in the Scan, then save to disk.
        """
        self.pointsProstate = []
        self.pointsBladder = []

        self.__saveToDisk(SAVE_POINT_DATA)

    def loadSaveData(self, saveName: str):
        """
        Load the saved PointData.JSON, BulletData.JSON, Editing.txt, and IPV.JSON  files from the directory selected.
        This will overwrite the current files in the recording directory.

        Args:
            saveName (str): Directory containing files to be loaded.

        Return:
            successFlags (bool): True if the load was successful, else False.
        """
        print(f'Loading Data: {saveName}')
        successFlags = [True, True, True, True]
        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.bulletPath.name), self.bulletPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading bullet data.', 'Cannot find file.')
            Path(self.bulletPath).unlink(missing_ok=True)
            successFlags[0] = False

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.pointPath.name), self.pointPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading point data', 'Cannot find file.')
            Path(self.pointPath).unlink(missing_ok=True)
            successFlags[1] = False

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.editPath.name), self.editPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading editing data', 'Cannot find file.')
            Path(self.editPath).unlink(missing_ok=True)
            successFlags[2] = False

        return successFlags

    def getSaveData(self):
        """
        Return a list of all the sub folders stored in the Save Data directory.

        Returns:
            folders (list): List of sub folders as strings.
        """
        folders = [vd.stem for vd in Path(f'{self.path}/Save Data').iterdir() if vd.is_dir()]

        return folders

    def deleteUserData(self, prefix):
        """
        Delete save data with the given prefix.

        Args:
            prefix: Save name to be deleted.
        """
        folders = [vd for vd in Path(f'{self.path}/Save Data').iterdir() if
                   vd.is_dir() and vd.stem.split('_')[0] == prefix]

        print(f'\tDeleting {len(folders)} save data folder(s)...')

        for f in folders:
            shutil.rmtree(f, onerror=su.remove_readonly)

    def checkSaveDataDirectory(self):
        """
        Check if the Save Data directory exists. If not, create it.

        Returns:
            saveDataPath: Path to SaveData directory.
        """
        # Create Save Data directory if not present.
        saveDataPath = su.checkSaveDataDirectory(self.path)

    def saveUserData(self, username: str, scan: int):
        """
        Save the current PointData.txt, BulletData.JSON, Editing.txt, and IPV.JSON files to the Save Data folder
        under the entered userName + time.

        Args:
            username (str): Username entered by user, will have time appended.
            scan: Scan number.
        """
        try:
            saveDataPath = f'{self.path}/Save Data'
            userPath = Path(saveDataPath, f'{username}_{int(time.time() * 1000)}')
            # Create directory with username and current time in milliseconds.
            userPath.mkdir(parents=True, exist_ok=True)
            # Copy current files to new user directory.
            shutil.copy(self.bulletPath, Path(userPath, self.bulletPath.name))
            shutil.copy(self.pointPath, Path(userPath, self.pointPath.name))
            shutil.copy(self.editPath, Path(userPath, self.editPath.name))
            shutil.copy(self.ipvPath, Path(userPath, self.ipvPath.name))
            print(f'\tScan {scan + 1} data saved to {userPath.name}')

        except Exception as e:
            print(f'\tError saving user data: {e}.')

    def frameAtTS1Centre(self):
        """
        Get the frame index that has been marked as the centre by Tristan (TS1) if the save data exists.
        Sagittal TS1 markings are spread over 5 frames, with the third frame taken as the centre.


        Returns:
            Index of frame used as TS1 centre.
        """
        folders = [vd.stem for vd in Path(f'{self.path}/Save Data').iterdir() if vd.is_dir()]
        frame = self.currentFrame
        for user in folders:
            if user.split('_')[0] == 'TS1':
                with open(f'{self.path}/Save Data/{user}/PointData.txt', 'r') as file:
                    framesWithPoints = []
                    for row in file.readlines():
                        framesWithPoints.append(int(row.split(',')[0]))
                framesWithPoints = sorted(set(framesWithPoints))
                if self.scanPlane == PLANE_TRANSVERSE:
                    frame = framesWithPoints[0]
                else:
                    frame = framesWithPoints[2]
        return frame

    def frameAtScanPercent(self, percentage: int):
        """
        Find the frame at a percentage of the scan, using the axisAngles/quaternion of the frames. NB: The result is
        based on indexing, and must be incremented by 1 to match frames stored from 0.

        Args:
            percentage (int): Percentage of scan to return as index.

        Returns:
            Index of frame at given percentage.
        """
        indexAtPercentage = 0
        # Find index.
        try:
            axisAngles = su.quaternionsToAxisAngles(self.quaternions)

            indexStart, indexEnd = su.estimateSlopeStartAndEnd(axisAngles)

            indexFromStart = int((indexEnd - indexStart) * (percentage / 100))

            indexAtPercentage = indexStart + indexFromStart
        except Exception as e:
            ErrorDialog(None, f'Error finding axis angle centre', e)

        return indexAtPercentage

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

    def copyFrameProstatePoints(self, location: str):
        """
        Copy frame points from previous or next frame on to current frame. Deletes any points on current frame
        then copies points on to frame.

        Args:
            location: Either NEXT frame or PREVIOUS frame.
        """
        # New frame.
        newFrame = self.currentFrame + 1 if location == NEXT else self.currentFrame - 1
        # Do not copy frames from opposite ends.
        if newFrame <= 0 or newFrame > self.frameCount:
            return
        # Points on new frame.
        newPoints = self.getProstatePointsOnFrame(position=newFrame - 1)

        if newPoints:
            # Delete points on current frame.
            self.clearFrameProstatePoints()
            for newPoint in newPoints:
                self.pointsProstate.append([self.frameNames[self.currentFrame - 1], newPoint[0], newPoint[1]])
            self.__saveToDisk(SAVE_POINT_DATA)

    def quaternionsToAxisAngles(self) -> list:
        """
        Convert quaternions to a list of axis angles (in degrees) in the following manner:
            1. Get the initial quaternion, to be used as the reference quaternion.
            2. Calculate the difference between all subsequent quaternions and the initial quaternion using:
                    r = p * conj(q)
               where r is the difference quaternion, p is the initial quaternion, and conj(q) is the conjugate of the
               current quaternion.
            3. Calculate the axis angle of r (the difference quaternion).

        Converting the raw quaternions to their axis angle representation for rotation comparisons is not the correct
        way to do it, the axis angle has to be calculated from the quaternion difference.

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

    def shrinkExpandProstatePoints(self, amount):
        """
        Shrink or expand points around their centre of mass. amount < 0 means shrink, amount > 0 means expand.

        Args:
            amount: How much to shrink or expand by.
        """
        points = self.getProstatePointsOnFrame()

        if points:
            newPoints = Utils.shrinkExpandPoints(points, amount)

            self.clearFrameProstatePoints()

            for newPoint in newPoints:
                self.pointsProstate.append([self.frameNames[self.currentFrame - 1], newPoint[0], newPoint[1]])

            self.__saveToDisk(SAVE_POINT_DATA)

    def countFrameProstatePoints(self, position=None):
        """
        Return number of prostate points on a frame. If position is None, return number of points on current frame.

        Args:
            position: Index of frame.

        Returns:
            count: Count of points on frame.
        """
        if position is not None:
            count = sum(1 for p in self.pointsProstate if p[0] == self.frameNames[position])
        else:
            count = sum(1 for p in self.pointsProstate if p[0] == self.frameNames[self.currentFrame - 1])

        return count

    def countFrameBladderPoints(self, position=None):
        """
        Return number of bladder points on a frame. If position is None, return number of points on current frame.

        Args:
            position: Index of frame.

        Returns:
            count: Count of points on frame.
        """
        if position is not None:
            count = sum(1 for p in self.pointsBladder if p[0] == self.frameNames[position])
        else:
            count = sum(1 for p in self.pointsBladder if p[0] == self.frameNames[self.currentFrame - 1])

        return count

    def printBulletDimensions(self):
        """
        Calculate available bullet dimensions and print them to screen.
        """
        l1 = self.bulletData['L1']
        l2 = self.bulletData['L2']
        w1 = self.bulletData['W1']
        w2 = self.bulletData['W2']
        h1 = self.bulletData['H1']
        h2 = self.bulletData['H2']

        # Calculate L.
        L = 0
        try:
            fd = self.frames[int(l1[0]) - 1].shape
            depths = self.depths[int(l1[0]) - 1]
            x = [depths[1] / fd[1] * l1[1], depths[1] / fd[1] * l2[1]]
            y = [depths[0] / fd[0] * l1[2], depths[0] / fd[0] * l2[2]]
            L = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        except Exception as e:
            pass
        # Calculate W.
        W = 0
        try:
            fd = self.frames[int(w1[0]) - 1].shape
            depths = self.depths[int(w1[0]) - 1]
            x = [depths[1] / fd[1] * w1[1], depths[1] / fd[1] * w2[1]]
            y = [depths[0] / fd[0] * w1[2], depths[0] / fd[0] * w2[2]]
            W = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        except Exception as e:
            pass
        # Calculate H.
        H = 0
        try:
            fd = self.frames[int(h1[0]) - 1].shape
            depths = self.depths[int(h1[0]) - 1]
            x = [depths[1] / fd[1] * h1[1], depths[1] / fd[1] * h2[1]]
            y = [depths[0] / fd[0] * h1[2], depths[0] / fd[0] * h2[2]]
            H = np.sqrt((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2)
        except Exception as e:
            pass
        print(f'\tLength = {L:0.2f}')
        print(f'\tWidth = {W:0.2f}')
        print(f'\tHeight = {H:0.2f}')
