# Scan.py

"""
Scan class with variables and methods for working with a single scan. Majority of the functionality of the application
is here.
"""
import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
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
# Prostate vs Bladder .
PROSTATE = '-PROSTATE-'
BLADDER = '-BLADDER-'
PROSTATE_BOX = '-PROSTATE-BOX-'
BLADDER_BOX = '-BLADDER-BOX-'
# Start or End.
BOX_START = '-START-'
BOX_DRAW = '-DRAW-'
BOX_END = '-END-'


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
        self.pointPath, self.pointsProstate, self.pointsBladder, self.boxProstate, self.boxBladder = None, None, None, None, None
        # IPV data from IPV.JSON.
        self.ipvPath, self.ipvData = None, None
        # Bullet data from Bullet.json
        self.bulletPath, self.bulletData = None, None
        # Has a Scan been loaded?
        self.loaded = False
        # SI estimate values.
        self.estimateSI = None
        # RL and AP ellipse.
        self.estimateRLAP = None

    def load(self, path: str, startingFrame=1):
        """
        Load a Scan object using the given path string. See __init__ for attribute details.

        Args:
            path: Path to Scan directory as a String.
            startingFrame: Starting frame position.
        """
        print(f'\tLoading {path}...')
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
        self.pointPath, self.pointsProstate, self.pointsBladder, self.boxProstate, self.boxBladder = su.getPointDataFromFile(
            self.path)
        self.ipvPath, self.ipvData = su.getIPVDataFromFile(self.path)
        self.bulletPath, self.bulletData = su.getBulletDataFromFile(self.path)
        self.estimateSI = None
        self.estimateRLAP = None
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
        su.drawScanDataOnAxis(axis, frameNumber=cfi + 1, frameCount=count, depths=depths, imuOff=imuOffset,
                              imuPos=imuPosition, dd=dd, frameProstatePoints=self.countFramePoints(PROSTATE),
                              frameBladderPoints=self.countFramePoints(BLADDER),
                              totalProstatePoints=len(self.pointsProstate), totalProstateBoxes=len(self.boxProstate),
                              totalBladderPoints=len(self.pointsBladder), totalBladderBoxes=len(self.boxBladder))

    def getBoxPointsOnFrame(self, prostateBladder, position=None):
        """
        Get start and end points for prostate or bladder bounding box on frame.

        Args:
            :param prostateBladder: Get either prostate or bladder points.
            :param position: Index of frame.
            :return: Start and end points of box as list.
        """
        points = []
        frameName = self.frameNames[position if position is not None else self.currentFrame - 1]
        if prostateBladder == PROSTATE:
            boxIndex = su.getIndexOfFrameInBoxPoints(self.boxProstate, frameName)
            if boxIndex > -1:
                points = self.boxProstate[boxIndex][1:]
        else:
            boxIndex = su.getIndexOfFrameInBoxPoints(self.boxBladder, frameName)
            if boxIndex > -1:
                points = self.boxBladder[boxIndex][1:]
        return points

    def getPointsOnFrame(self, prostateBladder, position=None):
        """
        Return a list of points (prostate or bladder) on the frame at 'position'. If position is None, use the
        current frame.

        :param prostateBladder: Get either prostate or bladder points.
        :param position: Index of frame.
        :return: List of points on frame.
        """
        if prostateBladder == PROSTATE:
            if position is not None:
                points = [p[1:] for p in self.pointsProstate if p[0] == self.frameNames[position]]
            else:
                points = [p[1:] for p in self.pointsProstate if p[0] == self.frameNames[self.currentFrame - 1]]
        else:
            if position is not None:
                points = [p[1:] for p in self.pointsBladder if p[0] == self.frameNames[position]]
            else:
                points = [p[1:] for p in self.pointsBladder if p[0] == self.frameNames[self.currentFrame - 1]]

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
            windowDimensions = [self.window.centralWidget().size().width(), self.window.centralWidget().size().height()]

            width = windowDimensions[0] * 0.48  # Width of frame is 48% of total screen width.

            ratio = width / self.frames[0].shape[1]

            height = self.frames[0].shape[0] * ratio  # Maintain aspect ratio of frame.

            # # If image is too high, then height must be used as the limiting dimension.
            if height > windowDimensions[1] * 0.8:
                height = windowDimensions[1] * 0.8
                ratio = height / self.frames[0].shape[0]
                width = self.frames[0].shape[1] * ratio

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
            ErrorDialog(None, f'Error opening Windows explorer.', e)

    def updateBoxPoints(self, prostateBladder, startDrawEnd, pointDisplay):
        """
        Update bounding box points for PROSTATE or BLADDER. On a down click the BOX_START point is altered, on release
        the BOX_END point is altered. On BOX_END alteration the points are organised such that the top left point is
        first and the bottom right point is second.

        Parameters
        ----------
        prostateBladder: PROSTATE or BLADDER bounding box points.
        startDrawEnd: Start point, draw point, or end point of the bounding box.
        pointDisplay: Point coordinates in display coordinates.
        """
        pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape, self.displayDimensions)
        frameName = self.frameNames[self.currentFrame - 1]
        if prostateBladder == PROSTATE:
            index = su.getIndexOfFrameInBoxPoints(self.boxProstate, frameName)
            if index > -1:
                if startDrawEnd == BOX_START:
                    self.boxProstate[index] = [frameName, pointPixel[0], pointPixel[1], pointPixel[0], pointPixel[1]]
                elif startDrawEnd == BOX_DRAW:
                    self.boxProstate[index][3:] = [pointPixel[0], pointPixel[1]]
                else:
                    self.boxProstate[index][1:] = su.getBoundingBoxStartAndEnd(self.boxProstate[index][1:])
                    self.__saveToDisk(SAVE_POINT_DATA)
            else:
                self.boxProstate.append([frameName, pointPixel[0], pointPixel[1], pointPixel[0], pointPixel[1]])
        else:
            index = su.getIndexOfFrameInBoxPoints(self.boxBladder, frameName)
            if index > -1:
                if startDrawEnd == BOX_START:
                    self.boxBladder[index] = [frameName, pointPixel[0], pointPixel[1], pointPixel[0], pointPixel[1]]
                elif startDrawEnd == BOX_DRAW:
                    self.boxBladder[index][3:] = [pointPixel[0], pointPixel[1]]
                else:
                    self.boxBladder[index][1:] = su.getBoundingBoxStartAndEnd(self.boxBladder[index][1:])
                    self.__saveToDisk(SAVE_POINT_DATA)
            else:
                self.boxBladder.append([frameName, pointPixel[0], pointPixel[1], pointPixel[0], pointPixel[1]])

    def generateBox(self, prostateBladder):
        """
        Generate either a prostate or bladder bounding box using the points available on the current frame. The
        boundaries of the frame are made 2 pixels wider than the rounded limits of the edges determined
        using the points on the current frame.

        Parameters
        ----------
        prostateBladder: Prostate or Bladder box.
        """
        points = self.getPointsOnFrame(PROSTATE if prostateBladder == PROSTATE_BOX else BLADDER,
                                       self.currentFrame - 1)

        if len(points) > 4:
            l = max([min(points, key=lambda coord: coord[0])[0] - 5, 1])
            t = max([min(points, key=lambda coord: coord[1])[1] - 5, 1])
            r = min([max(points, key=lambda coord: coord[0])[0] + 5, self.frameShape[1] - 2])
            b = min([max(points, key=lambda coord: coord[1])[1] + 5, self.frameShape[0] - 2])
            frameName = self.frameNames[self.currentFrame - 1]
            index = su.getIndexOfFrameInBoxPoints(
                self.boxProstate if prostateBladder == PROSTATE_BOX else self.boxBladder, frameName)
            if prostateBladder == PROSTATE_BOX:
                if index > -1:
                    self.boxProstate[index] = [frameName, l, t, r, b]
                else:
                    self.boxProstate.append([frameName, l, t, r, b])
            else:
                if index > -1:
                    self.boxBladder[index] = [frameName, l, t, r, b]
                else:
                    self.boxBladder.append([frameName, l, t, r, b])
            self.__saveToDisk(SAVE_POINT_DATA)

    def flipLR(self):
        """
        Flip the images in the Scan in the Left-Right dimension. This is for the IPV Scans as some of them have the
        prostate on the left while the patient data always has it on the right.

        Returns
        -------
        True if frames flipped without error, else False.
        """
        # Flip images.
        if not su.flipFrames(self.path, 'LR'):
            return False
        # Flip Point and Box data.
        width = self.frameShape[1]
        self.pointsProstate = su.flipPoints(width, self.pointsProstate, 'LR')
        self.pointsBladder = su.flipPoints(width, self.pointsBladder, 'LR')
        self.boxProstate = su.flipBoxes(width, self.boxProstate, 'LR')
        self.boxBladder = su.flipBoxes(width, self.boxBladder, 'LR')
        self.__saveToDisk(SAVE_ALL)
        # Flip save data.
        su.flipSaveData(f'{self.path}/Save Data', width, 'LR')

    def addOrRemovePoint(self, pointDisplay: list, prostateBladder, deleteRadius=10):
        """
        Add or remove a point to/from self.pointsProstate or self.pointsBladder. Point data is saved in pixel values.
        If the new point is within a radius of an old point, the old point is removed.

        Args:
            pointDisplay: x/width-, and y/height-coordinates returned by the canvas elements' event.
            prostateBladder: Add or remove points to either prostate points or bladder points.
            deleteRadius: Points within this radius will be deleted.
        """
        pointPixel = su.displayToPixels(pointDisplay, self.frames[self.currentFrame - 1].shape, self.displayDimensions)

        pointRemoved = False

        if prostateBladder == PROSTATE:
            for point in self.pointsProstate:
                if self.frameNames[self.currentFrame - 1] == point[0]:
                    # If within radius of another point, remove that point.
                    if su.pointInRadius(point[1:], pointPixel, deleteRadius):
                        self.pointsProstate.remove(point)
                        pointRemoved = True
                        break
            # If no point was removed, add the new point.
            if not pointRemoved:
                self.pointsProstate.append([self.frameNames[self.currentFrame - 1], pointPixel[0], pointPixel[1]])
        else:
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
                            'Bladder': self.pointsBladder,
                            'ProstateBox': self.boxProstate,
                            'BladderBox': self.boxBladder}
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

    def clearFrameBox(self, prostateBladder):
        """
        Clear either prostate bounding box or bladder bounding box frame frame.

        Parameters
        ----------
        prostateBladder: PROSTATE or BLADDER.
        """
        frameName = self.frameNames[self.currentFrame - 1]
        if prostateBladder == PROSTATE:
            self.boxProstate = [p for p in self.boxProstate if not p[0] == frameName]
        else:
            self.boxBladder = [p for p in self.boxBladder if not p[0] == frameName]

        self.__saveToDisk(SAVE_POINT_DATA)

    def clearFramePoints(self, prostateBladder):
        """
        Clear points on the currently displayed frame, then save to disk.
        """
        frameName = self.frameNames[self.currentFrame - 1]
        if prostateBladder == PROSTATE:
            self.pointsProstate = [p for p in self.pointsProstate if not p[0] == frameName]
        else:
            self.pointsBladder = [p for p in self.pointsBladder if not p[0] == frameName]

        self.__saveToDisk(SAVE_POINT_DATA)

    def clearScanBoxes(self):
        """
        Clear all boxes in the Scan, then save to disk.
        """
        self.boxProstate = []
        self.boxBladder = []

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
            ErrorDialog(None, 'Error loading bullet data.', e)
            Path(self.bulletPath).unlink(missing_ok=True)
            successFlags[0] = False

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.pointPath.name), self.pointPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading point data', e)
            Path(self.pointPath).unlink(missing_ok=True)
            successFlags[1] = False

        try:
            shutil.copy(Path(self.path, 'Save Data/' + saveName + '/' + self.editPath.name), self.editPath)
        except Exception as e:
            ErrorDialog(None, 'Error loading editing data', e)
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
        su.checkSaveDataDirectory(self.path)

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
                with open(f'{self.path}/Save Data/{user}/PointData.json', 'r') as file:
                    data = json.load(file)
                    prostateData = data.get('Prostate')
                    framesWithPoints = []
                    for row in prostateData:
                        framesWithPoints.append(row[0])
                framesWithPoints = sorted(set(framesWithPoints))
                if self.scanPlane == PLANE_TRANSVERSE or len(framesWithPoints) == 1:
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
            ErrorDialog(None, f'Error finding axis angle centre.', e)

        return indexAtPercentage

    def copyFramePoints(self, location: str, prostateBladder):
        """
        Copy frame points from previous or next frame on to current frame. Deletes any points on current frame
        then copies points on to frame. Either prostate points OR bladder points are copied, depending on which
        is being used for segmentation.

        Args:
            location: Either NEXT frame or PREVIOUS frame.
            prostateBladder: Either prostate or bladder points are copied.
        """
        # New frame.
        newFrame = self.currentFrame + 1 if location == NEXT else self.currentFrame - 1
        # Do not copy frames from opposite ends.
        if newFrame <= 0 or newFrame > self.frameCount:
            return
        # Points on new frame.
        newPoints = self.getPointsOnFrame(prostateBladder, position=newFrame - 1)

        if newPoints:
            # Delete points on current frame.
            self.clearFramePoints(prostateBladder)
            for newPoint in newPoints:
                if prostateBladder == PROSTATE:
                    self.pointsProstate.append([self.frameNames[self.currentFrame - 1], newPoint[0], newPoint[1]])
                else:
                    self.pointsBladder.append([self.frameNames[self.currentFrame - 1], newPoint[0], newPoint[1]])
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

    def shrinkExpandPoints(self, amount, prostateBladder):
        """
        Shrink or expand points around their centre of mass. amount < 0 means shrink, amount > 0 means expand.

        Args:
            amount: How much to shrink or expand by.
            prostateBladder: Shrink or expand either prostate or bladder points.
        """
        points = self.getPointsOnFrame(prostateBladder)

        if points:
            newPoints = Utils.shrinkExpandPoints(points, amount)

            self.clearFramePoints(prostateBladder)

            for newPoint in newPoints:
                self.pointsProstate.append([self.frameNames[self.currentFrame - 1], newPoint[0],
                                            newPoint[
                                                1]]) if prostateBladder == PROSTATE else self.pointsBladder.append(
                    [self.frameNames[self.currentFrame - 1], newPoint[0], newPoint[1]])

            self.__saveToDisk(SAVE_POINT_DATA)

    def countFramePoints(self, prostateBladder, position=None):
        """
        Return number of prostate points on a frame. If position is None, return number of points on current frame.

        Args:
            prostateBladder: Count either prostate or bladder points.
            position: Index of frame.

        Returns:
            count: Count of points on frame.
        """
        if prostateBladder == PROSTATE:
            if position is not None:
                count = sum(1 for p in self.pointsProstate if p[0] == self.frameNames[position])
            else:
                count = sum(1 for p in self.pointsProstate if p[0] == self.frameNames[self.currentFrame - 1])
        else:
            if position is not None:
                count = sum(1 for p in self.pointsBladder if p[0] == self.frameNames[position])
            else:
                count = sum(1 for p in self.pointsBladder if p[0] == self.frameNames[self.currentFrame - 1])

        return count

    def calculateRLAP(self, pointsWeight, angleWeight):
        """
        Estimate the RL and AP measurements of the ellipse equation. RL and AP are taken on the transverse plane,
        but no check is done to ensure the correct plane is being considered. User must just be aware.

        Parameters
        ----------
        pointsWeight: Weight applied to prostate points.
        angleWeight: Weight applied to desired angle (bladderCoM to prostateCoM line).
        """
        try:
            # Calculate bladder centre of mass (A).
            bladderCoM = su.calculateCentreOfMass(self.getPointsOnFrame(BLADDER))
            bladderCoMDisplay = su.pixelsToDisplay(bladderCoM, self.frames[self.currentFrame - 1].shape,
                                                   self.displayDimensions)
            # Find prostate bottom right point (C).
            prostateCoM = su.calculateCentreOfMass(self.getPointsOnFrame(PROSTATE))
            prostateCoMDisplay = su.pixelsToDisplay(prostateCoM, self.frames[self.currentFrame - 1].shape,
                                                    self.displayDimensions)
            # Get all prostate points on frame.
            prostatePoints = self.getPointsOnFrame(PROSTATE)
            prostatePointsDisplay = [su.pixelsToDisplay(point, self.frames[self.currentFrame - 1].shape,
                                                        self.displayDimensions) for point in prostatePoints]
            # Fit ellipse while trying to align ellipse to line AC.
            [[xc, yc], a, b, resultantPhi] = su.fitEllipseToPoints(prostatePointsDisplay, bladderCoMDisplay,
                                                                   prostateCoMDisplay, pointsWeight, angleWeight)
            self.estimateRLAP = {f'{self.currentFrame}': [[xc, yc], a, b, resultantPhi]}

            # Show points and resultant ellipse with major and minor axes to help ensure the correct values are being
            # solved for.
            # fig, ax = plt.subplots()
            # cos_phi = np.cos(resultantPhi)
            # sin_phi = np.sin(resultantPhi)
            # x_major1 = xc + a * cos_phi
            # y_major1 = yc + a * sin_phi
            # x_major2 = xc - a * cos_phi
            # y_major2 = yc - a * sin_phi
            #
            # x_minor1 = xc + b * sin_phi
            # y_minor1 = yc - b * cos_phi
            # x_minor2 = xc - b * sin_phi
            # y_minor2 = yc + b * cos_phi
            #
            # x = [p[0] for p in prostatePointsDisplay]
            # y = [p[1] for p in prostatePointsDisplay]
            # ax.plot([bladderCoMDisplay[0], prostateCoMDisplay[0]], [bladderCoMDisplay[1], prostateCoMDisplay[1]],
            #         marker='*')
            # # Plot the semi-major axis
            # ax.plot([x_major1, x_major2], [y_major1, y_major2], 'r--', lw=2)
            #
            # # Plot the semi-minor axis
            # ax.plot([x_minor1, x_minor2], [y_minor1, y_minor2], 'g--', lw=2)
            # ax.scatter(x, y, color='red', marker='o')
            # ellipse = Ellipse(xy=(xc, yc), width=2 * a, height=2 * b, angle=np.rad2deg(resultantPhi),
            #                   edgecolor='b', fc='None', lw=2)
            # ax.add_patch(ellipse)
            # ax.set_aspect('equal')
            # plt.show()
        except Exception as e:
            ErrorDialog(None, f'Error with RL/AP calculation.', e)

    def calculateSI(self):
        """
        Estimate the SI measurement of the ellipse equation. SI is taken on the sagittal plane, but no check is done
        to ensure the correct plane is being considered. User must just be aware.
        """
        try:
            # Calculate bladder centre of mass (A).
            bladderCoM = su.calculateCentreOfMass(self.getPointsOnFrame(BLADDER))
            # Find prostate bottom right point (C).
            prostateBottomRight = su.getBottomRightPoint(self.getPointsOnFrame(PROSTATE), 1, 1)
            # Find intersection of prostate boundary and line AC (B).
            intersections = su.findIntersectionsOfLineAndBoundary(self.getPointsOnFrame(PROSTATE),
                                                                  (bladderCoM, prostateBottomRight))
            self.estimateSI = {
                f'{self.currentFrame}': [bladderCoM, prostateBottomRight, intersections]
            }
        except Exception as e:
            ErrorDialog(None, f'Error with SI calculation.', e)

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
