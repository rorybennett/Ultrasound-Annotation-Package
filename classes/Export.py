import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import QInputDialog, QWidget

from classes import ExportUtil as eu, Utils
from classes import Scan
from classes.ErrorDialog import ErrorDialog
from classes.ExportDialog import ExportDialog


class Export:
    """
    Class for exporting Save Data and training data in the correct format.
    """

    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = [f'{x}' for x in range(1, self.totalPatients + 1)]

    @staticmethod
    def openExportDirectory(basedir):
        """Open the export folder in Windows explorer."""
        path = f"{basedir}/Export".replace('/', '\\')
        try:
            subprocess.Popen(f'explorer "{path}"')
        except Exception as e:
            ErrorDialog(None, f'Error opening Windows explorer', e)

    def exportIPVAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for ipv inference."""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportIPVAUSTransverseData(mainWindow)
        else:
            self._exportIPVAUSSagittalData(mainWindow)

    def exportnnUAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for nn-Unet inference"""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportnnUNettAUSData(mainWindow)
        else:
            self._exportnnUNetsAUSData(mainWindow)

    def exportYOLOAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for YOLO bounding box inference."""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportYOLOtAUSData(mainWindow)
        else:
            self._exportYOLOsAUSData(mainWindow)

    def exportAllSaveData(self):
        """Export all save data from all patients - For backup."""
        print(f'\tExporting all Save Data from all patients...')
        scanTypes = ['AUS']
        scanPlanes = ['Transverse', 'Sagittal']
        savePath = eu.createSaveDataExportDir()
        if not savePath:
            return
        # Loop through patients.
        for patient in self.patients:
            # Loop through scan types within patient.
            for scanType in scanTypes:
                for scanPlane in scanPlanes:
                    typeSrc = f'../Scans/{patient}/{scanType}/{scanPlane}'
                    try:
                        scans = os.listdir(typeSrc)
                        # Loop through scans in scan type in patient.
                        for scan in scans:
                            userSrc = f'{typeSrc}/{scan}/Save Data'
                            # Copy contents.
                            try:
                                dst = f'{savePath}/{patient}/{scanType}/{scanPlane}/{scan}'
                                shutil.copytree(userSrc, dst)
                            except FileNotFoundError as e:
                                print(f'{userSrc}  -  does not exist, skipping...')
                    except FileNotFoundError as e:
                        print(f'{typeSrc} - does not exist, skipping...')
        print(f'\tExporting completed.')
        return

    def _exportYOLOtAUSData(self, mainWindow: QWidget):
        """Export AUS Transverse frames for YOLO inference."""
        print('Export YOLO Transverse...')

    def _exportYOLOsAUSData(self, mainWindow: QWidget):
        """Export AUS Sagittal frames for YOLO inference."""
        print('Export YOLO Sagittal...')

    def _exportnnUNettAUSData(self, mainWindow: QWidget):
        """Export AUS Transverse frames for nn-Unet inference."""
        print(f'\tExporting tAUS frames for nn-UNet inference:', end=' ')
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save Prefix for tAUS Export', 'Enter Prefix:')
        if not ok:
            print('Create tAUS nnUNet Data Cancelled.')
            return
        # Create directories for transverse training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(Scan.PLANE_TRANSVERSE)
        print(imagesPath)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                print(f'\t\tCreating tAUS nnUNet data for patient {patient}...', end=' ')
                transversePath = f'{self.scansPath}/{patient}/AUS/transverse'
                scanDirs = Path(transversePath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f'{prefix} not found in {scanPath}. Skipping...', end=' ')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get point data from file.
                    prostatePoints, bladderPoints = eu.getPointData(pointDataPath)
                    if prostatePoints is None:
                        print(f'No frames with prostate point data available.', end=' ')
                    if bladderPoints is None:
                        print(f'No frames with bladder point data available.', end=' ')
                    # Get frames with prostate points and bladder points.
                    prostateFramesWithPoints, prostateFrameNumbers = eu.getFramesWithPoints(scanPath, prostatePoints)
                    # Get frames with prostate points and bladder points.
                    bladderFramesWithPoints, bladderFrameNumbers = eu.getFramesWithPoints(scanPath, bladderPoints)
                    # Combine frame lists into one.
                    framesWithPoints = []
                    framesWithPoints += prostateFrameNumbers if prostateFrameNumbers is not None else []
                    framesWithPoints += bladderFrameNumbers if bladderFrameNumbers is not None else []
                    framesWithPoints = list(dict.fromkeys(framesWithPoints))
                    # Loop through frames with points on them, gather points for mask creation.
                    for frameNumber in framesWithPoints:
                        pMask = None
                        if frameNumber in prostateFrameNumbers:
                            polygon = [(i[1], i[2]) for i in prostatePoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = prostateFramesWithPoints[prostateFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=1)
                            pMask = np.array(img)
                        bMask = None
                        if frameNumber in bladderFrameNumbers:
                            polygon = [(i[1], i[2]) for i in bladderPoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = bladderFramesWithPoints[bladderFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=2)
                            bMask = np.array(img)
                        # Combine masks and make overlaps only equal to prostate (prostate takes precedence).
                        finalMask = cv2.bitwise_or(pMask if pMask is not None else np.zeros(bMask.shape),
                                                   bMask if bMask is not None else np.zeros(pMask.shape))
                        finalMask[finalMask > 2] = 1

                        if pMask is not None:
                            finalFrame = prostateFramesWithPoints[prostateFrameNumbers.index(frameNumber)]
                        else:
                            finalFrame = bladderFramesWithPoints[bladderFrameNumbers.index(frameNumber)]

                        cv2.imwrite(f'{imagesPath}/t_P{patient}F{frameNumber}_0000.png', finalFrame)
                        cv2.imwrite(f'{labelsPath}/t_P{patient}F{frameNumber}.png', finalMask)
                        print('Complete.')
            except WindowsError as e:
                print(f'Error creating nnUNet tAUS data for patient {patient}', e)

    def _exportnnUNetsAUSData(self, mainWindow: QWidget):
        """Export sAUS frames for nn-UNet inference."""
        print(f'\tExporting sAUS frames for nn-UNet inference...', end=' ')
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save Prefix for sAUS Export', 'Enter Prefix:')
        if not ok:
            print('Create sAUS nnUNet Data Cancelled.')
            return
        # Create directories for sagittal training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(Scan.PLANE_SAGITTAL)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                print(f'\t\tCreating sAUS nnUNet data for patient {patient}...', end=' ')
                sagittalPath = f'{self.scansPath}/{patient}/AUS/sagittal'
                scanDirs = Path(sagittalPath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f'{prefix} not found in {scanPath}. Skipping...', end=' ')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get point data from file.
                    prostatePoints, bladderPoints = eu.getPointData(pointDataPath)
                    if prostatePoints is None:
                        print(f'No frames with prostate point data available.', end=' ')
                    if bladderPoints is None:
                        print(f'No frames with bladder point data available.', end=' ')
                    # Get frames with prostate points and bladder points.
                    prostateFramesWithPoints, prostateFrameNumbers = eu.getFramesWithPoints(scanPath, prostatePoints)
                    # Get frames with prostate points and bladder points.
                    bladderFramesWithPoints, bladderFrameNumbers = eu.getFramesWithPoints(scanPath, bladderPoints)
                    # Combine frame lists into one.
                    framesWithPoints = []
                    framesWithPoints += prostateFrameNumbers if prostateFrameNumbers is not None else []
                    framesWithPoints += bladderFrameNumbers if bladderFrameNumbers is not None else []
                    framesWithPoints = list(dict.fromkeys(framesWithPoints))
                    # Loop through frames with points on them, gather points for mask creation.
                    for frameNumber in framesWithPoints:
                        pMask = None
                        if frameNumber in prostateFrameNumbers:
                            polygon = [(i[1], i[2]) for i in prostatePoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = prostateFramesWithPoints[prostateFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=1)
                            pMask = np.array(img)
                        bMask = None
                        if frameNumber in bladderFrameNumbers:
                            polygon = [(i[1], i[2]) for i in bladderPoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = bladderFramesWithPoints[bladderFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=2)
                            bMask = np.array(img)
                        # Combine masks and make overlaps only equal to prostate (prostate takes precedence).
                        finalMask = cv2.bitwise_or(pMask if pMask is not None else np.zeros(bMask.shape),
                                                   bMask if bMask is not None else np.zeros(pMask.shape))
                        finalMask[finalMask > 2] = 1

                        if pMask is not None:
                            finalFrame = prostateFramesWithPoints[prostateFrameNumbers.index(frameNumber)]
                        else:
                            finalFrame = bladderFramesWithPoints[bladderFrameNumbers.index(frameNumber)]

                        cv2.imwrite(f'{imagesPath}/s_P{patient}F{frameNumber}_0000.png', finalFrame)
                        cv2.imwrite(f'{labelsPath}/s_P{patient}F{frameNumber}.png', finalMask)
                        print('Complete')
            except WindowsError as e:
                print(f'Error creating nnUNet sAUS data for patient {patient}', e)

    def _exportIPVAUSSagittalData(self, mainWindow: QWidget):
        """Export AUS sagittal frames for ipv inference."""
        print(f'\tExporting Sagittal frames for IPV inference...')
        # Get Export Settings.
        dlg = ExportDialog('IPV', 'Sagittal').executeDialog()
        if not dlg:
            print('\tCreate Sagittal IPV Data Cancelled.')
            return
        prefix, resample, pixelDensity = dlg
        # Create directories for sagittal training data.
        savePath = eu.createIPVTrainingDirs(Scan.PLANE_SAGITTAL)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                sagittalPath = f'{self.scansPath}/{patient}/AUS/sagittal'
                scanDirs = Path(sagittalPath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f'\t{prefix} not found in {scanPath}. Skipping...')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get point data from file.
                    prostatePoints, _ = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, prostatePoints)
                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return
                    # Loop through frames with points on them.
                    for index, frameNumber in enumerate(frameNumbers):
                        saveName = f'{patient}_{frameNumber}.jpg'
                        if resample:
                            # Resample Frame and Points.
                            depths = eu.getDepths(scanPath, int(frameNumber))
                            frame, points = eu.resampleImageAndPoints(framesWithPoints[index], depths,
                                                                      prostatePoints[2 * index:2 * index + 2],
                                                                      pixelDensity)
                        else:
                            frame = framesWithPoints[index]
                            points = prostatePoints[2 * index:2 * index + 2]
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/sagittal/{saveName}', frame)
                        # Sort points in IPV order (TOP, BOTTOM).
                        pointsList = []
                        for point in points:
                            pointsList.append(int(float(point[1])))
                            pointsList.append(int(float(point[2])))
                        pointsList = np.array(
                            [[pointsList[i], pointsList[i + 1]] for i in range(0, len(pointsList) - 1, 2)])
                        bottom = pointsList[np.argmin(pointsList[:, 1])]
                        top = pointsList[np.argmax(pointsList[:, 1])]

                        # Save point data.
                        with open(f'{savePath}/sagittal_mark_list.txt', 'a') as pointFile:
                            pointFile.write(
                                f'{saveName} ({bottom[0]}, {bottom[1]}) '
                                f'({top[0]}, {top[1]})\n')
            except WindowsError as e:
                print(f'\tError creating IPV sagittal data for patient {patient}: ', e)
        print(f'\tAUS sagittal exporting completed.')

    def _exportIPVAUSTransverseData(self, mainWindow: QWidget):
        """Export AUS transverse frames for ipv inference."""
        print(f'\tExporting Transverse frames for IPV inference...')
        # Get Export Settings.
        dlg = ExportDialog('IPV', 'Sagittal').executeDialog()
        if not dlg:
            print('\tCreate Sagittal IPV Data Cancelled.')
            return
        prefix, resample, pixelDensity = dlg
        # Create directories for sagittal training data.
        savePath = eu.createIPVTrainingDirs(Scan.PLANE_TRANSVERSE)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                transversePath = f'{self.scansPath}/{patient}/AUS/transverse'
                scanDirs = Path(transversePath).iterdir()
                # Go through each scan in the plane directory.
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f'\t{prefix} not found in {scanPath}. Skipping...')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get point data from file.
                    prostatePoints, _ = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, prostatePoints)

                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return

                    # Loop through frames with points on them.
                    for index, frameNumber in enumerate(frameNumbers):
                        saveName = f'{patient}_{frameNumbers[index]}.jpg'
                        if resample:
                            # Resample Frame and Points.
                            depths = eu.getDepths(scanPath, int(frameNumber))
                            frame, points = eu.resampleImageAndPoints(framesWithPoints[index], depths,
                                                                      prostatePoints[2 * index:2 * index + 4],
                                                                      pixelDensity)
                        else:
                            frame = framesWithPoints[index]
                            points = prostatePoints[2 * index:2 * index + 4]
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/transverse/{saveName}', frame)
                        # Sort points in IPV order (TOP, RIGHT, BOTTOM, LEFT).
                        pointsList = []
                        for point in points:
                            pointsList.append(int(float(point[1])))
                            pointsList.append(int(float(point[2])))
                        pointsList = np.array(
                            [[pointsList[i], pointsList[i + 1]] for i in range(0, len(pointsList) - 1, 2)])
                        bottom = pointsList[np.argmin(pointsList[:, 1])]
                        right = pointsList[np.argmax(pointsList[:, 0])]
                        top = pointsList[np.argmax(pointsList[:, 1])]
                        left = pointsList[np.argmin(pointsList[:, 0])]
                        # Save point data.
                        with open(f'{savePath}/transverse_mark_list.txt', 'a') as pointFile:
                            pointFile.write(
                                f'{saveName} ({bottom[0]}, {bottom[1]}) '
                                f'({right[0]}, {right[1]}) '
                                f'({top[0]}, {top[1]}) '
                                f'({left[0]}, {left[1]})\n')
            except WindowsError as e:
                print(f'\tError creating IPV transverse data for patient {patient}: ', e)
        print(f'\tAUS transverse exporting completed.')
