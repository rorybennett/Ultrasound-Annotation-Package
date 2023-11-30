import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import QInputDialog, QWidget

from classes import ExportUtil as eu
from classes import Scan


class Export:
    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = [f'{x}' for x in range(1, self.totalPatients + 1)]

    def exportIPVAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for ipv inference."""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportIPVAUSTransverseData(mainWindow)
        else:
            self._exportIPVAUSSagittalData(mainWindow)

    def exportnnUAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for nn-Unet inference"""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportnnUAUSTransverseData(mainWindow)
        else:
            self._exportnnUAUSSagittalData(mainWindow)

    def exportAllSaveData(self):
        """Export all save data from all patients - For backup."""
        print(f'\tExporting all Save Data from all patients...')
        scanTypes = ['Transverse', 'Sagittal']
        savePath = eu.createSaveDataExportDir()
        if not savePath:
            return
        # Loop through patients.
        for patient in self.patients:
            # Loop through scan types within patient.
            for scanType in scanTypes:
                src = f'../Scans/{patient}/{scanType}'
                scans = os.listdir(src)
                # Loop through scans in scan type in patient.
                for scan in scans:
                    src = f'{src}/{scan}/Save Data'
                    # Copy contents.
                    try:
                        dst = f'{savePath}/{patient}/{scanType}/{scan}'
                        shutil.copytree(src, dst)
                    except FileNotFoundError as e:
                        print(f'{src}  -  does not exist, skipping...')
        print(f'\tSExporting completed.')
        return

    def _exportnnUAUSTransverseData(self, mainWindow: QWidget):
        """Export AUS Transverse frames for nn-Unet inference."""
        print(f'\tExporting Transverse frames for nn-Unet inference...')
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save for Transverse Export', 'Enter Prefix:')
        if not ok:
            print('\tCreate Transverse nnUNet Data Cancelled.')
            return
        # Create directories for transverse training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(Scan.PLANE_TRANSVERSE)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                print(f'\t\tCreating transverse nnUNet data for patient {patient}...')
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
                        print(f'\t\t\t{prefix} not found in {scanPath}. Skipping...')
                        continue
                    # Path to PointData.txt file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'
                    # Get point data from file.
                    pointData = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointData)
                    if not framesWithPoints:
                        print(f'\t\t\tNo frames with point data available. Skipping...')
                        return
                    # Loop through frames with points on them, gather points for mask creation.
                    for index, frameNumber in enumerate(frameNumbers):
                        polygon = [(int(i[1]), int(i[2])) for i in pointData if i[0] == frameNumber]
                        frameShape = framesWithPoints[index].shape
                        img = Image.new('L', (frameShape[1], frameShape[0]), 255)
                        ImageDraw.Draw(img).polygon(polygon, fill=1)
                        mask = np.array(img)
                        cv2.imwrite(f'{imagesPath}/t_P{patient}F{frameNumber}_0000.png', framesWithPoints[index])
                        cv2.imwrite(f'{labelsPath}/t_P{patient}F{frameNumber}.png', mask)

            except WindowsError as e:
                print(f'\t\t\tError creating nnUNet transverse data for patient {patient}', e)

    def _exportnnUAUSSagittalData(self, mainWindow: QWidget):
        """Export AUS Sagittal frames for nn-Unet inference."""
        print(f'\tExporting Sagittal frames for nn-Unet inference...')
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save for Sagittal Export', 'Enter Prefix:')
        if not ok:
            print('\tCreate Sagittal nnUNet Data Cancelled.')
            return
        # Create directories for sagittal training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(Scan.PLANE_SAGITTAL)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                print(f'\t\tCreating sagittal nnUNet data for patient {patient}...')
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
                        print(f'\t\t\t{prefix} not found in {scanPath}. Skipping...')
                        continue
                    # Path to PointData.txt file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'
                    # Get point data from file.
                    pointData = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointData)
                    if not framesWithPoints:
                        print(f'\t\t\tNo frames with point data available. Skipping...')
                        return
                    # Loop through frames with points on them, gather points for mask creation.
                    for index, frameNumber in enumerate(frameNumbers):
                        polygon = [(int(i[1]), int(i[2])) for i in pointData if i[0] == frameNumber]
                        frameShape = framesWithPoints[index].shape
                        img = Image.new('L', (frameShape[1], frameShape[0]), 255)
                        ImageDraw.Draw(img).polygon(polygon, fill=1)
                        mask = np.array(img)
                        cv2.imwrite(f'{imagesPath}/s_P{patient}F{frameNumber}_0000.png', framesWithPoints[index])
                        cv2.imwrite(f'{labelsPath}/s_P{patient}F{frameNumber}.png', mask)

            except WindowsError as e:
                print(f'\t\t\tError creating nnUNet sagittal data for patient {patient}', e)

    def _exportIPVAUSSagittalData(self, mainWindow: QWidget):
        """Export AUS sagittal frames for ipv inference."""
        print(f'\tExporting Sagittal frames for IPV inference...')
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save for Sagittal Export', 'Enter Prefix:')
        if not ok:
            print('\tCreate Sagittal IPV Data Cancelled.')
            return
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
                    # Path to PointData.txt file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'
                    # Get point data from file.
                    pointData = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointData)
                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return
                    # Loop through frames with points on them.
                    for index, frameNumber in enumerate(frameNumbers):
                        saveName = f'{patient}_{frameNumber}.jpg'
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/sagittal/{saveName}', framesWithPoints[index])
                        # Get points on this frame (should only be 2).
                        points = pointData[2 * index:2 * index + 2]
                        # Sort points in IPV order (TOP, BOTTOM).
                        pointsList = []
                        for point in points:
                            pointsList.append(point[1])
                            pointsList.append(point[2])
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
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save for Transverse Export', 'Enter Prefix:')
        if not ok:
            print('\tCreate Transverse IPV Data Cancelled.')
            return
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
                    # Path to PointData.txt file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'
                    # Get point data from file.
                    pointData = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointData)
                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return

                    # Loop through frames with points on them.
                    for index, frame in enumerate(framesWithPoints):
                        saveName = f'{patient}_{frameNumbers[index]}.jpg'
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/transverse/{saveName}', frame)
                        # Get points on this frame (should only be 4).
                        points = pointData[2 * index:2 * index + 4]

                        # Sort points in IPV order (TOP, RIGHT, BOTTOM, LEFT).
                        pointsList = []
                        for point in points:
                            pointsList.append(point[1])
                            pointsList.append(point[2])
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
