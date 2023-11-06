import os
import shutil
from pathlib import Path

import cv2
from PyQt6.QtWidgets import QInputDialog, QWidget

from classes import ExportUtil as eu
from classes import Scan
from classes.ErrorDialog import ErrorDialog


class Export:
    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = [f'{x}' for x in range(1, self.totalPatients + 1)]

    def exportIPVAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for ipv inference."""
        if scanType == Scan.PLANE_TRANSVERSE:
            self.exportIPVAUSTransverseData(mainWindow)
        else:
            self.exportIPVAUSSagittalData(mainWindow)

    def exportAllSaveData(self):
        """Export all save data from all patients - For backup."""
        print(f'Exporting all Save Data from all patients...')
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

    def exportIPVAUSSagittalData(self, mainWindow: QWidget):
        """Export AUS sagittal frames for ipv inference."""
        print(f'Exporting Sagittal frames for IPV inference...')
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
                    # Get point data in mm from f()ile.
                    pointDataPix = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointDataPix)
                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return
                    # Loop through frames with points on them.
                    for index, frame in enumerate(framesWithPoints):
                        saveName = f'{patient}_{frameNumbers[index]}.png'
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/sagittal/{saveName}', frame)

                        # Save point data.
                        with open(f'{savePath}/sagittal_mark_list.txt', 'a') as pointFile:
                            pointFile.write(
                                f'{saveName} ({pointDataPix[0][1]}, {pointDataPix[0][2]}) '
                                f'({pointDataPix[1][1]}, {pointDataPix[1][2]})\n')
            except WindowsError as e:
                ErrorDialog(None, f'Error creating sagittal data', e)
        print(f'\tAUS sagittal exporting completed.')

    def exportIPVAUSTransverseData(self, mainWindow: QWidget):
        """Export AUS transverse frames for ipv inference."""
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
                    # Get point data in mm from f()ile.
                    pointDataPix = eu.getPointData(pointDataPath)
                    # Get frames with points on them.
                    framesWithPoints, frameNumbers = eu.getFramesWithPoints(scanPath, pointDataPix)
                    if not framesWithPoints:
                        print(f'\tNo frames with point data available.')
                        return

                    # Loop through frames with points on them.
                    for index, frame in enumerate(framesWithPoints):
                        saveName = f'{patient}_{frameNumbers[index]}.png'
                        # Save frame to disk.
                        cv2.imwrite(f'{savePath}/transverse/{saveName}', frame)
                        # Save point data.
                        with open(f'{savePath}/transverse_mark_list.txt', 'a') as pointFile:
                            pointFile.write(
                                f'{saveName} ({pointDataPix[0][1]}, {pointDataPix[0][2]}) '
                                f'({pointDataPix[1][1]}, {pointDataPix[1][2]}) '
                                f'({pointDataPix[2][1]}, {pointDataPix[2][2]}) '
                                f'({pointDataPix[3][1]}, {pointDataPix[3][2]})\n')
            except WindowsError as e:
                ErrorDialog(None, f'Error creating transverse data', e)
        print(f'\tAUS transverse exporting completed.')
