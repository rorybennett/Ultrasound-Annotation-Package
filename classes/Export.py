import os
from pathlib import Path

import cv2
from PyQt6.QtWidgets import QInputDialog, QWidget

import ExportUtil as eu
from classes import Scan


class Export:
    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = [f'{x:03d}' for x in range(1, self.totalPatients + 1)]

    def exportIPVSagittalData(self, mainWindow: QWidget):
        # Get Save prefix.
        prefix, ok = QInputDialog.getText(mainWindow, 'Select Save for Sagittal Export', 'Enter Prefix:')
        if not ok:
            print('Create Sagittal Data Cancelled.')
            return
        # Create directories for sagittal training data.
        savePath = eu.createTrainingDirs(Scan.TYPE_SAGITTAL)
        # Create training data from each patient.
        for patient in self.patients:
            sagittalPath = f'{self.scansPath}/{patient}/sagittal'
            framePath = os.listdir(Path(sagittalPath).absolute())[0]
            scanPath = f'{sagittalPath}/{framePath}'
            # Get save directory with given prefix.
            saveDir = eu.getSaveDirName(scanPath, prefix)

            if not saveDir:
                print(f'Patient {patient} does not have a matching Save directory with prefix {prefix}.')
                return
            # Path to PointData.txt file.
            pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'
            # Get point data in mm from file.
            pointDataMm = eu.getPointData(pointDataPath)
            # Get depths of scans.
            depths = eu.getDepths(scanPath)
            # Get required IMU data.
            imuOffset, imuPosition = eu.getIMUData(f'{scanPath}/Save Data/{saveDir}')
            # Get frames with points on them.
            framesWithPoints = eu.getFramesWithPoints(scanPath, pointDataMm)
            if not framesWithPoints:
                print(f'No frames with point data available.')
                return
            # Loop through frames with points on them.
            for index, frame in enumerate(framesWithPoints, start=1):
                saveName = f'{patient}_{index}.png'
                savePath = f'DATA/Sagittal/{saveName}'
                # Save frame to disk.
                cv2.imwrite(savePath, frame)
                # Convert points from mm to display/pixel coordinates.
                pointDataDisplay = []
                for point in pointDataMm:
                    pointDisplay = eu.mmToDisplayCoordinates([point[1], point[2]], depths, imuOffset, imuPosition,
                                                             [frame.shape[1], frame.shape[0]])
                    pointDataDisplay.append([point[0], pointDisplay[1], pointDisplay[1]])
                # Save point data.
                with open(f'DATA/Sagittal_mark_list.txt', 'a') as pointFile:
                    pointFile.write(
                        f'{saveName} ({pointDataDisplay[0][1]}, {pointDataDisplay[0][2]}) '
                        f'({pointDataDisplay[1][1]}, {pointDataDisplay[1][2]})\n')
