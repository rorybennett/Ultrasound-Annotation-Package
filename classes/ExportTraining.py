import os
from pathlib import Path

from PyQt6.QtWidgets import QInputDialog, QWidget

import ExportUtil as eu
from classes import Scan


class ExportTraining:
    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = [f'{x:03d}' for x in range(1, self.totalPatients + 1)]

    def createSagittalData(self, mainWindow: QWidget):
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

            pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.txt'



