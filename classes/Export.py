"""
Class for all exports:
    1. Exporting data in IPV format. This may be out of date.
    2. All SaveData (for backup purposes).
    3. IMU Data.
    4. Exporting data in YOLO format.
    5. Exporting data in YOLO format for use with nn-UNet.

This worked at one point, but not all of it may function 100% at the moment. Currently, only AUS export is possible.

Export are based on data in the Save Data directory, and not what is currently loaded.
"""
import glob
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from PyQt6.QtWidgets import QWidget
from natsort import natsort

from classes import ExportUtil as eu, Utils
from classes import Scan
from classes.ErrorDialog import ErrorDialog
from classes.ExportDialogs import ExportDialogs


class Export:
    """
    Class for exporting Save Data and training data in the correct format.
    """

    def __init__(self, scansPath: str):
        self.scansPath = scansPath
        self.totalPatients = eu.getTotalPatients(self.scansPath)
        self.patients = natsort.natsorted(os.listdir(self.scansPath))

    @staticmethod
    def openExportDirectory(basedir):
        """Open the export folder in Windows explorer."""
        path = f"{basedir}/Export".replace('/', '\\')
        try:
            subprocess.Popen(f'explorer "{path}"')
        except Exception as e:
            ErrorDialog(None, f'Error opening Windows explorer.', e)

    def exportIPVAUSData(self, scanType, mainWindow: QWidget):
        """Export AUS transverse or sagittal frames for ipv inference."""
        if scanType == Scan.PLANE_TRANSVERSE:
            self._exportIPVAUSTransverseData(mainWindow)
        else:
            self._exportIPVAUSSagittalData()

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

    def exportIMUData(self):
        """Export IMU data.txt file of prospective patients."""
        print(f'\tExporting IMU data from data.txt file...')
        scanTypes = ['AUS', 'PUS']
        scanPlanes = ['Transverse', 'Sagittal']
        savePath = eu.createIMUExportDir()
        if not savePath:
            return
        # Loop through prospective patients.
        for patient in self.patients[:19]:
            for scanType in scanTypes:
                for scanPlane in scanPlanes:
                    src = f'C:/Users/roryb/GDOffline/Research/Scans/{patient}/{scanType}/{scanPlane}/1/data.txt'
                    try:
                        dst = f'{savePath}/{patient}/{scanType}/{scanPlane}/1'
                        if not os.path.exists(dst):
                            Path(dst).mkdir(parents=True, exist_ok=False)
                        shutil.copyfile(src, f'{dst}/data.txt')
                    except FileNotFoundError as e:
                        print(f'{src} error: {e}')
        print(f'\tExporting completed.')
        return

    def exportYOLO3DAUSData(self, scanPlane):
        """ """
        print(f'\tExporting {scanPlane} AUS frames for YOLO 3D inference:', end=' ')
        # Get Save prefix.
        dlg = ExportDialogs().YOLODialog(scanPlane)
        if not dlg:
            print(f'\tCreate {scanPlane} YOLO Data Cancelled.')
            return
        prefix, prostate, bladder, exportName = dlg
        # Create export directory.
        imagesPath, labelsPath = eu.createYOLOTrainingDirs(scanPlane, prefix, exportName)
        print(imagesPath)
        for patient in self.patients:
            try:
                print(f'\t\tPatient {patient}...', end=' ')
                scanPath = f'{self.scansPath}/{patient}/AUS/{scanPlane}'
                scanDirs = Path(scanPath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f"{prefix} not found in {'/'.join(scanPath.split('/')[-4:])}, skipping.", end=' ')
                        continue
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get box data from file.
                    _, _, prostateBoxes, bladderBoxes = eu.getPointAndBoxData(pointDataPath)
                    if prostateBoxes is None:
                        print(f'No frames with prostate boxes...', end=' ')
                    if bladderBoxes is None:
                        print(f'No frames with bladder boxes...', end=' ')

                    prostateBoxes = {int(sublist[0]): sublist for sublist in prostateBoxes}
                    bladderBoxes = {int(sublist[0]): sublist for sublist in bladderBoxes}
                    # Get all frames.
                    allFrames = sorted(glob.glob(f'{scanPath}/*.png'))
                    # Loop through all frames.
                    for framePath in allFrames:
                        frame = cv2.imread(framePath, cv2.IMREAD_UNCHANGED)
                        frameNumber = int(os.path.splitext(os.path.basename(framePath))[0])
                        saveName = f'{scanPlane[0].lower()}P{patient}F{frameNumber.__str__()}'
                        labelText = ''
                        # Check if frame has a prostate box.
                        if frameNumber in prostateBoxes and prostate:
                            frameProstateBox = prostateBoxes[frameNumber]
                            yoloBoxes = eu.getYOLOBoxes(frameProstateBox[1:], frame.shape)
                            labelText += f'0 {yoloBoxes[0]} {yoloBoxes[1]} {yoloBoxes[2]} {yoloBoxes[3]}\n'
                        # Check if frame has a bladder box.
                        if frameNumber in bladderBoxes and bladder:
                            frameBladderBox = bladderBoxes[frameNumber]
                            yoloBoxes = eu.getYOLOBoxes(frameBladderBox[1:], frame.shape)
                            labelText += f'1 {yoloBoxes[0]} {yoloBoxes[1]} {yoloBoxes[2]} {yoloBoxes[3]}'
                        # Save frame as .png.
                        cv2.imwrite(f'{imagesPath}/{saveName}.png', frame)
                        with open(f'{labelsPath}/{saveName}.txt', 'w') as labelFile:
                            labelFile.write(labelText)
                print(f'Patient {patient} Complete.')
            except Exception as e:
                print(f'Error creating YOLO {scanPlane} AUS data for patient {patient}.', e)

    def exportYOLOAUSData(self, scanPlane):
        """Export AUS frames for YOLO inference."""
        print(f'\tExporting {scanPlane} AUS frames for YOLO inference:', end=' ')
        # Get Save prefix.
        dlg = ExportDialogs().YOLODialog(scanPlane)
        if not dlg:
            print(f'\tCreate {scanPlane} YOLO Data Cancelled.')
            return
        prefix, prostate, bladder, exportName = dlg
        # Create export directory.
        imagesPath, labelsPath = eu.createYOLOTrainingDirs(scanPlane, prefix, exportName)
        print(imagesPath)
        for patient in self.patients:
            try:
                print(f'\t\tPatient {patient}...', end=' ')
                scanPath = f'{self.scansPath}/{patient}/AUS/{scanPlane}'
                scanDirs = Path(scanPath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f"{prefix} not found in {'/'.join(scanPath.split('/')[-4:])}, skipping.", end=' ')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get box data from file.
                    _, _, prostateBoxes, bladderBoxes = eu.getPointAndBoxData(pointDataPath)
                    if prostateBoxes is None:
                        print(f'No frames with prostate boxes...', end=' ')
                    if bladderBoxes is None:
                        print(f'No frames with bladder boxes...', end=' ')
                    # Get frames with prostate boxes and bladder boxes.
                    pFrames, pFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, prostateBoxes)
                    bFrames, bFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, bladderBoxes)
                    # Combine frame lists into one.
                    framesWithBoxes = []
                    framesWithBoxes += pFrameNumbers if pFrameNumbers is not None else []
                    framesWithBoxes += bFrameNumbers if bFrameNumbers is not None else []
                    # Remove duplicates.
                    framesWithBoxes = list(dict.fromkeys(framesWithBoxes))
                    # Loop through frames with points on them, gather points for mask creation.
                    for frameNumber in framesWithBoxes:
                        pBox = [0]
                        if prostate and pFrameNumbers is not None and frameNumber in pFrameNumbers:
                            points = [i[1:] for i in prostateBoxes if i[0] == frameNumber][0]
                            pBox += eu.getYOLOBoxes(points, pFrames[pFrameNumbers.index(frameNumber)].shape)

                        bBox = [1]
                        if bladder and bFrameNumbers is not None and frameNumber in bFrameNumbers:
                            points = [i[1:] for i in bladderBoxes if i[0] == frameNumber][0]
                            bBox += eu.getYOLOBoxes(points, bFrames[bFrameNumbers.index(frameNumber)].shape)

                        # Save image and box data to image and label directory.
                        if len(pBox) > 1 or len(bBox) > 1:
                            if len(pBox) > 1:
                                finalFrame = pFrames[pFrameNumbers.index(frameNumber)]
                            else:
                                bBox[0] = 0
                                finalFrame = bFrames[bFrameNumbers.index(frameNumber)]

                            labels = [pBox, bBox]

                            cv2.imwrite(f'{imagesPath}/P{patient}F{frameNumber}.png', finalFrame)
                            with open(f'{labelsPath}/P{patient}F{frameNumber}.txt', 'w') as labelFile:
                                for label in labels:
                                    if len(label) > 1:
                                        labelFile.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

                print(f'Patient {patient} Complete.')
            except Exception as e:
                print(f'Error creating YOLO {scanPlane} AUS data for patient {patient}.', e)

    def exportYOLOfornnUNetAUS(self, scanPlane: str):
        """Export YOLO masks for use in nnUNet inference, does not include nnUNet label mask."""
        print(f'\tExporting {scanPlane} AUS frames for nn-UNet + YOLO inference:', end=' ')
        sp = scanPlane[0].lower()
        # Get Save prefix.
        dlg = ExportDialogs().nnUNetYOLODialog(scanPlane)
        if not dlg:
            print(f'\tCreate {scanPlane} nnUNet + YOLO Data Cancelled.')
            return
        prefix, prostate, bladder, exportName = dlg
        # Create directories for AUS training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(scanPlane, prefix, exportName)
        print(imagesPath)
        # Create training data from each patient.

        for patient in self.patients:
            try:
                print(f'\t\tPatient {patient}...', end=' ')
                scanPath = f'{self.scansPath}/{patient}/AUS/{scanPlane}'
                scanDirs = Path(scanPath).iterdir()
                for scan in scanDirs:
                    # Some files are .avi, they can be skipped.
                    if scan.is_file():
                        continue
                    # Get save directory with given prefix.
                    scanPath = scan.as_posix()
                    saveDir = eu.getSaveDirName(scanPath, prefix)
                    if not saveDir:
                        print(f"{prefix} not found in {'/'.join(scanPath.split('/')[-4:])}, skipping.", end=' ')
                        continue
                    # Path to PointData.json file.
                    pointDataPath = f'{scanPath}/Save Data/{saveDir}/PointData.json'
                    # Get box data from file.
                    _, _, prostateBoxes, bladderBoxes = eu.getPointAndBoxData(pointDataPath)
                    if prostateBoxes is None:
                        print(f'No frames with prostate boxes...', end=' ')
                    if bladderBoxes is None:
                        print(f'No frames with bladder boxes...', end=' ')
                    # Get frames with prostate boxes and bladder boxes.
                    pFrames, pFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, prostateBoxes)
                    bFrames, bFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, bladderBoxes)
                    # Combine frame lists into one.
                    framesWithBoxes = []
                    framesWithBoxes += pFrameNumbers if pFrameNumbers is not None else []
                    framesWithBoxes += bFrameNumbers if bFrameNumbers is not None else []
                    # Remove duplicates.
                    framesWithBoxes = list(dict.fromkeys(framesWithBoxes))
                    # Loop through frames with boxes on them, gather points for mask creation.
                    for frameNumber in framesWithBoxes:
                        pBox = [0]
                        if prostate and pFrameNumbers is not None and frameNumber in pFrameNumbers:
                            points = [i[1:] for i in prostateBoxes if i[0] == frameNumber][0]
                            pBox += points

                        bBox = [1]
                        if bladder and bFrameNumbers is not None and frameNumber in bFrameNumbers:
                            points = [i[1:] for i in bladderBoxes if i[0] == frameNumber][0]
                            bBox += points

                        mask = np.zeros(pFrames[pFrameNumbers.index(frameNumber)].shape if len(pBox) > 1 else
                                        bFrames[bFrameNumbers.index(frameNumber)].shape)

                        labels = [pBox, bBox]

                        boxMask = eu.getYOLOMasks(labels, mask)

                        if len(pBox) > 1:
                            finalFrame = pFrames[pFrameNumbers.index(frameNumber)]
                        else:
                            finalFrame = bFrames[bFrameNumbers.index(frameNumber)]

                        cv2.imwrite(f'{imagesPath}/{sp}_P{patient}F{frameNumber}_0001.png', boxMask)
                        cv2.imwrite(f'{imagesPath}/{sp}_P{patient}F{frameNumber}_0000.png', finalFrame)

                print(f'Patient {patient} Complete.')
            except Exception as e:
                print(f'Error creating nnUNet + YOLO {scanPlane} AUS data for patient {patient}.', e)

    def exportnnUNetAUSData(self, scanPlane: str):
        """Export AUS frames for nn-Unet inference, either sagittal or transverse."""
        print(f'\tExporting {scanPlane} AUS frames for nn-UNet inference:', end=' ')
        sp = scanPlane[0].lower()
        # Get Save prefix.
        dlg = ExportDialogs().nnUNetDialog(scanPlane)
        if not dlg:
            print(f'\tCreate {scanPlane} nnUNet Data Cancelled.')
            return
        prefix, prostate, bladder, exportName = dlg
        # Create directories for AUS training data.
        imagesPath, labelsPath = eu.creatennUAUSTrainingDirs(scanPlane, prefix, exportName)
        print(imagesPath)
        # Create training data from each patient.
        for patient in self.patients:
            try:
                print(f'\t\tPatient {patient}...', end=' ')
                scanPath = f'{self.scansPath}/{patient}/AUS/{scanPlane}'
                scanDirs = Path(scanPath).iterdir()
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
                    prostatePoints, bladderPoints, _, _ = eu.getPointAndBoxData(pointDataPath)
                    if prostatePoints is None:
                        print(f'No frames with prostate points...', end=' ')
                    if bladderPoints is None:
                        print(f'No frames with bladder points...', end=' ')
                    # Get frames with prostate points and bladder points.
                    pFrames, pFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, prostatePoints)
                    # Get frames with prostate points and bladder points.
                    bFrames, bFrameNumbers = eu.getFramesWithPointsOrBoxes(scanPath, bladderPoints)
                    # Combine frame lists into one.
                    framesWithPoints = []
                    framesWithPoints += pFrameNumbers if pFrameNumbers is not None else []
                    framesWithPoints += bFrameNumbers if bFrameNumbers is not None else []
                    # Remove duplicates.
                    framesWithPoints = list(dict.fromkeys(framesWithPoints))
                    # Loop through frames with points on them, gather points for mask creation.
                    for frameNumber in framesWithPoints:
                        pMask = None
                        if prostate and pFrameNumbers is not None and frameNumber in pFrameNumbers:
                            polygon = [[i[1], i[2]] for i in prostatePoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = pFrames[pFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=1)
                            pMask = np.array(img)
                        bMask = None
                        if bladder and bFrameNumbers is not None and frameNumber in bFrameNumbers:
                            polygon = [[i[1], i[2]] for i in bladderPoints if i[0] == frameNumber]
                            polygon = [(i[0], i[1]) for i in Utils.distributePoints(polygon, len(polygon))]
                            frameShape = bFrames[bFrameNumbers.index(frameNumber)].shape
                            img = Image.new('L', (frameShape[1], frameShape[0]))
                            ImageDraw.Draw(img).polygon(polygon, fill=2 if pMask is not None else 1)
                            bMask = np.array(img)

                        # Combine masks and make overlaps only equal to prostate (prostate takes precedence).
                        finalMask = cv2.bitwise_or(pMask if pMask is not None else np.zeros_like(bMask),
                                                   bMask if bMask is not None else np.zeros_like(pMask))
                        finalMask[finalMask > 2] = 1

                        if pMask is not None:
                            finalFrame = pFrames[pFrameNumbers.index(frameNumber)]
                        else:
                            finalFrame = bFrames[bFrameNumbers.index(frameNumber)]

                        cv2.imwrite(f'{imagesPath}/{sp}_P{patient}F{frameNumber}_0000.png', finalFrame)
                        cv2.imwrite(f'{labelsPath}/{sp}_P{patient}F{frameNumber}.png', finalMask)
                    print('Complete.')
            except WindowsError as e:
                print(f'Error creating nnUNet {scanPlane} AUS data for patient {patient}.', e)

    def _exportIPVAUSSagittalData(self):
        """Export AUS sagittal frames for ipv inference."""
        print(f'\tExporting Sagittal frames for IPV inference...')
        # Get Export Settings.
        dlg = ExportDialogs().IPVDialog()
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
                print(f'\tError creating IPV sagittal data for patient {patient}.', e)
        print(f'\tAUS sagittal exporting completed.')

    def _exportIPVAUSTransverseData(self, mainWindow: QWidget):
        """Export AUS transverse frames for ipv inference."""
        print(f'\tExporting Transverse frames for IPV inference...')
        # Get Export Settings.
        dlg = ExportDialogs('IPV', 'Sagittal').executeDialog()
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
                print(f'\tError creating IPV transverse data for patient {patient}.', e)
        print(f'\tAUS transverse exporting completed.')
