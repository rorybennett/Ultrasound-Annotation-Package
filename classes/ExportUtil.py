import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from classes import Scan
from classes.ErrorDialog import ErrorDialog


def creatennUAUSTrainingDirs(scanPlane: str, saveName: str, exportName: str):
    """
    Create directories using nn-UNet training structure.

    Args:
        scanPlane: Plane of scan.
        saveName: Save name used in SaveData.
        exportName: Name to be included in folder.

    Returns:
        String to main directory if successful, else False.
    """
    # Create new, empty directories.
    try:
        dataPath = f"Export/nnUNet/{saveName}/{int(time.time())}_{scanPlane[0].lower()}AUS{exportName if exportName else '-'}"
        imagesPath = Path(f'{dataPath}/imagesTr')
        imagesPath.mkdir(parents=True, exist_ok=False)
        labelsPath = Path(f'{dataPath}/labelsTr')
        labelsPath.mkdir(parents=True, exist_ok=False)
        return imagesPath, labelsPath
    except FileExistsError as e:
        ErrorDialog(None, 'Error creating nnUNet export directories.', e)
        return False


def getYOLOBoxes(points, frameShape):
    """
    Convert points (top_left, bottom_right) of a rectangle into normalised [centre_x, centre_y, width, height].

    Parameters
    ----------
    points: Top left and bottom right points of rectangle.
    frameShape: Shape of frame, used for normalising.

    Returns
    -------
    centre_x, centre_y, width, height
    """
    a = points[:2]
    b = points[2:]

    centre = [((a[0] + b[0]) / 2) / frameShape[1], ((a[1] + b[1]) / 2) / frameShape[0]]
    c = (b[0] - a[0]) / frameShape[1]
    d = (b[1] - a[1]) / frameShape[0]

    return [centre[0], centre[1], c, d]


def createYOLOTrainingDirs(scanType: str, saveName: str, exportName: str):
    """
    Create training directories using YOLO training structure.

    Parameters
    ----------
    scanType: Plane of Scan, Transverse or Sagittal
    saveName: Name used in SaveData.
    exportName: Main directory name, can be blank.

    Returns
    -------
    Paths to images and labels directories.
    """
    # Create new, empty directories.
    try:
        parentPath = Path(f"Export/YOLO/{saveName}/{int(time.time())}_{exportName if exportName else '-'}_{scanType}")
        imagesPath = Path(parentPath, 'images')
        imagesPath.mkdir(parents=True, exist_ok=False)
        labelsPath = Path(parentPath, 'labels')
        labelsPath.mkdir(parents=True, exist_ok=False)
        return imagesPath, labelsPath
    except Exception as e:
        ErrorDialog(None, 'Error creating YOLO export directories.', e)
        return False


def createIPVTrainingDirs(scanPlane: str):
    """
    Create directories using IPV training structure.

    Args:
        scanPlane: Plane of scan.

    Returns:
        String to main directory if successful, else False.
    """
    # Create new, empty directories.
    try:
        dataPath = f'Export/IPV/{int(time.time())}_IPV_{scanPlane}_export/DATA'
        Path(f'{dataPath}/fold_lists').mkdir(parents=True, exist_ok=False)
        if scanPlane == Scan.PLANE_TRANSVERSE:
            Path(f'{dataPath}/transverse').mkdir(exist_ok=False)
        else:
            Path(f'{dataPath}/sagittal').mkdir(exist_ok=False)
    except FileExistsError as e:
        ErrorDialog(None, 'Error creating IPV export directories.', e)
        return False
    return dataPath


def getTotalPatients(scansPath: str):
    """
    Count the number of folders at the given path, which will return the total number of patients.

    Args:
        scansPath: Path to the Scans directory as a String.

    Returns:
        totalPatients: Total number of patients.
    """
    totalPatients = 0
    for i in os.listdir(Path(scansPath)):
        if os.path.isdir(Path(scansPath, i)):
            totalPatients += 1
    return totalPatients


def getSaveDirName(scanPath: str, prefix: str):
    """
    Get name of the save data directory given the prefix (the timestamp is unknown). Will only return the first match.

    Args:
        scanPath: Path to Scan directory (including Scan time directory).
        prefix: Save prefix.

    Returns:
        Either String Path to save directory with given prefix, or False.
    """
    try:
        savePath = f'{scanPath}/Save Data'
        saveDirs = os.listdir(savePath)

        for saveDir in saveDirs:
            pre = ''.join(saveDir.split('_')[:-1])

            if pre == prefix:
                return saveDir
    except WindowsError:
        return False


def getPointAndBoxData(filePath: str):
    """
    Get point and box data from given file path.  The data is not always sorted in order as that was only added later,
    so it is sorted by frame name here.

    Parameters
    ----------
    filePath: Path to PointData.json file.

    Returns
    -------
    Prostate and bladder points and boxes.
    """
    with open(filePath, newline='\n') as pointFile:
        data = json.load(pointFile)
        prostatePoints = data.get('Prostate')
        bladderPoints = data.get('Bladder')
        prostateBoxes = data.get('ProstateBox')
        bladderBoxes = data.get('BladderBox')

    if prostatePoints is not None:
        prostatePoints.sort(key=lambda row: ([row[0]]))

    if bladderPoints is not None:
        bladderPoints.sort(key=lambda row: ([row[0]]))

    if prostateBoxes is not None:
        prostateBoxes.sort(key=lambda row: ([row[0]]))

    if bladderBoxes is not None:
        bladderBoxes.sort(key=lambda row: ([row[0]]))

    return prostatePoints, bladderPoints, prostateBoxes, bladderBoxes


def getDepths(scanPath: str, frameNumber: int):
    """
    Get the depth dimensions of a scan [height, width].

    Args:
        scanPath: Path as String to Scan directory.
        frameNumber: Frame number.

    Returns:
        depths: List of depth dimensions.
    """
    with open(f'{scanPath}/data.txt', 'r') as dataFile:
        lines = dataFile.readlines()
        depths = [int(lines[frameNumber - 1].split(',')[-3]), int(lines[frameNumber - 1].split(',')[-2])]

    return depths


def getIMUData(saveDir):
    """
    Get the imu offset and position from the EditingData.txt file of the save directory.

    Args:
        saveDir: Path (str) to save directory.

    Returns:
        Imu offset and position as floats.
    """
    with open(f'{saveDir}/EditingData.txt', 'r') as editingFile:
        lines = editingFile.readlines()
        imuOffset = float(lines[0].split(':')[-1])
        imuPosition = float(lines[1].split(':')[-1])
    return imuOffset, imuPosition


def getFramesWithPointsOrBoxes(scanPath, pointData):
    """
    Return frames in specified recording path that contain points or boxes on them. Frames are returned as grayscale images.

    Args:
        scanPath: Path (str) to scan (directory in scan type).
        pointData: Point data with file names, used to find frames to return.

    Returns:
        frames: np.array of frames corresponding to point data frame names, else False.
    """
    if pointData is None or len(pointData) == 0:
        return None, None

    # Remove duplicates
    frame_names = [row[0] for row in pointData]
    frame_names = list(dict.fromkeys(frame_names))

    # Read frames into array
    frames = []
    frameNumbers = []
    for row in frame_names:
        framePath = scanPath + '/' + row + '.png'
        frames.append(cv2.imread(framePath, cv2.IMREAD_GRAYSCALE))
        frameNumbers.append(row.split('.')[0])

    return frames, frameNumbers


def createSaveDataExportDir():
    """
    Create a directory where all patient Save Data can be stored.

    Returns:
        path: Path (str) to newly created directory, else False if there was a problem.
    """
    try:
        path = f'../Export/Save Data Exports/{int(time.time())}_save_data_export'
        Path(f'{path}').mkdir(parents=True, exist_ok=False)
    except Exception as e:
        ErrorDialog(None, f'Error creating Save Data Export Directory.', e)
        return False
    return path


def resampleImageAndPoints(frame: np.ndarray, scanDimensions, points, density=4):
    """
    Resample a given image to the required pixel density.

    Args:
        frame: Image to be resampled.
        scanDimensions: Scan dimensions (height, width)
        points: Points to be resampled.
        density: Required pixel density (pixels / mm)

    Returns:
        Resampled/Resized frame and resampled points.
    """
    oldShape = frame.shape
    # Resample Frame.
    newHeight = scanDimensions[0] * density
    newWidth = scanDimensions[1] * density

    frame = cv2.resize(frame, dsize=(newWidth, newHeight), interpolation=cv2.INTER_CUBIC)

    # Resample Points.
    resampledPoints = []
    newShape = frame.shape
    for point in points:
        # x value.
        x = int(float(point[1])) / oldShape[1] * newShape[1]
        # y value.
        y = int(float(point[2])) / oldShape[0] * newShape[0]
        resampledPoints.append([point[0], round(x), round(y)])

    return frame, resampledPoints
