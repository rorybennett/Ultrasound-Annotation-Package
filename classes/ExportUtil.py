import csv
import os
import time
from pathlib import Path

import cv2

from classes import Scan
from classes.ErrorDialog import ErrorDialog


def createIPVTrainingDirs(scanType: str):
    """
    Create directories using training structure.

    Args:
        scanType: Type of scan.

    Returns:
        String to main directory if successful, else False

    """
    # Create new, empty directories.
    try:
        dataPath = f'../Export/IPV/{int(time.time())}_IPV_{scanType}_export/DATA'
        Path(f'{dataPath}/fold_lists').mkdir(parents=True, exist_ok=False)
        if scanType == Scan.PLANE_TRANSVERSE:
            Path(f'{dataPath}/transverse').mkdir(exist_ok=False)
        else:
            Path(f'{dataPath}/sagittal').mkdir(exist_ok=False)
    except FileExistsError as e:
        ErrorDialog(None, 'Error creating directories', e)
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
    Get name of the save data directory given the prefix (the timestamp is unknown).

    Args:
        scanPath: Path to Scan directory (including Scan time directory).
        prefix: Save prefix.

    Returns:
        Either String Path to save directory with given prefix, or False.
    """
    savePath = f'{scanPath}/Save Data'
    saveDirs = os.listdir(savePath)

    for saveDir in saveDirs:
        pre = ''.join(saveDir.split('_')[:-1])

        if pre == prefix:
            return saveDir

    return False


def getPointData(filePath: str):
    """
        Extract frame name and point data from given filePath (str). The point data is not always sorted in order as
        that was only added later, so it is sorted by frame name here.

        Args:
            filePath: Path to file, as a string, including file type.

        Returns:
            pointData: list of file names and associated point data, sorted by file name for grouping reasons.
        """
    with open(filePath, newline='\n') as pointFile:
        pointData = list(csv.reader(pointFile))

    for i, r in enumerate(pointData):
        pointData[i] = [r[0], round(float(r[1])), round(float(r[2]))]

    pointData.sort(key=lambda row: ([row[0]]))

    return pointData


def getDepths(scanPath: str):
    """
    Get the depth dimensions of a scan [height, width]

    Args:
        scanPath: Path as String to Scan directory.

    Returns:
        depths: List of depth dimensions.
    """
    with open(f'{scanPath}/data.txt', 'r') as dataFile:
        lines = dataFile.readlines()
        depths = [int(lines[0].split(',')[-3]), int(lines[0].split(',')[-2])]

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


def getFramesWithPoints(scanPath, pointData):
    """
    Return frames in specified recording path that contain points on them.

    Args:
        scanPath: Path (str) to scan (directory in scan type).
        pointData: Point data with file names, used to find frames to return.

    Returns:
        frames: np.array of frames corresponding to point data frame names, else False.
    """
    if len(pointData) == 0:
        return False

    # Remove duplicates
    frame_names = [row[0] for row in pointData]
    frame_names = list(dict.fromkeys(frame_names))

    # Read frames into array
    frames = []
    frameNumbers = []
    for row in frame_names:
        framePath = scanPath + '/' + row + '.png'
        frames.append(cv2.imread(framePath))
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
        ErrorDialog(None, f'Error creating Save Data Export Directory', e)
        return False
    return path
