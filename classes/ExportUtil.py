import os
import time
from pathlib import Path

from classes import Scan


def createTrainingDirs(scanType: str):
    """
    Create directories using training structure.

    Args:
        scanType: Type of scan.

    Returns:
        String to main directory if successful, else False

    """
    # Create new, empty directories.
    try:
        dataPath = f'../Export/IPV/{int(time.time())}/DATA'
        Path(f'{dataPath}/fold_lists').mkdir(parents=True, exist_ok=False)
        if scanType == Scan.TYPE_TRANSVERSE:
            Path(f'{dataPath}/Transverse').mkdir(exist_ok=False)
        else:
            Path(f'{dataPath}/Sagittal').mkdir(exist_ok=False)
    except FileExistsError as e:
        print(f'Error creating directories: {e}.')
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
