import csv
import os

import cv2
import numpy as np
from matplotlib.axes import Axes
from natsort import natsorted

from classes import Scan


def loadFrames(scanPath: str):
    """
    Load all .png images in recording_path as frames into a multidimensional list/

    :param scanPath: String representation of the recording path.

    :return frames: List of all the .png frames saved in the recording path directory.
    """
    # All files and subdirectories at given path.
    allContents = natsorted(os.listdir(scanPath))

    frames = []

    for file in allContents:
        if file.split('.')[-1] == 'png':
            img = cv2.imread(f'{scanPath}/{file}', cv2.IMREAD_UNCHANGED)
            frames.append(img)
    return frames


def getScanType(scanPath: str):
    """
    Return the type of scan based on scan parent directory.

    :param scanPath: String representation of the scan path.

    :return scan_type: Type of scan.
    """

    scan_type = scanPath.split('/')[-2]

    if scan_type == 'Transverse':
        scan_type = Scan.TYPE_TRANSVERSE
    elif scan_type == 'Sagittal':
        scan_type = Scan.TYPE_SAGITTAL

    return scan_type


def drawFrameOnAxis(axis: Axes, frame: np.ndarray):
    """
    Clear the given axis and plot a new frame with imshow. Enforce axis limits.

    :param axis: Axis used to display frame.
    :param frame: Frame to be drawn on axis.
    """
    axis.cla()
    axis.axis('off')
    axis.imshow(frame, cmap='gray')
    axis.set_xlim(-0.5, frame.shape[1])
    axis.set_ylim(frame.shape[0], -0.5)


def drawScanDataOnAxis(axis: Axes, frame: np.ndarray, fNo: int, fCount: int, depths: list, imuOff: float, imuPos: float,
                       dd: list):
    """
    Function for plotting extra details about the scan on the frame.

    :param axis: Axis displaying frame.
    :param frame: Frame currently displayed.
    :param fNo: Current frame number.
    :param fCount: Total number of frames in scan.
    :param depths: Height and Width of scan in mm.
    :param imuOff: Offset of the imu from the end of the probe.
    :param imuPos: IMU position as a percent of frame width.
    :param dd: Display dimensions - shape of the frame, first and second value swapped.
    """
    # Scan width and depth in mm.
    axis.text(dd[0] - 80, 30, f'{int(depths[1])}mm', color='white')
    axis.text(dd[0] - 30, 90, f'{int(depths[0])}mm', color='white', rotation=-90)
    # IMU offset and position details.
    axis.text(20, 40, f'IMU Offset: {imuOff:.1f}mm', color='blue')
    axis.text(20, 60, f'IMU Position: {imuPos:.1f}%', color='blue')
    # Current frame number over total frames.
    axis.text(dd[0] - 70, frame.shape[0] - 20, f'{fNo}/{fCount}', color='white')
    # Scan position indicator.
    axis.plot([dd[0] - 20, dd[0] - 20], [dd[1] - 40, dd[1] - 240], color='white', linewidth=1)
    axis.plot([dd[0] - 22, dd[0] - 16], [dd[1] - 40 - 201 * (fNo - 1) / fCount, dd[1] - 40 - 201 * (fNo - 1) / fCount],
              color='white')
    # Location of IMU in relation to width, a percentage of width.
    axis.plot([imuPos / 100 * dd[0], imuPos / 100 * dd[0]], [0, 10], color='white', linewidth=2)


def getIMUDataFromFile(scanPath: str):
    """
     Helper function for acquiring information from the data.txt file. This includes:
            accelerations   -       All acceleration values stored during recording.
            quaternions     -       All quaternion values stored during recording.
            imu_count       -       Total rows in the file.
            duration        -       Duration of recording based on first and last frame names.
    If there is no data file, all returned values will be None.

    :param scanPath: String representation of the scan path.

    :return accelerations: List of accelerations.
    :return quaternions: List of quaternions.
    :return depths: List of scan depths.
    :return duration: Duration of recording.
    """
    names = []
    accelerations = []
    quaternions = []
    depths = []
    # Information acquired from the IMU data.txt file.
    with open(scanPath + '/data.txt', 'r') as dataFile:
        reader = csv.reader(dataFile)
        for row in reader:
            names.append(row[0])
            accelerations.append([float(row[2]), float(row[3]), float(row[4])])
            quaternions.append([float(row[6]), float(row[7]), float(row[8]), float(row[9])])
            depths.append([float(row[14]), float(row[15])])
        duration = int(names[-1].split('-')[1].split('.')[0]) - int(names[0].split('-')[1].split('.')[0])
        if duration == 0:
            duration = 1

    return accelerations, quaternions, depths, duration
