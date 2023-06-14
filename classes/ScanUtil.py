import csv
import os
from pathlib import Path

import cv2
import numpy as np
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from natsort import natsorted

from classes import Scan

# Rotates the '+' marker by 45 degrees.
m = MarkerStyle('+')
m._transform.rotate_deg(45)


def loadFrames(scanPath: str):
    """
    Load all .png images in recording_path as frames into a multidimensional list.

    Args:
        scanPath: String representation of the recording path.

    Returns:
        List of all the .png frames saved in the recording path directory.
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
    Get the type of scan based on scan parent directory.

    Args:
        scanPath: String representation of the scan path.

    Returns:
        Type of scan as a String.
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

    Args:
        axis: Axis used to display frame.
        frame: Frame to be drawn on axis.

    Returns:
        Nothing returned as data is drawn directly on axis.
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

    Args:
        axis: Axis displaying frame.
        frame: Frame currently displayed.
        fNo: Current frame number.
        fCount: Total number of frames in scan.
        depths: Height and Width of scan in mm.
        imuOff: Offset of the imu from the end of the probe.
        imuPos: IMU position as a percent of frame width.
        dd: Display dimensions - shape of the frame, first and second value swapped.

    Returns:
        Nothing returned as data is drawn directly on axis.
    """
    # Scan width and depth in mm.
    axis.text(dd[0] - 120, 30, f'<- {int(depths[1])}mm ->', color='white')
    axis.text(dd[0] - 30, 130, f'<- {int(depths[0])}mm ->', color='white', rotation=-90)
    # IMU offset and position details.
    axis.text(20, 40, f'IMU Offset: {imuOff:.1f}mm', color='lightblue')
    axis.text(20, 60, f'IMU Position: {imuPos:.1f}%', color='lightblue')
    # Current frame number over total frames.
    axis.text(dd[0] - 120, frame.shape[0] - 20, f'Frame {fNo} of {fCount}', color='white')
    # Scan position indicator.
    axis.plot([dd[0] - 20, dd[0] - 20], [dd[1] - 40, dd[1] - 240], color='white', linewidth=1)
    axis.plot([dd[0] - 22, dd[0] - 16], [dd[1] - 40 - 201 * (fNo - 1) / fCount, dd[1] - 40 - 201 * (fNo - 1) / fCount],
              color='white')
    # Location of IMU in relation to width, a percentage of width.
    axis.plot([imuPos / 100 * dd[0], imuPos / 100 * dd[0]], [0, 10], color='white', linewidth=2)


def getIMUDataFromFile(scanPath: str):
    """
     Helper function for acquiring information from the data.txt file. This includes:
            accelerations   -       All acceleration values stored during recording.\n
            quaternions     -       All quaternion values stored during recording.\n
            imu_count       -       Total rows in the file.\n
            duration        -       Duration of recording based on first and last frame names.
    If there is no data file, all returned values will be None.

    Args:
        scanPath: String representation of the scan path.

    Returns:
         List of frame names, list of accelerations, list of quaternions, list of scan depths, duration of recording.
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

    return names, accelerations, quaternions, depths, duration


def getEditDataFromFile(scanPath: str):
    """
    Helper function to get any editing details that are already stored. If there is no file it is created. Values
    stored in the file:
            imuOffset      ->      Offset between probe end and IMU.\n
            imuPosition    ->      Position of IMU as percentage of width of scan.

    Args:
        scanPath: String representation of the scan path.

    Returns:
        Path to EditingData.txt file, imuOffset, imuPosition.
    """
    editPath = checkEditDataFile(scanPath)
    imuOffset = 0
    imuPosition = 50

    with open(editPath, 'r') as editFile:
        for line in editFile.readlines():
            lineSplit = line.split(':')
            parameter = lineSplit[0]
            value = lineSplit[1]

            if parameter == 'imuOffset':
                imuOffset = float(value)
            elif parameter == 'imuPosition':
                imuPosition = float(value)
                if imuPosition == 0:
                    print('IMU Position is 0, changing it to 50.')
                    imuPosition = 50

        return editPath, imuOffset, imuPosition


def checkEditDataFile(scanPath: str) -> Path:
    """
    Check if the given folder contains an EditingData.txt file, if True return a Path object to it, else create
    the file and return a Path object to it. When creating the file add some default values to it.

    Args:
        scanPath: String representation of the scan path.

    Returns:
        Path object to existing or newly created EditingData.txt file.
    """
    editPath = Path(scanPath, 'EditingData.txt')
    if not editPath.is_file():
        print(f'EditingData.txt does not exist, creating now.')
        with open(editPath, 'a') as editingFile:
            editingFile.write('imuOffset:0\n')
            editingFile.write('imuPosition:50\n')

    return editPath


def getPointDataFromFile(scanPath: str):
    """
        Helper function to get point data that has already been saved. If there is no file it is created. The
        frame name and coordinates of a point are stored on each line. The coordinates are stored as x-, and y-values
        that represent the location of the point in real-world coordinates (mm).

        Args:
            scanPath: String representation of the scan path.

        Returns:
            Path object to existing or newly created PointData.txt file, list of points placed in current scan.
        """
    pointPath = checkPointDataFile(scanPath)

    pointData = []

    with open(pointPath, 'r') as pointFile:
        for line in pointFile.readlines():
            lineSplit = line.split(',')
            pointData.append([lineSplit[0], float(lineSplit[1]), float(lineSplit[2])])

    return pointPath, pointData


def checkPointDataFile(scanPath: str):
    """
        Check if the given directory contains a PointData.txt file, if True return a Path object to it, else create
        the file and return a Path object to it.

        Args:
            scanPath: Path to a recording directory.

        Returns:
            Path object to existing or newly created PointData.txt file.
        """
    pointPath = Path(scanPath, 'PointData.txt')
    if not pointPath.is_file():
        with open(pointPath, 'a'):
            pass

    return pointPath


def drawPointDataOnAxis(axis, points, depths, imuOffset, imuPosition, dd):
    """
    Plot given points on the frame. Points are currently stored in mm (including the IMU offset from the edge
    of the probe). These need to be converted to display coordinates.

    Args:
        axis: Axis displaying frame.
        points: List of points in mm.
        depths: Height and Width of recording in mm.
        imuOffset: Offset of the imu from the end of the probe.
        imuPosition: IMU position as a percent of frame width.
        dd: Display dimension - shape of the frame, first and second value swapped.

    Returns:
        Nothing returned as data is drawn directly on axis.
    """
    for point in points:
        point_display = mmToDisplay(point, depths, imuOffset, imuPosition, dd)

        axis.plot(point_display[0], point_display[1], marker=m, color='lime')


def mmToDisplay(pointMM: list, depths: list, imuOffset: float, imuPosition: float, dd: list):
    """
    Convert a point in mm to a point in the display coordinates. First convert the point in mm to a display ratio,
    then to display coordinates.

    Args:
        pointMM: x and y coordinates of the point in mm.
        depths: Scan depth.
        imuOffset: IMU offset.
        imuPosition: Position of IMU shown by ticks.
        dd: Display dimensions, based on frame.

    Returns:
        Point coordinates in display dimensions.
    """
    pointDisplay = ratioToDisplay(mmToRatio(pointMM, depths, imuOffset, imuPosition),
                                  dd)

    return pointDisplay


def mmToRatio(pointMm: list, depths: list, imuOffset: float, imuPosition: float):
    """
    Convert the given point in mm to a display ratio. Calculated using the imu offset and the depths of the scan. Remove
    the IMU offset in the y direction.

    Args:
        pointMm: Point as x and y coordinates in mm.
        depths : Depth and width of scan, used to get the point ratio.
        imuOffset: IMU Offset.
        imuPosition: Position of IMU shown by ticks.

    Returns:
        x and y coordinates of the point as a ratio of the display.
    """
    point_ratio = [(pointMm[0] + depths[1] / (depths[1] * imuPosition / 100)) / depths[1],
                   (pointMm[1] - imuOffset) / depths[0]]

    return point_ratio


def ratioToDisplay(pointRatio: list, dd: list):
    """
    Convert a point given as a ratio of the display dimensions to display coordinates. Rounding is done as display
    coordinates have to be integers.

    Args:
        pointRatio: Width and Height ratio of a point in relation to the display dimensions.
        dd: Display dimensions, based on frame.

    Returns:
        Point coordinates in display dimensions (int rounding).
    """
    pointDisplay = [int(pointRatio[0] * dd[0]),
                    int(pointRatio[1] * dd[1])]

    return pointDisplay


def displayToMm(pointDisplay: list, depths: list, imuOffset: float, imuPosition: int, dd: list):
    """
    Convert a point in display coordinates to mm. Include the offset in the y direction.

    Args:
        pointDisplay (list): x and y coordinates of the point in display dimensions.
        depths (list): Scan depth.
        imuOffset (float): IMU offset.
        imuPosition (int): Position of IMU shown by ticks.
        dd (list): Dimensions of display, based on frame.

    Returns:
        point_mm (list): Point coordinates in mm.
    """
    point_ratio = displayToRatio(pointDisplay, dd)

    # Point in mm using depth of scan as reference.
    point_mm = [point_ratio[0] * depths[1] - (depths[1] / (depths[1] * (imuPosition / 100))),
                point_ratio[1] * depths[0] + imuOffset]

    return point_mm


def displayToRatio(pointDisplay: list, dd: list):
    """
    Convert a point in display coordinates into a ratio of the display dimensions.

    Args:
        pointDisplay (list): x and y coordinates of the point in display dimensions.
        dd (list): Dimensions of frame being displayed.

    Returns:
        pointRatio (list): Point as a ratio of the display dimensions.
    """
    pointRatio = [pointDisplay[0] / dd[0], (dd[1] - pointDisplay[1]) / dd[1]]

    return pointRatio

def pointInRadius(centre: list, point: list, radius: int) -> bool:
    """
    Checks whether the point is within the constant radius of the centre, if it is return True, else False.

    Args:
        centre (float, float): Location of centre point, as a percentage.
        point (float, float): Location of test point, as a percentage.

    Returns:
        withinRadius (bool): True if within radius, else False.
    """
    withinRadius = False

    if (point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2 < radius ** 2:
        withinRadius = True

    return withinRadius
