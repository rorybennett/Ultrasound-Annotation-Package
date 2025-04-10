"""
Utils used by the Scan class.
"""

import csv
import json
import math
import os
import stat
from pathlib import Path

import cv2
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Polygon
from natsort import natsorted
from numpy.lib.stride_tricks import sliding_window_view
from pyquaternion import Quaternion
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, Bounds
from shapely import MultiLineString
from shapely.affinity import scale
from shapely.geometry import LineString, Point
from shapely.geometry import Polygon as PolygonShapely

from classes.ErrorDialog import ErrorDialog

# Rotates the '+' marker by 45 degrees. Could probably use 'x' instead.
m = MarkerStyle('+')
m._transform.rotate_deg(45)

# Bullet marking colours.
# Colours used by axis plots for plane data.
BULLET_COL = {
    'L1': (1, 1, 1),
    'L2': (1, 1, 1),
    'W1': (1, 0, 0),
    'W2': (1, 0, 0),
    'H1': (.2, 1, 1),
    'H2': (.2, 1, 1)
}


def loadFrames(scanPath: str):
    """
    Load all .png images in scanPath as frames into a multidimensional list.

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
    axis.set_xlim(-.1, frame.shape[1])
    axis.set_ylim(frame.shape[0], -0.5)


def drawScanDataOnAxis(axis: Axes, frameNumber: int, frameCount: int, depths: list, imuOff: float, imuPos: float,
                       frameProstatePoints: int, frameBladderPoints: int, totalProstatePoints: int,
                       totalBladderPoints: int, totalProstateBoxes: int, totalBladderBoxes: int, dd: list):
    """
    Plot extra details about the Scan on the frame.

    Parameters
    ----------
    axis: Axis displaying frame.
    frameNumber: Current frame number.
    frameCount: Total number of frames in scan.
    depths: Height and Width of scan in mm.
    imuOff: Offset of the imu from the end of the probe.
    imuPos: IMU position as a percent of frame width.
    frameProstatePoints: Total prostate points on current frame.
    frameBladderPoints: Total bladder points on current frame.
    totalProstatePoints: Total prostate points on all frames in Scan.
    totalBladderPoints: Total bladder points on all frames in Scan.
    totalProstateBoxes: Total prostate bounding boxes in Scan.
    totalBladderBoxes: Total bladder bounding boxes in Scan.
    dd: Display dimensions - shape of the frame, first and second value swapped.

    Returns
    -------
    None
    """
    # Scan width and depth in mm.
    axis.text(dd[0] - 120, 30, f'<- {int(depths[1])}mm ->', color='white')
    axis.text(dd[0] - 30, 130, f'<- {int(depths[0])}mm ->', color='white', rotation=-90)
    # IMU offset and position details.
    axis.text(20, 40, f'IMU Offset: {imuOff:.1f}mm', color='lightblue')
    axis.text(20, 60, f'IMU Position: {imuPos:.1f}%', color='lightblue')
    # Current frame number over total frames.
    axis.text(dd[0] - 150, dd[1] - 20, f'Frame {frameNumber} of {frameCount}', color='white')
    # Location of IMU in relation to width, a percentage of width.
    axis.plot([imuPos / 100 * dd[0], imuPos / 100 * dd[0]], [0, 10], color='white', linewidth=2)
    # Points' indicator.
    axis.text(20, dd[1] - 80, f'Prostate Points: {frameProstatePoints}/{totalProstatePoints}', color='white')
    axis.text(20, dd[1] - 60, f'Bladder Points: {frameBladderPoints}/{totalBladderPoints}', color='white')
    axis.text(20, dd[1] - 40, f'Prostate Boxes: {totalProstateBoxes}', color='white')
    axis.text(20, dd[1] - 20, f'Bladder Boxes: {totalBladderBoxes}', color='white')


def drawMaskOnAxis(axis: Axes, points: list, fd: list, dd: list, color):
    """
    Draw a polygon mask using the points given. If there are too few points the mask will not be drawn.

    Args:
        axis: Axis displaying frame.
        points: Points on the currently displayed frame.
        fd: Frame dimensions.
        dd: Display dimensions.
        color: Colour of mask.
    """
    # Convert points from frame coordinates to canvas coordinates.
    points = [pixelsToDisplay(point, fd, dd) for point in points]
    if len(points) > 1:
        polygon = Polygon(points, closed=True, alpha=0.2, color=color)
        axis.add_patch(polygon)


def drawBoxOnAxis(axis: Axes, points: list, fd: list, dd: list, color):
    """
    Draw a bounding box using the points given. Top left and bottom right.

    Args:
        axis: Axis displaying frame.
        points: Box end points for current frame
        fd: Frame dimensions.
        dd: Display dimensions.
        color: Colour of box.
    """
    # Convert points from frame coordinates to canvas coordinates.
    if len(points) > 1:
        points = [points[:2], points[2:]]
        points = [pixelsToDisplay(point, fd, dd) for point in points]
        axis.text(points[0][0], points[0][1], 'A', color=color)
        axis.text(points[1][0], points[1][1], 'B', color=color)
        rect = patches.Rectangle(points[0], points[1][0] - points[0][0], points[1][1] - points[0][1], linewidth=1,
                                 edgecolor=color, facecolor='none')
        axis.add_patch(rect)


def flipSaveData(savePath, widthHeight, direction):
    """
    Flip the points and boxes in the PointData.json file of the Save Data directories.

    Parameters
    ----------
    savePath: Path to Save Data directory.
    widthHeight: Width of frame for LR, or height of frame for UD.
    direction: Direction (LR or UD).
    """
    for saveDir in os.listdir(savePath):
        with open(f'{savePath}/{saveDir}/PointData.json', 'r') as file:
            pointData = json.load(file)

        pointData['Prostate'] = flipPoints(widthHeight, pointData['Prostate'], 'LR')
        pointData['Bladder'] = flipPoints(widthHeight, pointData['Bladder'], 'LR')
        pointData['ProstateBox'] = flipBoxes(widthHeight, pointData['ProstateBox'], 'LR')
        pointData['BladderBox'] = flipBoxes(widthHeight, pointData['BladderBox'], 'LR')

        with open(f'{savePath}/{saveDir}/PointData.json', 'w') as newFile:
            json.dump(pointData, newFile, indent=4)


def flipPoints(widthHeight, points, direction):
    """
    Flip the points in the direction given.

    Parameters
    ----------
    widthHeight: Width of frame for LR, or height of frame for UD.
    points: Points to flip, with frame number.
    direction: Direction (LR or UD).

    Returns
    -------
    Flipped points.
    """
    for i, point in enumerate(points):
        points[i] = [point[0], widthHeight - point[1], point[2]]
    return points


def flipBoxes(widthHeight, boxes, direction):
    """
    Flip the boxes in the direction given.

    Parameters
    ----------
    widthHeight: Width of frame for LR, or height of frame for UD.
    boxes: boxes to flip, with frame number.
    direction: Direction (LR or UD).

    Returns
    -------
    Flipped boxes.
    """
    for i, box in enumerate(boxes):
        boxes[i] = [box[0], widthHeight - box[3], box[2], widthHeight - box[1], box[4]]
    return boxes


def flipFrames(framePath, direction):
    """
    Flip the frames in the given directory either the Left-Right direction or the Top-Down direction.

    Parameters
    ----------
    framePath: Path to frames.
    direction: Direction (LR or UD).

    Returns
    -------
    True if frames flipped, else False.
    """
    try:
        files = natsorted(os.listdir(framePath))

        for f in files:
            if f.__contains__('.png'):
                # Read frame.
                img = cv2.imread(f'{framePath}/{f}', cv2.IMREAD_UNCHANGED)
                # Flip frame.
                imgFlipped = np.flip(img, axis=0 if direction == 'UD' else 1)
                # Save frame.
                cv2.imwrite(f'{framePath}/{f}', imgFlipped)
        return True
    except Exception:
        return False


def getBoundingBoxStartAndEnd(points):
    """
    Find the start and end of the bounding box, assuming top left is start and bottom right is end. This does not work
    as intended, it just forces a left to right, top to bottom bounding box to be drawn.
    Parameters
    ----------
    points: Current start and end points of the bounding box.

    Returns
    -------
    Start and end points of the bounding box.
    """
    # Convert 2 points into 4 corners of rectangle.
    pointsArray = np.array([[points[0], points[1]],
                            [points[0], points[3]],
                            [points[2], points[3]],
                            [points[2], points[1]]])

    # Find start point (top left: min_x then min_y).
    x_min = np.where(pointsArray[:, 0] == pointsArray[:, 0].min())[0]
    y_min = np.where(pointsArray[x_min, 1] == pointsArray[x_min, 1].min())[0]
    top_left = (pointsArray[x_min[y_min[0]]])

    # Find end point (bottom right: max_x then max_y).
    x_max = np.where(pointsArray[:, 0] == pointsArray[:, 0].max())[0]
    y_max = np.where(pointsArray[x_max, 1] == pointsArray[x_max, 1].max())[0]
    bottom_right = pointsArray[x_max[y_max[0]]]

    results = [int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])]

    return results


def drawIPVDataOnAxis(axis: Axes, ipv: dict, name: str, fd: list, dd: list):
    """
        Plot the IPV data onto the frame.

        Args:
            axis: Axis displaying frame.
            ipv: Dictionary of ipv data.
            name: Frame name as a string.
            fd: Dimensions of original frame (x, y).
            dd: Display dimension - shape of the displayed frame.
    """

    if name == ipv['centre'][0]:
        # Plot the centre circle.
        pointDisplay = pixelsToDisplay(ipv['centre'][1:], fd, dd)
        axis.plot(pointDisplay[0], pointDisplay[1], marker='+', color='white', markersize=5)
        circle = plt.Circle((pointDisplay[0], pointDisplay[1]), ipv['radius'], fill=False, color='white',
                            linestyle='--')
        axis.add_artist(circle)
    fd = [fd[1], fd[0]]
    if name == ipv['inferred_points'][0]:
        for point in ipv['inferred_points'][1]:
            # If a point is [1, 1] it failed inference.
            if not (point[0] == 1 and point[1] == 1):
                # Inferred points are in relation to the original image dimensions, not the resized frame
                # that is displayed.
                point_display = (dd[0] / fd[0] * point[0], dd[1] / fd[1] * point[1])
                # circle = plt.Circle(point_display, 40, fill=False, color='red', linestyle='--')
                # ax.add_artist(circle)
                axis.plot(point_display[0], point_display[1], marker='o', color='red')


def drawBulletDataOnAxis(axis: Axes, name: str, bullet: dict, fd: list, dd: list):
    """
    Plot the bullet data onto the frame.

    Args:
        axis: Axis displaying frame.
        name: Frame name as a String.
        bullet: Dictionary of bullet data.
        fd: Dimensions of original frame (x, y).
        dd: Display dimension - shape of the displayed frame.
    """
    # Plot all the bullet end points.
    for k in bullet:
        if bullet[k][0] == name:
            pointDisplay = pixelsToDisplay(bullet[k][1:], fd, dd)
            axis.plot(pointDisplay[0], pointDisplay[1], marker='*', color=BULLET_COL[k], markersize=5)
            axis.text(pointDisplay[0] - 20, pointDisplay[1] - 10, k, color=BULLET_COL[k])
    # Plot connecting lines where necessary.
    if name == bullet['L1'][0] and name == bullet['L2'][0]:
        l1 = pixelsToDisplay(bullet['L1'][1:], fd, dd)
        l2 = pixelsToDisplay(bullet['L2'][1:], fd, dd)
        axis.plot([l1[0], l2[0]], [l1[1], l2[1]], color=BULLET_COL['L1'], linewidth=1, linestyle='--')
    if name == bullet['W1'][0] and name == bullet['W2'][0]:
        w1 = pixelsToDisplay(bullet['W1'][1:], fd, dd)
        w2 = pixelsToDisplay(bullet['W2'][1:], fd, dd)
        axis.plot([w1[0], w2[0]], [w1[1], w2[1]], color=BULLET_COL['W1'], linewidth=1, linestyle='--')
    if name == bullet['H1'][0] and name == bullet['H2'][0]:
        h1 = pixelsToDisplay(bullet['H1'][1:], fd, dd)
        h2 = pixelsToDisplay(bullet['H2'][1:], fd, dd)
        axis.plot([h1[0], h2[0]], [h1[1], h2[1]], color=BULLET_COL['H1'], linewidth=1, linestyle='--')


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
        try:
            duration = int(names[-1].split('-')[1].split('.')[0]) - int(names[0].split('-')[1].split('.')[0])
        except IndexError:
            duration = 1
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
        print(f'\tEditingData.txt does not exist, creating now.')
        with open(editPath, 'a') as editingFile:
            editingFile.write('imuOffset:0\n')
            editingFile.write('imuPosition:50\n')

    return editPath


def getPointDataFromFile(scanPath: str):
    """
        Helper function to get point data that has already been saved. If there is no file it is created. The prostate
        and bladder points are stored in a dictionary format.

        Args:
            scanPath: String representation of the scan path.

        Returns:
            Path object to existing or newly created PointData.json file, dict of prostate points placed in current
            scan, and dict of bladder points placed in current scan.
        """
    pointPath = checkPointDataFile(scanPath)

    f = open(pointPath)

    data = json.load(f)

    f.close()

    prostatePoints = data.get('Prostate')

    if prostatePoints is None:
        prostatePoints = []

    bladderPoints = data.get('Bladder')

    if bladderPoints is None:
        bladderPoints = []

    prostateBox = data.get('ProstateBox')

    if prostateBox is None:
        prostateBox = []

    bladderBox = data.get('BladderBox')

    if bladderBox is None:
        bladderBox = []

    return pointPath, prostatePoints, bladderPoints, prostateBox, bladderBox


def checkPointDataFile(scanPath: str):
    """
        Check if the given directory contains a PointData.json file, if True return a Path object to it, else create
        the file and return a Path object to it.

        Args:
            scanPath: Path to a recording directory.

        Returns:
            Path object to existing or newly created PointData.json file.
        """
    pointPath = Path(scanPath, 'PointData.json')
    if not pointPath.is_file():
        print(f'\tPointData.json not found, creating...')
        with open(pointPath, 'a') as new_file:
            initial_json = {'Prostate': [], 'Bladder': []}
            json.dump(initial_json, new_file, indent=4)

    return pointPath


def drawPointDataOnAxis(axis, points, fd, dd, colour):
    """
    Plot given points on the frame. Points are currently stored in pixels (including the IMU offset from the edge
    of the probe). These need to be converted to display coordinates.

    Args:
        axis: Axis displaying frame.
        points: List of points in pixels.
        fd: Original frame dimensions (before resizing)
        dd: Display dimension - shape of the frame, first and second value swapped.
        colour: Colour of points (limegreen - prostate, lightblue - bladder)

    Returns:
        Nothing returned as data is drawn directly on axis.
    """
    for point in points:
        pointDisplay = pixelsToDisplay(point, fd, dd)

        axis.plot(pointDisplay[0], pointDisplay[1], marker=m, color=colour, markersize=15)


def drawRLAPEstimateData(axis, RLAPData):
    """
    Draw the Rl/AP ellipse with prostate boundary end points.

    Parameters
    ----------
    axis: Axis displaying frame.
    RLAPData

    Returns
    -------

    """
    try:
        centreDisplay = RLAPData[0]
        a = RLAPData[1]
        b = RLAPData[2]
        phi = RLAPData[3]

        ellipse = patches.Ellipse(xy=centreDisplay, width=2 * a, height=2 * b,
                                  angle=np.rad2deg(phi), edgecolor='r', fc='None', lw=2)

        axis.add_patch(ellipse)

    except Exception as e:
        ErrorDialog(None, 'Error drawing RL/AP ellipse.', e)


def drawSIEstimateData(axis, SIData, fd, dd):
    """
    Plot the SI estimate data onto the axis.
        - Centre of mass of bladder (A).
        - Bottom right point of prostate (C).
        - Top left intersection of line AC with prostate (B).

    Parameters
    ----------
    axis: Axis displaying frame.
    SIData: Dictionary of data used to display SI values
    fd: Original frame dimensions (before resizing)
    dd: Display dimension - shape of the frame, first and second value swapped.
    """
    try:
        # Plot bottom right point of prostate as target.
        bottomRightDisplay = pixelsToDisplay(SIData[1], fd, dd)
        axis.plot(bottomRightDisplay[0], bottomRightDisplay[1], marker='+', color='red', markersize=25)
        axis.plot(bottomRightDisplay[0], bottomRightDisplay[1], marker='o', markerfacecolor='none', color='red',
                  markersize=25)
    except Exception as e:
        ErrorDialog(None, 'Error drawing prostate bottom right point.', e)
    try:
        # Plot arc centred at prostate end.
        bladderCoMDisplay = pixelsToDisplay(SIData[0], fd, dd)
        bottomRightDisplay = pixelsToDisplay(SIData[1], fd, dd)
        r = np.sqrt((bladderCoMDisplay[0] - bottomRightDisplay[0]) ** 2
                    + (bladderCoMDisplay[1] - bottomRightDisplay[1]) ** 2)
        theta_centre = np.arctan2(bladderCoMDisplay[1] - bottomRightDisplay[1],
                                  bladderCoMDisplay[0] - bottomRightDisplay[0])
        theta = np.linspace(theta_centre - np.pi / 4, theta_centre + np.pi / 4, 100)

        x = bottomRightDisplay[0] + r * np.cos(theta)
        y = bottomRightDisplay[1] + r * np.sin(theta)
        axis.plot(x, y, linestyle='--', linewidth=1, color='red')
        axis.plot(bladderCoMDisplay[0], bladderCoMDisplay[1], marker='o', color='red')
    except Exception as e:
        ErrorDialog(None, 'Error drawing arc of bladder CoM.', e)

    try:
        # Plot intersection of line and top of prostate as target.
        topProstateDisplay = pixelsToDisplay(SIData[2], fd, dd)
        axis.plot(topProstateDisplay[0], topProstateDisplay[1], marker='o', color='red', markersize=25,
                  markerfacecolor='none')
        axis.plot(topProstateDisplay[0], topProstateDisplay[1], marker='+', color='red', markersize=25)
    except Exception as e:
        ErrorDialog(None, 'Error drawing intersection point between bladder CoM and prostate end.', e)


def pixelsToDisplay(pointPix: list, fd: list, dd: list):
    """
    Convert a point from frame relative pixels to display coordinates. Since the frames are resized before being
    displayed the point coordinates need to be updated.

    Args:
        pointPix: Point in frame relative coordinates.
        fd: Frame dimensions.
        dd: Display dimensions.

    Returns:
        Point in display coordinates.
    """
    pointRatio = [pointPix[0] / fd[1], pointPix[1] / fd[0]]
    pointDisplay = ratioToCoordinates(pointRatio, dd)
    return pointDisplay


def displayToPixels(pointDisplay: list, fd: list, dd: list):
    """
    Convert a point from display coordinates to frame relative pixel coordinates.
    Args:
        pointDisplay: Point in display coordinates.
        fd: Frame dimensions.
        dd: Display dimensions.

    Returns:
        Point in frame relative pixel coordinates.
    """
    pointRatio = [pointDisplay[0] / dd[0], (pointDisplay[1]) / dd[1]]
    pointPix = ratioToCoordinates(pointRatio, [fd[1], fd[0]])

    return pointPix


def ratioToCoordinates(pointRatio: list, dimensions: list):
    """
    Convert a point given as a ratio to coordinates. Rounding is done as display and pixel coordinates have to be
    integers. If dimensions are related to the display dimensions, the result will be a point in display relative
    dimensions. If dimensions are related to frame dimensions, the result will be a point in frame relative dimensions.

    Args:
        pointRatio: Width and Height ratio of a point.
        dimensions: Either display or frame dimensions.

    Returns:
        Point coordinates in either frame or display relative dimensions (int rounding).
    """
    pointDisplay = [round(pointRatio[0] * dimensions[0]), round(pointRatio[1] * dimensions[1])]

    return pointDisplay


def pointInRadius(centre: list, point: list, radius: int) -> bool:
    """
    Checks whether the point is within the constant radius of the centre, if it is return True, else False.

    Args:
        centre (float, float): Location of centre point, as a percentage.
        point (float, float): Location of test point, as a percentage.
        radius: Radius around point.

    Returns:
        withinRadius (bool): True if within radius, else False.
    """
    withinRadius = False

    if (point[0] - centre[0]) ** 2 + (point[1] - centre[1]) ** 2 < radius ** 2:
        withinRadius = True

    return withinRadius


def checkSaveDataDirectory(scanPath: str) -> Path:
    """
    Check if the given directory contains a Save Data folder, if not then it must be created. This folder is used to
    save BulletData, PointData, and EditingData for a user.

    Args:
        scanPath (str): Path to a recording directory.

    Returns:
        saveDataPath (Path): Path to the newly created directory.
    """
    saveDataPath = Path(scanPath, 'Save Data')
    saveDataPath.mkdir(parents=True, exist_ok=True)

    return saveDataPath


def getBottomRightPoint(points: list, rightWeight=1, bottomWeight=1):
    """
    Given a list of points in 2D, (x, y), find the bottom right point with further right taking precedence over
    lower points (concavity should favour further right points).

    Parameters
    ----------
    points: List of points in (x, y) coordinates.
    rightWeight: Importance of rightwardness.
    bottomWeight: Importance of downwardness.

    Returns
    -------
    Bottom right point.
    """
    bottomRight = points[0]
    max_weighted_value = rightWeight * points[0][0] + bottomWeight * points[0][1]

    for p in points:
        weighted_value = rightWeight * p[0] + bottomWeight * p[1]

        if weighted_value > max_weighted_value:
            max_weighted_value = weighted_value
            bottomRight = p

    return bottomRight


def findIntersectionsOfLineAndBoundary(boundaryPoints, linePoints):
    """
    Find the intersection between a line and the boundary line of an irregular shape. The boundary points belong
    to the prostate. The linePoints are made up of: [BladderCoM, ProstateBottomRight]. ProstateBottomRight will
    be on the boundary, but BladderCoM could be anywhere. If it is within the boundaryPoints the line must be
    extended (a scale of 10 should be enough). If the bladderCoM is down and to the right of the
    ProstateBottomRight then something will probably go wrong.
    Parameters
    ----------
    boundaryPoints
    linePoints

    Returns
    -------

    """
    # Create a polygon (use shapely, not matplotlib).
    polygon = PolygonShapely(boundaryPoints)
    # Get point centre of arc and point on arc.
    centrePoint = linePoints[1]
    arcPoint = linePoints[0]
    # Calculate arc radius length.
    r = np.sqrt((arcPoint[0] - centrePoint[0]) ** 2
                + (arcPoint[1] - centrePoint[1]) ** 2)
    # Generate 100 point along the arc, with arcPoint at centre.
    thetaCentre = np.arctan2(arcPoint[1] - centrePoint[1],
                             arcPoint[0] - centrePoint[0])
    theta = np.linspace(thetaCentre - np.pi / 4, thetaCentre + np.pi / 4, 100)
    x = centrePoint[0] + r * np.cos(theta)
    y = centrePoint[1] + r * np.sin(theta)

    largestPoint = []
    largestLength = 0
    for xPoint, yPoint in zip(x, y):
        endPoint = [xPoint, yPoint]

        # Create a line.
        line = LineString((centrePoint, endPoint))

        # Check if the end point is inside the polygon
        if polygon.contains(Point(endPoint)):
            # Extend the line.
            print('\t\tThe line between centrePoint and endPoint needs to be extended.')
            line = scale(line, xfact=10, yfact=10, origin=Point(centrePoint))

        # Find the intersection points.
        intersection = polygon.intersection(line)

        if isinstance(intersection, MultiLineString):
            intersection = intersection.geoms[0]
        # Extract the intersection points that are not the original boundary point.
        intersectionPoints = [Point(coords) for coords in intersection.coords]
        intersectionPoints = [point for point in intersectionPoints if not Point(centrePoint).equals(point)]

        if intersectionPoints:
            # Get the left most point.
            leftmost_point = min(intersectionPoints, key=lambda point: point.x)
            leftIntersection = [leftmost_point.x, leftmost_point.y]
            lineLength = np.sqrt((leftIntersection[0] - centrePoint[0]) ** 2 +
                                 (leftIntersection[1] - centrePoint[1]) ** 2)
            if lineLength >= largestLength:
                largestLength = lineLength
                largestPoint = leftIntersection

    return largestPoint


def fitEllipseToPoints(pointsDisplay: list, bladderCoMDisplay, prostateCoMDisplay, pointsWeight, angleWeight):
    """
    Using the LsqEllipse function, fit an ellipse to the points given. The line between bladderCoMDisplay and
    prostateCoMDisplay is used to calculate the desired phi of the ellipse (vertical taken as zero).

    Parameters
    ----------
    pointsDisplay: List of (x, y) points in display coordinates.
    bladderCoMDisplay: Bladder center-of-mass.
    prostateCoMDisplay: Prostate center-of-mass.
    pointsWeight: Weight applied to prostate points.
    angleWeight: Weight applied to desired angle (bladderCoM to prostateCoM line).

    Returns
    -------
    center, width, height, phi of the fit ellipse.
    """
    # Convert prostate display point into an array.
    pointsArray = np.array(pointsDisplay)
    # Calculate desired angle, relative to horizontal. When plotting, must add pi/2 as Ellipse is from horizontal.
    desiredPhi = angleRelativeToVertical(prostateCoMDisplay, bladderCoMDisplay)

    initialGuess = [np.mean(pointsArray[:, 0]), np.mean(pointsArray[:, 1]), 50, 30, desiredPhi]
    # Define bounds for the parameters
    bounds = Bounds([-np.inf, -np.inf, 0, 0, -2 * np.pi], [np.inf, np.inf, np.inf, np.inf, 2 * np.pi])

    result = minimize(ellipseCostFunction, np.array(initialGuess),
                      args=(pointsArray, desiredPhi, pointsWeight, angleWeight),
                      bounds=bounds)

    xc, yc, a, b, resultantPhi = result.x
    return [[xc, yc], a, b, resultantPhi]


def ellipseCostFunction(params, points, desiredPhi, pointWeight, angleWeight):
    """
    Cost function when trying to fit an ellipse with a desired rotation.

    Parameters
    ----------
    params: [(x_centre, y_centre), Semi-major, Semi-minor] of ellipse.
    points: Boundary points.
    desiredPhi: Desired rotation of ellipse.
    pointWeight: Weight applied to boundary points.
    angleWeight: Weight applied to desired rotation.

    Returns
    -------
    Cost for ellipse fitting.
    """
    xc, yc, a, b, phi = params

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x, y = points[:, 0], points[:, 1]
    cost = pointWeight * (((x - xc) * cos_phi + (y - yc) * sin_phi) ** 2 / a ** 2 +
                          ((x - xc) * sin_phi - (y - yc) * cos_phi) ** 2 / b ** 2 - 1)

    rotationConstraint = angleWeight * (phi - desiredPhi) ** 2

    return np.sum(cost ** 2) + rotationConstraint


def angleRelativeToVertical(point1, point2):
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = np.arctan2(dy, dx)
    return angle


def calculateCentreOfMass(points: list):
    """
    Calculate the centre of mass given a list of points in 2D. Assumes equal weighting and points are used as is.

    Parameters
    ----------
    points: List of points in (x, y) coordinates.

    Returns
    -------
    Centre of mass in (x, y) coordinates.
    """
    com = [sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)]

    return com


def estimateSlopeStartAndEnd(axisAngles: list):
    """
    Estimate where the scan starts and stops, based on the gradient of the axis angle. The average of a sliding window
    is used as the average mean of the gradient. This is compared with a threshold to find when the gradient goes
    above a certain value, and when it drops below the same value. The section in between the threshold is considered
    the scan of the prostate. Very rough estimate.

    Args:
        axisAngles (list): All axis angles of the recording (degrees).

    Returns:
        slopeIndexStart (int): Index of the __start of the slope/scan.
        slopeIndexEnd (int): Index of the end of the slope/scan.
    """
    threshold = 0.15
    window = 5
    mean = np.average(sliding_window_view(np.gradient(axisAngles), window_shape=window), axis=1)
    aboveThreshold = np.argwhere(mean > threshold)
    # Slope started when threshold first passed, add in half window size, -1 as threshold has already been passed.
    slopeStartIndex = aboveThreshold[0] + int(math.floor(window / 2)) - 1
    slopeStartIndex = max(0, slopeStartIndex[0])
    # Slope ended when threshold last passed, add in half window size, +1 so that we are passed threshold.
    slopeEndIndex = aboveThreshold[-1] + int(math.floor(window / 2)) + 1
    slopeEndIndex = min(len(axisAngles) - 1, slopeEndIndex[0])

    return slopeStartIndex, slopeEndIndex


def quaternionsToAxisAngles(quaternions: list) -> list:
    """
    Convert the given list of quaternions to a list of axis angles (in degrees) in the following manner:
        1. Get the initial quaternion, to be used as the reference quaternion.
        2. Calculate the difference between all subsequent quaternions and the initial quaternion using:
                r = p * conj(q)
           where r is the difference quaternion, p is the initial quaternion, and conj(q) is the conjugate of the
           current quaternion.
        3. Calculate the axis angle of r (the difference quaternion).

    Converting the raw quaternions to their axis angle representation for rotation comparisons is not the correct way
    to do it, the axis angle has to be calculated from the quaternion difference.

    Args:
        quaternions: List of quaternion values.

    Returns:
        axisAngles: List of axis angles (in degrees) relative to the first rotation (taken as 0 degrees).
    """
    initialQ = Quaternion(quaternions[0])
    axisAngles = []
    # Get angle differences (as quaternion rotations).
    for row in quaternions:
        q = Quaternion(row)
        r = initialQ * q.conjugate

        axisAngles.append(180 / np.pi * 2 * np.arctan2(np.sqrt(r[1] ** 2 + r[2] ** 2 + r[3] ** 2), r[0]))

    return axisAngles


def getBulletDataFromFile(scanPath: str):
    """
    Helper function to get Bullet data from file.

    Args:
        scanPath (str): String representation of the current Scan path.

    Returns:
        bulletPath: Path to Bullet.JSON file.
        bulletData: Bullet data stored in Bullet.json file.
    """
    bulletPath = checkBulletDataFile(scanPath)

    with open(bulletPath, 'r') as bulletFile:
        bulletData = json.load(bulletFile)
    # Changing H from 3D point to 2D, must change older values.
    for k in bulletData:
        if isinstance(bulletData[k][0], float):
            print('\tOld 3-plane data detected, updating...')
            bulletData[k] = ['', 0, 0]
    return bulletPath, bulletData


def getIPVDataFromFile(scanPath: str):
    """
    Helper function to get IPV data from file.

    Args:
        scanPath (str): String representation of the current Scan path.

    Returns:
        ipvPath: Path to IPV.JSON file.
        ipvData: IPV data stored in IPV.JSON file.
    """
    ipvPath = checkIPVDataFile(scanPath)

    with open(ipvPath, 'r') as ipvFile:
        ipvData = json.load(ipvFile)
        if 'radius' not in ipvData:
            ipvData['radius'] = 100

    return ipvPath, ipvData


def checkBulletDataFile(scanPath: str) -> Path:
    """
    Check if the given folder contains a Bullet.json file, if True return a Path object to it, else create the file.

    Args:
        scanPath: String representation of the Scan path.

    Returns:
        bulletPath: Path object to existing or newly created Bullet.json file.
    """
    bulletPath = Path(scanPath, 'BulletData.json')
    if not bulletPath.is_file():
        print('\tNo Bullet data found. Creating...')
        initialBullet = {
            'L1': ['', 0, 0],
            'L2': ['', 0, 0],
            'W1': ['', 0, 0],
            'W2': ['', 0, 0],
            'H1': ['', 0, 0],
            'H2': ['', 0, 0]
        }
        with open(bulletPath, 'w') as bulletFile:
            json.dump(initialBullet, bulletFile, indent=4)
    return bulletPath


def checkIPVDataFile(scanPath: str) -> Path:
    """
    Check if the given folder contains an IPV.JSON file, if True return a Path object to it, else create the file
    and return a Path object to it.

    Args:
        scanPath: Path to a recording directory.

    Returns:
        ipvPath: Path object to existing or newly created IPV.json file.
    """
    ipvPath = Path(scanPath, 'IPV.json')
    if not ipvPath.is_file():
        print('\tNo IPV data found. Creating...')
        initialIPV = {
            'centre': ['', 0, 0],
            'radius': 100,
            'inferred_points': ['', []]
        }
        with open(ipvPath, 'w') as ipvFile:
            json.dump(initialIPV, ipvFile, indent=4)

    return ipvPath


def getIndexOfFrameInBoxPoints(boxPoints, frameName):
    """
    Get the index of the box points with given frame name.

    Parameters
    ----------
    boxPoints: List of frames and box start and end points.
    frameName: Frame to search for.

    Returns
    -------
    Index of frameName if in list, else -1.
    """
    for index, row in enumerate(boxPoints):
        if row[0] == frameName:
            return index
    return -1


def getIndexOfApexFrames(pointsProstate: list):
    """
    Find the index of the first and last frame with prostate points on it. For use in navigating to frames at a
    certain percentage of the prostate size.

    Parameters
    ----------
    pointsProstate: List of prostate points [frameIndex, x, y].

    Returns
    -------
    Apex 1 index, Apex 2 index.
    """
    sortedByFrame = sorted(pointsProstate, key=lambda x: int(x[0]))

    indexStart = int(sortedByFrame[0][0])
    indexEnd = int(sortedByFrame[-1][0])

    return indexStart, indexEnd


def fit_spline_to_boundary(points):
    """
    Fit a spline to the given points (boundary).
    Returns the spline parameters and the parameterized boundary points (u).
    """
    points = np.append(points, points[0])
    n = len(points)
    u = np.linspace(0, 1, n)  # Parameterize the points uniformly along the boundary
    tck = CubicSpline(u, points, bc_type='natural')  # Natural boundary conditions
    return tck, u


def interpolate_spline(tck_start, tck_end, u_start, u_end, t, numPoints):
    """
    Interpolate between two splines (start and end) using the interpolation factor t.
    Returns the interpolated points along the boundary.
    """
    # Interpolate between start and end splines
    interp_x_start = tck_start(u_start)
    interp_x_end = tck_end(u_end)

    # Morph the splines towards each other using the interpolation factor
    interp_x = (1 - t) * interp_x_start + t * interp_x_end
    return interp_x

def plot_points_with_labels(points, color='blue', label='Points'):
    """
    Debugging the order in which points are being sorted in the sortClockWise function.

    Parameters
    ----------
    points
    color
    label

    Returns
    -------

    """
    xs, ys = zip(*points)
    plt.scatter(xs, ys, c=color, label=label)

    # Annotate each point with its index
    for i, (x, y) in enumerate(points):
        plt.text(x + 1, y + 1, str(i), fontsize=9, color=color)


def remove_readonly(func, path, excinfo):
    """
    Passed to shutil.rmtree to handle deleting read only files.
    """
    os.chmod(path, stat.S_IWRITE)
    func(path)


