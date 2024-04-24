import os
from operator import itemgetter
from pathlib import Path

import natsort
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSpacerItem, QSizePolicy
from matplotlib.patches import Polygon
from scipy.interpolate import splprep, splev

from classes.ErrorDialog import ErrorDialog

labelFont = QFont('Arial', 14)
spacer = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
stylesheet = """QToolTip { background-color: black; 
                                   color: white; 
                                   border: black solid 1px }"""


def resetEditingData(scansPath: str):
    """
    Reset all the Editing Data by deleting IPV.json, PointData.txt, and BulletData.json.

    To return this data it will need to be loaded from Save Data folder.

    Args:
        scansPath: Path to Scans directory.

    Returns:
        result: True if no errors, else False.
    """
    # Get total patients.
    totalPatients = 0
    for i in os.listdir(Path(scansPath)):
        if os.path.isdir(Path(scansPath, i)):
            totalPatients += 1
    print(f'Total patients to reset: {totalPatients}')

    patients = natsort.natsorted(Path(scansPath).iterdir())
    for patient in patients:
        try:
            for scanType in natsort.natsorted(patient.iterdir()):
                for scanPlane in natsort.natsorted(scanType.iterdir()):
                    for scan in natsort.natsorted(scanPlane.iterdir()):
                        if scan.is_dir():
                            # Delete files.
                            if Path(scan, 'BulletData.json').exists():
                                os.remove(Path(scan, 'BulletData.json'))

                            if Path(scan, 'IPV.json').exists():
                                os.remove(Path(scan, 'IPV.json'))

                            if Path(scan, 'PointData.json').exists():
                                os.remove(Path(scan, 'PointData.json'))

        except Exception as e:
            print(f'Error resetting editing data: {e}.')
            return False
    return True


def getCOM(points: np.array):
    """
    Calculate the centre-of-mass of a list of points.

    Args:
        points (np.array): Array of x- and y-coordinates used to calculate the centre-of mass.

    Returns:
        com (list): x- and y-coordinate of the centre-of-mass.
    """
    com = np.sum(points, axis=0) / len(points)

    com = list(com)

    return com


def cartesianToPolar(point: list):
    """
    Convert the given point (x, y) in cartesian coordinates to polar coordinates (rho, phi).

    Args:
        point (list): (x, y) point to be converted.

    Returns:
        polar (list): (rho, phi) polar coordinates.
    """
    polar = [np.sqrt(point[0] ** 2 + point[1] ** 2), np.arctan2(point[1], point[0])]

    return polar


def polarToCartesian(point: list):
    """
        Convert the given point (rho, phi) in polar coordinates to cartesian coordinates (x, y).

        Args:
            point (list): (rho, phi) polar coordinates to be converted.

        Returns:
            cartesian (list): (x, y) cartesian point.
    """
    cartesian = [point[0] * np.cos(point[1]), point[0] * np.sin(point[1])]

    return cartesian


def shrinkExpandPoints(points: list, amount: int):
    """
    Shrink or expand points around their centre of mass. Once shrinking or expanding is complete, convert to integer
    (pixel coordinates)

    Args:
        points: List of points.
        amount: How much to shrink or expand the points by.

    Returns:
        points: Points after shrinking.
    """
    # Find centre-of-mass of points.
    com = getCOM(points)
    # Shift points to centre-of-mass origin.
    shiftedPoints = []
    for point in points:
        shiftedPoints.append([point[0] - com[0], point[1] - com[1]])
    # Convert shifted points to polar coordinates.
    polarPoints = []
    for point in shiftedPoints:
        polarPoints.append(cartesianToPolar(point))
    # Shrink by amount.
    shrinkPolarPoints = []
    for point in polarPoints:
        shrinkPolarPoints.append([point[0] + amount, point[1]])
    # Convert shrink polar coordinates back to cartesian coordinates.
    cartesianPoints = []
    for point in shrinkPolarPoints:
        cartesianPoints.append(polarToCartesian(point))
    # Shift back to original position.
    newPoints = []
    for point in cartesianPoints:
        newPoints.append([round(point[0] + com[0]), round(point[1] + com[1])])

    return newPoints


def organiseClockwise(points: np.array):
    """
    Sort the given points in a clockwise manner. The origin is taken as the centre of mass of the points. This method
    requires the points to be spread out in a more-or-less circular fashion to get a reasonable centre.

    Args:
        points (np.array): Array of x- and y-coordinates to be sorted.

    Returns:
        orderedPoints (list): Clockwise ordered list of points.
    """
    # Find centre-of-mass of points.
    com = getCOM(points)
    # Shift points to centre-of-mass origin.
    shiftedPoints = []
    for point in points:
        shiftedPoints.append([point[0] - com[0], point[1] - com[1]])
    # Convert shifted points to polar coordinates.
    polarPoints = []
    for point in shiftedPoints:
        polarPoints.append(cartesianToPolar(point))
    # Sort polar coordinates by phi (angle).
    polarPoints.sort(key=itemgetter(1))
    # Convert sorted polar coordinates back to cartesian coordinates.
    cartesian_points = []
    for point in polarPoints:
        cartesian_points.append(polarToCartesian(point))
    # Shift back to original position.
    orderedPoints = []
    for point in cartesian_points:
        orderedPoints.append([point[0] + com[0], point[1] + com[1]])

    return orderedPoints


def distributePoints(pointsPix, count):
    """
    Distribute points evenly around a spline generated from the original points. Point order is NOT corrected.

    Args:
        pointsPix: Points in pixel coordinates.
        count: Number of points to be placed throughout spline.

    Return:
        Return all points, distributed evenly and in order.
    """

    pointsPix = np.asfarray(pointsPix)
    # Add extra point on end to complete spline.
    pointsPix = np.append(pointsPix, [pointsPix[0, :]], axis=0)
    # Polygon, acting as spline.
    poly = Polygon(np.column_stack([pointsPix[:, 0], pointsPix[:, 1]]))
    # Extract points from polygon.
    xs, ys = poly.xy.T
    # Evenly space points along spline line.
    xn, yn = interpolate(xs, ys, len(xs) if len(xs) > count else count + 1)
    if xn is None:
        return
    # Get all points except the last one, which is a repeat.
    endPointsPix = np.column_stack([xn, yn])[:-1]

    return endPointsPix


def interpolate(x, y, total_points):
    """
    Interpolate x and y using splprep and splev. Used by the Spline class.
    """
    try:
        [tck, _] = splprep([x, y], s=0, per=True)

        xi, yi = splev(np.linspace(0, 1, total_points), tck)

        return xi, yi
    except Exception as e:
        ErrorDialog(None, 'Error interpolating points (too few, must be > 2).',
                    f'{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}.')
        return None, None


def createTitleLayout():
    """Create title layout area."""
    layout = QHBoxLayout()

    patientLabel = QLabel(f'Patient: ')
    patientLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    patientLabel.setFont(labelFont)
    typeLabel = QLabel(f'Type:')
    typeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    typeLabel.setFont(labelFont)
    planeLabel = QLabel(f'Plane:')
    planeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    planeLabel.setFont(labelFont)
    numberLabel = QLabel(f'Number:')
    numberLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    numberLabel.setFont(labelFont)
    frameLabel = QLabel(f'Frames:')
    frameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
    frameLabel.setFont(labelFont)

    layout.addWidget(patientLabel, 0)
    layout.addWidget(typeLabel, 0)
    layout.addWidget(planeLabel, 0)
    layout.addWidget(numberLabel, 0)
    layout.addWidget(frameLabel, 0)
    layout.setSpacing(10)

    return layout
