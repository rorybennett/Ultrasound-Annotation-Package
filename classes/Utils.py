import os
from operator import itemgetter
from pathlib import Path

import natsort
import numpy as np
from scipy.interpolate import splprep, splev


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

    # Limit patients to first 19.
    patients = natsort.natsorted(Path(scansPath).iterdir())[0:19]
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

                            if Path(scan, 'PointData.txt').exists():
                                os.remove(Path(scan, 'PointData.txt'))

        except Exception as e:
            print(f'Error resetting editing data: {e}.')
    return True


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
    cartesianPoints = []
    for point in polarPoints:
        cartesianPoints.append(polarToCartesian(point))
    # Shift back to original position.
    orderedPoints = []
    for point in cartesianPoints:
        orderedPoints.append([point[0] + com[0], point[1] + com[1]])

    return orderedPoints


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


def interpolate(x, y, total_points):
    """
    Interpolate x and y using splprep and splev. Used by the Spline class.
    """
    [tck, _] = splprep([x, y], s=0, per=True)

    xi, yi = splev(np.linspace(0, 1, total_points), tck)

    return xi, yi

def pointDistance(point1: list, point2: list) -> float:
    """
    Calculate the distance between two points. The z coordinate is not always necessary.

    Args:
        point1 (list): (x, y, z) coordinates of the first point.
        point2 (list): (x, y, z) coordinates of the second point.

    Returns:
        distance (float): Calculated distance.
    """
    distance = np.linalg.norm(np.array(point2) - np.array(point1), axis=0)

    return distance
def pointSegmentDistance(p, s0, s1):
    """
    Get the distance of a point to a segment. Used by the Spline class.
      *p*, *s0*, *s1* are *xy* sequences
    This algorithm from
    http://geomalgorithms.com/a02-_lines.html

    Args:
        p ():
        s0 ():
        s1 ():
    """
    v = s1 - s0
    w = p - s0
    c1 = np.dot(w, v)

    if c1 <= 0:
        return pointDistance(p, s0)

    c2 = np.dot(v, v)

    if c2 <= c1:
        return pointDistance(p, s1)

    b = c1 / c2
    pb = s0 + b * v

    return pointDistance(p, pb)
