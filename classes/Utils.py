import os
from pathlib import Path

import natsort
import numpy as np


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
    Shrink points around their centre of mass.

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
        newPoints.append([point[0] + com[0], point[1] + com[1]])

    return newPoints
