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


def getScanType(recording_path: str):
    """
    Return the type of scan based on scan parent directory.

    :param recording_path: String representation of the recording path.

    :return scan_type: Type of scan.
    """

    scan_type = recording_path.split('/')[-2]

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


