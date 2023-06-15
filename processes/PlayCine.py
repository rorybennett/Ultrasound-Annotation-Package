# PlayCine.py
"""Play a Cine of the given Scan."""
import multiprocessing

import cv2

from classes import Scan


class PlayCine:
    def __init__(self):
        """
        Initialise a ProcessCine object.
        """
        self.async_process = None
        self.queue = None
        self.pool = None

    def start_process(self, frames, dimensions, ):
        """
        Start the process that will display the frames on a loop. Convert all frames to byte representation - for Graph.
        """

        dimensions = [scan.frames[0].shape[1], scan.frames[0].shape[0]]
        framesAsBytes = []
        for frame in scan.frames:
            framesAsBytes.append(cv2.imencode(".png", frame)[1].tobytes())
        patient, scanType, _ = scan.getScanDetails()
        title = f'Patient: {}, Scan Type: {scanType}'
        self.pool = multiprocessing.Pool(1)
        self.async_process = self.pool.apply_async(process, args=(framesAsBytes, dimensions, title))

def process(framesAsBytes, dimensions, title):
