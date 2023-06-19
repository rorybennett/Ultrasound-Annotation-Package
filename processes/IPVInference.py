"""
Class for handling inference of ellipse points.
"""
import csv
import datetime as dt
import multiprocessing
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from IPV.Code import parameters as pms, InferPoints
from IPV.Code.DataCreator import DataCreator
from classes import Scan


def createTemporaryIPVDirectory():
    """
    Create a directory for storing files needed for inference on the current frame. This folder will be deleted after
    inference, with only some files stored for later use. They are stored in the IPV/Inference directory.

    Returns:
        ipvPath: Path of the temporary IPV directory.
    """
    cwd = Path.cwd().parent

    ipvPath = Path(cwd, f"IPV/Inference/{dt.datetime.now().strftime('%d %m %Y %H-%M-%S')}")
    ipvPath.mkdir(parents=True, exist_ok=True)

    return ipvPath


class IPVInference:
    def __init__(self):
        """
        Initialise a IPVInference object.
        """
        self.asyncProcess = None
        self.queue = None
        self.pool = None

    def start(self, scanPath: str, keepData: bool = False):
        """
        Start the process that will infer the points on the given frame, centred at centre.
        """
        self.pool = multiprocessing.Pool(1)
        self.asyncProcess = self.pool.apply_async(process, args=(scanPath, keepData))


def process(scanPath, keepData):
    """
    Method to be run in an async_process pool for inferring points.

    1. Create data.
    2. Infer points.
    3. Delete created data:
        a. Delete patches.
        b. Delete .txt, .csv, and other supplementary files. Get final point positions before deleting.
        c. Keep the point distribution image, convolution images, arc maps, and point positions.

    """
    print(f'Starting inference for frame...')
    try:
        # =========================================================================================================
        # Create data. Create patches around centre point of prostate.
        # =========================================================================================================
        startTime = dt.datetime.now()
        # Create directory for frame inference.
        ipvPath = createTemporaryIPVDirectory()
        # Load recording without frames
        scan = Scan.Scan(scanPath)
        radius = scan.ipvData['radius']
        frameName = scan.ipvData['centre'][0]
        centre = scan.getIPVCentreInFrameDimensions()
        frame = scan.frames[scan.frameNames.index(frameName)]

        # Save frame to directory.
        cv2.imwrite(str(ipvPath) + f'/{frameName}.png', cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))
        if scan.scanType == Scan.TYPE_TRANSVERSE:
            dc = DataCreator(distance_intervals=pms.tasks_classes[0], angle_intervals=pms.tasks_classes[1],
                             subpatch_scales=pms.sub_patch_scales)
            dc.create(frame_name=frameName, step=10, data_path=str(ipvPath), centre=centre, radius=radius,
                      num_of_points=4)

            infoCSV = open(str(ipvPath) + '/test_info.csv', 'w')
            infoFile = csv.writer(infoCSV)
            infoFile.writerow(
                ['DISTANCE_INTERVALS', 'ANGLE_INTERVALS', 'HAVE_SUB_PATCHES', 'SUB_PATCH_SCALES',
                 'PATCH_SIZE'])
            infoFile.writerow(
                [str(pms.tasks_classes[0]), str(pms.tasks_classes[1]), pms.sub_patches, pms.sub_patch_scales,
                 pms.sub_patch_scales[0]])
            infoCSV.close()
            endTime = dt.datetime.now()
            print('\tPatches created in:', endTime - startTime)
            # =========================================================================================================
            # Infer points. Run the model to infer the required points using generated patches.
            # =========================================================================================================
            startTime = dt.datetime.now()
            InferPoints.infer(str(ipvPath), "../IPV/Models/transverse",
                              frameName, 4)
            endTime = dt.datetime.now()
            print('\tInference completed in:', endTime - startTime)

            detectedPoints = np.genfromtxt(str(ipvPath) + '/TestResults/DetectedPoints.csv', delimiter=',')
            scan.updateIPVInferredPoints(detectedPoints[1:])
            print('\tIPV data saved.')
        else:
            dc = DataCreator(distance_intervals=pms.tasks_classes[0], angle_intervals=pms.tasks_classes[1],
                             subpatch_scales=pms.sub_patch_scales)

            dc.create(frame_name=frameName, step=10, data_path=str(ipvPath), centre=centre, radius=radius,
                      num_of_points=2)

            infoCSV = open(str(ipvPath) + '/test_info.csv', 'w')
            infoFile = csv.writer(infoCSV)
            infoFile.writerow(
                ['DISTANCE_INTERVALS', 'ANGLE_INTERVALS', 'HAVE_SUB_PATCHES', 'SUB_PATCH_SCALES', 'PATCH_SIZE'])
            infoFile.writerow(
                [str(pms.tasks_classes[0]), str(pms.tasks_classes[1]), pms.sub_patches, pms.sub_patch_scales,
                 pms.sub_patch_scales[0]])
            infoCSV.close()
            endTime = dt.datetime.now()
            print('\tPatches created in:', endTime - startTime)
            # =========================================================================================================
            # Infer points. Run the model to infer the required points using generated patches.
            # =========================================================================================================
            startTime = dt.datetime.now()
            InferPoints.infer(str(ipvPath), "../IPV/Models/sagittal",
                              frameName, 2)
            endTime = dt.datetime.now()
            print('\tInference completed in:', endTime - startTime)

            detectedPoints = np.genfromtxt(str(ipvPath) + '/TestResults/DetectedPoints.csv', delimiter=',')
            scan.updateIPVInferredPoints(detectedPoints[1:])
            print('\tIPV data saved.')

        if not keepData:
            try:
                shutil.rmtree(f'{str(ipvPath)}/test_patches')
            except Exception as e:
                print(e)
            try:
                os.remove(f'{str(ipvPath)}/test_info.csv')
            except Exception as e:
                print(e)
            try:
                os.remove(f'{str(ipvPath)}/test_fold.csv')
            except Exception as e:
                print(e)


    except Exception as e:
        print(e)
