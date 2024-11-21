"""
Convert PointData.txt to PointData.json.
"""
import json
import os
from pathlib import Path

from natsort import natsorted


def text_to_json(point_dir):
    print(f'Converting: {point_dir}/PointData.txt...')
    try:
        # Get data from PointData.txt.
        point_data = []
        with open(f'{point_dir}/PointData.txt', 'r') as pointFile:
            for line in pointFile.readlines():
                lineSplit = line.split(',')
                point_data.append([lineSplit[0], float(lineSplit[1]), float(lineSplit[2])])
        # Delete PointData.txt.
        os.remove(f'{point_dir}/PointData.txt')
        # Convert to json.
        point_json = {'Prostate': point_data}
        # Save json
        with open(f'{point_dir}/PointData.json', 'w') as json_file:
            json.dump(point_json, json_file, indent=4)
    except Exception as ex:
        print(f'Error: {ex}')


scans_dir = '../Scans'
# Get all patient directories.
patients = natsorted(Path(scans_dir).iterdir())

# Deal with one patient at a time (only AUS).
for p in patients:
    positions = ['AUS', 'PUS']
    # Deal with one position at a time.
    for pos in positions:
        position_dir = f'{p}/{pos}'
        planes = ['sagittal', 'transverse']
        # Deal with one plane at a time (sagittal and transverse).
        for plane in planes:
            plane_dir = f'{position_dir}/{plane}'
            try:
                scans = sorted([i for i in natsorted(Path(plane_dir).iterdir())])
                # Deal with one scan at a time (either 1 or Clarius).
                for scan in [i for i in scans if i.is_dir()]:
                    text_to_json(scan)
                    # Deal with Sava Data folder.
                    for save_dir in [i for i in Path(f'{scan}/Save Data').iterdir()]:
                        text_to_json(save_dir)
            except FileNotFoundError as e:
                print(f'{e}.')
