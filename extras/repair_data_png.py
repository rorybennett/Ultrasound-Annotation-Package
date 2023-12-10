"""Repair data.txt file by removing .png from frame names."""

# Get all patient directories.
from pathlib import Path

from natsort import natsorted


def fix_data_file(dataPath):
    with open(dataPath, 'r') as file:
        data = file.read().splitlines()

    fixed_data = []
    for line in data:
        fixed_data.append(line.replace('.png', ''))

    with open(dataPath, 'w') as file:
        for line in fixed_data:
            file.write(line + '\n')


def fix_point_file(pointPath):
    with open(pointPath, 'r') as file:
        point_data = file.read().splitlines()

    fixed_point_data = []
    for line in point_data:
        fixed_point_data.append(line.replace('.png', ''))

    with open(pointPath, 'w') as file:
        for line in fixed_point_data:
            file.write(line + '\n')

scans_dir = '../Scans'

patients = natsorted(Path(scans_dir).iterdir())[19:]
# Deal with one patient at a time (only AUS).
for p in patients:
    position_dir = f'{p}/AUS'
    planes = ['sagittal', 'transverse']
    # Deal with one plane at a time (sagittal and transverse).
    for plane in planes:
        plane_dir = f'{position_dir}/{plane}'
        try:
            scans = natsorted([i for i in natsorted(Path(plane_dir).iterdir())])
            # Deal with one scan at a time (either 1 or Clarius).
            for scan in [i for i in scans if i.is_dir()]:
                # Deal with active PointData.txt
                fix_data_file(f'{scan}/data.txt')
                fix_point_file(f'{scan}/PointData.txt')
                # Deal with Sava Data PointData files.
                for save_dir in [i for i in Path(f'{scan}/Save Data').iterdir()]:
                    fix_point_file(f'{save_dir}/PointData.txt')
        except FileNotFoundError as e:
            print(f'{e}.')
