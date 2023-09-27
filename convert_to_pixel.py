import json
from pathlib import Path

import numpy as np


def mmToFrame(pointMM: list, depths: list, imuOffset: float, imuPosition: float, fd: list):
    """
    Convert a point in mm to a point in the frame coordinates. First convert the point in mm to a display ratio,
    then to frame coordinates.

    Args:
        pointMM: x and y coordinates of the point in mm.
        depths: Scan depth.
        imuOffset: IMU offset.
        imuPosition: Position of IMU shown by ticks.
        fd: Frame dimensions

    Returns:
        Point coordinates in display dimensions.
    """
    pointDisplay = ratioToFrame(mmToRatio(pointMM, depths, imuOffset, imuPosition), fd)

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


def ratioToFrame(pointRatio: list, fd: list):
    """
    Convert a point given as a ratio of the display dimensions to frame coordinates. Rounding is done as frame
    coordinates have to be integers.

    Args:
        pointRatio: Width and Height ratio of a point in relation to the display dimensions.
        fd: Frame dimensions, based on frame.

    Returns:
        Point coordinates in frame dimensions (int rounding).
    """
    pointDisplay = [int(pointRatio[0] * fd[0]),
                    int(pointRatio[1] * fd[1])]

    return pointDisplay


def bullet_to_pixels(scan_dir, dd):
    # Get data.txt info.
    data = []
    with open(f'{dd}', 'r') as f:
        for line in [l.rstrip() for l in f.readlines()]:
            data.append(line.split(','))
    data = np.array(data)
    # Get BulletData.json.
    with open(f'{scan_dir}/BulletData.json', 'r') as f:
        bullet_m = json.load(f)
    # Get EditingData.txt info.
    with open(f'{scan_dir}/EditingData.txt', 'r') as f:
        for line in f.readlines():
            lineSplit = line.split(':')
            parameter = lineSplit[0]
            value = lineSplit[1]

            if parameter == 'imuOffset':
                imuOffset = float(value)
            elif parameter == 'imuPosition':
                imuPosition = float(value)
    # Convert mm to pixel
    bullet_p = {}
    for k in bullet_m:
        frame_name = bullet_m[k][0]
        pixel = [0, 0]
        if frame_name:
            point = [float(bullet_m[k][1]), float(bullet_m[k][2])]

            index = np.where(data[:, 0] == frame_name)
            depths = [float(data[index, 14]), float(data[index, 15])]
            fd = [float(data[index, 11]), float(data[index, 12])]
            pixel = mmToFrame(point, depths, imuOffset, imuPosition, fd)
        bullet_p[k] = [frame_name, pixel[0], pixel[1]]
    with open(f'{scan_dir}/BulletData.json', 'w') as plane_file:
        json.dump(bullet_p, plane_file, indent=4)


def points_to_pixels(scan_dir, dd):
    # Get data.txt info.
    data = []
    with open(f'{dd}', 'r') as f:
        for line in [l.rstrip() for l in f.readlines()]:
            data.append(line.split(','))
    data = np.array(data)
    # Get PointData.txt info.
    point_m = []
    with open(f'{scan_dir}/PointData.txt', 'r') as f:
        for line in [l.rstrip() for l in f.readlines()]:
            point_m.append(line.split(','))
    point_m = np.array(point_m)
    # Get EditingData.txt info.
    with open(f'{scan_dir}/EditingData.txt', 'r') as f:
        for line in f.readlines():
            lineSplit = line.split(':')
            parameter = lineSplit[0]
            value = lineSplit[1]

            if parameter == 'imuOffset':
                imuOffset = float(value)
            elif parameter == 'imuPosition':
                imuPosition = float(value)
    # Convert mm to pixel
    point_pixel = []
    for p in point_m:
        frame_name = p[0]
        point = [float(p[1]), float(p[2])]

        index = np.where(data[:, 0] == frame_name)
        depths = [float(data[index, 14]), float(data[index, 15])]
        fd = [float(data[index, 11]), float(data[index, 12])]
        pixel = mmToFrame(point, depths, imuOffset, imuPosition, fd)
        point_pixel.append([frame_name, pixel[0], pixel[1]])

    with open(f'{scan_dir}/PointData.txt', 'w') as pointFile:
        for point in point_pixel:
            pointFile.write(f'{point[0]},{point[1]},{point[2]}\n')


scans_dir = 'Scans'
# Get all patient directories.
patients = sorted(Path(scans_dir).iterdir())
# Deal with one patient at a time (only AUS).
for p in patients:
    position_dir = f'{p}/AUS'
    planes = ['sagittal', 'transverse']
    # Deal with one plane at a time (sagittal and transverse).
    for plane in planes:
        plane_dir = f'{position_dir}/{plane}'
        try:
            scans = sorted([i for i in sorted(Path(plane_dir).iterdir())])
            # Deal with one scan at a time (either 1 or Clarius).
            for scan in [i for i in scans if i.is_dir()]:
                data_dir = f'{scan}/data.txt'
                # Convert bullet to pixels.
                bullet_to_pixels(scan, data_dir)
                # Convert points to pixels.
                points_to_pixels(scan, data_dir)
                # Deal with Sava Data folder.
                for save_dir in [i for i in Path(f'{scan}/Save Data').iterdir()]:
                    bullet_to_pixels(save_dir, data_dir)
                    points_to_pixels(save_dir, data_dir)
        except FileNotFoundError as e:
            print(f'{e}.')
