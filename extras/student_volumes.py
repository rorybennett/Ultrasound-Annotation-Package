import json
from pathlib import Path

import numpy as np


def distance(point_1, point_2):
    """Calculate the Euclidean distance between 2 points."""
    d = np.sqrt((point_1[0] - point_2[0]) ** 2 + (point_1[1] - point_2[1]) ** 2)
    return d


students = ['AS', 'KL', 'n', 'NS']

# Work with 1 student at a time.
for student in students:
    print(f'================================================================')
    print(f'Student: {student}')
    print(f'================================================================')
    # All 19 patients per student.
    patients = range(1, 20)
    for p in patients:
        print(f'Patient {p} - ', end=' ')
        scan_planes = ['transverse', 'sagittal']
        for sp in scan_planes:
            scan_dir = Path(f'Scans/{p}/AUS/{sp}/1')
            save_dir = Path(scan_dir, 'Save Data')

            for save in save_dir.iterdir():
                if save.name.split('_')[0] == student:
                    # Load data.txt.
                    with open(f'{scan_dir}/data.txt', 'r') as data:
                        line = data.readline().split(',')
                        depths = [int(line[-2]), int(line[-3])]
                        fd = [int(line[-6]), int(line[-5])]
                    # Load BulletData.json.
                    with open(f'{save}/BulletData.json', 'r') as file:
                        bd = json.load(file)
                    # Calculate RL and AP
                    rl = 0
                    ap = 0
                    if sp == 'transverse':
                        points_pix = [bd['L1'][1], bd['L1'][2],
                                      bd['L2'][1], bd['L2'][2],
                                      bd['W1'][1], bd['W1'][2],
                                      bd['W2'][1], bd['W2'][2]]
                        # Convert to mm.
                        points_mm = []
                        for i, pp in enumerate(points_pix):
                            # X values.
                            if i % 2 == 0:
                                points_mm.append(depths[0] / fd[0] * points_pix[i])
                            # Y values.
                            else:
                                points_mm.append(depths[1] / fd[1] * points_pix[i])

                        points_mm = np.array(
                            [[points_mm[i], points_mm[i + 1]] for i in range(0, len(points_mm) - 1, 2)])
                        p1 = points_mm[np.argmin(points_mm[:, 1])]
                        p2 = points_mm[np.argmax(points_mm[:, 0])]
                        p3 = points_mm[np.argmax(points_mm[:, 1])]
                        p4 = points_mm[np.argmin(points_mm[:, 0])]

                        rl = distance(p4, p2)
                        ap = distance(p3, p1)
                        print(f'RL = {rl:0.2f}. AP = {ap:0.2f}.', end=' ')
                    # Calculate SI
                    si = 0
                    if sp == 'sagittal':
                        points_pix = [bd['H1'][1], bd['H1'][2],
                                      bd['H2'][1], bd['H2'][2]]
                        # Convert to mm.
                        points_mm = []
                        for i, pp in enumerate(points_pix):
                            # X values.
                            if i % 2 == 0:
                                points_mm.append(depths[0] / fd[0] * points_pix[i])
                            # Y values.
                            else:
                                points_mm.append(depths[1] / fd[1] * points_pix[i])

                        si = distance(points_mm[0:2], points_mm[2:4])
                        print(f'SI = {si:0.2f}.')

    print('')
