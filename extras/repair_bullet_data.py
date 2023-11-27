import json
from pathlib import Path


def fix_bullet_sagittal(dir):
    with open(f'{dir}/BulletData.json', 'r') as bulletFile:
        bulletData = json.load(bulletFile)
    try:
        if int(bulletData['L1'][0]) > 0:
            bulletData['H1'] = bulletData['L1']
            bulletData['H2'] = bulletData['L2']

            bulletData['L1'] = ['', 0, 0]
            bulletData['L2'] = ['', 0, 0]
    except:
        pass

    if bulletData['W1'][0] == 0:
        bulletData['W1'][0] = ''
    if bulletData['W2'][0] == 0:
        bulletData['W2'][0] = ''
    if bulletData['L1'][0] == 0:
        bulletData['L1'][0] = ''
    if bulletData['L2'][0] == 0:
        bulletData['L2'][0] = ''

    with open(f'{dir}/BulletData.json', 'w') as bulletFile:
        json.dump(bulletData, bulletFile, indent=4)

def fix_bullet_transverse(dir):
    with open(f'{dir}/BulletData.json', 'r') as bulletFile:
        bulletData = json.load(bulletFile)

    if bulletData['H1'][0] == 0:
        bulletData['H1'][0] = ''
    if bulletData['H2'][0] == 0:
        bulletData['H2'][0] = ''

    with open(f'{dir}/BulletData.json', 'w') as bulletFile:
        json.dump(bulletData, bulletFile, indent=4)

scans_dir = '../Scans'
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
                # Deal with Sava Data folder.
                for save_dir in [i for i in Path(f'{scan}/Save Data').iterdir()]:
                    if plane == 'sagittal':
                        fix_bullet_sagittal(save_dir)
                    elif plane == 'transverse':
                        fix_bullet_transverse(save_dir)
        except FileNotFoundError as e:
            print(f'{e}.')
