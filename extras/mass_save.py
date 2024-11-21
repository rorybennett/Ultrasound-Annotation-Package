"""
Add extra files to IPV folders to enable usage by this application.
"""
import os
import shutil

from natsort import natsort

scan_dir = f'../../../../Scans'

save_name = 'YOLOIPV_1719225669105'

planes = ['Sagittal', 'Transverse']

all_patients = [i for i in natsort.natsorted(os.listdir(scan_dir)) if i.startswith('A')]

for patient in all_patients[1:]:
    for p in planes:
        from_dir = f'{scan_dir}/{patient}/AUS/{p}/1'

        to_dir = f'{from_dir}/Save Data/{save_name}'

        if not os.path.isdir(to_dir):
            os.mkdir(to_dir)

        # Copy files to Save Data folder.
        shutil.copy(f'{from_dir}/PointData.json', f'{to_dir}/PointData.json')
        shutil.copy(f'{from_dir}/BulletData.json', f'{to_dir}/BulletData.json')
        shutil.copy(f'{from_dir}/EditingData.txt', f'{to_dir}/EditingData.json')
        shutil.copy(f'{from_dir}/IPV.json', f'{to_dir}/IPV.json')
