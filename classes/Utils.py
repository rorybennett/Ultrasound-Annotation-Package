import os
from pathlib import Path

import natsort


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
