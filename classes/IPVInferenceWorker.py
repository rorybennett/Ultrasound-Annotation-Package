import requests
from PyQt6.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject

from classes import Scan


class Signals(QObject):
    finished = pyqtSignal()
    started = pyqtSignal()


class IPVInferenceWorker(QRunnable):
    def __init__(self, scan: Scan, address: str, modelName: str):
        super(IPVInferenceWorker, self).__init__()
        self.scan = scan
        self.address = address
        self.modelName = modelName
        self.signals = Signals()

    @pyqtSlot()
    def run(self):
        """
                Send the current IPV centre frame for inference at the given address. This address can either be online
                or local (if local the server must be running on localhost (http://127.0.0.1:5000/)).
                """
        self.signals.started.emit()
        address = f'{self.address.strip()}/infer'
        print(f'Sending frame for IPV inference at: {address}')
        # If IPV centre has been placed, else use currently displayed frame with no ROI.
        if self.scan.ipvData['centre'][0]:
            frameName = self.scan.ipvData['centre'][0]
            frameNumber = int(frameName.split('-')[0])
            patient, _, _, _, _ = self.scan.getScanDetails()
            print(f"\tSending IPV centre frame ({self.scan.ipvData['centre'][0].split('-')[0]}) for inference...")
            with open(f"{self.scan.path}/{frameName}.png", 'rb') as imageFile:
                frame = imageFile.read()
            centre = self.scan.ipvData['centre'][1:]
            data = {
                'model_name': f'transverse_{self.modelName}' if self.scan.scanPlane == Scan.PLANE_TRANSVERSE else f'sagittal_{self.modelName}',
                'frame': frame,
                'ipv_centre': f'[{centre[0]}, {centre[1]}]',
                'ipv_radius': self.scan.ipvData['radius'],
                'scan_plane': self.scan.scanPlane,
                'patient_number': patient,
                'frame_number': frameNumber}
        else:
            frameName = self.scan.frameNames[self.scan.currentFrame - 1]
            frameNumber = int(frameName.split('-')[0])
            patient, _, _, _, _ = self.scan.getScanDetails()
            print(f'Sending currently displayed frame ({self.scan.currentFrame}) for inference...')
            with open(f"{self.scan.path}/{frameName}.png", 'rb') as imageFile:
                frame = imageFile.read()
            data = {
                'model_name': f'transverse_{self.modelName}' if self.scan.scanPlane == Scan.PLANE_TRANSVERSE else f'sagittal_{self.modelName}',
                'frame': frame,
                'ipv_centre': f'[0, 0]',
                'ipv_radius': 0,
                'scan_plane': self.scan.scanPlane,
                'patient_number': patient,
                'frame_number': frameNumber}
        try:
            result = requests.post(address, files=data, timeout=3600)
            if result.ok:
                print(f'\tIPV result returned: {result.json()}')
                self.scan.updateIPVInferredPoints(result.json()['result'], frameName)
            else:
                # ErrorDialog(None, f'Error with inference', result.status_code)
                print('Error with inference.')
        except Exception as e:
            # ErrorDialog(None, f'Error in http request', e)
            print('Error with http request')

        self.signals.finished.emit()
