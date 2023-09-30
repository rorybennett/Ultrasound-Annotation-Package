import matplotlib
from PyQt6.QtCore import QRunnable, pyqtSlot, pyqtSignal, QObject
from matplotlib import pyplot as plt


from classes import Scan


class Signals(QObject):
    finished = pyqtSignal()
    started = pyqtSignal()


class AxisAngleWorker(QRunnable):
    def __init__(self, scan: Scan):
        super(AxisAngleWorker, self).__init__()
        self.scan = scan
        self.signals = Signals()

    @pyqtSlot()
    def run(self):
        self.signals.started.emit()

        axis_angles = self.scan.quaternionsToAxisAngles()

        fig, ax = plt.subplots(1)

        ax.set_xlabel('Frame Number')
        ax.set_ylabel('Probe Axis Angle (degrees)')

        ax.plot(range(1, len(axis_angles) + 1), axis_angles, c='blue')

        ax.set_xlim([0, len(axis_angles) + 1])
        ax.set_ylim([min(axis_angles) - 2, max(axis_angles) + 2])

        plt.show()

        self.signals.finished.emit()
