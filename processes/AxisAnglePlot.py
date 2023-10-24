"""
Class for handling plotting the axis angle of the probe. This plot if used to identify the middle frame.

A custom multiprocessing.manager class is created to enable a LIFO queue approach.
"""
import multiprocessing
import queue
from multiprocessing.managers import BaseManager
from queue import LifoQueue

from matplotlib import pyplot as plt

from classes import Scan


class MyManager(BaseManager):
    """
    Custom class used to register LifoQueue to the multiprocessing manager class.
    """
    pass


MyManager.register('LifoQueue', LifoQueue)


class AxisAnglePlot:
    def __init__(self):
        """
        Initialise a ProcessAnglePlot object.
        """
        self.async_process = None
        self.manager = MyManager()
        self.manager.start()
        self.queue = None
        self.pool = None

    def start(self, scan: Scan):
        """
        Create the queue object, processing pool, and start the plottingProcess running in the process pool.
        """
        self.queue = self.manager.LifoQueue()
        self.pool = multiprocessing.Pool(1)

        self.async_process = self.pool.apply_async(plottingProcess, args=(self.queue, scan.path, scan.currentFrame - 1))

    def updateIndex(self, index: int):
        """
        Add an index to the queue, used to determine the current quaternion.

        Args:
            index (int): Index of quaternion to be focused on (current frame quaternion).
        """
        if self.queue:
            self.queue.put(index)

    def end(self):
        """
        Close and join the plottingProcess. First the while loop in the plottingProcess method is broken, then the
        plottingQueue is deleted (helps with memory release), and the process pool is closed and joined.
        """
        del self.queue
        self.pool.close()
        self.pool.join()


def plottingProcess(lifoQueue, scanPath: str, frameIndex: int):
    """
    Method to be run in an async_process pool for plotting the points of a recording.

    Args:
        frameIndex: Current frame index.
        scanPath: Path to Scan object.
        lifoQueue (LifoQueue): MyManager queue object operating with LIFO principle.
    """
    scan = Scan.Scan(scanPath)
    axisAngles = scan.quaternionsToAxisAngles()
    patient, scanType, scanPlane, scanNumber, _ = scan.getScanDetails()

    fig, ax = plt.subplots(1)
    fig.canvas.manager.set_window_title(f'Axis Angle Plot')

    ax.set_title(f'Patient: {patient}, Scan Type: {scanType},\n'
                 f'Scan Plane: {scanPlane}, Scan Number: {scanNumber}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Probe Axis Angle (degrees)')
    ax.set_xlim([0, len(axisAngles) + 1])
    ax.set_ylim([min(axisAngles) - 2, max(axisAngles) + 2])

    ax.plot(range(1, len(axisAngles) + 1), axisAngles, c='blue')

    # 2 lines that show current location and some text, they are removed before first drawing and thus must exist.
    ax.plot([], [])
    ax.plot([], [])
    ax.text(0, 0, '')
    while plt.fignum_exists(fig.number):
        try:
            # Remove text label from previous point.
            ax.texts[0].remove()
            # Remove 'crosshair' from previous point.
            # ax.lines[0].remove()
            ax.lines[2].remove()
            ax.lines[1].remove()

            angle = axisAngles[frameIndex]
            indexMax = len(axisAngles)
            # Plot 'crosshair' for current point.
            ax.plot(frameIndex + 1, angle, marker='o', markersize=15, color='r', fillstyle='none', alpha=0.5)
            ax.plot(frameIndex + 1, angle, marker='+', markersize=15, color='r', alpha=0.5)
            # Add text label for current point, change rotation based on where 'crosshair' is placed.
            if frameIndex < indexMax / 6:
                ax.text(frameIndex + 1, angle + 3, f'[{frameIndex + 1}, {angle:0.1f}]', rotation=90)
            elif indexMax / 6 <= frameIndex <= 4 * indexMax / 6:
                ax.text(frameIndex + 6, angle - .5, f'[{frameIndex + 1}, {angle:0.1f}]')
            else:
                ax.text(frameIndex - 2, angle - 9.5, f'[{frameIndex + 1}, {angle:0.1f}]', rotation=-90)

            plt.pause(0.05)

            frameIndex = lifoQueue.get(False)
            # Empty the queue of older variables.
            while not lifoQueue.empty():
                try:
                    lifoQueue.get(False)
                except queue.Empty:
                    continue
                lifoQueue.task_done()
        except Exception as e:
            if isinstance(e, queue.Empty):
                pass
            else:
                print(f'ProcessAxisAnglePlot Error: {e}.')
