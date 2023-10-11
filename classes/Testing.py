from classes import Scan
from classes.ProcessAxisAnglePlot import ProcessAxisAnglePlot


def main():
    scan = Scan.Scan(
        "C:/Users/roryb/GDOffline/Research/Coding/Python/Ultrasound-Data-Edit-V2/Scans/001/AUS/Transverse/1", 1)

    axisAngleProcess = ProcessAxisAnglePlot()
    axisAngleProcess.start(scan)

    while True:
        pass


if __name__ == '__main__':
    main()
