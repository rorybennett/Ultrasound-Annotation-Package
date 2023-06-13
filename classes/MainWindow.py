# MainWindow.py

"""Main Window for viewing and editing ultrasound scans."""
import sys
from pathlib import Path

import qdarktheme
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout, QPushButton

from classes import Scan
from classes.FrameCanvas import FrameCanvas


class MainWindow(QMainWindow):

    def __init__(self):
        """Initialise MainWindow."""
        # Setup GUI.
        super().__init__()
        self.setWindowTitle("Ultrasound Scan Editing")
        # Display 2 scans side-by-side as central widget.
        self.mainWidget = QWidget(self)
        self.layoutMain = QHBoxLayout(self.mainWidget)
        self.mainWidget.installEventFilter(self)

        self.left = QVBoxLayout()
        self.leftButtons = self._createTopButtons(1)
        self.axis1 = FrameCanvas(self)
        self.cidButton1 = self.axis1.mpl_connect('button_press_event', self._axisButtonPressEvent)
        self.left.addLayout(self.leftButtons)
        self.left.addWidget(self.axis1)

        self.right = QVBoxLayout()
        self.rightButtons = self._createTopButtons(2)
        self.axis2 = FrameCanvas(self)
        self.cidButton2 = self.axis2.mpl_connect('button_press_event', self._axisButtonPressEvent)
        self.right.addLayout(self.rightButtons)
        self.right.addWidget(self.axis2)

        self.layoutMain.addLayout(self.left)
        self.layoutMain.addLayout(self.right)

        self.setCentralWidget(self.mainWidget)

        self._createMenu()

        # Scan directory Path.
        self.scansPath = Path(Path.cwd().parent, 'Scans')
        # Scan 1.
        self.s1: Scan = None
        # Scan 2.
        self.s2: Scan = None

    def _createTopButtons(self, scanNumber: int):
        """Create the layout for the top row of buttons"""
        layout = QHBoxLayout()
        cineButton = QPushButton('Cine', self)
        cineButton.clicked.connect(lambda: self._onCineClicked(scanNumber))
        layout.addWidget(cineButton)

        return layout

    def _onCineClicked(self, scanNumber: int):
        """Play a cine of the scan in a separate window."""
        if scanNumber == 1:
            print("Play Cine of Scan 1")
        else:
            print("Play Cine of Scan 2")

    def _axisButtonPressEvent(self, event):
        """Handle clicks on axes (canvas displaying image)."""
        displayPoint = [event.x - 1 if event.x > 0 else 0,
                        event.y - 1 if event.y > 0 else 0]
        if event.button in [1, 3]:
            if self.s1:
                print(displayPoint)
                return

            if self.s2:
                print(displayPoint)
                return

    def _createMenu(self):
        """Create menus."""
        # First scan menu.
        self.menuScan1 = self.menuBar().addMenu("Scan 1")
        self.menuScan1.addAction("Select Scan Folder", lambda: self._selectScanDialog(1))
        self.menuScan1.addAction("Open Scan Directory", lambda: self._openScanDirectory(1)).setDisabled(True)

        # Second scan menu.
        self.menuScan2 = self.menuBar().addMenu("Scan 2")
        self.menuScan2.addAction("Select Scan Folder", lambda: self._selectScanDialog(2))
        self.menuScan2.addAction("Open Scan Directory", lambda: self._openScanDirectory(2)).setDisabled(True)

    def _selectScanDialog(self, scanNumber: int):
        """Show dialog for selecting a scan folder."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Scan {scanNumber}',
                                                    directory=str(self.scansPath))

        if scanNumber == 1:
            self.s1 = Scan.Scan(scanPath)
            self.menuScan1.actions()[1].setEnabled(True)
            self.left.itemAt(1).widget().setFixedSize(self.s1.displayDimensions[0],
                                                      self.s1.displayDimensions[1])
        else:
            self.s2 = Scan.Scan(scanPath)
            self.menuScan2.actions()[1].setEnabled(True)
            self.right.itemAt(1).widget().setFixedSize(self.s2.displayDimensions[0],
                                                       self.s2.displayDimensions[1])

        self._updateDisplay(scanNumber)

    def _openScanDirectory(self, scanNumber: int):
        """Open directory of Scan."""
        if scanNumber == 1:
            self.s1.openDirectory()
        else:
            self.s2.openDirectory()

    def _updateDisplay(self, scanNumber: int):
        """Update the shown frame."""
        if scanNumber == 1:
            self.s1.drawFrameOnAxis(self.axis1)
        else:
            self.s2.drawFrameOnAxis(self.axis2)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if self.s1 and self.axis1.underMouse():
            if event.text() == 'w':
                self.s1.navigate(Scan.NAVIGATION['w'])
            elif event.text() == 's':
                self.s1.navigate(Scan.NAVIGATION['s'])
            self._updateDisplay(1)
        elif self.s2 and self.axis2.underMouse():
            if event.text() == 'w':
                self.s2.navigate(Scan.NAVIGATION['w'])
            elif event.text() == 's':
                self.s2.navigate(Scan.NAVIGATION['s'])
            self._updateDisplay(2)


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


def main():
    editingApp = QApplication([])

    qdarktheme.setup_theme()

    mainWindow = MainWindow()
    mainWindow.showMaximized()

    sys.exit(editingApp.exec())


if __name__ == "__main__":
    sys.excepthook = except_hook
    main()
