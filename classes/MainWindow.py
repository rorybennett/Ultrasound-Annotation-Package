# MainWindow.py

"""Main Window for viewing and editing ultrasound scans."""
import sys
from pathlib import Path

import qdarktheme
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, \
    QLabel, QGridLayout, QSpacerItem, QSizePolicy, QCheckBox, QMenu

from classes import Scan
from classes.FrameCanvas import FrameCanvas


class MainWindow(QMainWindow):
    labelFont = QFont('Arial', 18)

    def __init__(self):
        """Initialise MainWindow."""
        # Setup GUI.
        super().__init__()
        self.setWindowTitle("Ultrasound Scan Editing")
        # Display 2 scans side-by-side inside central widget.
        self.mainWidget = QWidget(self)
        self.mainLayout = QHBoxLayout(self.mainWidget)
        self.mainWidget.installEventFilter(self)

        spacer = QSpacerItem(1, 1, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        # Left side.
        self.left = QVBoxLayout()
        self.left.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.leftTitle = self._createTitle()
        self.leftButtons = self._createTopButtons(1)
        self.axis1 = FrameCanvas(self)
        self.axis1.mpl_connect('button_press_event', self._axis1PressEvent)
        self.leftBoxes = self._createBoxes(1)
        self.left.addLayout(self.leftTitle)
        self.left.addLayout(self.leftButtons)
        self.left.addWidget(self.axis1)
        self.left.addLayout(self.leftBoxes)
        self.left.addItem(spacer)
        # Right side.
        self.right = QVBoxLayout()
        self.right.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.rightTitle = self._createTitle()
        self.rightButtons = self._createTopButtons(2)
        self.axis2 = FrameCanvas(self)
        self.axis2.mpl_connect('button_press_event', self._axis2PressEvent)
        self.rightBoxes = self._createBoxes(2)
        self.right.addLayout(self.rightTitle)
        self.right.addLayout(self.rightButtons)
        self.right.addWidget(self.axis2)
        self.right.addLayout(self.rightBoxes)
        self.right.addItem(spacer)

        self.mainLayout.addLayout(self.left)
        self.mainLayout.addLayout(self.right)
        self.setCentralWidget(self.mainWidget)

        self._createMainMenu()

        # Scan directory Path.
        self.scansPath = Path(Path.cwd().parent, 'Scans')
        # Scan 1.
        self.s1: Scan = None
        # Scan 2.
        self.s2: Scan = None

    def _createTopButtons(self, scan: int):
        """Create the layout for the top row of buttons"""
        layout = QHBoxLayout()
        cineButton = QPushButton('Cine', self)
        cineButton.clicked.connect(lambda: self._onCineClicked(scan))
        layout.addWidget(cineButton)

        return layout

    @staticmethod
    def _createTitle():
        """Create title layout area."""
        layout = QGridLayout()

        patientLabel = QLabel(f'Patient: ')
        patientLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        patientLabel.setFont(MainWindow.labelFont)
        scanLabel = QLabel(f'Scan Type:')
        scanLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scanLabel.setFont(MainWindow.labelFont)
        frameLabel = QLabel(f'Total Frames:')
        frameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frameLabel.setFont(MainWindow.labelFont)

        layout.addWidget(patientLabel, 0, 0)
        layout.addWidget(scanLabel, 0, 1)
        layout.addWidget(frameLabel, 0, 2)

        return layout

    def _updateTitle(self, scan: int):
        """Update title information."""
        if scan == 1:
            patient, scanType, frameCount = self.s1.getScanDetails()
            self.leftTitle.itemAt(0).widget().setText(f'Patient: {patient}')
            self.leftTitle.itemAt(1).widget().setText(f'Scan Type: {scanType}')
            self.leftTitle.itemAt(2).widget().setText(f'Total Frames: {frameCount}')
        else:
            patient, scanType, frameCount = self.s2.getScanDetails()
            self.rightTitle.itemAt(0).widget().setText(f'Patient: {patient}')
            self.rightTitle.itemAt(1).widget().setText(f'Scan Type: {scanType}')
            self.rightTitle.itemAt(2).widget().setText(f'Total Frames: {frameCount}')

    def _createBoxes(self, scan: int):
        """Create checkboxes below canvas."""
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        points = QCheckBox('Show Points')
        points.setChecked(True)
        points.stateChanged.connect(lambda: self._updateDisplay(scan))
        layout.addWidget(points)

        return layout

    def _onCineClicked(self, scan: int):
        """Play a cine of the scan in a separate window."""
        if scan == 1:
            print("Play Cine of Scan 1")
        else:
            print("Play Cine of Scan 2")

    def _createMainMenu(self):
        """Create menus."""
        # Load scans menu.
        self.menuLoadScan = self.menuBar().addMenu("Load Scans")
        self.menuLoadScan.addAction("Select Scan 1 Folder...", lambda: self._selectScanDialog(1))
        self.menuLoadScan.addAction("Open Scan 1 Directory...", lambda: self._openScanDirectory(1)).setDisabled(True)
        self.menuLoadScan.addSeparator()
        self.menuLoadScan.addAction("Select Scan 2 Folder...", lambda: self._selectScanDialog(2))
        self.menuLoadScan.addAction("Open Scan 2 Directory...", lambda: self._openScanDirectory(2)).setDisabled(True)

        # Load data menu
        self.menuLoad = self.menuBar().addMenu("Load Data")
        self.menuLoad.addAction('Load Scan 1 Data', lambda: None).setDisabled(True)
        self.menuLoad.addSeparator()
        self.menuLoad.addAction('Load Scan 2 Data', lambda: None).setDisabled(True)

    def _selectScanDialog(self, scan: int):
        """Show dialog for selecting a scan folder."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Scan {scan}',
                                                    directory=str(self.scansPath))

        if scan == 1:
            self.s1 = Scan.Scan(scanPath)
            self.menuLoadScan.actions()[1].setEnabled(True)
            self.left.itemAt(2).widget().setFixedSize(self.s1.displayDimensions[0],
                                                      self.s1.displayDimensions[1])
            self._updateTitle(1)
            self.menuLoad.actions()[0].setEnabled(True)
        else:
            self.s2 = Scan.Scan(scanPath)
            self.menuLoadScan.actions()[3].setEnabled(True)
            self.right.itemAt(2).widget().setFixedSize(self.s2.displayDimensions[0],
                                                       self.s2.displayDimensions[1])
            self._updateTitle(2)
            self.menuLoad.actions()[2].setEnabled(True)

        self._updateDisplay(scan)

    def _axis1PressEvent(self, event):
        """Handle left clicks on axis 1 (canvas displaying image)."""
        displayPoint = [event.x - 1 if event.x > 0 else 0,
                        event.y - 1 if event.y > 0 else 0]
        # Left click.
        if event.button == 1 and self.s1 and self.leftBoxes.itemAt(0).widget().isChecked():
            self.s1.addOrRemovePoint(displayPoint)
            self._updateDisplay(1)
            return

    def _axis2PressEvent(self, event):
        """Handle left clicks on axis 2 (canvas displaying image)."""
        displayPoint = [event.x - 1 if event.x > 0 else 0,
                        event.y - 1 if event.y > 0 else 0]
        # Left click.
        if event.button == 1 and self.s2 and self.rightBoxes.itemAt(0).widget().isChecked():
            self.s2.addOrRemovePoint(displayPoint)
            self._updateDisplay(2)
            return

    def _openScanDirectory(self, scan: int):
        """Open directory of Scan."""
        if scan == 1:
            self.s1.openDirectory()
        else:
            self.s2.openDirectory()

    def _updateDisplay(self, scan: int):
        """Update the shown frame."""
        if scan == 1:
            self.s1.drawFrameOnAxis(self.axis1, self.leftBoxes.itemAt(0).widget().isChecked())
        else:
            self.s2.drawFrameOnAxis(self.axis2, self.rightBoxes.itemAt(0).widget().isChecked())

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

    def _clearFramePoints(self, scan: int):
        """Clear frame points from scan, then update display."""
        self.s1.clearFramePoints() if scan == 1 else self.s2.clearFramePoints()
        self._updateDisplay(scan)

    def contextMenuEvent(self, event):
        if self.s1 and self.axis1.underMouse():
            menu = QMenu()
            menu.addAction('Clear Points', lambda: self._clearFramePoints(1))
            menu.exec(event.globalPos())
        elif self.s2 and self.axis2.underMouse():
            menu = QMenu()
            menu.addAction('Clear Points', lambda: self._clearFramePoints(2))
            menu.exec(event.globalPos())


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
