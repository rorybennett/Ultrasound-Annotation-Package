# MainWindow.py

"""Main Window for viewing and editing ultrasound scans."""
import sys
from pathlib import Path

import qdarktheme
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, \
    QLabel, QSpacerItem, QSizePolicy, QCheckBox, QMenu, QInputDialog, QStyle

import Scan
from classes import Export
from classes.FrameCanvas import FrameCanvas
from processes import PlayCine

INFER_LOCAL = 'INFER-LOCAL'
INFER_ONLINE = 'INFER-ONLINE'


class MainWindow(QMainWindow):
    labelFont = QFont('Arial', 14)

    def __init__(self):
        """Initialise MainWindow."""
        # Setup GUI.
        super().__init__()
        self.setWindowTitle("Ultrasound Scan Editing")
        # Tooltip style.
        self.setStyleSheet("""QToolTip { background-color: black; 
                                   color: white; 
                                   border: black solid 1px }""")

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
        self.axis1.mpl_connect('button_press_event', lambda x: self._axisClickEvent(x, 1))
        self.axis1.mpl_connect('scroll_event', lambda x: self._axisScrollEvent(x, 1))
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
        self.axis2.mpl_connect('button_press_event', lambda x: self._axisClickEvent(x, 2))
        self.axis2.mpl_connect('scroll_event', lambda x: self._axisScrollEvent(x, 2))
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
        self.scansPath = str(Path(Path.cwd().parent, 'Scans'))
        # Scan 1.
        self.s1: Scan = None
        # Scan 2.
        self.s2: Scan = None
        # Class for exporting data for training.
        self.export = Export.Export(self.scansPath)

    def _createTopButtons(self, scan: int):
        """Create the layout for the top row of buttons"""
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        cineButton = QPushButton('', self)
        cineButton.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        cineButton.setToolTip('Play Cine of Scan')
        cineButton.clicked.connect(lambda: self._onCineClicked(scan))
        cineButton.setDisabled(True)
        layout.addWidget(cineButton)

        nav50Button = QPushButton('50%')
        nav50Button.setToolTip('Show frame at 50% (based on IMU data).')
        nav50Button.clicked.connect(lambda: self._onNav50Clicked(scan))
        nav50Button.setDisabled(True)
        layout.addWidget(nav50Button)

        return layout

    def _onNav50Clicked(self, scan: int):
        """Travel to the frame at 50%."""
        if scan == 1:
            self.s1.navigate(self.s1.frameAtScanPercent(50))
        else:
            self.s2.navigate(self.s2.frameAtScanPercent(50))

        self._updateDisplay(scan)

    @staticmethod
    def _createTitle():
        """Create title layout area."""
        layout = QHBoxLayout()

        patientLabel = QLabel(f'Patient: ')
        patientLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        patientLabel.setFont(MainWindow.labelFont)
        typeLabel = QLabel(f'Scan Type:')
        typeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        typeLabel.setFont(MainWindow.labelFont)
        planeLabel = QLabel(f'Scan Plane:')
        planeLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        planeLabel.setFont(MainWindow.labelFont)
        frameLabel = QLabel(f'Total Frames:')
        frameLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        frameLabel.setFont(MainWindow.labelFont)

        layout.addWidget(patientLabel, 0)
        layout.addWidget(typeLabel, 0)
        layout.addWidget(planeLabel, 0)
        layout.addWidget(frameLabel, 0)
        layout.setSpacing(10)

        return layout

    def _updateTitle(self, scan: int):
        """Update title information."""
        if scan == 1:
            patient, scanType, scanPlane, frameCount = self.s1.getScanDetails()
            self.leftTitle.itemAt(0).widget().setText(f'Patient: {patient}')
            self.leftTitle.itemAt(1).widget().setText(f'Scan Type: {scanType}')
            self.leftTitle.itemAt(2).widget().setText(f'Scan Plane: {scanPlane}')
            self.leftTitle.itemAt(3).widget().setText(f'Total Frames: {frameCount}')
        else:
            patient, scanType, scanPlane, frameCount = self.s2.getScanDetails()
            self.rightTitle.itemAt(0).widget().setText(f'Patient: {patient}')
            self.rightTitle.itemAt(1).widget().setText(f'Scan Type: {scanType}')
            self.leftTitle.itemAt(2).widget().setText(f'Scan Plane: {scanPlane}')
            self.rightTitle.itemAt(3).widget().setText(f'Total Frames: {frameCount}')

    def _createBoxes(self, scan: int):
        """Create checkboxes below canvas."""
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        points = QCheckBox('Show Points')
        points.setChecked(True)
        points.stateChanged.connect(lambda: self._updateDisplay(scan))
        points.setDisabled(True)
        layout.addWidget(points)

        ipv = QCheckBox('Show IPV Data')
        ipv.setChecked(False)
        ipv.stateChanged.connect(lambda: self._updateDisplay(scan))
        ipv.setDisabled(True)
        layout.addWidget(ipv)

        return layout

    def _onCineClicked(self, scan: int):
        """Play a cine of the scan in a separate window."""
        if scan == 1:
            patient, scanType, scanPlane, _ = self.s1.getScanDetails()
            cine = PlayCine.PlayCine(self.s1.frames, patient, scanType, scanPlane)
            cine.startProcess()
        else:
            patient, scanType, scanPlane, _ = self.s2.getScanDetails()
            cine = PlayCine.PlayCine(self.s2.frames, patient, scanType, scanPlane)
            cine.startProcess()

    def _createMainMenu(self):
        """Create menus."""
        # Load scans menu.
        self.menuLoadScans = self.menuBar().addMenu("Load Scans")
        self.menuLoadScans.addAction("Select Scan 1 Folder...", lambda: self._selectScanDialog(1))
        self.menuLoadScans.addAction("Open Scan 1 Directory...", lambda: self._openScanDirectory(1)).setDisabled(True)
        self.menuLoadScans.addSeparator()
        self.menuLoadScans.addAction("Select Scan 2 Folder...", lambda: self._selectScanDialog(2))
        self.menuLoadScans.addAction("Open Scan 2 Directory...", lambda: self._openScanDirectory(2)).setDisabled(True)
        # Inference menu.
        menuInference = self.menuBar().addMenu('Inference')
        menuIPV = menuInference.addMenu('IPV Inference')
        self.menuIPV1 = menuIPV.addAction('Infer Scan 1', lambda: self._ipvInference(1))
        self.menuIPV1.setDisabled(True)
        self.menuIPV2 = menuIPV.addAction('Infer Scan 2', lambda: self._ipvInference(2))
        self.menuIPV2.setDisabled(True)
        # Load data menu
        menuLoadData = self.menuBar().addMenu("Load Data")
        self.menuLoadData1 = menuLoadData.addMenu('Load Scan 1 Data')
        self.menuLoadData1.setDisabled(True)
        self.menuLoadData1.aboutToShow.connect(lambda: self._populateLoadScanData(1))
        menuLoadData.addSeparator()
        self.menuLoadData2 = menuLoadData.addMenu('Load Scan 2 Data')
        self.menuLoadData2.setDisabled(True)
        self.menuLoadData2.aboutToShow.connect(lambda: self._populateLoadScanData(2))
        # Save data menu.
        self.menuSaveData = self.menuBar().addMenu("Save Data")
        self.menuSaveData.addAction('Save Scan 1 Data', lambda: self._saveData(1)).setDisabled(True)
        self.menuSaveData.addSeparator()
        self.menuSaveData.addAction('Save Scan 2 Data', lambda: self._saveData(2)).setDisabled(True)
        # Export data menu.
        self.menuExport = self.menuBar().addMenu("Export Data")
        menuExportIPV = self.menuExport.addMenu('IPV')
        menuExportIPV.addAction('Transverse', lambda: self._exportDataIPV(Scan.TYPE_TRANSVERSE))
        menuExportIPV.addAction('Sagittal', lambda: self._exportDataIPV(Scan.TYPE_SAGITTAL))
        self.menuExport.addAction('Save Data', lambda: self.export.exportAllSaveData())

    def _exportDataIPV(self, scanType: str):
        """Export save data for model training."""
        if scanType == Scan.TYPE_TRANSVERSE:
            self.export.exportIPVTransverseData(self)
        else:
            self.export.exportIPVSagittalData(self)

    def _ipvInference(self, scan: int):
        """Send current frame for IPV inference, either online or locally."""

        dlg = QInputDialog(self)
        dlg.resize(500, 200)
        dlg.setWindowTitle('IPV Inference')
        dlg.setLabelText('Enter Root Address:')
        dlg.setTextValue('http://127.0.0.1:5000/')
        ok = dlg.exec()
        address = dlg.textValue()

        if not ok or not address:
            return

        self.s1.inferenceIPV(address) if scan == 1 else self.s2.inferenceIPV(address)

        self._updateDisplay(scan)

    def _saveData(self, scan: int):
        """Save Scan point data. Check for overwrite"""
        userName, ok = QInputDialog.getText(self, 'Save Current Data', 'Enter User Name:')

        if ok:
            if not userName:
                print(f'User Name is empty...')
                self._saveData(scan)
                return
            self.s1.saveUserData(userName) if scan == 1 else self.s2.saveUserData(userName)

    def _loadSaveData(self, scan: int, fileName: str):
        """Load save data of scan and update display."""
        if scan == 1:
            self.s1.loadSaveData(fileName)
            self.s1 = Scan.Scan(self.s1.path, self.s1.currentFrame)
        else:
            self.s2.loadSaveData(fileName)
            self.s2 = Scan.Scan(self.s2.path, self.s2.currentFrame)

        self._updateDisplay(scan)

    def _populateLoadScanData(self, scan: int):
        """Populate the load submenu just before opening."""
        if scan == 1:
            self.menuLoadData1.clear()
            actions = []
            for fileName in self.s1.getSaveData():
                action = QAction(fileName, self)
                action.triggered.connect(lambda _, x=fileName: self._loadSaveData(scan, x))
                actions.append(action)
            self.menuLoadData1.addActions(actions)
        else:
            self.menuLoadData2.clear()
            actions = []
            for fileName in self.s2.getSaveData():
                action = QAction(fileName, self)
                action.triggered.connect(lambda _, x=fileName: self._loadSaveData(scan, x))
                actions.append(action)
            self.menuLoadData2.addActions(actions)

    def _selectScanDialog(self, scan: int):
        """Show dialog for selecting a scan folder."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Scan {scan}', directory=self.scansPath)

        if not scanPath:
            return

        try:
            if scan == 1:
                self.s1 = Scan.Scan(scanPath)
                self.menuLoadScans.actions()[1].setEnabled(True)
                self.menuIPV1.setEnabled(True)
                self.menuLoadData1.setEnabled(True)
                self.menuSaveData.actions()[0].setEnabled(True)
                for i in range(self.leftButtons.count()):
                    self.leftButtons.itemAt(i).widget().setEnabled(True)
                self.left.itemAt(2).widget().setFixedSize(self.s1.displayDimensions[0],
                                                          self.s1.displayDimensions[1])
                for i in range(self.leftBoxes.count()):
                    self.leftBoxes.itemAt(i).widget().setEnabled(True)
                self._updateTitle(1)
            else:
                self.s2 = Scan.Scan(scanPath)
                self.menuLoadScans.actions()[4].setEnabled(True)
                self.menuIPV2.setEnabled(True)
                self.menuLoadData2.setEnabled(True)
                self.menuSaveData.actions()[1].setEnabled(True)
                for i in range(self.rightButtons.count()):
                    self.rightButtons.itemAt(i).widget().setEnabled(True)
                self.right.itemAt(2).widget().setFixedSize(self.s2.displayDimensions[0],
                                                           self.s2.displayDimensions[1])
                self.rightBoxes.itemAt(0).widget().setEnabled(True)
                for i in range(self.rightBoxes.count()):
                    self.rightBoxes.itemAt(i).widget().setEnabled(True)
                self._updateTitle(2)
            self._updateDisplay(scan)
        except Exception as e:
            print(f'Error opening scan folder: {e}.')

    def _axisClickEvent(self, event, scan: int):
        """Handle left clicks on axis 1 and 2 (canvas displaying image)."""
        displayPoint = [event.x - 1 if event.x > 0 else 0,
                        event.y - 1 if event.y > 0 else 0]
        # Left click.
        if event.button == 1:
            if scan == 1 and self.s1 and self.leftBoxes.itemAt(0).widget().isChecked():
                self.s1.addOrRemovePoint(displayPoint)
                self._updateDisplay(1)
                return
            elif scan == 2 and self.s2 and self.rightBoxes.itemAt(0).widget().isChecked():
                self.s2.addOrRemovePoint(displayPoint)
                self._updateDisplay(2)
                return

    def _axisScrollEvent(self, event, scan: int):
        """Handle scroll events on axis 1 (canvas displaying image)."""
        if scan == 1 and self.s1:
            if event.button == 'up':
                self.s1.navigate(Scan.NAVIGATION['w'])
            else:
                self.s1.navigate(Scan.NAVIGATION['s'])
            self._updateDisplay(1)
            return
        elif scan == 2 and self.s2:
            if event.button == 'up':
                self.s2.navigate(Scan.NAVIGATION['w'])
            else:
                self.s2.navigate(Scan.NAVIGATION['s'])
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
            self.s1.drawFrameOnAxis(self.axis1,
                                    showPoints=self.leftBoxes.itemAt(0).widget().isChecked(),
                                    showIPV=self.leftBoxes.itemAt(1).widget().isChecked())
        else:
            self.s2.drawFrameOnAxis(self.axis2,
                                    showPoints=self.rightBoxes.itemAt(0).widget().isChecked(),
                                    showIPV=self.rightBoxes.itemAt(1).widget().isChecked())

    def _clearScanPoints(self, scan: int):
        """Clear all points in a Scan, then update display."""
        self.s1.clearScanPoints() if scan == 1 else self.s2.clearScanPoints()
        self._updateDisplay(scan)

    def _clearFramePoints(self, scan: int):
        """Clear frame points from scan, then update display."""
        self.s1.clearFramePoints() if scan == 1 else self.s2.clearFramePoints()
        self._updateDisplay(scan)

    def _updateIPVCentre(self, scan: int, addOrRemove: str, position: QPoint):
        """Add or remove IPV centre then update display."""
        position = [position.x(), self.s1.displayDimensions[1] - position.y()]
        self.s1.updateIPVCentre(position, addOrRemove) if scan == 1 else self.s2.updateIPVCentre(position, addOrRemove)
        self._updateDisplay(scan)

    def _updateIPVRadius(self, scan: int):
        """Dialog for updating IPV centre radius."""
        radius, ok = QInputDialog.getText(self, 'Update IPV Radius', 'Enter Radius:')

        if ok:
            try:
                radius = int(radius)
                self.s1.updateIPVRadius(radius) if scan == 1 else self.s2.updateIPVRadius(radius)
                self._updateDisplay(scan)
            except Exception as e:
                print(f'Error converting radius to int: {e}.')

    def _removeIPVData(self, scan: int):
        """Remove all IPV data of the Scan."""
        self.s1.removeIPVData() if scan == 1 else self.s2.removeIPVData()

        self._updateDisplay(scan)

    def _refreshScanData(self, scan: int):
        """Refresh scan data by re-reading files."""
        if scan == 1:
            self.s1 = Scan.Scan(self.s1.path, self.s1.currentFrame)
        else:
            self.s2 = Scan.Scan(self.s2.path, self.s2.currentFrame)

        self._updateDisplay(scan)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press events."""
        if self.s1 and self.axis1.underMouse():
            if event.key() == Qt.Key.Key_W:
                self.s1.navigate(Scan.NAVIGATION['w'])
            elif event.key() == Qt.Key.Key_S:
                self.s1.navigate(Scan.NAVIGATION['s'])
            self._updateDisplay(1)
        elif self.s2 and self.axis2.underMouse():
            if event.key() == Qt.Key.Key_W:
                self.s1.navigate(Scan.NAVIGATION['w'])
            elif event.key() == Qt.Key.Key_D:
                self.s1.navigate(Scan.NAVIGATION['s'])
            self._updateDisplay(2)

    def contextMenuEvent(self, event):
        if self.s1 and self.axis1.underMouse():
            menu = QMenu()
            menuPoints = menu.addMenu('Points')
            menuPoints.addAction('Clear Frame Points', lambda: self._clearFramePoints(1))
            menuPoints.addSeparator()
            menuPoints.addAction('Clear All Points', lambda: self._clearScanPoints(1))
            menuIPV = menu.addMenu('IPV')
            menuIPV.addAction('Add Center',
                              lambda: self._updateIPVCentre(1, Scan.ADD_POINT,
                                                            self.axis1.mapFromGlobal(event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Remove Center', lambda: self._updateIPVCentre(1, Scan.REMOVE_POINT,
                                                                             self.axis1.mapFromGlobal(
                                                                                 event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Edit IPV Centre Radius', lambda: self._updateIPVRadius(1))
            menuIPV.addSeparator()
            menuIPV.addAction('Clear IPV Data', lambda: self._removeIPVData(1))
            menu.addAction('Refresh Scan Data', lambda: self._refreshScanData(1))
            menu.exec(event.globalPos())
        elif self.s2 and self.axis2.underMouse():
            menu = QMenu()
            menuPoints = menu.addMenu('Points')
            menuPoints.addAction('Clear Frame Points', lambda: self._clearFramePoints(2))
            menuPoints.addSeparator()
            menuPoints.addAction('Clear All Points', lambda: self._clearScanPoints(2))
            menuIPV = menu.addMenu('IPV')
            menuIPV.addAction('Add Center',
                              lambda: self._updateIPVCentre(2, Scan.ADD_POINT,
                                                            self.axis2.mapFromGlobal(event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Remove Center', lambda: self._updateIPVCentre(1, Scan.REMOVE_POINT,
                                                                             self.axis2.mapFromGlobal(
                                                                                 event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Edit IPV Centre Radius', lambda: self._updateIPVRadius(2))
            menuIPV.addSeparator()
            menuIPV.addAction('Clear IPV Data', lambda: self._removeIPVData(2))
            menu.addAction('Refresh Scan Data', lambda: self._refreshScanData(2))
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
