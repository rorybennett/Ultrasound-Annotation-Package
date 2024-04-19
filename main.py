# main.py

"""Main Window for viewing and editing ultrasound scans."""
import multiprocessing
import os
import sys

import qdarktheme
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, \
    QCheckBox, QMenu, QInputDialog, QStyle, QMessageBox, QToolBar, QSpinBox, QRadioButton, QButtonGroup
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from classes import Export, Utils
from classes import Scan
from classes.ErrorDialog import ErrorDialog
from classes.FrameCanvas import FrameCanvas
from processes import PlayCine, AxisAnglePlot

INFER_LOCAL = 'INFER-LOCAL'
INFER_ONLINE = 'INFER-ONLINE'

basedir = os.path.dirname(__file__)


# noinspection PyUnresolvedReferences
class Main(QMainWindow):

    def __init__(self):
        """Initialise MainWindow."""
        # Setup GUI.
        super().__init__()
        self.setWindowTitle("Ultrasound Scan Editing")
        self.setWindowIcon(QIcon(f'{basedir}/res/main.png'))
        # Tooltip style.
        self.setStyleSheet(Utils.stylesheet)

        # Display 2 scans side-by-side inside central widget.
        self.mainWidget = QWidget(self)
        self.mainLayout = QHBoxLayout(self.mainWidget)
        self.mainWidget.installEventFilter(self)

        # Menu load data.
        self.menuLoadData = []
        # 2 vertical layouts.
        self.layouts = [QVBoxLayout(), QVBoxLayout()]
        # Toolbars.
        self.toolbars = [self._createToolBars(0), self._createToolBars(1)]
        # Title rows.
        self.titles = [Utils.createTitleLayout(), Utils.createTitleLayout()]
        # Buttons above canvas.
        self.buttons = [self._createTopButtons(0), self._createTopButtons(1)]
        # Boxes below canvas.
        self.boxes = [self._createBoxes(0), self._createBoxes(1)]
        # Canvases for displaying frames.
        self.canvases = [FrameCanvas(updateDisplay=lambda: self._updateDisplay(0),
                                     showProstatePointsBox=self.boxes[0].itemAt(0).widget(),
                                     showBladderPointsBox=self.boxes[0].itemAt(1).widget(),
                                     showProstateMaskBox=self.boxes[0].itemAt(2).widget(),
                                     showBladderMaskBox=self.boxes[0].itemAt(3).widget(),
                                     segmentProstateBox=self.toolbars[0].actions()[8].defaultWidget(),
                                     segmentBladderBox=self.toolbars[0].actions()[9].defaultWidget()),
                         FrameCanvas(updateDisplay=lambda: self._updateDisplay(1),
                                     showProstatePointsBox=self.boxes[1].itemAt(0).widget(),
                                     showBladderPointsBox=self.boxes[1].itemAt(1).widget(),
                                     showProstateMaskBox=self.boxes[1].itemAt(2).widget(),
                                     showBladderMaskBox=self.boxes[1].itemAt(3).widget(),
                                     segmentProstateBox=self.toolbars[1].actions()[8].defaultWidget(),
                                     segmentBladderBox=self.toolbars[1].actions()[9].defaultWidget())]
        # Canvas navigation toolbars.
        self.navBars = [NavigationToolbar(self.canvases[0], self),
                        NavigationToolbar(self.canvases[1], self)]

        # Left side.
        self.layouts[0].setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layouts[0].addLayout(self.titles[0])
        self.layouts[0].addLayout(self.buttons[0])
        self.layouts[0].addWidget(self.canvases[0])
        self.layouts[0].addWidget(self.navBars[0])
        self.layouts[0].addLayout(self.boxes[0])
        self.layouts[0].addItem(Utils.spacer)
        # Right side.
        self.layouts[1].setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layouts[1].addLayout(self.titles[1])
        self.layouts[1].addLayout(self.buttons[1])
        self.layouts[1].addWidget(self.canvases[1])
        self.layouts[1].addWidget(self.navBars[1])
        self.layouts[1].addLayout(self.boxes[1])
        self.layouts[1].addItem(Utils.spacer)
        # Add left and right to mainLayout.
        self.mainLayout.addLayout(self.layouts[0])
        self.mainLayout.addLayout(self.layouts[1])
        self.setCentralWidget(self.mainWidget)

        self._createMainMenu()

        # Scan directory Path.
        self.scansPath = f'C:/Users/roryb/GDOffline/Research/Scans'
        # Scans.
        self.scans = [Scan.Scan(self), Scan.Scan(self)]
        # Class for exporting data for training.
        self.export = Export.Export(self.scansPath)
        # Thread pool.
        self.threadPool = QThreadPool()
        # Processes.
        self.axisAngleProcess = [AxisAnglePlot.AxisAnglePlot(),
                                 AxisAnglePlot.AxisAnglePlot()]

        self.showMaximized()

    def _createMainMenu(self):
        """Create menus."""
        # Load scans menu.
        self.menuLoadScans = self.menuBar().addMenu("Load Scans")
        for i in range(2):
            self.menuLoadScans.addAction(f"Select Scan {i + 1} Folder...", lambda x=i: self._selectScanDialog(x))
            self.menuLoadScans.addAction(f"Open Scan {i + 1} Directory...",
                                         lambda x=i: self.scans[x].openDirectory()).setDisabled(True)

            self.menuLoadScans.addSeparator()
        self.menuLoadScans.addAction('Load AUS Patient', lambda: self._selectAUSPatientDialog())
        self.menuLoadScans.addSeparator()
        self.menuLoadScans.addAction('Load PUS Patient', lambda: self._selectPUSPatientDialog())

        # Load data menu
        menuLoadData = self.menuBar().addMenu("Load Data")
        self.menuLoadData.append(menuLoadData.addMenu('Load Scan 1 Data'))
        menuLoadData.addSeparator()
        self.menuLoadData.append(menuLoadData.addMenu('Load Scan 2 Data'))
        self.menuLoadData[0].setDisabled(True)
        self.menuLoadData[1].setDisabled(True)
        self.menuLoadData[0].aboutToShow.connect(lambda: self._populateLoadScanData(0))
        self.menuLoadData[1].aboutToShow.connect(lambda: self._populateLoadScanData(1))
        # Save data menu.
        self.menuSaveData = self.menuBar().addMenu("Save Data")
        self.menuSaveData.addAction('Save Scan 1 Data', lambda: self._saveData([0])).setDisabled(True)
        self.menuSaveData.addSeparator()
        self.menuSaveData.addAction('Save Scan 2 Data', lambda: self._saveData([1])).setDisabled(True)
        self.menuSaveData.addSeparator()
        self.menuSaveData.addAction('Save Both', lambda: self._saveData([0, 1])).setDisabled(True)
        # Export data menu.
        self.menuExport = self.menuBar().addMenu("Export Data")
        menuExportIPV = self.menuExport.addMenu('IPV')
        menuExportIPV.addAction('Transverse', lambda: self.export.exportIPVAUSData(Scan.PLANE_TRANSVERSE, self))
        menuExportIPV.addAction('Sagittal', lambda: self.export.exportIPVAUSData(Scan.PLANE_SAGITTAL, self))
        menuExportnnU = self.menuExport.addMenu('nnUNet')
        menuExportnnU.addAction('Transverse', lambda: self.export.exportnnUAUSData(Scan.PLANE_TRANSVERSE, self))
        menuExportnnU.addAction('Sagittal', lambda: self.export.exportnnUAUSData(Scan.PLANE_SAGITTAL, self))
        self.menuExport.addAction('Save Data', lambda: self.export.exportAllSaveData())
        self.menuExport.addSeparator()
        self.menuExport.addAction('Open Export Directory', lambda: self.export.openExportDirectory(basedir))
        # Reset data menu.
        self.menuReset = self.menuBar().addMenu("Reset Data")
        self.menuReset = self.menuReset.addAction("Reset Editing Data", lambda: self._resetEditingData())
        # Extra functions menu.
        self.menuExtras = self.menuBar().addMenu("Extras")
        self.menuExtras.addAction('Bullet Scan 1', lambda: self.scans[0].printBulletDimensions()).setDisabled(True)
        self.menuExtras.addAction('Bullet Scan 2', lambda: self.scans[1].printBulletDimensions()).setDisabled(True)
        self.menuExtras.addSeparator()
        menuExtrasNext = self.menuExtras.addMenu("Next")
        menuExtrasNext.addAction(f'Scan 1', lambda: self._navigatePatients(0, Scan.NEXT)).setDisabled(True)
        menuExtrasNext.addAction(f'Scan 2', lambda: self._navigatePatients(1, Scan.NEXT)).setDisabled(True)
        menuExtrasNext.addAction(f'Patient', lambda: self._navigatePatients(-1, Scan.NEXT)).setDisabled(True)
        menuExtrasPrevious = self.menuExtras.addMenu("Previous")
        menuExtrasPrevious.addAction(f'Scan 1', lambda: self._navigatePatients(0, Scan.PREVIOUS)).setDisabled(True)
        menuExtrasPrevious.addAction(f'Scan 2', lambda: self._navigatePatients(1, Scan.PREVIOUS)).setDisabled(True)
        menuExtrasPrevious.addAction(f'Patient', lambda: self._navigatePatients(-1, Scan.NEXT)).setDisabled(True)

    def _createToolBars(self, scan):
        """Create left and right toolbars (mirrored)."""
        toolbar = QToolBar(f'ToolBar {scan}')
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea if scan == 0 else Qt.ToolBarArea.RightToolBarArea, toolbar)

        copyPreviousAction = QAction(QIcon(f"{basedir}/res/copy_previous.png"), "Copy previous frame points.", self)
        copyPreviousAction.triggered.connect(lambda: self._copyFramePoints(scan, Scan.PREVIOUS))
        toolbar.addAction(copyPreviousAction)

        copyNextAction = QAction(QIcon(f"{basedir}/res/copy_next.png"), "Copy points from next frame.", self)
        copyNextAction.triggered.connect(lambda: self._copyFramePoints(scan, Scan.NEXT))
        toolbar.addAction(copyNextAction)

        shrinkAction = QAction(QIcon(f"{basedir}/res/shrink.png"), "Shrink points around CoM.", self)
        shrinkAction.triggered.connect(lambda: self._shrinkExpandPoints(scan, -shrinkSpinBox.value()))
        toolbar.addAction(shrinkAction)

        shrinkSpinBox = QSpinBox(minimum=1, maximum=50, value=5)
        shrinkSpinBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        shrinkSpinBox.setToolTip("Shrink Scale (minimum 1).")
        toolbar.addWidget(shrinkSpinBox)

        expandAction = QAction(QIcon(f"{basedir}/res/expand.png"), "Expand points around CoM.", self)
        expandAction.triggered.connect(lambda: self._shrinkExpandPoints(scan, expandSpinBox.value()))
        toolbar.addAction(expandAction)

        expandSpinBox = QSpinBox(minimum=1, maximum=50, value=5)
        expandSpinBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        expandSpinBox.setToolTip("Expand Scale (minimum 1).")
        toolbar.addWidget(expandSpinBox)

        distAction = QAction(QIcon(f'{basedir}/res/distribute.png'), 'Distribute points along spline.', self)
        distAction.triggered.connect(lambda: self._distributePoints(scan, distSpinBox.value()))
        toolbar.addAction(distAction)

        distSpinBox = QSpinBox(minimum=5, maximum=100, value=50)
        distSpinBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        distSpinBox.setToolTip("Number of points in even distribution.")
        toolbar.addWidget(distSpinBox)

        prostateSegmentationButton = QRadioButton("Prostate")
        prostateSegmentationButton.setToolTip("Segment the prostate.")
        prostateSegmentationButton.setChecked(True)
        toolbar.addWidget(prostateSegmentationButton)

        bladderSegmentationButton = QRadioButton("Bladder")
        bladderSegmentationButton.setToolTip("Segment the bladder.")
        toolbar.addWidget(bladderSegmentationButton)

        segmentationGroup = QButtonGroup()
        segmentationGroup.addButton(prostateSegmentationButton)
        segmentationGroup.addButton(bladderSegmentationButton)

        toolbar.setDisabled(True)
        toolbar.setMovable(False)

        return toolbar

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

        nav50IMUButton = QPushButton('IMU Centre')
        nav50IMUButton.setToolTip('Show frame at 50% of sweep (based on IMU data).')
        nav50IMUButton.clicked.connect(lambda: self._onNav50Clicked(scan, Scan.NAV_TYPE_IMU))
        nav50IMUButton.setDisabled(True)
        layout.addWidget(nav50IMUButton)

        nav50TS1Button = QPushButton('TS1 Centre')
        nav50TS1Button.setToolTip('Show frame at 50% of prostate (based on TS1 data).')
        nav50TS1Button.clicked.connect(lambda: self._onNav50Clicked(scan, Scan.NAV_TYPE_TS1))
        nav50TS1Button.setDisabled(True)
        layout.addWidget(nav50TS1Button)

        axisAngleButton = QPushButton('Axis Angle Plot')
        axisAngleButton.setToolTip('Show axis angle plot.')
        axisAngleButton.clicked.connect(lambda: self._onAxisAngleClicked(scan))
        axisAngleButton.setDisabled(True)
        layout.addWidget(axisAngleButton)

        return layout

    def _createBoxes(self, scan: int):
        """Create checkboxes below canvas."""
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        prostatePoints = QCheckBox('Show Prostate\nPoints')
        prostatePoints.setChecked(True)
        prostatePoints.stateChanged.connect(lambda: self._updateDisplay(scan))
        prostatePoints.setDisabled(True)
        layout.addWidget(prostatePoints)

        bladderPoints = QCheckBox('Show Bladder\nPoints')
        bladderPoints.setChecked(True)
        bladderPoints.stateChanged.connect(lambda: self._updateDisplay(scan))
        bladderPoints.setDisabled(True)
        layout.addWidget(bladderPoints)

        prostateMask = QCheckBox('Show Prostate\nMask')
        prostateMask.setChecked(False)
        prostateMask.stateChanged.connect(lambda: self._updateDisplay(scan))
        prostateMask.setDisabled(True)
        layout.addWidget(prostateMask)

        bladderMask = QCheckBox('Show Bladder\nMask')
        bladderMask.setChecked(False)
        bladderMask.stateChanged.connect(lambda: self._updateDisplay(scan))
        bladderMask.setDisabled(True)
        layout.addWidget(bladderMask)

        return layout

    def _distributePoints(self, scan: int, count: int):
        """Distribute points along a generated spline."""
        pb = Scan.PROSTATE if self.toolbars[scan].actions()[8].defaultWidget().isChecked() else Scan.BLADDER
        self.canvases[scan].distributeFramePoints(count, pb)
        self._updateDisplay(scan)

    def _onAxisAngleClicked(self, scan):
        """Start Axis Angle plotting process."""
        self.axisAngleProcess[scan].start(self.scans[scan])

    def _onNav50Clicked(self, scan: int, navType: str):
        """Travel to the frame at 50% of scan or prostate."""
        if navType == Scan.NAV_TYPE_IMU:
            self.scans[scan].navigate(self.scans[scan].frameAtScanPercent(50))
        else:
            self.scans[scan].navigate(self.scans[scan].frameAtTS1Centre())
        self._updateDisplay(scan)

    def _updateTitle(self, scan: int):
        """Update title information."""
        patient, scanType, scanPlane, scanNumber, scanFrames = self.scans[scan].getScanDetails()
        self.titles[scan].itemAt(0).widget().setText(f'Patient: {patient}')
        self.titles[scan].itemAt(1).widget().setText(f'Type: {scanType}')
        self.titles[scan].itemAt(2).widget().setText(f'Plane: {scanPlane}')
        self.titles[scan].itemAt(3).widget().setText(f'Number: {scanNumber}')
        self.titles[scan].itemAt(4).widget().setText(f'Frames: {scanFrames}')

    def _onCineClicked(self, scan: int):
        """Play a cine of the scan in a separate window."""
        patient, scanType, scanPlane, _, _ = self.scans[scan].getScanDetails()
        cine = PlayCine.PlayCine(self.scans[scan].frames, patient, scanType, scanPlane)
        cine.startProcess()

    def _resetEditingData(self):
        """Reset all editing data after confirmation."""
        confirm = QMessageBox.question(self, 'Reset Editing Data', 'Are you sure you wan to reset all editing data?',
                                       buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        if confirm == QMessageBox.StandardButton.Ok:
            result = Utils.resetEditingData(self.scansPath)
            text = 'Editing Data has been reset!' if result else 'An error occurred while resetting Editing Data!'
            dialog = QMessageBox(parent=self, text=text)
            dialog.setWindowTitle('Reset Editing Data')
            dialog.exec()
            [self._refreshScanData(i) for i in [0, 1]]

    def _saveData(self, scans: list):
        """Save Scan point data. Check for overwrite"""
        for scan in scans:
            saveName, ok = QInputDialog.getText(self, f'Save Scan {scan + 1} Data', 'Enter User Name:')

            if ok:
                if not saveName:
                    ErrorDialog(self, 'User name is empty.', '')
                    self._saveData(scan)
                    return
                self.scans[scan].checkSaveDataDirectory()
                # "Overwrite" old save directories.
                if saveName in [i.split('_')[0] for i in self.scans[scan].getSaveData()]:
                    confirm = QMessageBox.question(self, 'Overwrite Save Data', 'Overwrite old Save Data?',
                                                   buttons=QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

                    if confirm == QMessageBox.StandardButton.Ok:
                        self.scans[scan].deleteUserData(saveName)
                    else:
                        return
                self.scans[scan].saveUserData(saveName, scan)

    def _populateLoadScanData(self, scan: int):
        """Populate the load submenu just before opening."""
        self.menuLoadData[scan].clear()
        actions = []
        for fileName in self.scans[scan].getSaveData():
            action = QAction(fileName.split('_')[0], self)
            action.triggered.connect(lambda _, x=fileName: self._loadSaveData(scan, x))
            actions.append(action)
        self.menuLoadData[scan].addActions(actions)

    def _loadSaveData(self, scan: int, fileName: str):
        """Load save data of scan and update display."""
        self.scans[scan].loadSaveData(fileName)
        self._refreshScanData(scan)

    def _navigatePatients(self, scan: int, direction: str):
        """Load previous or next patient."""
        if scan > -1:
            patient, scanType, scanPlane, scanNumber, _ = self.scans[scan].getScanDetails()
            nextScanPath = (f'{self.scansPath}/'
                            f'{int(patient) - 1 if direction == Scan.PREVIOUS else int(patient) + 1}/'
                            f'{scanType}/{scanPlane}/{scanNumber}')
            self._loadScan(scan, nextScanPath)
        else:
            for i in range(2):
                if self.scans[i].loaded:
                    patient, scanType, scanPlane, scanNumber, _ = self.scans[i].getScanDetails()
                    nextScanPath = (f'{self.scansPath}/'
                                    f'{int(patient) - 1 if direction == Scan.PREVIOUS else int(patient) + 1}/'
                                    f'{scanType}/{scanPlane}/{scanNumber}')
                    self._loadScan(i, nextScanPath)



    def _selectAUSPatientDialog(self):
        """Load both scans of an AUS patient."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Patient', directory=self.scansPath)

        if not scanPath:
            return

        transverse = f'{scanPath}/AUS/Transverse/1'
        self._loadScan(0, transverse)
        sagittal = f'{scanPath}/AUS/Sagittal/1'
        self._loadScan(1, sagittal)

    def _selectPUSPatientDialog(self):
        """Load both scans of a PUS patient."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Patient', directory=self.scansPath)

        if not scanPath:
            return

        transverse = f'{scanPath}/PUS/Transverse/1'
        self._loadScan(0, transverse)
        sagittal = f'{scanPath}/PUS/Sagittal/1'
        self._loadScan(1, sagittal)

    def _selectScanDialog(self, scan: int):
        """Show dialog for selecting a scan folder."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Scan {scan + 1}', directory=self.scansPath)

        if not scanPath:
            return

        self._loadScan(scan, scanPath)

    def _loadScan(self, scan: int, scanPath: str):
        """Load a scan."""
        try:
            self.scans[scan].load(scanPath)
            self.toolbars[scan].setEnabled(True)
            self.navBars[scan].setMaximumWidth(self.scans[scan].displayDimensions[0])
            self.canvases[scan].linkedScan = self.scans[scan]
            for i in range(self.buttons[scan].count()):
                self.buttons[scan].itemAt(i).widget().setEnabled(True)
            self.layouts[scan].itemAt(2).widget().setFixedSize(self.scans[scan].displayDimensions[0],
                                                               self.scans[scan].displayDimensions[1])
            for i in range(self.boxes[scan].count()):
                self.boxes[scan].itemAt(i).widget().setEnabled(True)

            self.menuLoadData[scan].setEnabled(True)
            self.menuLoadScans.actions()[1 if scan == 0 else 4].setEnabled(True)
            self.menuSaveData.actions()[0 if scan == 0 else 2].setEnabled(True)
            self.menuSaveData.actions()[4].setEnabled(True) if self.scans[0].loaded and self.scans[1].loaded else \
                self.menuSaveData.actions()[4].setDisabled(True)
            self.menuExtras.actions()[scan].setEnabled(True)
            self.menuExtras.menuInAction(self.menuExtras.actions()[3]).actions()[scan].setEnabled(True)
            self.menuExtras.menuInAction(self.menuExtras.actions()[3]).actions()[2].setEnabled(True)
            self.menuExtras.menuInAction(self.menuExtras.actions()[4]).actions()[scan].setEnabled(True)
            self.menuExtras.menuInAction(self.menuExtras.actions()[4]).actions()[2].setEnabled(True)

            self._updateTitle(scan)
            self._updateDisplay(scan, new=True)
        except Exception as e:
            ErrorDialog(self, 'Error loading Scan data', e)

    def _updateDisplay(self, scan: int, new=False):
        """Update the shown frame and position on plot."""
        self.canvases[scan].updateAxis(new)
        self.axisAngleProcess[scan].updateIndex(self.scans[scan].currentFrame - 1)

    def _shrinkExpandPoints(self, scan: int, amount):
        """Expand or shrink points around centre of mass."""
        pb = Scan.PROSTATE if self.toolbars[scan].actions()[8].defaultWidget().isChecked() else Scan.BLADDER
        self.scans[scan].shrinkExpandPoints(amount, pb)
        self._updateDisplay(scan)

    def _clearScanPoints(self, scan: int):
        """Clear all points in a Scan, then update display."""
        self.scans[scan].clearScanPoints()
        self._updateDisplay(scan)

    def _clearFramePoints(self, scan: int, prostateBladder):
        """Clear frame prostate points from scan, then update display."""
        self.scans[scan].clearFramePoints(prostateBladder)
        self._updateDisplay(scan)

    def _copyFramePoints(self, scan: int, location):
        """Copy points from either previous or next frame."""
        pb = Scan.PROSTATE if self.toolbars[scan].actions()[8].defaultWidget().isChecked() else Scan.BLADDER
        self.scans[scan].copyFramePoints(location, pb)
        self._updateDisplay(scan)

    def _refreshScanData(self, scan: int):
        """Refresh scan data by re-reading files."""
        if self.scans[scan].loaded:
            self.scans[scan].load(self.scans[scan].path, self.scans[scan].currentFrame)
            self._updateDisplay(scan)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key press events."""
        if self.scans[0].loaded and self.canvases[0].underMouse():
            if event.key() == Qt.Key.Key_W:
                self.scans[0].navigate(Scan.NAVIGATION['w'])
            elif event.key() == Qt.Key.Key_S:
                self.scans[0].navigate(Scan.NAVIGATION['s'])
            elif self.buttons[0].itemAt(3).widget().isChecked() and event.key() == Qt.Key.Key_D:
                self.toolbars[0].actions()[7].trigger()
            self._updateDisplay(0)

        elif self.scans[1].loaded and self.canvases[1].underMouse():
            if event.key() == Qt.Key.Key_W:
                self.scans[1].navigate(Scan.NAVIGATION['w'])
            elif event.key() == Qt.Key.Key_S:
                self.scans[1].navigate(Scan.NAVIGATION['s'])
            elif self.buttons[1].itemAt(3).widget().isChecked() and event.key() == Qt.Key.Key_D:
                self.toolbars[1].actions()[7].trigger()
            self._updateDisplay(1)

    def contextMenuEvent(self, event):
        for i in [0, 1]:
            if self.scans[i].loaded and self.canvases[i].underMouse():
                menu = QMenu()
                menuPoints = menu.addMenu('Points')
                menuPoints.addAction('Clear Frame Prostate Points', lambda: self._clearFramePoints(i, Scan.PROSTATE))
                menuPoints.addAction('Clear Frame Bladder Points', lambda: self._clearFramePoints(i, Scan.BLADDER))
                menuPoints.addSeparator()
                menuPoints.addAction('Clear All Points', lambda: self._clearScanPoints(i))
                menu.addAction('Refresh Scan Data', lambda: self._refreshScanData(i))
                menu.exec(event.globalPos())


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.excepthook = except_hook
    # main()
    qdarktheme.enable_hi_dpi()
    editingApp = QApplication([])

    qdarktheme.setup_theme()

    mainWindow = Main()

    sys.exit(editingApp.exec())

# To create an executable:
# pyinstaller main.py
# Add res to main .spec - a=[..., datas=[('res', 'res')],...]
# pyinstaller main.spec
