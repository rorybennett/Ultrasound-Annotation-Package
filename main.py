# main.py

"""Main Window for viewing and editing ultrasound scans."""
import multiprocessing
import os
import sys
from pathlib import Path

import qdarktheme
from PyQt6 import QtGui
from PyQt6.QtCore import Qt, QPoint, QThreadPool
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMainWindow, QApplication, QFileDialog, QHBoxLayout, QWidget, QVBoxLayout, QPushButton, \
    QCheckBox, QMenu, QInputDialog, QStyle, QMessageBox, QToolBar, QSpinBox

from classes import Export, Utils
from classes import Scan
from classes.ErrorDialog import ErrorDialog
from classes.FrameCanvas import FrameCanvas
from classes.IPVInferenceWorker import IPVInferenceWorker
from classes.InputDialog import InputDialog
from classes.LoadingDialog import LoadingDialog
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
                                     showPointsBox=self.boxes[0].itemAt(0).widget(),
                                     showIPVBox=self.boxes[0].itemAt(1).widget(),
                                     dragButton=self.toolbars[0].actions()[6]),
                         FrameCanvas(updateDisplay=lambda: self._updateDisplay(1),
                                     showPointsBox=self.boxes[1].itemAt(0).widget(),
                                     showIPVBox=self.boxes[1].itemAt(1).widget(),
                                     dragButton=self.toolbars[1].actions()[6])]

        # Left side.
        self.layouts[0].setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layouts[0].addLayout(self.titles[0])
        self.layouts[0].addLayout(self.buttons[0])
        self.layouts[0].addWidget(self.canvases[0])
        self.layouts[0].addLayout(self.boxes[0])
        self.layouts[0].addItem(Utils.spacer)
        # Right side.
        self.layouts[1].setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.layouts[1].addLayout(self.titles[1])
        self.layouts[1].addLayout(self.buttons[1])
        self.layouts[1].addWidget(self.canvases[1])
        self.layouts[1].addLayout(self.boxes[1])
        self.layouts[1].addItem(Utils.spacer)
        # Add left and right to mainLayout.
        self.mainLayout.addLayout(self.layouts[0])
        self.mainLayout.addLayout(self.layouts[1])
        self.setCentralWidget(self.mainWidget)

        self._createMainMenu()

        # Scan directory Path.
        self.scansPath = f'{basedir}/Scans'
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

    def _distributePoints(self, scan: int, count: int):
        """Distribute points along a generated spline."""
        self.canvases[scan].distributeFramePoints(count)
        self._updateDisplay(scan)

    def _toggleSegmentationTB(self, scan: int):
        """Toggle segmentation tool bar for scan."""
        self.toolbars[scan].setEnabled(True if self.buttons[scan].itemAt(3).widget().isChecked() else False)

    def _onAxisAngleClicked(self, scan):
        """Start Axis Angle plotting process."""
        self.axisAngleProcess[scan].start(self.scans[scan])

    def _onNav50Clicked(self, scan: int):
        """Travel to the frame at 50%."""
        self.scans[scan].navigate(self.scans[scan].frameAtScanPercent(50))
        self._updateDisplay(scan)

    def _updateTitle(self, scan: int):
        """Update title information."""
        patient, scanType, scanPlane, scanNumber, scanFrames = self.scans[scan].getScanDetails()
        self.titles[scan].itemAt(0).widget().setText(f'Patient: {patient}')
        self.titles[scan].itemAt(0).widget().setText(f'Type: {scanType}')
        self.titles[scan].itemAt(0).widget().setText(f'Plane: {scanPlane}')
        self.titles[scan].itemAt(0).widget().setText(f'Number: {scanNumber}')
        self.titles[scan].itemAt(0).widget().setText(f'Frames: {scanFrames}')

    def _createToolBars(self, scan):
        """Create left and right toolbars (mirrored)."""
        toolbar = QToolBar(f'ToolBar {scan}')
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea if scan == 0 else Qt.ToolBarArea.RightToolBarArea, toolbar)

        copyPreviousAction = QAction(QIcon(f"{basedir}/res/copy_previous.png"), "Copy previous frame points.",
                                     self)
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

        dragButton = QAction(QIcon(f"{basedir}/res/drag.png"), "Drag all points on frame.", self)
        dragButton.setCheckable(True)
        toolbar.addAction(dragButton)

        distAction = QAction(QIcon(f'{basedir}/res/distribute.png'), 'Distribute points along spline.', self)
        distAction.triggered.connect(lambda: self._distributePoints(scan, distSpinBox.value()))
        toolbar.addAction(distAction)

        distSpinBox = QSpinBox(minimum=5, maximum=100, value=20)
        distSpinBox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        distSpinBox.setToolTip("Number of points in even distribution.")
        toolbar.addWidget(distSpinBox)

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

        nav50Button = QPushButton('50%')
        nav50Button.setToolTip('Show frame at 50% (based on IMU data).')
        nav50Button.clicked.connect(lambda: self._onNav50Clicked(scan))
        nav50Button.setDisabled(True)
        layout.addWidget(nav50Button)

        axisAngleButton = QPushButton('Axis Angle Plot')
        axisAngleButton.setToolTip('Show axis angle plot.')
        axisAngleButton.clicked.connect(lambda: self._onAxisAngleClicked(scan))
        axisAngleButton.setDisabled(True)
        layout.addWidget(axisAngleButton)

        segmentationBox = QCheckBox('Segmentation')
        segmentationBox.setToolTip('Enable segmentation tool bar.')
        segmentationBox.stateChanged.connect(lambda: self._toggleSegmentationTB(scan))
        segmentationBox.setDisabled(True)
        layout.addWidget(segmentationBox)

        return layout

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
        patient, scanType, scanPlane, _, _ = self.scans[scan].getScanDetails()
        cine = PlayCine.PlayCine(self.scans[scan].frames, patient, scanType, scanPlane)
        cine.startProcess()

    def _createMainMenu(self):
        """Create menus."""
        # Load scans menu.
        self.menuLoadScans = self.menuBar().addMenu("Load Scans")
        self.menuLoadScans.addAction("Select Scan 1 Folder...", lambda: self._selectScanDialog(0))
        self.menuLoadScans.addAction("Open Scan 1 Directory...",
                                     lambda: self.scans[0].openDirectory()).setDisabled(True)
        self.menuLoadScans.addSeparator()
        self.menuLoadScans.addAction("Select Scan 2 Folder...", lambda: self._selectScanDialog(1))
        self.menuLoadScans.addAction("Open Scan 2 Directory...",
                                     lambda: self.scans[1].openDirectory()).setDisabled(True)
        # Inference menu.
        menuInference = self.menuBar().addMenu('Inference')
        menuIPV = menuInference.addMenu('IPV Inference')
        self.menuIPV1 = menuIPV.addAction('Infer Scan 1', lambda: self._ipvInference(0))
        self.menuIPV1.setDisabled(True)
        self.menuIPV2 = menuIPV.addAction('Infer Scan 2', lambda: self._ipvInference(1))
        self.menuIPV2.setDisabled(True)
        # Load data menu
        menuLoadData = self.menuBar().addMenu("Load Data")
        self.menuLoadData1 = menuLoadData.addMenu('Load Scan 1 Data')
        self.menuLoadData1.setDisabled(True)
        self.menuLoadData1.aboutToShow.connect(lambda: self._populateLoadScanData(0))
        menuLoadData.addSeparator()
        self.menuLoadData2 = menuLoadData.addMenu('Load Scan 2 Data')
        self.menuLoadData2.setDisabled(True)
        self.menuLoadData2.aboutToShow.connect(lambda: self._populateLoadScanData(1))
        # Save data menu.
        self.menuSaveData = self.menuBar().addMenu("Save Data")
        self.menuSaveData.addAction('Save Scan 1 Data', lambda: self._saveData(0)).setDisabled(True)
        self.menuSaveData.addSeparator()
        self.menuSaveData.addAction('Save Scan 2 Data', lambda: self._saveData(1)).setDisabled(True)
        # Export data menu.
        self.menuExport = self.menuBar().addMenu("Export Data")
        menuExportIPV = self.menuExport.addMenu('IPV')
        menuExportIPV.addAction('Transverse', lambda: self.export.exportIPVAUSData(Scan.PLANE_TRANSVERSE, self))
        menuExportIPV.addAction('Sagittal', lambda: self.export.exportIPVAUSData(Scan.PLANE_SAGITTAL, self))
        self.menuExport.addAction('Save Data', lambda: self.export.exportAllSaveData())
        # Reset data menu.
        self.menuReset = self.menuBar().addMenu("Reset Data")
        self.menuReset = self.menuReset.addAction("Reset Editing Data", lambda: self._resetEditingData())

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

    def _ipvInference(self, scan: int):
        """Send current frame for IPV inference, either online or locally."""
        dlg = InputDialog()
        ok = dlg.exec()
        address, modelName = dlg.getInputs()

        if not ok or not address or not modelName:
            return

        loading = LoadingDialog(loadingMessage='Inferring IPV Points...', basedir=basedir)

        worker = IPVInferenceWorker(self.scans[scan], address, modelName)

        worker.signals.started.connect(lambda: self.setDisabled(True))
        worker.signals.started.connect(lambda: loading.start())
        worker.signals.finished.connect(lambda: loading.stop())
        worker.signals.finished.connect(lambda: self.setDisabled(False))
        worker.signals.finished.connect(lambda: self._updateDisplay(scan))

        self.threadPool.start(worker)

    def _saveData(self, scan: int):
        """Save Scan point data. Check for overwrite"""
        userName, ok = QInputDialog.getText(self, 'Save Current Data', 'Enter User Name:')

        if ok:
            if not userName:
                ErrorDialog(self, 'User name is empty.', '')
                self._saveData(scan)
                return
            self.scans[scan].saveUserData(userName)

    def _populateLoadScanData(self, scan: int):
        """Populate the load submenu just before opening."""
        if scan == 0:
            self.menuLoadData1.clear()
            actions = []
            for fileName in self.scans[scan].getSaveData():
                action = QAction(fileName, self)
                action.triggered.connect(lambda _, x=fileName: self._loadSaveData(scan, x))
                actions.append(action)
            self.menuLoadData1.addActions(actions)
        else:
            self.menuLoadData2.clear()
            actions = []
            for fileName in self.scans[scan].getSaveData():
                action = QAction(fileName, self)
                action.triggered.connect(lambda _, x=fileName: self._loadSaveData(scan, x))
                actions.append(action)
            self.menuLoadData2.addActions(actions)

    def _loadSaveData(self, scan: int, fileName: str):
        """Load save data of scan and update display."""
        self.scans[scan].loadSaveData(fileName)
        self._refreshScanData(scan)

    def _selectScanDialog(self, scan: int):
        """Show dialog for selecting a scan folder."""
        scanPath = QFileDialog.getExistingDirectory(self, caption=f'Select Scan {scan + 1}', directory=self.scansPath)

        if not scanPath:
            return

        try:
            self.scans[scan].load(scanPath)
            self.canvases[scan].linkedScan = self.scans[scan]
            for i in range(self.buttons[scan].count()):
                self.buttons[scan].itemAt(i).widget().setEnabled(True)
            self.layouts[scan].itemAt(2).widget().setFixedSize(self.scans[scan].displayDimensions[0],
                                                               self.scans[scan].displayDimensions[1])
            for i in range(self.boxes[scan].count()):
                self.boxes[scan].itemAt(i).widget().setEnabled(True)

            if scan == 0:
                self.menuLoadScans.actions()[1].setEnabled(True)
                self.menuIPV1.setEnabled(True)
                self.menuLoadData1.setEnabled(True)
                self.menuSaveData.actions()[0].setEnabled(True)
            else:
                self.menuLoadScans.actions()[4].setEnabled(True)
                self.menuIPV2.setEnabled(True)
                self.menuLoadData2.setEnabled(True)
                self.menuSaveData.actions()[2].setEnabled(True)

            self._updateTitle(scan)
            self._updateDisplay(scan)
        except Exception as e:
            ErrorDialog(self, 'Error loading Scan data', e)

    def _updateDisplay(self, scan: int):
        """Update the shown frame and position on plot."""
        self.canvases[scan].updateAxis()
        self.axisAngleProcess[scan].updateIndex(self.scans[scan].currentFrame)

    def _shrinkExpandPoints(self, scan: int, amount):
        """Expand or shrink points around centre of mass."""
        self.scans[scan].shrinkExpandPoints(amount)
        self._updateDisplay(scan)

    def _clearScanPoints(self, scan: int):
        """Clear all points in a Scan, then update display."""
        self.scans[scan].clearScanPoints()
        self._updateDisplay(scan)

    def _clearFramePoints(self, scan: int):
        """Clear frame points from scan, then update display."""
        self.scans[scan].clearFramePoints()
        self._updateDisplay(scan)

    def _updateIPVCentre(self, scan: int, addOrRemove: str, position: QPoint):
        """Add or remove IPV centre then update display."""
        position = [position.x(), self.scans[scan].displayDimensions[1] - position.y()]
        self.scans[scan].updateIPVCentre(position, addOrRemove)
        self._updateDisplay(scan)

    def _updateIPVRadius(self, scan: int):
        """Dialog for updating IPV centre radius."""
        radius, ok = QInputDialog.getText(self, 'Update IPV Radius', 'Enter Radius:')

        if ok:
            try:
                radius = int(radius)
                self.scans[scan].updateIPVRadius(radius)
                self._updateDisplay(scan)
            except Exception as e:
                ErrorDialog(self, f'Error converting radius to int', e)

    def _removeIPVData(self, scan: int):
        """Remove all IPV data of the Scan."""
        self.scans[scan].removeIPVData()
        self._updateDisplay(scan)

    def _copyFramePoints(self, scan: int, location):
        """Copy points from either previous or next frame."""
        self.scans[scan].copyFramePoints(location)
        self._updateDisplay(scan)

    def _refreshScanData(self, scan: int):
        """Refresh scan data by re-reading files."""
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
            self._updateDisplay(1)

    # def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
    #     QMainWindow.resizeEvent(self, a0)
    #     if self.s1:
    #         self.s1 = Scan.Scan(self.s1.path, startingFrame=self.s1.currentFrame, window=self)
    #         self.left.itemAt(2).widget().setFixedSize(self.s1.displayDimensions[0],
    #                                                   self.s1.displayDimensions[1])
    #         self._updateDisplay(1)
    #     if self.s2:
    #         self.s2 = Scan.Scan(self.s2.path, startingFrame=self.s2.currentFrame, window=self)
    #         self.right.itemAt(2).widget().setFixedSize(self.s2.displayDimensions[0],
    #                                                    self.s2.displayDimensions[1])
    #         self._updateDisplay(2)

    def contextMenuEvent(self, event):
        if self.scans[0].loaded and self.canvases[0].underMouse():
            menu = QMenu()
            menuPoints = menu.addMenu('Points')
            menuPoints.addAction('Clear Frame Points', lambda: self._clearFramePoints(0))
            menuPoints.addSeparator()
            menuPoints.addAction('Clear All Points', lambda: self._clearScanPoints(0))
            menuIPV = menu.addMenu('IPV')
            menuIPV.addAction('Add Center',
                              lambda: self._updateIPVCentre(0, Scan.ADD_POINT,
                                                            self.canvases[0].mapFromGlobal(event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Remove Center', lambda: self._updateIPVCentre(0, Scan.REMOVE_POINT,
                                                                             self.canvases[0].mapFromGlobal(
                                                                                 event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Edit IPV Centre Radius', lambda: self._updateIPVRadius(0))
            menuIPV.addSeparator()
            menuIPV.addAction('Clear IPV Data', lambda: self._removeIPVData(0))
            menu.addAction('Refresh Scan Data', lambda: self._refreshScanData(0))
            menu.exec(event.globalPos())
        elif self.scans[1].loaded and self.canvases[1].underMouse():
            menu = QMenu()
            menuPoints = menu.addMenu('Points')
            menuPoints.addAction('Clear Frame Points', lambda: self._clearFramePoints(1))
            menuPoints.addSeparator()
            menuPoints.addAction('Clear All Points', lambda: self._clearScanPoints(1))
            menuIPV = menu.addMenu('IPV')
            menuIPV.addAction('Add Center',
                              lambda: self._updateIPVCentre(1, Scan.ADD_POINT,
                                                            self.canvases[1].mapFromGlobal(event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Remove Center', lambda: self._updateIPVCentre(1, Scan.REMOVE_POINT,
                                                                             self.canvases[1].mapFromGlobal(
                                                                                 event.globalPos())))
            menuIPV.addSeparator()
            menuIPV.addAction('Edit IPV Centre Radius', lambda: self._updateIPVRadius(1))
            menuIPV.addSeparator()
            menuIPV.addAction('Clear IPV Data', lambda: self._removeIPVData(1))
            menu.addAction('Refresh Scan Data', lambda: self._refreshScanData(1))
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
