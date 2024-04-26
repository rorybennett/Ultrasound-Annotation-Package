from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QFormLayout, QLineEdit, QCheckBox


class ExportDialogs(QDialog):
    """
    Custom dialog for exporting frames.
    """

    def YOLODialog(self, parent=None):
        """
        Dialog for YOLO exports. Can choose Save Data Prefix, include prostate boxes and/or bladder boxes, and export
        save folder name.

        Parameters
        ----------
        parent: Main window, if available.

        Returns
        -------
        Save prefix, include prostate boxes, include bladder boxes, export name.
        """
        super().__init__(parent)

        self.setWindowTitle('YOLO Export Settings')

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        buttonBox = QDialogButtonBox(QBtn)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout = QFormLayout(self)
        # Save Data Prefix.
        prefixLE = QLineEdit()
        layout.addRow(QLabel('Save Prefix:'), prefixLE)
        prostateCB = QCheckBox()
        prostateCB.setChecked(True)
        layout.addRow(QLabel('Prostate Boxes:'), prostateCB)
        bladderCB = QCheckBox()
        layout.addRow(QLabel('Bladder Boxes:'), bladderCB)
        nameLE = QLineEdit()
        layout.addRow(QLabel('Export Name:'), nameLE)

        layout.addWidget(buttonBox)
        self.setLayout(layout)

        ok = self.exec()

        if ok:
            prefix = prefixLE.text()
            prostate = prostateCB.isChecked()
            bladder = bladderCB.isChecked()
            export = nameLE.text()
            return prefix, prostate, bladder, export
        else:
            return False

    def nnUNetDialog(self, parent=None):
        """
        Dialog for nnUNet exports. Can choose Save Data Prefix, include prostate points and/or bladder points, and
        export save folder name.

        Parameters
        ----------
        parent: Main window, if available.

        Returns
        -------
        Save prefix, include prostate points, include bladder points, export name.
        """
        super().__init__(parent)

        self.setWindowTitle('nnUNet Export Settings')

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        buttonBox = QDialogButtonBox(QBtn)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout = QFormLayout(self)
        # Save Data Prefix.
        prefixLE = QLineEdit()
        layout.addRow(QLabel('Save Prefix:'), prefixLE)
        prostateCB = QCheckBox()
        prostateCB.setChecked(True)
        layout.addRow(QLabel('Prostate Points:'), prostateCB)
        bladderCB = QCheckBox()
        layout.addRow(QLabel('Bladder Points:'), bladderCB)
        nameLE = QLineEdit()
        layout.addRow(QLabel('Export Name:'), nameLE)

        layout.addWidget(buttonBox)
        self.setLayout(layout)

        ok = self.exec()

        if ok:
            prefix = prefixLE.text()
            prostate = prostateCB.isChecked()
            bladder = bladderCB.isChecked()
            export = nameLE.text()
            return prefix, prostate, bladder, export
        else:
            return False

    def IPVDialog(self, parent=None):
        """
        Dialog for IPV exports. Can choose Save Data prefix, if images should be resampled, and their resample
        pixel density.

        Parameters
        ----------
        parent: Main window, if available.

        Returns
        -------
        Save prefix, if resampling should be used, and resample density.
        """
        super().__init__(parent)

        self.setWindowTitle(f'IPV Export Settings')

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        buttonBox = QDialogButtonBox(QBtn)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout = QFormLayout(self)
        # Save Data Prefix.
        prefixLE = QLineEdit()
        layout.addRow(QLabel('Save Prefix:'), prefixLE)
        # Resample Images CheckBox.
        resampleCB = QCheckBox()
        resampleCB.setChecked(True)
        layout.addRow(QLabel('Resample Images:'), resampleCB)
        # Resample Pixel Density.
        pixelDensityLE = QLineEdit('4')
        resampleCB.toggled.connect(lambda: pixelDensityLE.setEnabled(resampleCB.isChecked()))
        layout.addRow(QLabel('Pixel Density (Pixels/mm):'), pixelDensityLE)

        layout.addWidget(buttonBox)
        self.setLayout(layout)

        ok = self.exec()

        if ok:
            prefix = prefixLE.text()
            resample = resampleCB.isChecked()
            density = int(pixelDensityLE.text())
            return prefix, resample, density
        else:
            return False
