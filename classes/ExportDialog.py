from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QFormLayout, QLineEdit, QCheckBox


class ExportDialog(QDialog):
    """
    Custom dialog for exporting frames.
    """

    def __init__(self, model, plane, parent=None):
        super().__init__(parent)

        self.setWindowTitle(f'{model} Export Settings')

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QFormLayout(self)
        # Save Data Prefix.
        self.prefixLE = QLineEdit()
        self.layout.addRow(QLabel('Save Prefix:'), self.prefixLE)
        # Resample Images CheckBox.
        self.resampleCB = QCheckBox()
        self.resampleCB.setChecked(True)
        self.layout.addRow(QLabel('Resample Images:'), self.resampleCB)
        # Resample Pixel Density.
        self.pixelDensityLE = QLineEdit('4')
        self.resampleCB.toggled.connect(lambda: self.pixelDensityLE.setEnabled(self.resampleCB.isChecked()))
        self.layout.addRow(QLabel('Pixel Density (Pixels/mm):'), self.pixelDensityLE)

        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def executeDialog(self):
        ok = self.exec()

        if ok:
            prefix = self.prefixLE.text()
            resample = self.resampleCB.isChecked()
            density = int(self.pixelDensityLE.text())
            return prefix, resample, density
        else:
            return False
