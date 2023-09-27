from PyQt6.QtWidgets import QDialog, QLineEdit, QDialogButtonBox, QFormLayout


class InputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.address = QLineEdit(self)
        self.address.setText('http://127.0.0.1:5000/')
        self.modelName = QLineEdit(self)
        self.modelName.setText('TS1_1')
        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)

        layout = QFormLayout(self)
        layout.addRow("Enter Root Address:", self.address)
        layout.addRow("Enter Model Name:", self.modelName)
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

    def getInputs(self):
        return self.address.text(), self.modelName.text()
