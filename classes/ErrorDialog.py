"""
Error Dialog that is used throughout the application. Largely used for debugging, but also used for alerts.
"""

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel


class ErrorDialog(QDialog):
    """
    Custom dialog to show an error message with the exact error below the message.
    """

    def __init__(self, parent=None, message=None, e=None, f_name=None):
        super().__init__(parent)

        self.setWindowTitle("Error")

        buttonBox = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok, self)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f'Message: {message}'))
        layout.addWidget(QLabel(f'Error: {type(e).__name__} at line {e.__traceback__.tb_lineno} of {f_name}.\n'
                                f'{e}.'))
        layout.addWidget(buttonBox)

        buttonBox.accepted.connect(self.accept)

        self.exec()
