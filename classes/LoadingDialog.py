from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QLabel, QDialog, QVBoxLayout


class LoadingDialog(QDialog):
    def __init__(self, parent=None, loadingMessage='Loading...'):
        super().__init__(parent)
        # self.setFixedSize(400, 400)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet('border : 2px solid white;')

        labelMessage = QLabel(self)
        labelMessage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        labelMessage.setText(loadingMessage)
        labelMessage.setStyleSheet('border:none')
        labelAnimation = QLabel(self)
        labelAnimation.setStyleSheet('border:none')

        self.movie = QMovie('loading.gif')
        self.movie.setScaledSize(QSize(200, 200))
        labelAnimation.setMovie(self.movie)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(labelMessage)
        self.layout.addWidget(labelAnimation)
        self.setLayout(self.layout)

    def start(self):
        self.movie.start()
        self.exec()

    def stop(self):
        self.movie.stop()
        self.close()
