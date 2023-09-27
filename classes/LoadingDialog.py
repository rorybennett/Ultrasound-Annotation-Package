from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QWidget, QLabel


class LoadingDialog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 200)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.CustomizeWindowHint)


        self.labelAnimation = QLabel(self)

        self.movie = QMovie('loading.gif')
        self.movie.setScaledSize(QSize(200, 200))
        self.labelAnimation.setMovie(self.movie)

        self.start()
        self.show()

    def start(self):
        self.movie.start()

    def stop(self):
        self.movie.stop()
        self.close()

