from PyQt6.QtWidgets import QApplication
from views.main_window import MainWindow

from utils.logs import setup_logging


import sys
from logging import getLogger

setup_logging()
logger = getLogger(__name__)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        with open("src/ui/resources/style.qss", "r") as f:
            style = f.read()
            app.setStyleSheet(style)
    except FileNotFoundError:
        logger.warning("Stylesheet not found. Please create a 'style.qss' file.")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())