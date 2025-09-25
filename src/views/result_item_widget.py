from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap

import os
import logging

from ui.result_item_widget_ui import Ui_ResultItemWidget


logger = logging.getLogger(__name__)


class ResultItemWidget(QWidget):
    """
    A custom widget to display a single recognition result, including
    a name and a confidence progress bar.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_ResultItemWidget()
        self.ui.setupUi(self)

    def set_data(self, name: str, confidence: float, icon_path: str = None):
        """
        Populates the widget with the result data.

        Args:
            name (str): The name of the recognized person.
            confidence (float): The similarity score (from 0.0 to 1.0).
            icon_path (str, optional): Path to an icon for the user. Defaults to None.
        """
        
        self.ui.profile_name.setText(name)
        
        progress_value = int(confidence * 100)
        self.ui.confidance_bar.setValue(progress_value)
        
        if icon_path:
            pixmap = QPixmap(icon_path)
            self.ui.profile_image.setPixmap(pixmap)
        elif os.path.exists("./data/students/default.png"):

            logger.info("No Student Profile Image found, Loading Default Profile Image.")

            self.ui.profile_image.setPixmap(QPixmap("./data/students/default.png"))
        
        else:

            logger.warning(f"Default Profile Image Not Found, Program started at: {os.getcwd()}")