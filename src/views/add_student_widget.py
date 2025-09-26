import os
from PyQt6.QtWidgets import QDialog, QFileDialog
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from ui.add_student_ui import Ui_add_user_dialog

class AddStudentDialog(QDialog):
    """
    A dialog for adding a new student to the database.
    It handles user input for name and image selection.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.ui = Ui_add_user_dialog()
        self.ui.setupUi(self)
        
        self.selected_image_path = None
        
        self.ui.profile_image_browse.clicked.connect(self.open_image_dialog)
        
        self.ui.buttonBox.accepted.connect(self.accept)
        self.ui.buttonBox.rejected.connect(self.reject)
        
    def open_image_dialog(self):
        """
        Opens a QFileDialog to allow the user to select an image file.
        """
        
        start_path = os.path.expanduser("~")

        file_dialog = QFileDialog()
        
        options = file_dialog.options()
        options |= QFileDialog.Option.DontUseNativeDialog
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Student Image",
            start_path,
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )
        
        if file_path:
            self.selected_image_path = file_path
            
            self.ui.profile_image_input.setText(file_path)
            
            pixmap = QPixmap(file_path)
            scaled_pixmap = pixmap.scaled(self.ui.profile_image_preview.size(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)
            self.ui.profile_image_preview.setPixmap(scaled_pixmap)

    def get_student_data(self):
        """
        Returns the entered student data if the dialog was accepted.
        
        Returns:
            A dictionary containing the student's name and image path,
            or None if the data is invalid or the dialog was cancelled.
        """
        if self.result() == QDialog.DialogCode.Accepted:
            student_name = self.ui.profile_name_input.text().strip()
            
            if student_name and self.selected_image_path:
                return {
                    "name": student_name,
                    "image_path": self.selected_image_path
                }
        return None