from PyQt6.QtWidgets import QListWidget, QLabel, QListWidgetItem, QFrame
from PyQt6.QtCore import Qt

from .result_item_widget import ResultItemWidget
from database.db_models import StudentResult

import logging


logger = logging.getLogger(__name__)


class ResultListWidget(QListWidget):
    """
    A custom QListWidget that displays grouped items from a dictionary of
    StudentResult objects.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

    def populate_from_dict(self, data: dict[int, list[StudentResult]]):
        """
        Clears the list and populates it with new data from a dictionary.
        The dictionary is expected to have track_ids as keys and lists of
        StudentResult objects as values.
        """
        self.clear()

        for track_id in sorted(data.keys()):
            self.add_separator(f"Track ID: {track_id}")

            student_results = data[track_id]
            for result in student_results:
                item_widget = ResultItemWidget(self)
                logger.info(f"Result name: {result.student_name}, Score: {result.similarity_score}")
                item_widget.set_data(
                    name=result.student_name,
                    confidence=result.similarity_score,
                    icon_path=result.student_image_path
                )

                list_item = QListWidgetItem(self)
                list_item.setSizeHint(item_widget.sizeHint())

                self.addItem(list_item)
                self.setItemWidget(list_item, item_widget)

    def add_separator(self, text: str):
        """
        Adds a non-selectable separator item with a title to the list.
        """
        separator_line = QFrame()
        separator_line.setFrameShape(QFrame.Shape.HLine)
        separator_line.setFrameShadow(QFrame.Shadow.Sunken)
        
        line_item = QListWidgetItem(self)
        line_item.setSizeHint(separator_line.sizeHint())
        self.addItem(line_item)
        self.setItemWidget(line_item, separator_line)
        line_item.setFlags(Qt.ItemFlag.NoItemFlags)

        header_label = QLabel(text)
        header_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #CCCCCC; /* Light gray color for visibility */
                background-color: transparent; /* Ensure it has no background */
                border: none;
                padding-top: 8px;
                padding-bottom: 4px;
            }
        """)
        
        label_item = QListWidgetItem(self)
        label_item.setSizeHint(header_label.sizeHint())
        self.addItem(label_item)
        self.setItemWidget(label_item, header_label)
        label_item.setFlags(Qt.ItemFlag.NoItemFlags)

    def resizeEvent(self, event):
        """
        Override the resizeEvent to update item widgets.
        """
        super().resizeEvent(event)
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if widget:
                item.setSizeHint(widget.sizeHint())