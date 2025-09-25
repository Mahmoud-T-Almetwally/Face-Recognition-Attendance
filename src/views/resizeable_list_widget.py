from PyQt6.QtWidgets import QListWidget

class ResizableListWidget(QListWidget):
    """
    A QListWidget subclass that automatically resizes its item widgets
    to fit the width of the list view during a resize event.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

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