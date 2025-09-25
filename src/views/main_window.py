from ui.main_window_ui import Ui_MainWindow
from views.result_item_widget import ResultItemWidget

from PyQt6.QtWidgets import QMainWindow, QSizePolicy, QListWidgetItem
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap

import cv2
import numpy as np

from logging import getLogger
from vision.camera_manager import CameraWorker

logger = getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Main Window UI Displaying the Camera Feed and a Results list.
    """

    start_worker_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        size_policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.ui.video_display_label.setSizePolicy(size_policy)

        self.is_camera_running = False

        self.setup_camera()

        self.ui.actionAttendance.triggered.connect(self.display_attendance)
        self.ui.actionEnroll.triggered.connect(self.enroll_student)
        self.ui.actionLogs.triggered.connect(self.display_logs)
        self.ui.actionStart.triggered.connect(self.start_camera_feed)
        self.ui.actionStop.triggered.connect(self.stop_camera_feed)

        self.ui.actionStop.setEnabled(False)

        self.show()

    def setup_camera(self):
        """
        Initializes the camera worker and thread.
        """
        
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(camera_index=0)
        
        self.camera_worker.moveToThread(self.camera_thread)
        
        self.start_worker_signal.connect(self.camera_worker.start_capture)
        
        self.camera_worker.frame_ready.connect(self.update_frame)
        self.camera_worker.error.connect(self.handle_camera_error)
        self.camera_worker.finished.connect(self.on_worker_finished)
        
        self.camera_thread.start()
        logger.info("Camera thread started and is waiting for tasks.")

    def start_camera_feed(self):
        """
        Starts the camera thread.
        """
        if not self.is_camera_running:
            self.ui.statusbar.showMessage("Starting camera feed...")
            logger.info("Starting Camera Feed...")
            
            self.start_worker_signal.emit()
            self.is_camera_running = True
            self.ui.actionStart.setEnabled(False)
            self.ui.video_display_label.setText("")
            self.ui.actionStop.setEnabled(True)

    def stop_camera_feed(self):
        """
        Signals the camera worker to stop.
        """
        if self.is_camera_running:
            logger.info("Stopping Camera Feed...")
            self.ui.statusbar.showMessage("Stopping camera feed...")
            self.camera_worker.stop()

    def on_worker_finished(self):
        """Slot to handle the worker's finished signal."""
        self.is_camera_running = False
        self.ui.statusbar.showMessage("Camera feed stopped.", 3000)
        logger.info("Stopped Camera Feed")
        self.ui.actionStart.setEnabled(True)
        self.ui.video_display_label.setText("Press Start to begin the Camera Feed")
        self.ui.actionStop.setEnabled(False)

    def update_frame(self, frame: np.ndarray):
        """
        Receives a frame from the worker and displays it in the QLabel.
        
        Args:
            frame: The captured video frame as a NumPy array.
        """
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        container_size = self.ui.video_display_label.parentWidget().size()
        scaled_pixmap = pixmap.scaled(container_size, 
                                      Qt.AspectRatioMode.KeepAspectRatio, 
                                      Qt.TransformationMode.SmoothTransformation)
        

        self.ui.video_display_label.setPixmap(scaled_pixmap)


    def handle_camera_error(self, error_msg: str):
        """
        Displays an error message in the status bar.
        """
        logger.error(error_msg)
        self.ui.statusbar.showMessage(error_msg, 5000)
        self.on_worker_finished()

    def display_logs(self):
        logger.info("Displaying Attendance Logs")

    def display_attendance(self):
        logger.info("Displaying Full Attendance Table")

    def enroll_student(self):
        logger.info("Starting Enroll Student")
        
    def closeEvent(self, event):
        """
        Handles the main window's close event for graceful shutdown.
        """
        logger.info("Closing application...")
        if self.camera_thread.isRunning():
            logger.info("Shutting down camera thread...")
            self.camera_worker.stop()
            self.camera_thread.quit()
            if not self.camera_thread.wait(3000):
                logger.warning("Thread did not terminate in time. Forcing termination.")
                self.camera_thread.terminate()
                self.camera_thread.wait()

        logger.info("Shutdown complete.")
        event.accept()