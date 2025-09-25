import cv2
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np

from .face_analyzer import FaceAnalyzer

from logging import getLogger


logger = getLogger(__name__)


class CameraWorker(QObject):
    """
    A worker that captures video frames. It's designed to live in a
    long-running QThread.
    """
    frame_ready = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._is_running = False
        self.face_analyzer = FaceAnalyzer()

    @pyqtSlot()
    def start_capture(self):
        """
        Starts the camera capture loop. This is a slot that can be
        triggered from the main thread.
        """
        if self._is_running:
            return

        self._is_running = True
        self.face_analyzer.prepare()
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            self.error.emit(f"Error: Could not open camera with index {self.camera_index}.")
            self._is_running = False
            self.finished.emit()
            return
        
        while self._is_running:
            ret, frame = cap.read()
            if not ret:
                self.error.emit("Error: Could not read frame from camera.")
                self._is_running = False
                break

            processed_frame, faces = self.face_analyzer.process_frame(frame)
            
            self.frame_ready.emit(processed_frame, faces)

        if cap.isOpened():
            cap.release()
            
        self.finished.emit()
        logger.info("Camera worker loop has finished.")

    def stop(self):
        """
        Stops the camera capture loop. This can be called from any thread.
        """
        logger.info("Stopping camera worker...")
        self._is_running = False