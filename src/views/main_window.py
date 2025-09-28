from ui.main_window_ui import Ui_MainWindow
from vision.face_analyzer import FaceAnalyzer

from PyQt6.QtWidgets import QMainWindow, QSizePolicy, QDialog
from PyQt6.QtCore import QThread, Qt, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap

from pydantic_core import ValidationError

import os
import shutil
import uuid
import cv2
import numpy as np
from PIL import Image
from logging import getLogger

from vision.camera_manager import CameraWorker
from database.db_models import StudentResult, Student
from database.database_manager import DatabaseManager
from database.face_cache import FaceCache
from .add_student_widget import AddStudentDialog


DB_PATH = "./data/attendance.db"

logger = getLogger(__name__)


class Worker(QObject):
    finished = pyqtSignal()
    task = None

    def run(self):
        if self.task:
            self.task()
        self.finished.emit()


class MainWindow(QMainWindow):
    """
    Main Window UI Displaying the Camera Feed and a Results list.
    """

    start_worker_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        try:
            self.db_manager = DatabaseManager(db_path=DB_PATH)
            self.face_cache = FaceCache(db_manager=self.db_manager)
            logger.info("DatabaseManager and FaceCache initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize database or cache: {e}")
            self.ui.statusbar.showMessage(f"Error: Could not connect to database at {DB_PATH}")

        self.face_analyzer = FaceAnalyzer()
        self.is_analyzer_ready = False

        self.ui.actionEnroll.setEnabled(False)
        self.ui.statusbar.showMessage("Loading AI model, please wait...")

        size_policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.ui.video_display_label.setSizePolicy(size_policy)

        self.is_camera_running = False

        ## For Debugging ResultListWidget

        # if not os.path.exists("student1.png"):
        #     pixmap = QPixmap(64, 64)
        #     pixmap.fill(Qt.GlobalColor.blue)
        #     pixmap.save("student1.png", "PNG")

        # if not os.path.exists("student2.png"):
        #     pixmap = QPixmap(64, 64)
        #     pixmap.fill(Qt.GlobalColor.green)
        #     pixmap.save("student2.png", "PNG")

        # sample_data = {
        #     0: [
        #         StudentResult(student_id="s1", student_name="Alice", student_image_path="student1.png", similarity_score=0.95),
        #         StudentResult(student_id="s2", student_name="Bob", student_image_path="student2.png", similarity_score=0.88),
        #     ],
        #     1: [
        #         StudentResult(student_id="s3", student_name="Charlie", student_image_path="non_existent.png", similarity_score=0.76),
        #     ],
        #     2: [
        #         StudentResult(student_id="s4", student_name="David", student_image_path="student1.png", similarity_score=0.99),
        #         StudentResult(student_id="s5", student_name="Eve", student_image_path="student2.png", similarity_score=0.91),
        #         StudentResult(student_id="s6", student_name="Frank", student_image_path="student1.png", similarity_score=0.82),
        #     ]
        # }

        # self.ui.results_list.populate_from_dict(sample_data)

        self.setup_camera()

        self.initialize_analyzer()

        self.ui.actionAttendance.triggered.connect(self.display_attendance)
        self.ui.actionEnroll.triggered.connect(self.enroll_student)
        self.ui.actionLogs.triggered.connect(self.display_logs)
        self.ui.actionStart.triggered.connect(self.start_camera_feed)
        self.ui.actionStop.triggered.connect(self.stop_camera_feed)

        self.ui.actionStop.setEnabled(False)

        self.show()

    def initialize_analyzer(self):
        """
        Initializes the FaceAnalyzer in a background thread to prevent UI freezing.
        """
        self.analyzer_thread = QThread()
        self.analyzer_worker = Worker()
        self.analyzer_worker.task = self.face_analyzer.prepare
        
        self.analyzer_worker.moveToThread(self.analyzer_thread)
        self.analyzer_thread.started.connect(self.analyzer_worker.run)
        self.analyzer_worker.finished.connect(self.on_analyzer_ready)
        
        self.analyzer_worker.finished.connect(self.analyzer_thread.quit)
        self.analyzer_worker.finished.connect(self.analyzer_worker.deleteLater)
        self.analyzer_thread.finished.connect(self.analyzer_thread.deleteLater)
        
        self.analyzer_thread.start()
        
    def on_analyzer_ready(self):
        """Slot called when the FaceAnalyzer has finished loading."""
        self.is_analyzer_ready = True
        self.ui.actionEnroll.setEnabled(True)
        self.ui.statusbar.showMessage("AI Model Loaded. Ready.", 5000)
        logger.info("FaceAnalyzer is ready.")

    def setup_camera(self):
        """
        Initializes the camera worker and thread.
        """
        
        self.camera_thread = QThread()
        self.camera_worker = CameraWorker(face_analyzer=self.face_analyzer, camera_index=0)
        
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
        self.ui.results_list.clear()

    def update_frame(self, frame: np.ndarray, faces: list):
        """
        Receives a frame from the worker and displays it in the QLabel.
        
        Args:
            frame: The captured video frame as a NumPy array.
        """

        if faces:
            recognition_results = self.face_cache.recognize_faces(faces)
            
            self.ui.results_list.populate_from_dict(recognition_results)
        else:
            self.face_cache.recognize_faces([]) 
            self.ui.results_list.clear()
        
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
        """
        Creates and shows the Add Student dialog. If the user clicks OK,
        it processes the new student's data.
        """

        if not self.is_analyzer_ready:
            self.ui.statusbar.showMessage("Please wait, AI model is still loading.")
            return

        logger.info("Starting Enroll Student")
        dialog = AddStudentDialog(self)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            student_data = dialog.get_student_data()
            if student_data:
                logger.info("New student to be added:")
                logger.info(f"Name: {student_data['student_name']}")
                logger.info(f"Image: {student_data['student_image_path']}")
                
                profile_image = Image.open(student_data['student_image_path'])
                
                face_embeddings = self.face_analyzer.get_face_embeddings(profile_image)
                
                if len(face_embeddings):
                    embedding = face_embeddings[0]
                    student_data["student_id"] = str(uuid.uuid4())
                    student_data["student_face_embedding"] = embedding

                    image_path = shutil.copy(src=student_data['student_image_path'], dst=f"./data/students/{student_data['student_id']}.png")

                    student_data['student_image_path'] = image_path

                    try:
                        new_student = Student(**student_data)
                        if self.db_manager.add_student(new_student):
                            logger.info(f"Successfully added {new_student.student_name} to the database.")
                            self.ui.statusbar.showMessage(f"Student '{new_student.student_name}' enrolled successfully.", 5000)
                        else:
                            logger.error(f"Failed to add {new_student.student_name} to the database.")
                            self.ui.statusbar.showMessage("Error: Could not enroll student.", 5000)
                    except ValidationError as e:
                        logger.error(f"Validation Error Caught in student data, original error message: {e}")

                else:
                    print(face_embeddings)
                    logger.warning("No Faces Found in profile image, Skipping student.")
                    self.ui.statusbar.showMessage("No Faces Found in profile image, Skipping student.")
                
                self.ui.statusbar.showMessage(f"Student '{student_data['student_name']}' ready to be processed.", 5000)
        
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