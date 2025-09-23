from ui.main_window_ui import Ui_MainWindow
from PyQt6.QtWidgets import QMainWindow 
from logging import getLogger

logger = getLogger(__name__)


class MainWindow(QMainWindow):

    """
    Main Window UI Displaying the Camera Feed and a Results list.
    """

    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()

        self.ui.setupUi(self)

        self.ui.actionAttendance.triggered.connect(self.display_attendance)
        self.ui.actionEnroll.triggered.connect(self.enroll_student)
        self.ui.actionLogs.triggered.connect(self.display_logs)
        self.ui.actionStart.triggered.connect(self.start_camera_feed)
        self.ui.actionStop.triggered.connect(self.stop_camera_feed)

        self.ui.actionStop.setEnabled(False)

        self.show()

    def start_camera_feed(self):
        logger.info("Started Camera Feed")
        self.ui.actionStart.setEnabled(False)
        self.ui.video_display_label.setText("")
        self.ui.actionStop.setEnabled(True)
        self.ui.statusbar.showMessage("Camera feed started.")

    def stop_camera_feed(self):
        logger.info("Stopped Camera Feed")
        self.ui.actionStart.setEnabled(True)
        self.ui.video_display_label.setText("Press Start to begin the Camera Feed")
        self.ui.actionStop.setEnabled(False)
        self.ui.statusbar.showMessage("Camera feed stopped.")


    def display_logs(self):
        logger.info("Displaying Attendance Logs")

    def display_attendance(self):
        logger.info("Displaying Full Attendance Table")

    def enroll_student(self):
        logger.info("Starting Enroll Student")