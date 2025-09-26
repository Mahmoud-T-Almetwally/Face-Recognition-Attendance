import sqlite3
import sqlite_vec
import numpy as np
import logging
import json
from typing import List, Literal

from .db_models import Student, StudentResult, StudentRecord, AttendanceRecord


logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manages all interactions with the SQLite database, using the sqlite-vec extension.
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        try:
            self.conn = sqlite3.connect(db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            
            logger.info("Successfully connected to database and loaded sqlite-vec extension.")
            self._create_tables()

        except sqlite3.Error as e:
            logger.error(f"Database connection or vec extension loading failed: {e}")
            raise

    def _create_tables(self):
        with self.conn:
            self.conn.execute('PRAGMA foreign_keys = ON;')
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    student_image_path TEXT
                )
            """)
            
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_students USING vec0(
                    face_embedding float[512]
                )
            """)

            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    attend_id TEXT PRIMARY KEY,
                    student_id TEXT NOT NULL,
                    recorded_frame TEXT NOT NULL,
                    attend_datetime TEXT,
                    FOREIGN KEY (student_id) REFERENCES students (student_id)
                )
            """)
            logger.info("Tables created or already exist.")

    def add_student(self, student: Student) -> bool:
        try:
            with self.conn:
                cursor = self.conn.execute(
                    "INSERT INTO students (student_id, student_name, student_image_path) VALUES (?, ?, ?)",
                    (student.student_id, student.student_name, student.student_image_path)
                )
                student_rowid = cursor.lastrowid
                embedding_json = json.dumps(student.student_face_embedding.tolist())
                self.conn.execute(
                    "INSERT INTO vec_students (rowid, face_embedding) VALUES (?, ?)",
                    (student_rowid, embedding_json)
                )
            logger.info(f"Successfully added student: {student.student_name} ({student.student_id})")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Failed to add student. ID '{student.student_id}' may already exist.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while adding a student: {e}")
            return False

    def get_all_students(self, order_by: Literal["student_name", "student_id"] = "student_name", page: int = 1, page_size: int = 20) -> List[StudentRecord]:
        offset = (page - 1) * page_size
        
        allowed_order_columns = ["student_name", "student_id"]
        if order_by not in allowed_order_columns:
            raise ValueError(f"Invalid order_by column. Must be one of {allowed_order_columns}")
        
        query = f"SELECT student_id, student_name, student_image_path FROM students ORDER BY {order_by} LIMIT ? OFFSET ?"
        
        cursor = self.conn.execute(query, (page_size, offset))
        return [StudentRecord(**dict(row)) for row in cursor.fetchall()]

    def find_similar_students(self, query_embedding: np.ndarray, k: int = 5) -> List[StudentResult]:
        query_json = json.dumps(query_embedding.tolist())
        
        cursor = self.conn.execute("""
            SELECT
                s.student_id, s.student_name, s.student_image_path,
                v.face_embedding, v.distance
            FROM vec_students v
            JOIN students s ON s.rowid = v.rowid
            WHERE v.face_embedding MATCH ?
            AND k = ?
        """, (query_json, k))
        
        results = []
        for row in cursor.fetchall():
            l2_distance = row['distance']
            similarity = 1 - (l2_distance**2) / 2
            
            student_data = {
                "student_id": row['student_id'],
                "student_name": row['student_name'],
                "student_image_path": row['student_image_path'],
                "student_face_embedding": np.frombuffer(row['face_embedding'], dtype=np.float32),
                "similarity_score": similarity
            }
            results.append(StudentResult(**student_data))
            
        return results
    
    def get_all_attendance(self, page: int = 1, page_size: int = 20) -> List[AttendanceRecord]:
        offset = (page - 1) * page_size
        cursor = self.conn.execute(
            "SELECT attend_id, student_id, recorded_frame, attend_datetime FROM attendance ORDER BY attend_datetime DESC LIMIT ? OFFSET ?",
            (page_size, offset)
        )
        return [AttendanceRecord(**dict(row)) for row in cursor.fetchall()]

    def add_attendance_record(self, attendance: AttendanceRecord) -> bool:
        try:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO attendance (attend_id, student_id, recorded_frame, attend_datetime) VALUES (?, ?, ?, ?)",
                    (attendance.attend_id, attendance.student_id, attendance.recorded_frame, str(attendance.attend_datetime))
                )
            logger.info(f"Successfully recorded attendance for student: {attendance.student_id}")
            return True
        except sqlite3.IntegrityError:
            logger.error(f"Failed to record attendance. Attendance ID '{attendance.attend_id}' may already exist.")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while adding attendance: {e}")
            return False

    def close(self):
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")