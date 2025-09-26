import pytest
import numpy as np
import uuid
from datetime import datetime
import sqlite3

from src.database.database_manager import DatabaseManager
from src.database.db_models import Student, AttendanceRecord


@pytest.fixture(scope="function")
def db_manager():
    """
    Pytest fixture to set up a clean, in-memory DatabaseManager for each test.
    The database is automatically torn down after the test runs.
    """
    
    manager = DatabaseManager(db_path=":memory:")
    yield manager
    manager.close()

def create_dummy_student(name_prefix: str = "Student") -> Student:
    """Helper function to create a valid Student object with random data."""
    return Student(
        student_id=str(uuid.uuid4()),
        student_name=f"{name_prefix} {uuid.uuid4().hex[:6]}",
        student_image_path=f"/path/to/{uuid.uuid4().hex[:6]}.jpg",
        student_face_embedding=np.random.rand(512).astype(np.float32)
    )


class TestDatabaseManager:

    def test_initialization_and_table_creation(self, db_manager: DatabaseManager):
        """
        Tests if the DatabaseManager initializes correctly and creates all necessary tables.
        """
        
        cursor = db_manager.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}

        assert "students" in tables
        assert "attendance" in tables
        assert any(name.startswith("vec_students") for name in tables)


    def test_add_student_success(self, db_manager: DatabaseManager):
        """
        Tests that a valid student can be added successfully.
        """
        student = create_dummy_student()

        result = db_manager.add_student(student)
        retrieved_students = db_manager.get_all_students()

        assert result is True
        assert len(retrieved_students) == 1
        assert retrieved_students[0].student_id == student.student_id
        assert retrieved_students[0].student_name == student.student_name

    def test_add_student_duplicate_id_fails(self, db_manager: DatabaseManager):
        """
        Tests that adding a student with a duplicate ID fails gracefully.
        """
        student1 = create_dummy_student()
        db_manager.add_student(student1)

        student2 = create_dummy_student()
        student2.student_id = student1.student_id 

        result = db_manager.add_student(student2)
        retrieved_students = db_manager.get_all_students()

        assert result is False
        assert len(retrieved_students) == 1

    def test_get_all_students_empty_db(self, db_manager: DatabaseManager):
        """
        Tests that getting all students from an empty database returns an empty list.
        """
        students = db_manager.get_all_students()
        assert students == []

    def test_get_all_students_pagination(self, db_manager: DatabaseManager):
        """
        Tests the pagination logic of get_all_students.
        """
        for _ in range(15):
            db_manager.add_student(create_dummy_student())

        page1 = db_manager.get_all_students(page=1, page_size=10)
        page2 = db_manager.get_all_students(page=2, page_size=10)
        page3 = db_manager.get_all_students(page=3, page_size=10)

        assert len(page1) == 10
        assert len(page2) == 5
        assert len(page3) == 0

    def test_get_all_students_ordering(self, db_manager: DatabaseManager):
        """
        Tests the order_by logic of get_all_students.
        """
        student_c = Student(student_id="S03", student_name="Charlie", student_image_path="path/to/image1.png", student_face_embedding=np.random.rand(512))
        student_a = Student(student_id="S01", student_name="Alice", student_image_path="path/to/image2.png", student_face_embedding=np.random.rand(512))
        student_b = Student(student_id="S02", student_name="Bob", student_image_path="path/to/image3.png", student_face_embedding=np.random.rand(512))
        db_manager.add_student(student_c)
        db_manager.add_student(student_a)
        db_manager.add_student(student_b)

        by_name = db_manager.get_all_students(order_by="student_name")
        by_id = db_manager.get_all_students(order_by="student_id")

        assert by_name[0].student_name == "Alice"
        assert by_name[1].student_name == "Bob"
        assert by_name[2].student_name == "Charlie"
        
        assert by_id[0].student_id == "S01"
        assert by_id[1].student_id == "S02"
        assert by_id[2].student_id == "S03"

    def test_get_all_students_invalid_order_by_raises_error(self, db_manager: DatabaseManager):
        """
        Tests that an invalid order_by parameter raises a ValueError (prevents SQL injection).
        """
        with pytest.raises(ValueError):
            db_manager.get_all_students(order_by="invalid_column")

    def test_find_similar_students_exact_match(self, db_manager: DatabaseManager):
        """
        Tests that searching for an exact embedding returns that student with max similarity.
        """
        student = create_dummy_student()
        db_manager.add_student(student)

        results = db_manager.find_similar_students(student.student_face_embedding, k=1)

        assert len(results) == 1
        assert results[0].student_id == student.student_id
        assert np.isclose(results[0].similarity_score, 1.0, atol=1e-5)

    def test_find_similar_students_closest_match(self, db_manager: DatabaseManager):
        """
        Tests that the function returns the correct closest match.
        """
        student_target = create_dummy_student("Target")
        student_far = create_dummy_student("Far")
        student_far.student_face_embedding = np.ones(512, dtype=np.float32)
        
        db_manager.add_student(student_target)
        db_manager.add_student(student_far)

        query_embedding = student_target.student_face_embedding + np.random.normal(0, 0.01, 512)

        results = db_manager.find_similar_students(query_embedding, k=1)

        assert len(results) == 1
        assert results[0].student_id == student_target.student_id
        assert results[0].similarity_score > 0.95

    def test_find_similar_students_k_parameter(self, db_manager: DatabaseManager):
        """
        Tests that the 'k' parameter correctly limits the number of results.
        """
        for i in range(5):
            db_manager.add_student(create_dummy_student(f"Student_{i}"))
        
        query_embedding = np.random.rand(512).astype(np.float32)
        
        results = db_manager.find_similar_students(query_embedding, k=3)
        
        assert len(results) == 3


    def test_add_attendance_record_success(self, db_manager: DatabaseManager):
        """
        Tests that an attendance record can be added for an existing student.
        """
        student = create_dummy_student()
        db_manager.add_student(student)
        
        attendance = AttendanceRecord(
            attend_id=str(uuid.uuid4()),
            student_id=student.student_id,
            recorded_frame="/path/to/frame.jpg",
            attend_datetime=datetime.now()
        )

        result = db_manager.add_attendance_record(attendance)
        records = db_manager.get_all_attendance()

        assert result is True
        assert len(records) == 1
        assert records[0].student_id == student.student_id

    def test_add_attendance_record_fails_for_nonexistent_student(self, db_manager: DatabaseManager):
        """
        Tests that adding an attendance record for a student who does not exist fails.
        Note: This relies on the FOREIGN KEY constraint in the table definition.
        """
        attendance = AttendanceRecord(
            attend_id=str(uuid.uuid4()),
            student_id="non-existent-student-id",
            recorded_frame="/path/to/frame.jpg",
            attend_datetime=datetime.now()
        )

        result = db_manager.add_attendance_record(attendance)

        records = db_manager.get_all_attendance()

        assert result is False
        assert len(records) == 0