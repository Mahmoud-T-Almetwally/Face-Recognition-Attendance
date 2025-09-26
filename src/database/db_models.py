from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from datetime import datetime


class Student(BaseModel):
    """Represents a record of a student"""
    model_config = ConfigDict(arbitrary_types_allowed = True)
    
    student_id: str = Field(..., description="The unique ID of the student")
    student_name: str = Field(..., description="The name of the student")
    student_image_path: str = Field(..., description="The profile image path of the student")
    student_face_embedding: np.ndarray = Field(..., description="The 512-dim embedding vector")


class StudentRecord(BaseModel):
    """Represents Only the human read-able data of a Student"""
    
    student_id: str = Field(..., description="The unique ID of the student")
    student_name: str = Field(..., description="The name of the student")
    student_image_path: str = Field(..., description="The profile image path of the student")

class StudentResult(BaseModel):
    """Represents a search result of a student"""
    model_config = ConfigDict(arbitrary_types_allowed = True)
    
    student_id: str = Field(..., description="The unique ID of the student")
    student_name: str = Field(..., description="The name of the student")
    student_image_path: str = Field(..., description="The profile image path of the student")
    student_face_embedding: np.ndarray = Field(..., description="The 512-dim embedding vector")
    similarity_score: float = Field(..., description="The similarity score to the search vector")


class AttendanceRecord(BaseModel):
    """Represents a Record of a Student's Attendance"""
    model_config = ConfigDict(arbitrary_types_allowed = True)

    attend_id: str = Field(..., description="The unique ID of the attending record")
    student_id: str = Field(..., description="The ID of the student attending")
    recorded_frame: str = Field(..., description="The path to the frame recording of the student attending")
    attend_datetime: datetime = Field(..., description="The date & time of the attendence")
