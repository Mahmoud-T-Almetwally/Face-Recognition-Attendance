import pytest
import numpy as np
from unittest.mock import call, MagicMock

from src.database.face_cache import FaceCache
from src.database.db_models import StudentResult
from src.database.database_manager import DatabaseManager

class MockFace:
    def __init__(self, track_id, embedding_val=0.1):
        self.track_id = track_id
        self.embedding = np.full(512, embedding_val, dtype=np.float32)

@pytest.fixture
def mock_db_manager():
    """Provides a MagicMock instance of the DatabaseManager."""
    db_manager = MagicMock(spec=DatabaseManager)
    return db_manager

@pytest.fixture
def face_cache(mock_db_manager):
    """Provides a clean instance of FaceCache for each test."""
    return FaceCache(db_manager=mock_db_manager, cache_size_per_id=3, similarity_threshold=0.7)

def create_student_results(scores, start_id=1):
    """Helper function to generate mock StudentResult objects."""
    results = []
    for i, score in enumerate(scores):
        results.append(
            StudentResult(
                student_id=f"id_{start_id + i}",
                student_name=f"Student {start_id + i}",
                student_image_path=f"/path/to/image_{start_id + i}.jpg",
                similarity_score=score
            )
        )
    return results

class TestFaceCache:
    """Test suite for the FaceCache class."""

    def test_initialization(self, mock_db_manager):
        """Tests successful initialization and default parameter setting."""
        cache = FaceCache(db_manager=mock_db_manager, cache_size_per_id=5, similarity_threshold=0.8)
        assert cache.db_manager == mock_db_manager
        assert cache.cache_size_per_id == 5
        assert cache.similarity_threshold == 0.8
        assert not cache.recognition_cache
        assert not cache.last_seen_track_ids

    def test_initialization_type_error(self):
        """Tests that a TypeError is raised for an invalid db_manager."""
        with pytest.raises(TypeError):
            FaceCache(db_manager="not_a_db_manager_instance")

    def test_first_frame_with_new_faces(self, face_cache, mock_db_manager):
        """
        Tests the primary scenario: new faces appear, the DB is queried,
        and the cache is populated correctly.
        """
        faces = [MockFace(track_id=1), MockFace(track_id=2)]
        
        mock_db_manager.find_similar_students_bulk.return_value = {
            0: create_student_results([0.9, 0.85, 0.75, 0.6]),
            1: create_student_results([0.95, 0.65]),
        }

        results = face_cache.recognize_faces(faces)

        mock_db_manager.find_similar_students_bulk.assert_called_once()
        call_args = mock_db_manager.find_similar_students_bulk.call_args[0][0]
        assert len(call_args) == 2

        # 2. Cache is populated correctly (filtered, sorted, and trimmed)
        assert len(face_cache.recognition_cache) == 2
        
        # Check track_id 1: should keep top 3 results above 0.7
        assert len(face_cache.recognition_cache[1]) == 3 
        assert face_cache.recognition_cache[1][0].similarity_score == 0.9
        
        # Check track_id 2: should filter out the 0.65 score
        assert len(face_cache.recognition_cache[2]) == 1
        assert face_cache.recognition_cache[2][0].similarity_score == 0.95

        # 3. Returned results match the cache state
        assert results == face_cache.recognition_cache
        
        # 4. Internal state is updated
        assert face_cache.last_seen_track_ids == {1, 2}

    def test_existing_faces_are_served_from_cache(self, face_cache, mock_db_manager):
        """
        Tests that if the same faces are seen again, the DB is NOT queried.
        """
        # Arrange: First frame to populate the cache
        faces_frame1 = [MockFace(track_id=1)]
        mock_db_manager.find_similar_students_bulk.return_value = {
            0: create_student_results([0.9])
        }
        face_cache.recognize_faces(faces_frame1)
        mock_db_manager.find_similar_students_bulk.assert_called_once()

        # Act: Second frame with the same face
        faces_frame2 = [MockFace(track_id=1)]
        results = face_cache.recognize_faces(faces_frame2)

        # Assert
        # 1. DB was NOT called again
        mock_db_manager.find_similar_students_bulk.assert_called_once()

        # 2. The result is the cached data
        assert len(results) == 1
        assert results[1][0].similarity_score == 0.9

    def test_mixed_frame_new_and_existing_faces(self, face_cache, mock_db_manager):
        """
        Tests a frame with both old and new faces, ensuring only new ones are queried.
        """
        # Arrange: First frame with face 1
        faces_frame1 = [MockFace(track_id=1)]
        mock_db_manager.find_similar_students_bulk.return_value = {0: create_student_results([0.9], start_id=1)}
        face_cache.recognize_faces(faces_frame1)
        
        # Reset mock for the next call
        mock_db_manager.reset_mock()
        
        # Arrange: Second frame with faces 1 and 2
        faces_frame2 = [MockFace(track_id=1), MockFace(track_id=2)]
        mock_db_manager.find_similar_students_bulk.return_value = {0: create_student_results([0.88], start_id=2)}

        # Act
        results = face_cache.recognize_faces(faces_frame2)

        # Assert
        # 1. DB was called, but only with the embedding for the new face (track_id=2)
        mock_db_manager.find_similar_students_bulk.assert_called_once()
        call_args = mock_db_manager.find_similar_students_bulk.call_args[0][0]
        assert len(call_args) == 1 # Only one new embedding was queried

        # 2. Cache contains data for both track IDs
        assert len(face_cache.recognition_cache) == 2
        assert face_cache.recognition_cache[1][0].student_id == "id_1"
        assert face_cache.recognition_cache[2][0].student_id == "id_2"
        
        # 3. Internal state is updated
        assert face_cache.last_seen_track_ids == {1, 2}

    def test_dropped_faces_are_cleared_from_cache(self, face_cache, mock_db_manager):
        """
        Tests that when a face disappears, its data is purged from the cache.
        """
        # Arrange: Frame 1 with faces 1 and 2
        faces_frame1 = [MockFace(track_id=1), MockFace(track_id=2)]
        mock_db_manager.find_similar_students_bulk.return_value = {
            0: create_student_results([0.9]),
            1: create_student_results([0.85]),
        }
        face_cache.recognize_faces(faces_frame1)
        assert face_cache.last_seen_track_ids == {1, 2}
        assert 1 in face_cache.recognition_cache
        assert 2 in face_cache.recognition_cache

        # Act: Frame 2 where face 1 has disappeared
        faces_frame2 = [MockFace(track_id=2)]
        results = face_cache.recognize_faces(faces_frame2)

        # Assert
        # 1. Cache for the dropped track_id is cleared
        assert 1 not in face_cache.recognition_cache
        assert 2 in face_cache.recognition_cache # The remaining one is still there
        
        # 2. Returned dictionary only contains active tracks
        assert 1 not in results
        assert 2 in results
        
        # 3. Internal state is updated
        assert face_cache.last_seen_track_ids == {2}
        
    def test_no_matches_from_db(self, face_cache, mock_db_manager):
        """
        Tests that if the DB returns no matches, the cache entry is an empty list.
        """
        # Arrange
        faces = [MockFace(track_id=1)]
        mock_db_manager.find_similar_students_bulk.return_value = {0: []} # DB finds nothing

        # Act
        results = face_cache.recognize_faces(faces)

        # Assert
        mock_db_manager.find_similar_students_bulk.assert_called_once()
        assert 1 in face_cache.recognition_cache
        assert face_cache.recognition_cache[1] == []
        assert results[1] == []

    def test_empty_face_list_clears_cache(self, face_cache, mock_db_manager):
        """
        Tests that providing an empty list of faces clears any existing cache.
        """
        # Arrange: Populate cache
        faces_frame1 = [MockFace(track_id=5)]
        mock_db_manager.find_similar_students_bulk.return_value = {0: create_student_results([0.9])}
        face_cache.recognize_faces(faces_frame1)
        assert face_cache.recognition_cache

        # Act: Process an empty list of faces
        results = face_cache.recognize_faces([])

        # Assert
        assert not face_cache.recognition_cache # Cache is now empty
        assert not face_cache.last_seen_track_ids
        assert not results