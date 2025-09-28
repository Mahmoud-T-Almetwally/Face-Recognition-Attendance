import cv2
import numpy as np
from PIL.Image import Image as PILImage
from insightface.app import FaceAnalysis
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    A class to handle face detection and recognition using InsightFace,
    with a custom tracking mechanism based on face embeddings.
    """
    def __init__(self, frame_history_size=5, similarity_threshold=0.5, max_unseen_frames=10, persistent_face_grouping=False):
        """
        Initializes the FaceAnalyzer.

        Args:
            frame_history_size (int): The number of recent frames to consider for face grouping.
            similarity_threshold (int): The cosine similarity threshold to consider two faces as the same person.
            max_unseen_frames (int): The maximum number of consecutive frames a face can be unseen before its track is dropped.
            persistence_face_grouping (bool): Turns off face group dropping.
        """
        self.app = None
        self.frame_history = deque(maxlen=frame_history_size)
        self.persistent_face_grouping = persistent_face_grouping
        self.face_groups = []
        self.next_group_id = 0
        self.similarity_threshold = similarity_threshold
        self.max_unseen_frames = max_unseen_frames

    def prepare(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        """
        Loads the InsightFace models. This can take some time.
        
        Args:
            providers: A list of ONNX Runtime execution providers.
        """
        if self.app is not None:
            return
            
        logger.info("Loading InsightFace models... This may take a moment.")
        self.app = FaceAnalysis(name='buffalo_l', root="./model_cache", providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("InsightFace models loaded.")

    def get_face_embeddings(self, image: np.ndarray | PILImage) -> list[np.ndarray]:
        """
        Processes a single image to return face embeddings.

        Args:
            image: The input image as a NumPy array or a PIL Image.

        Returns:
            A list containing:
            - The face embeddings found in the image.
        """

        if isinstance(image, PILImage):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        if self.app is None:
            return image, []

        faces = self.app.get(image)

        return [face.embedding for face in faces]


    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Processes a frame to detect and track faces based on embeddings.

        Args:
            frame: The input video frame as a NumPy array.

        Returns:
            A tuple containing:
            - The frame with bounding boxes and track IDs.
            - A list of 'face' objects from InsightFace with an added 'track_id' attribute.
        """
        if self.app is None:
            return frame, []

        faces = self.app.get(frame)
        self.frame_history.append(faces)

        current_face_embeddings = {i: face.embedding for i, face in enumerate(faces)}

        if not self.face_groups:
            for i, embedding in current_face_embeddings.items():
                self.face_groups.append({
                    "id": self.next_group_id,
                    "representative_embedding": embedding,
                    "unseen_frames": 0
                })
                faces[i].track_id = self.next_group_id
                self.next_group_id += 1
        else:
            self._update_face_groups(faces, current_face_embeddings)
        if not self.persistent_face_grouping:
            self._cleanup_stale_groups()
        
        processed_frame = self.draw_on_frame(frame, faces)

        return processed_frame, faces

    def _update_face_groups(self, faces, current_face_embeddings):
        """
        Updates existing face groups with new faces from the current frame.
        """
        matched_face_indices = set()

        for group in self.face_groups:
            best_match_similarity = -1
            best_match_index = -1

            for i, embedding in current_face_embeddings.items():
                if i in matched_face_indices:
                    continue
                
                similarity = cosine_similarity([group["representative_embedding"]], [embedding])[0][0]

                if similarity > best_match_similarity:
                    best_match_similarity = similarity
                    best_match_index = i
            
            if best_match_similarity > self.similarity_threshold:
                matched_face_indices.add(best_match_index)
                faces[best_match_index].track_id = group["id"]
                
                alpha = 0.2
                group["representative_embedding"] = (1 - alpha) * group["representative_embedding"] + alpha * current_face_embeddings[best_match_index]
                group["unseen_frames"] = 0
            else:
                group["unseen_frames"] += 1

        for i, embedding in current_face_embeddings.items():
            if i not in matched_face_indices:
                self.face_groups.append({
                    "id": self.next_group_id,
                    "representative_embedding": embedding,
                    "unseen_frames": 0
                })
                faces[i].track_id = self.next_group_id
                self.next_group_id += 1


    def _cleanup_stale_groups(self):
        """
        Removes face groups that haven't been seen for a while.
        """
        self.face_groups = [g for g in self.face_groups if g["unseen_frames"] <= self.max_unseen_frames]


    def draw_on_frame(self, frame: np.ndarray, faces: list):
        """
        Draws bounding boxes and track IDs on the frame.
        """
        for face in faces:
            if hasattr(face, 'track_id'):
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                det_score = face.det_score
                track_id = face.track_id
                cv2.putText(frame, f"{det_score * 100:.2f}% id: {track_id}", (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame