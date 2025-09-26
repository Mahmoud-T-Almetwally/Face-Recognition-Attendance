import cv2
import numpy as np
from PIL.Image import Image as PILImage
from insightface.app import FaceAnalysis
from sort_tracker import Sort

import logging


logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    A class to handle face detection and recognition using InsightFace.
    """
    def __init__(self):
        self.app = None

        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
    def prepare(self, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        """
        Loads the InsightFace models. This can take some time.
        
        Args:
            providers: A list of ONNX Runtime execution providers.
                       Defaults to ['CPUExecutionProvider'].
                       Use ['CUDAExecutionProvider', 'CPUExecutionProvider'] for GPU.
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
        Processes a frame to detect and analyze faces.

        Args:
            frame: The input video frame as a NumPy array.

        Returns:
            A tuple containing:
            - The frame with bounding boxes and information drawn on it.
            - A list of 'face' objects from InsightFace for each detected face.
        """
        if self.app is None:
            return frame, []

        faces = self.app.get(frame)

        if not faces:
            tracked_objects = self.tracker.update(np.empty((0, 5)))
            return frame, []

        detections = np.array([
            list(face.bbox) + [face.det_score]
            for face in faces
        ])

        tracked_objects = self.tracker.update(detections)

        self.associate_tracker_ids(faces, tracked_objects)
        
        processed_frame = self.draw_on_frame(frame, faces)

        return processed_frame, faces
    
    def associate_tracker_ids(self, faces, tracked_objects):
        """
        Assigns the track_id from SORT to the corresponding insightface Face object.
        """
        
        unmatched_tracks = list(tracked_objects)

        for face in faces:
            best_match_iou = 0
            best_match_track = None
            
            for track in unmatched_tracks:
                track_bbox = track[:4]
                iou = self.calculate_iou(face.bbox, track_bbox)
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_track = track
            
            if best_match_track is not None and best_match_iou > 0.3:
                face.track_id = int(best_match_track[4])
                unmatched_tracks.remove(best_match_track)
            else:
                face.track_id = None

    def calculate_iou(self, boxA, boxB):
        """Calculates Intersection over Union for two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def draw_on_frame(self, frame: np.ndarray, faces: list):
        """
        Draws bounding boxes and keypoints on the frame.
        """
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            det_score = face.det_score
            cv2.putText(frame, f"{det_score * 100:.2f}% id: {face.track_id}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame