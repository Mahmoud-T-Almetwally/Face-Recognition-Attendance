
import cv2
import numpy as np
from insightface.app import FaceAnalysis

import logging


logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    A class to handle face detection and recognition using InsightFace.
    """
    def __init__(self):
        self.app = None
        
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

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Processes a single frame to detect and analyze faces.

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
        
        processed_frame = self.draw_on_frame(frame, faces)

        return processed_frame, faces

    def draw_on_frame(self, frame: np.ndarray, faces: list):
        """
        Draws bounding boxes and keypoints on the frame.
        """
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            det_score = face.det_score
            cv2.putText(frame, f"{det_score * 100:.2f}%", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

        return frame