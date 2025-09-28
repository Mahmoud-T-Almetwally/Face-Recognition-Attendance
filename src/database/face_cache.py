from typing import List, Dict, Set

from insightface.app.common import Face 

from .database_manager import DatabaseManager
from .db_models import StudentResult

import logging

logger = logging.getLogger(__name__)


class FaceCache:
    """
    Manages face recognition by caching results for tracked faces.

    This class reduces database load by only querying for newly detected faces
    and maintaining a cache of potential identities for each ongoing track.
    """

    def __init__(self, db_manager: DatabaseManager, cache_size_per_id: int = 5, similarity_threshold: float = 0.0):
        """
        Initializes the FaceCache.

        Args:
            db_manager: An instance of the DatabaseManager to query for students.
            cache_size_per_id: The maximum number of top candidate results to cache for each track ID.
            similarity_threshold: The minimum similarity score required to consider a result valid.
        """
        if not isinstance(db_manager, DatabaseManager):
            raise TypeError("db_manager must be an instance of DatabaseManager")

        self.db_manager = db_manager
        self.cache_size_per_id = cache_size_per_id
        self.similarity_threshold = similarity_threshold
    
        self.recognition_cache: Dict[int, List[StudentResult]] = {}
        
        self.last_seen_track_ids: Set[int] = set()
        
        logger.info(f"FaceRecognizer initialized with cache_size={cache_size_per_id} and threshold={similarity_threshold}")

    def recognize_faces(self, faces: List[Face]) -> Dict[int, List[StudentResult]]:
        """
        Recognizes a list of tracked faces, utilizing an internal cache.

        Args:
            faces: A list of Face objects from FaceAnalyzer, each with 'track_id' and 'embedding'.

        Returns:
            A dictionary mapping each track_id to a list of sorted StudentResult objects.
        """
        current_track_ids = {face.track_id for face in faces if face.track_id is not None}
        
        # 1. Identify dropped tracks and clean the cache
        dropped_track_ids = self.last_seen_track_ids - current_track_ids
        for track_id in dropped_track_ids:
            if track_id in self.recognition_cache:
                del self.recognition_cache[track_id]
                logger.debug(f"Dropped track_id {track_id} from cache.")

        # 2. Identify new tracks that need to be processed
        new_track_ids = current_track_ids - self.last_seen_track_ids
        
        new_faces_to_query = [face for face in faces if face.track_id in new_track_ids]

        # 3. Perform a bulk query for all new faces
        if new_faces_to_query:
            logger.info(f"Found {len(new_faces_to_query)} new faces to recognize.")
            
            query_embeddings = [face.embedding for face in new_faces_to_query]
            track_id_map = [face.track_id for face in new_faces_to_query]
            
            bulk_results = self.db_manager.find_similar_students_bulk(
                query_embeddings, k=self.cache_size_per_id
            )

            for idx, track_id in enumerate(track_id_map):
                results = bulk_results.get(idx, [])
                

                filtered_results = [res for res in results if res.similarity_score >= self.similarity_threshold]
                sorted_results = sorted(filtered_results, key=lambda x: x.similarity_score, reverse=True)
                
                logger.info(f"Found {len(sorted_results)} results for ID: {track_id}")

                self.recognition_cache[track_id] = sorted_results[:self.cache_size_per_id]
                logger.debug(f"Cached {len(self.recognition_cache[track_id])} results for new track_id {track_id}.")

        # 5. Update the set of last seen track IDs for the next frame
        self.last_seen_track_ids = current_track_ids

        # 6. Return the current state of the cache for all active tracks
        return {track_id: self.recognition_cache.get(track_id, []) for track_id in current_track_ids}