import numpy as np
from typing import List

class VectorOperations:
    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 0.0
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        return np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))
    
    @staticmethod
    def normalize_vector(vector: List[float]) -> List[float]:
        if not vector:
            return []
        v_array = np.array(vector)
        norm = np.linalg.norm(v_array)
        if norm == 0:
            return vector
        return (v_array / norm).tolist()
