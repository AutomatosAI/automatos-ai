import numpy as np
from typing import List

class DistanceMetrics:
    @staticmethod
    def cosine_distance(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return 1.0
        v1_array = np.array(v1)
        v2_array = np.array(v2)
        similarity = np.dot(v1_array, v2_array) / (np.linalg.norm(v1_array) * np.linalg.norm(v2_array))
        return 1.0 - similarity
    
    @staticmethod
    def euclidean_distance(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return float('inf')
        return np.linalg.norm(np.array(v1) - np.array(v2))
    
    @staticmethod
    def manhattan_distance(v1: List[float], v2: List[float]) -> float:
        if not v1 or not v2:
            return float('inf')
        return np.sum(np.abs(np.array(v1) - np.array(v2)))

