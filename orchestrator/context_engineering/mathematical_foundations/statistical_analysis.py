import numpy as np
from typing import List, Dict, Any

class StatisticalAnalysis:
    @staticmethod
    def calculate_mean(values: List[float]) -> float:
        return np.mean(values) if values else 0.0
    
    @staticmethod
    def calculate_std(values: List[float]) -> float:
        return np.std(values) if values else 0.0
    
    @staticmethod
    def calculate_variance(values: List[float]) -> float:
        return np.var(values) if values else 0.0
