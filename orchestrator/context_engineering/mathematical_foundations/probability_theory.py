import numpy as np
from typing import List, Dict, Any

class ProbabilityTheory:
    @staticmethod
    def calculate_probability(events: List[Any], total_events: int) -> float:
        if total_events == 0:
            return 0.0
        return len(events) / total_events
    
    @staticmethod
    def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
        if evidence == 0:
            return 0.0
        return (prior * likelihood) / evidence
