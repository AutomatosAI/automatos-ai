

"""
Evaluation Engine - Multi-dimensional System Evaluation
========================================================

Implements comprehensive system evaluation with:
- Multi-dimensional quality assessment
- Emergence detection algorithms
- Performance tracking and analysis
- Statistical variance calculations
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class EvaluationDimension(Enum):
    """Evaluation dimensions for system assessment"""
    PERFORMANCE = "performance"
    EFFICIENCY = "efficiency" 
    EMERGENCE = "emergence"
    RELIABILITY = "reliability"
    ADAPTABILITY = "adaptability"

@dataclass
class SystemQualityResult:
    """Result structure for system quality evaluation"""
    overall_quality: float
    dimension_scores: Dict[str, float]
    task_scores: List[Dict[str, Any]]
    emergence_indicators: Dict[str, float]
    evaluation_timestamp: datetime
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        return result

@dataclass 
class EvaluationMetrics:
    """Statistical metrics for evaluation analysis"""
    mean_score: float
    variance: float
    standard_deviation: float
    coefficient_of_variation: float
    confidence_level: float
    sample_size: int

@dataclass
class EmergenceIndicator:
    """Emergence detection result"""
    emergence_score: float
    variance_ratio: float
    prediction_error: float
    complexity_index: float
    novelty_score: float

class EmergenceDetector:
    """Advanced emergence detection in system behavior"""
    
    def __init__(self, sensitivity_threshold: float = 0.1):
        self.sensitivity_threshold = sensitivity_threshold
        self.baseline_behaviors = {}
        
    async def detect_emergence(self, 
                             observed_values: List[float],
                             predicted_values: List[float],
                             timestamps: List[datetime]) -> EmergenceIndicator:
        """
        Detect emergent behavior using variance analysis
        
        Formula: Emergence_Score = |Observed - Predicted| / Variance
        """
        try:
            if len(observed_values) != len(predicted_values):
                raise ValueError("Observed and predicted values must have same length")
                
            # Calculate emergence score
            differences = np.array([abs(o - p) for o, p in zip(observed_values, predicted_values)])
            variance = np.var(observed_values) if len(observed_values) > 1 else 1.0
            emergence_score = np.mean(differences) / variance if variance > 0 else 0.0
            
            # Calculate complexity index
            complexity_index = self._calculate_complexity(observed_values, timestamps)
            
            # Calculate novelty score  
            novelty_score = self._calculate_novelty(observed_values)
            
            # Calculate variance ratio
            observed_var = np.var(observed_values)
            predicted_var = np.var(predicted_values)
            variance_ratio = observed_var / predicted_var if predicted_var > 0 else float('inf')
            
            # Calculate prediction error
            prediction_error = np.mean(np.abs(differences))
            
            return EmergenceIndicator(
                emergence_score=emergence_score,
                variance_ratio=variance_ratio,
                prediction_error=prediction_error,
                complexity_index=complexity_index,
                novelty_score=novelty_score
            )
            
        except Exception as e:
            logger.error(f"Error in emergence detection: {e}")
            return EmergenceIndicator(0.0, 0.0, 0.0, 0.0, 0.0)
    
    def _calculate_complexity(self, values: List[float], timestamps: List[datetime]) -> float:
        """Calculate complexity index based on temporal patterns"""
        if len(values) < 3:
            return 0.0
            
        # Calculate rate of change
        rates = []
        for i in range(1, len(values)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if time_diff > 0:
                rate = abs(values[i] - values[i-1]) / time_diff
                rates.append(rate)
        
        return np.var(rates) if rates else 0.0
    
    def _calculate_novelty(self, values: List[float]) -> float:
        """Calculate novelty score based on historical baselines"""
        if not values:
            return 0.0
            
        current_pattern = np.array(values[-5:])  # Last 5 values
        
        # Compare with stored baselines
        max_novelty = 0.0
        for baseline_name, baseline_values in self.baseline_behaviors.items():
            if len(baseline_values) >= len(current_pattern):
                baseline_pattern = np.array(baseline_values[-len(current_pattern):])
                similarity = np.corrcoef(current_pattern, baseline_pattern)[0, 1]
                novelty = 1 - abs(similarity) if not np.isnan(similarity) else 1.0
                max_novelty = max(max_novelty, novelty)
        
        return max_novelty

class EvaluationEngine:
    """
    Core evaluation engine for multi-dimensional system assessment
    
    Implements the formula:
    System_Quality = Σ wᵢ × Qᵢ(S, E, T)
    
    Where:
    - wᵢ: Weight for dimension i
    - Qᵢ: Quality function for dimension i
    - S: System state
    - E: Environment
    - T: Time
    """
    
    def __init__(self, 
                 dimension_weights: Optional[Dict[str, float]] = None,
                 confidence_level: float = 0.95):
        """
        Initialize evaluation engine
        
        Args:
            dimension_weights: Custom weights for evaluation dimensions
            confidence_level: Statistical confidence level for results
        """
        self.dimension_weights = dimension_weights or {
            EvaluationDimension.PERFORMANCE.value: 0.3,
            EvaluationDimension.EFFICIENCY.value: 0.25,
            EvaluationDimension.EMERGENCE.value: 0.2,
            EvaluationDimension.RELIABILITY.value: 0.15,
            EvaluationDimension.ADAPTABILITY.value: 0.1
        }
        self.confidence_level = confidence_level
        self.emergence_detector = EmergenceDetector()
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.dimension_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            logger.warning(f"Dimension weights sum to {weight_sum}, normalizing...")
            for key in self.dimension_weights:
                self.dimension_weights[key] /= weight_sum
    
    async def evaluate_system(self, 
                             db: Session,
                             tasks: List[Any],
                             context_data: Optional[Dict[str, Any]] = None) -> SystemQualityResult:
        """
        Comprehensive system evaluation across multiple dimensions
        
        Args:
            db: Database session
            tasks: List of tasks for evaluation
            context_data: Additional context information
            
        Returns:
            SystemQualityResult with comprehensive evaluation metrics
        """
        try:
            logger.info(f"Starting system evaluation with {len(tasks)} tasks")
            
            if not tasks:
                logger.warning("No tasks provided for evaluation")
                return self._empty_result()
            
            # Calculate dimension scores
            dimension_scores = {}
            task_scores = []
            
            # Performance evaluation
            performance_scores = await self._evaluate_performance(tasks)
            dimension_scores[EvaluationDimension.PERFORMANCE.value] = np.mean(performance_scores)
            
            # Efficiency evaluation  
            efficiency_scores = await self._evaluate_efficiency(tasks)
            dimension_scores[EvaluationDimension.EFFICIENCY.value] = np.mean(efficiency_scores)
            
            # Emergence evaluation
            emergence_result = await self._evaluate_emergence(tasks)
            dimension_scores[EvaluationDimension.EMERGENCE.value] = emergence_result.emergence_score
            
            # Reliability evaluation
            reliability_scores = await self._evaluate_reliability(tasks)
            dimension_scores[EvaluationDimension.RELIABILITY.value] = np.mean(reliability_scores)
            
            # Adaptability evaluation
            adaptability_scores = await self._evaluate_adaptability(tasks)
            dimension_scores[EvaluationDimension.ADAPTABILITY.value] = np.mean(adaptability_scores)
            
            # Calculate individual task scores
            for i, task in enumerate(tasks):
                task_score = {
                    "task_id": getattr(task, 'id', i),
                    "performance": performance_scores[i] if i < len(performance_scores) else 0.0,
                    "efficiency": efficiency_scores[i] if i < len(efficiency_scores) else 0.0,
                    "reliability": reliability_scores[i] if i < len(reliability_scores) else 0.0,
                    "adaptability": adaptability_scores[i] if i < len(adaptability_scores) else 0.0
                }
                
                # Calculate weighted quality score for task
                task_quality = sum(
                    self.dimension_weights[dim] * task_score[dim]
                    for dim in task_score.keys() if dim != "task_id"
                )
                task_score["quality"] = task_quality
                task_scores.append(task_score)
            
            # Calculate overall system quality
            overall_quality = sum(
                self.dimension_weights[dim] * score
                for dim, score in dimension_scores.items()
            )
            
            # Calculate confidence interval
            quality_values = [task["quality"] for task in task_scores]
            confidence_interval = self._calculate_confidence_interval(quality_values)
            
            # Emergence indicators
            emergence_indicators = {
                "emergence_score": emergence_result.emergence_score,
                "variance_ratio": emergence_result.variance_ratio,
                "complexity_index": emergence_result.complexity_index,
                "novelty_score": emergence_result.novelty_score
            }
            
            result = SystemQualityResult(
                overall_quality=overall_quality,
                dimension_scores=dimension_scores,
                task_scores=task_scores,
                emergence_indicators=emergence_indicators,
                evaluation_timestamp=datetime.utcnow(),
                confidence_interval=confidence_interval
            )
            
            logger.info(f"System evaluation completed. Overall quality: {overall_quality:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in system evaluation: {e}")
            return self._empty_result()
    
    async def _evaluate_performance(self, tasks: List[Any]) -> List[float]:
        """Evaluate performance dimension"""
        scores = []
        for task in tasks:
            # Basic performance metrics
            completion_rate = 1.0 if getattr(task, 'status', '') == 'completed' else 0.5
            
            # Task complexity factor
            description_length = len(getattr(task, 'description', ''))
            complexity_factor = min(description_length / 500.0, 1.0)  # Normalize to max 1.0
            
            # Time efficiency
            if hasattr(task, 'created_at') and hasattr(task, 'updated_at'):
                duration = (task.updated_at - task.created_at).total_seconds()
                time_efficiency = max(0.1, 1.0 / (1.0 + duration / 3600.0))  # Hours
            else:
                time_efficiency = 0.5
            
            performance_score = (completion_rate * 0.5 + 
                               complexity_factor * 0.3 + 
                               time_efficiency * 0.2)
            
            scores.append(min(1.0, max(0.0, performance_score)))
        
        return scores
    
    async def _evaluate_efficiency(self, tasks: List[Any]) -> List[float]:
        """Evaluate efficiency dimension"""
        scores = []
        for task in tasks:
            # Resource utilization metrics
            if hasattr(task, 'created_at') and hasattr(task, 'updated_at'):
                duration_hours = (task.updated_at - task.created_at).total_seconds() / 3600.0
                efficiency = 1.0 / (1.0 + duration_hours) if duration_hours > 0 else 1.0
            else:
                efficiency = 0.5
            
            # Success rate factor
            success_factor = 1.0 if getattr(task, 'status', '') == 'completed' else 0.3
            
            efficiency_score = efficiency * success_factor
            scores.append(min(1.0, max(0.0, efficiency_score)))
        
        return scores
    
    async def _evaluate_emergence(self, tasks: List[Any]) -> EmergenceIndicator:
        """Evaluate emergence dimension"""
        if not tasks:
            return EmergenceIndicator(0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Extract observable metrics from tasks
        observed_values = []
        predicted_values = []
        timestamps = []
        
        for task in tasks:
            # Use task complexity as observed value
            observed_complexity = len(getattr(task, 'description', '')) / 100.0
            observed_values.append(observed_complexity)
            
            # Use simple prediction baseline
            predicted_values.append(0.5)  # Baseline prediction
            
            # Use creation timestamp
            timestamp = getattr(task, 'created_at', datetime.utcnow())
            timestamps.append(timestamp)
        
        return await self.emergence_detector.detect_emergence(
            observed_values, predicted_values, timestamps
        )
    
    async def _evaluate_reliability(self, tasks: List[Any]) -> List[float]:
        """Evaluate reliability dimension using exponential decay"""
        scores = []
        for task in tasks:
            if hasattr(task, 'created_at'):
                # Reliability decays over time (R(t) = e^(-λt))
                age_hours = (datetime.utcnow() - task.created_at).total_seconds() / 3600.0
                reliability = np.exp(-0.1 * age_hours)  # λ = 0.1
            else:
                reliability = 0.5
            
            # Adjust for success
            if getattr(task, 'status', '') == 'completed':
                reliability *= 1.2
            elif getattr(task, 'status', '') == 'failed':
                reliability *= 0.5
            
            scores.append(min(1.0, max(0.0, reliability)))
        
        return scores
    
    async def _evaluate_adaptability(self, tasks: List[Any]) -> List[float]:
        """Evaluate adaptability dimension"""
        scores = []
        for task in tasks:
            # Measure adaptability based on task variation handling
            task_type_diversity = 0.7  # Placeholder - could analyze actual task types
            
            # Context adaptation
            description_uniqueness = len(set(getattr(task, 'description', '').split())) / 100.0
            context_adaptation = min(1.0, description_uniqueness)
            
            adaptability_score = (task_type_diversity * 0.6 + context_adaptation * 0.4)
            scores.append(min(1.0, max(0.0, adaptability_score)))
        
        return scores
    
    def _calculate_confidence_interval(self, values: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for quality scores"""
        if not values:
            return (0.0, 0.0)
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        n = len(values)
        
        # Z-score for 95% confidence
        z_score = 1.96 if self.confidence_level == 0.95 else 2.576
        margin = z_score * (std_val / np.sqrt(n))
        
        return (max(0.0, mean_val - margin), min(1.0, mean_val + margin))
    
    def _empty_result(self) -> SystemQualityResult:
        """Return empty result for error cases"""
        return SystemQualityResult(
            overall_quality=0.0,
            dimension_scores={dim: 0.0 for dim in self.dimension_weights.keys()},
            task_scores=[],
            emergence_indicators={},
            evaluation_timestamp=datetime.utcnow(),
            confidence_interval=(0.0, 0.0)
        )
    
    def get_evaluation_metrics(self, values: List[float]) -> EvaluationMetrics:
        """Calculate statistical metrics for evaluation results"""
        if not values:
            return EvaluationMetrics(0.0, 0.0, 0.0, 0.0, self.confidence_level, 0)
        
        mean_score = np.mean(values)
        variance = np.var(values)
        std_dev = np.std(values)
        cv = std_dev / mean_score if mean_score > 0 else 0.0
        
        return EvaluationMetrics(
            mean_score=mean_score,
            variance=variance,
            standard_deviation=std_dev,
            coefficient_of_variation=cv,
            confidence_level=self.confidence_level,
            sample_size=len(values)
        )

