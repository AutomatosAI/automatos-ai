

"""
Benchmark Design Framework - Validity, Reliability & Discriminatory Power Analysis
=================================================================================

Implements comprehensive benchmark design with:
- Validity calculation using Validity(B) = α * Content + β * Construct + γ * Criterion
- Reliability analysis using Reliability = 1 - Variance_error / Variance_total  
- Discriminatory Power using Discriminatory_Power = |Score_high - Score_low| / Range
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pandas as pd

logger = logging.getLogger(__name__)

class ValidityType(Enum):
    """Types of validity assessment"""
    CONTENT = "content"
    CONSTRUCT = "construct" 
    CRITERION = "criterion"
    FACE = "face"
    CONVERGENT = "convergent"
    DISCRIMINANT = "discriminant"

class ReliabilityType(Enum):
    """Types of reliability assessment"""
    INTERNAL_CONSISTENCY = "internal_consistency"
    TEST_RETEST = "test_retest"
    INTER_RATER = "inter_rater"
    SPLIT_HALF = "split_half"

class BenchmarkType(Enum):
    """Types of benchmarks"""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    SCALABILITY = "scalability"
    RELIABILITY = "reliability"

@dataclass
class ValidityResult:
    """Validity assessment results"""
    overall_validity: float
    content_validity: float
    construct_validity: float
    criterion_validity: float
    validity_components: Dict[str, float]
    confidence_interval: Tuple[float, float]
    validity_classification: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ReliabilityResult:
    """Reliability assessment results"""
    overall_reliability: float
    internal_consistency: float
    temporal_stability: float  
    inter_rater_reliability: float
    measurement_error: float
    reliability_classification: str
    confidence_interval: Tuple[float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class DiscriminatoryPowerResult:
    """Discriminatory power analysis results"""
    discriminatory_power: float
    separation_index: float
    effect_size: float
    statistical_significance: float
    power_classification: str
    group_statistics: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class BenchmarkQualityResult:
    """Complete benchmark quality assessment"""
    validity_result: ValidityResult
    reliability_result: ReliabilityResult
    discriminatory_power_result: DiscriminatoryPowerResult
    overall_benchmark_quality: float
    benchmark_classification: str
    recommendations: List[str]
    assessment_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['validity_result'] = self.validity_result.to_dict()
        result['reliability_result'] = self.reliability_result.to_dict()
        result['discriminatory_power_result'] = self.discriminatory_power_result.to_dict()
        result['assessment_timestamp'] = self.assessment_timestamp.isoformat()
        return result

class ValidityCalculator:
    """
    Comprehensive validity assessment calculator
    
    Implements: Validity(B) = α * Content + β * Construct + γ * Criterion
    """
    
    def __init__(self):
        self.validity_weights = {
            ValidityType.CONTENT: 0.4,
            ValidityType.CONSTRUCT: 0.3,
            ValidityType.CRITERION: 0.3
        }
        
        self.validity_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "adequate": 0.7,
            "questionable": 0.6,
            "poor": 0.0
        }
    
    async def calculate_validity(self, 
                               benchmark_data: Dict[str, Any],
                               theoretical_framework: Dict[str, Any],
                               criterion_data: Optional[Dict[str, Any]] = None) -> ValidityResult:
        """
        Calculate comprehensive validity using multiple validity types
        """
        try:
            logger.info("Calculating benchmark validity")
            
            # Content validity assessment
            content_validity = await self._calculate_content_validity(
                benchmark_data, theoretical_framework
            )
            
            # Construct validity assessment  
            construct_validity = await self._calculate_construct_validity(
                benchmark_data, theoretical_framework
            )
            
            # Criterion validity assessment
            criterion_validity = await self._calculate_criterion_validity(
                benchmark_data, criterion_data
            ) if criterion_data else 0.5
            
            # Calculate overall validity
            overall_validity = (
                self.validity_weights[ValidityType.CONTENT] * content_validity +
                self.validity_weights[ValidityType.CONSTRUCT] * construct_validity +
                self.validity_weights[ValidityType.CRITERION] * criterion_validity
            )
            
            # Additional validity components
            validity_components = await self._calculate_additional_validities(
                benchmark_data, theoretical_framework
            )
            
            # Calculate confidence interval
            validity_scores = [content_validity, construct_validity, criterion_validity]
            confidence_interval = self._calculate_confidence_interval(validity_scores)
            
            # Classify validity level
            validity_classification = self._classify_validity(overall_validity)
            
            return ValidityResult(
                overall_validity=overall_validity,
                content_validity=content_validity,
                construct_validity=construct_validity,
                criterion_validity=criterion_validity,
                validity_components=validity_components,
                confidence_interval=confidence_interval,
                validity_classification=validity_classification
            )
            
        except Exception as e:
            logger.error(f"Error calculating validity: {e}")
            return self._create_empty_validity_result()
    
    async def _calculate_content_validity(self, 
                                        benchmark_data: Dict[str, Any],
                                        theoretical_framework: Dict[str, Any]) -> float:
        """
        Calculate content validity based on coverage of theoretical framework
        """
        # Get benchmark components and theoretical requirements
        benchmark_components = benchmark_data.get('components', [])
        theoretical_requirements = theoretical_framework.get('required_capabilities', [])
        
        if not theoretical_requirements:
            return 0.7  # Default when no framework provided
        
        # Calculate coverage
        covered_requirements = 0
        total_requirements = len(theoretical_requirements)
        
        for requirement in theoretical_requirements:
            # Check if requirement is covered by any benchmark component
            requirement_covered = any(
                requirement.lower() in str(component).lower()
                for component in benchmark_components
            )
            if requirement_covered:
                covered_requirements += 1
        
        # Content validity = coverage ratio
        coverage_ratio = covered_requirements / total_requirements if total_requirements > 0 else 0.0
        
        # Adjust for comprehensiveness
        comprehensiveness_factor = min(1.0, len(benchmark_components) / max(1, total_requirements))
        
        content_validity = (coverage_ratio * 0.7 + comprehensiveness_factor * 0.3)
        
        return min(1.0, content_validity)
    
    async def _calculate_construct_validity(self, 
                                          benchmark_data: Dict[str, Any],
                                          theoretical_framework: Dict[str, Any]) -> float:
        """
        Calculate construct validity using correlation with theoretical predictions
        """
        # Get benchmark scores and theoretical predictions
        benchmark_scores = benchmark_data.get('scores', [])
        theoretical_scores = theoretical_framework.get('predicted_scores', [])
        
        if not benchmark_scores or not theoretical_scores:
            return 0.6  # Default when insufficient data
        
        # Ensure same length for correlation
        min_length = min(len(benchmark_scores), len(theoretical_scores))
        if min_length < 2:
            return 0.6
        
        actual_scores = benchmark_scores[:min_length]
        predicted_scores = theoretical_scores[:min_length]
        
        try:
            # Calculate Pearson correlation
            correlation, p_value = pearsonr(actual_scores, predicted_scores)
            
            # Construct validity is the absolute correlation
            construct_validity = abs(correlation) if not np.isnan(correlation) else 0.5
            
            # Adjust for statistical significance
            if p_value < 0.05:
                construct_validity *= 1.1  # Boost for significance
            
            return min(1.0, construct_validity)
            
        except Exception as e:
            logger.warning(f"Error calculating construct validity correlation: {e}")
            return 0.5
    
    async def _calculate_criterion_validity(self, 
                                          benchmark_data: Dict[str, Any],
                                          criterion_data: Dict[str, Any]) -> float:
        """
        Calculate criterion validity using external criterion
        """
        # Get benchmark predictions and actual outcomes
        benchmark_predictions = benchmark_data.get('predictions', [])
        actual_outcomes = criterion_data.get('actual_outcomes', [])
        
        if not benchmark_predictions or not actual_outcomes:
            return 0.5
        
        # Ensure same length
        min_length = min(len(benchmark_predictions), len(actual_outcomes))
        if min_length < 2:
            return 0.5
        
        predictions = benchmark_predictions[:min_length]
        outcomes = actual_outcomes[:min_length]
        
        try:
            # For categorical data, use Cohen's Kappa
            if all(isinstance(x, (str, bool, int)) and isinstance(y, (str, bool, int)) 
                   for x, y in zip(predictions, outcomes)):
                kappa_score = cohen_kappa_score(predictions, outcomes)
                criterion_validity = (kappa_score + 1.0) / 2.0  # Normalize to 0-1
            
            # For continuous data, use correlation
            else:
                correlation, _ = pearsonr(predictions, outcomes)
                criterion_validity = abs(correlation) if not np.isnan(correlation) else 0.5
            
            return min(1.0, max(0.0, criterion_validity))
            
        except Exception as e:
            logger.warning(f"Error calculating criterion validity: {e}")
            return 0.5
    
    async def _calculate_additional_validities(self, 
                                             benchmark_data: Dict[str, Any],
                                             theoretical_framework: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional validity metrics"""
        additional_validities = {}
        
        # Face validity (expert assessment)
        expert_ratings = benchmark_data.get('expert_ratings', [])
        if expert_ratings:
            additional_validities['face_validity'] = np.mean(expert_ratings)
        else:
            additional_validities['face_validity'] = 0.7  # Default assumption
        
        # Convergent validity (correlation with similar benchmarks)
        similar_benchmark_scores = benchmark_data.get('similar_benchmark_scores', [])
        current_scores = benchmark_data.get('scores', [])
        
        if similar_benchmark_scores and current_scores:
            min_len = min(len(similar_benchmark_scores), len(current_scores))
            if min_len >= 2:
                try:
                    convergent_corr, _ = pearsonr(
                        similar_benchmark_scores[:min_len],
                        current_scores[:min_len]
                    )
                    additional_validities['convergent_validity'] = abs(convergent_corr) if not np.isnan(convergent_corr) else 0.6
                except:
                    additional_validities['convergent_validity'] = 0.6
            else:
                additional_validities['convergent_validity'] = 0.6
        else:
            additional_validities['convergent_validity'] = 0.6
        
        # Discriminant validity (low correlation with dissimilar constructs)
        dissimilar_scores = benchmark_data.get('dissimilar_construct_scores', [])
        if dissimilar_scores and current_scores:
            min_len = min(len(dissimilar_scores), len(current_scores))
            if min_len >= 2:
                try:
                    discriminant_corr, _ = pearsonr(
                        dissimilar_scores[:min_len],
                        current_scores[:min_len]
                    )
                    # For discriminant validity, we want LOW correlation
                    additional_validities['discriminant_validity'] = 1.0 - abs(discriminant_corr) if not np.isnan(discriminant_corr) else 0.6
                except:
                    additional_validities['discriminant_validity'] = 0.6
            else:
                additional_validities['discriminant_validity'] = 0.6
        else:
            additional_validities['discriminant_validity'] = 0.6
        
        return additional_validities
    
    def _classify_validity(self, validity_score: float) -> str:
        """Classify validity level"""
        for level, threshold in self.validity_thresholds.items():
            if validity_score >= threshold:
                return level
        return "poor"
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for validity scores"""
        if not scores:
            return (0.0, 0.0)
        
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        n = len(scores)
        
        # 95% confidence interval
        margin = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0.0
        
        return (max(0.0, mean_val - margin), min(1.0, mean_val + margin))
    
    def _create_empty_validity_result(self) -> ValidityResult:
        """Create empty validity result for error cases"""
        return ValidityResult(
            overall_validity=0.0,
            content_validity=0.0,
            construct_validity=0.0,
            criterion_validity=0.0,
            validity_components={},
            confidence_interval=(0.0, 0.0),
            validity_classification="error"
        )

class ReliabilityCalculator:
    """
    Comprehensive reliability assessment calculator
    
    Implements: Reliability = 1 - Variance_error / Variance_total
    """
    
    def __init__(self):
        self.reliability_weights = {
            ReliabilityType.INTERNAL_CONSISTENCY: 0.4,
            ReliabilityType.TEST_RETEST: 0.3,
            ReliabilityType.INTER_RATER: 0.3
        }
        
        self.reliability_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "acceptable": 0.7,
            "questionable": 0.6,
            "poor": 0.0
        }
    
    async def calculate_reliability(self, 
                                  benchmark_scores: List[float],
                                  repeated_measures: Optional[List[List[float]]] = None,
                                  rater_scores: Optional[List[List[float]]] = None) -> ReliabilityResult:
        """
        Calculate comprehensive reliability using multiple reliability types
        """
        try:
            logger.info("Calculating benchmark reliability")
            
            # Internal consistency (Cronbach's alpha approximation)
            internal_consistency = await self._calculate_internal_consistency(
                benchmark_scores, repeated_measures
            )
            
            # Test-retest reliability
            temporal_stability = await self._calculate_temporal_stability(
                repeated_measures
            ) if repeated_measures else 0.7
            
            # Inter-rater reliability
            inter_rater_reliability = await self._calculate_inter_rater_reliability(
                rater_scores
            ) if rater_scores else 0.7
            
            # Calculate overall reliability
            overall_reliability = (
                self.reliability_weights[ReliabilityType.INTERNAL_CONSISTENCY] * internal_consistency +
                self.reliability_weights[ReliabilityType.TEST_RETEST] * temporal_stability +
                self.reliability_weights[ReliabilityType.INTER_RATER] * inter_rater_reliability
            )
            
            # Calculate measurement error
            measurement_error = await self._calculate_measurement_error(
                benchmark_scores, repeated_measures
            )
            
            # Classify reliability
            reliability_classification = self._classify_reliability(overall_reliability)
            
            # Calculate confidence interval
            reliability_scores = [internal_consistency, temporal_stability, inter_rater_reliability]
            confidence_interval = self._calculate_confidence_interval(reliability_scores)
            
            return ReliabilityResult(
                overall_reliability=overall_reliability,
                internal_consistency=internal_consistency,
                temporal_stability=temporal_stability,
                inter_rater_reliability=inter_rater_reliability,
                measurement_error=measurement_error,
                reliability_classification=reliability_classification,
                confidence_interval=confidence_interval
            )
            
        except Exception as e:
            logger.error(f"Error calculating reliability: {e}")
            return self._create_empty_reliability_result()
    
    async def _calculate_internal_consistency(self, 
                                            scores: List[float],
                                            repeated_measures: Optional[List[List[float]]]) -> float:
        """
        Calculate internal consistency using variance analysis
        """
        if not scores:
            return 0.5
        
        # If we have repeated measures, use them for better estimation
        if repeated_measures and len(repeated_measures) > 1:
            all_scores = []
            for measures in repeated_measures:
                all_scores.extend(measures)
            
            # Variance-based reliability estimation
            total_variance = np.var(all_scores) if all_scores else 1.0
            
            # Calculate variance within each measurement set
            within_variances = []
            for measures in repeated_measures:
                if len(measures) > 1:
                    within_variances.append(np.var(measures))
            
            error_variance = np.mean(within_variances) if within_variances else total_variance * 0.3
            
            # Reliability = 1 - error_variance / total_variance
            reliability = 1.0 - (error_variance / total_variance) if total_variance > 0 else 0.5
            
        else:
            # Simple estimation based on score distribution
            score_variance = np.var(scores)
            score_mean = np.mean(scores)
            
            # Coefficient of variation as reliability proxy
            cv = (np.std(scores) / score_mean) if score_mean > 0 else 1.0
            reliability = max(0.0, 1.0 - cv)
        
        return min(1.0, max(0.0, reliability))
    
    async def _calculate_temporal_stability(self, 
                                          repeated_measures: List[List[float]]) -> float:
        """
        Calculate test-retest reliability using temporal stability
        """
        if not repeated_measures or len(repeated_measures) < 2:
            return 0.7  # Default assumption
        
        # Calculate correlations between consecutive measurements
        correlations = []
        
        for i in range(len(repeated_measures) - 1):
            measure1 = repeated_measures[i]
            measure2 = repeated_measures[i + 1]
            
            if len(measure1) >= 2 and len(measure2) >= 2:
                # Ensure same length for correlation
                min_len = min(len(measure1), len(measure2))
                try:
                    corr, _ = pearsonr(measure1[:min_len], measure2[:min_len])
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        # Average correlation as stability measure
        if correlations:
            temporal_stability = np.mean(correlations)
        else:
            temporal_stability = 0.7  # Default when correlation fails
        
        return min(1.0, max(0.0, temporal_stability))
    
    async def _calculate_inter_rater_reliability(self, 
                                               rater_scores: List[List[float]]) -> float:
        """
        Calculate inter-rater reliability using intraclass correlation approximation
        """
        if not rater_scores or len(rater_scores) < 2:
            return 0.7  # Default assumption
        
        # Convert to matrix format
        try:
            # Pad shorter lists with mean values to ensure same length
            max_len = max(len(scores) for scores in rater_scores)
            padded_scores = []
            
            for scores in rater_scores:
                if len(scores) < max_len:
                    mean_score = np.mean(scores) if scores else 0.5
                    padded = scores + [mean_score] * (max_len - len(scores))
                else:
                    padded = scores[:max_len]
                padded_scores.append(padded)
            
            # Calculate correlation matrix
            correlations = []
            for i in range(len(padded_scores)):
                for j in range(i + 1, len(padded_scores)):
                    try:
                        corr, _ = pearsonr(padded_scores[i], padded_scores[j])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    except:
                        continue
            
            # Average correlation as inter-rater reliability
            if correlations:
                inter_rater_reliability = np.mean(correlations)
            else:
                inter_rater_reliability = 0.7
                
        except Exception as e:
            logger.warning(f"Error calculating inter-rater reliability: {e}")
            inter_rater_reliability = 0.7
        
        return min(1.0, max(0.0, inter_rater_reliability))
    
    async def _calculate_measurement_error(self, 
                                         scores: List[float],
                                         repeated_measures: Optional[List[List[float]]]) -> float:
        """
        Calculate measurement error as proportion of total variance
        """
        if not scores:
            return 0.5
        
        total_variance = np.var(scores)
        
        if repeated_measures and len(repeated_measures) > 1:
            # Calculate error variance from repeated measures
            error_variances = []
            for measures in repeated_measures:
                if len(measures) > 1:
                    error_variances.append(np.var(measures))
            
            error_variance = np.mean(error_variances) if error_variances else total_variance * 0.3
        else:
            # Estimate error variance as 30% of total variance
            error_variance = total_variance * 0.3
        
        # Measurement error as proportion
        measurement_error = error_variance / total_variance if total_variance > 0 else 0.3
        
        return min(1.0, max(0.0, measurement_error))
    
    def _classify_reliability(self, reliability_score: float) -> str:
        """Classify reliability level"""
        for level, threshold in self.reliability_thresholds.items():
            if reliability_score >= threshold:
                return level
        return "poor"
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for reliability scores"""
        if not scores:
            return (0.0, 0.0)
        
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        n = len(scores)
        
        # 95% confidence interval
        margin = 1.96 * (std_val / np.sqrt(n)) if n > 0 else 0.0
        
        return (max(0.0, mean_val - margin), min(1.0, mean_val + margin))
    
    def _create_empty_reliability_result(self) -> ReliabilityResult:
        """Create empty reliability result for error cases"""
        return ReliabilityResult(
            overall_reliability=0.0,
            internal_consistency=0.0,
            temporal_stability=0.0,
            inter_rater_reliability=0.0,
            measurement_error=1.0,
            reliability_classification="error",
            confidence_interval=(0.0, 0.0)
        )

class DiscriminatoryPowerAnalyzer:
    """
    Discriminatory power analysis
    
    Implements: Discriminatory_Power = |Score_high - Score_low| / Range
    """
    
    def __init__(self):
        self.power_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "moderate": 0.4,
            "weak": 0.2,
            "poor": 0.0
        }
    
    async def analyze_discriminatory_power(self, 
                                         high_performance_scores: List[float],
                                         low_performance_scores: List[float],
                                         all_scores: Optional[List[float]] = None) -> DiscriminatoryPowerResult:
        """
        Analyze discriminatory power using group differences
        """
        try:
            logger.info("Analyzing discriminatory power")
            
            if not high_performance_scores or not low_performance_scores:
                return self._create_empty_discriminatory_result()
            
            # Calculate basic discriminatory power
            high_mean = np.mean(high_performance_scores)
            low_mean = np.mean(low_performance_scores)
            
            # Calculate score range
            if all_scores:
                score_range = max(all_scores) - min(all_scores)
            else:
                combined_scores = high_performance_scores + low_performance_scores
                score_range = max(combined_scores) - min(combined_scores)
            
            # Discriminatory power formula
            discriminatory_power = abs(high_mean - low_mean) / score_range if score_range > 0 else 0.0
            
            # Calculate separation index (standardized difference)
            high_std = np.std(high_performance_scores)
            low_std = np.std(low_performance_scores)
            pooled_std = np.sqrt((high_std**2 + low_std**2) / 2)
            
            separation_index = abs(high_mean - low_mean) / pooled_std if pooled_std > 0 else 0.0
            
            # Calculate effect size (Cohen's d)
            effect_size = abs(high_mean - low_mean) / pooled_std if pooled_std > 0 else 0.0
            
            # Statistical significance test
            try:
                t_stat, p_value = stats.ttest_ind(high_performance_scores, low_performance_scores)
                statistical_significance = 1.0 - p_value if not np.isnan(p_value) else 0.5
            except:
                statistical_significance = 0.5
            
            # Group statistics
            group_statistics = {
                'high_performance': {
                    'mean': high_mean,
                    'std': high_std,
                    'n': len(high_performance_scores),
                    'min': min(high_performance_scores),
                    'max': max(high_performance_scores)
                },
                'low_performance': {
                    'mean': low_mean,
                    'std': low_std,
                    'n': len(low_performance_scores),
                    'min': min(low_performance_scores),
                    'max': max(low_performance_scores)
                }
            }
            
            # Classify power level
            power_classification = self._classify_power(discriminatory_power)
            
            return DiscriminatoryPowerResult(
                discriminatory_power=discriminatory_power,
                separation_index=separation_index,
                effect_size=effect_size,
                statistical_significance=statistical_significance,
                power_classification=power_classification,
                group_statistics=group_statistics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing discriminatory power: {e}")
            return self._create_empty_discriminatory_result()
    
    def _classify_power(self, power_score: float) -> str:
        """Classify discriminatory power level"""
        for level, threshold in self.power_thresholds.items():
            if power_score >= threshold:
                return level
        return "poor"
    
    def _create_empty_discriminatory_result(self) -> DiscriminatoryPowerResult:
        """Create empty discriminatory power result for error cases"""
        return DiscriminatoryPowerResult(
            discriminatory_power=0.0,
            separation_index=0.0,
            effect_size=0.0,
            statistical_significance=0.0,
            power_classification="error",
            group_statistics={}
        )

class BenchmarkDesignFramework:
    """
    Main benchmark design framework integrating all analysis components
    """
    
    def __init__(self):
        self.validity_calculator = ValidityCalculator()
        self.reliability_calculator = ReliabilityCalculator()
        self.discriminatory_power_analyzer = DiscriminatoryPowerAnalyzer()
        
        self.quality_weights = {
            "validity": 0.4,
            "reliability": 0.35,
            "discriminatory_power": 0.25
        }
        
        self.benchmark_thresholds = {
            "excellent": 0.9,
            "good": 0.8,
            "adequate": 0.7,
            "needs_improvement": 0.6,
            "poor": 0.0
        }
    
    async def evaluate_benchmark_quality(self, 
                                       benchmark_data: Dict[str, Any],
                                       theoretical_framework: Dict[str, Any],
                                       validation_data: Optional[Dict[str, Any]] = None) -> BenchmarkQualityResult:
        """
        Comprehensive benchmark quality evaluation
        """
        try:
            logger.info("Evaluating comprehensive benchmark quality")
            
            # Validity assessment
            validity_result = await self.validity_calculator.calculate_validity(
                benchmark_data, theoretical_framework, validation_data
            )
            
            # Reliability assessment
            benchmark_scores = benchmark_data.get('scores', [])
            repeated_measures = benchmark_data.get('repeated_measures', None)
            rater_scores = benchmark_data.get('rater_scores', None)
            
            reliability_result = await self.reliability_calculator.calculate_reliability(
                benchmark_scores, repeated_measures, rater_scores
            )
            
            # Discriminatory power assessment
            high_scores = benchmark_data.get('high_performance_scores', [])
            low_scores = benchmark_data.get('low_performance_scores', [])
            all_scores = benchmark_data.get('all_scores', benchmark_scores)
            
            discriminatory_power_result = await self.discriminatory_power_analyzer.analyze_discriminatory_power(
                high_scores, low_scores, all_scores
            )
            
            # Calculate overall benchmark quality
            overall_quality = (
                self.quality_weights["validity"] * validity_result.overall_validity +
                self.quality_weights["reliability"] * reliability_result.overall_reliability +
                self.quality_weights["discriminatory_power"] * discriminatory_power_result.discriminatory_power
            )
            
            # Classify benchmark quality
            benchmark_classification = self._classify_benchmark_quality(overall_quality)
            
            # Generate recommendations
            recommendations = await self._generate_quality_recommendations(
                validity_result, reliability_result, discriminatory_power_result
            )
            
            return BenchmarkQualityResult(
                validity_result=validity_result,
                reliability_result=reliability_result,
                discriminatory_power_result=discriminatory_power_result,
                overall_benchmark_quality=overall_quality,
                benchmark_classification=benchmark_classification,
                recommendations=recommendations,
                assessment_timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error evaluating benchmark quality: {e}")
            return self._create_empty_benchmark_result()
    
    def _classify_benchmark_quality(self, quality_score: float) -> str:
        """Classify overall benchmark quality"""
        for level, threshold in self.benchmark_thresholds.items():
            if quality_score >= threshold:
                return level
        return "poor"
    
    async def _generate_quality_recommendations(self, 
                                              validity: ValidityResult,
                                              reliability: ReliabilityResult,
                                              discriminatory_power: DiscriminatoryPowerResult) -> List[str]:
        """Generate recommendations for benchmark improvement"""
        recommendations = []
        
        # Validity recommendations
        if validity.overall_validity < 0.7:
            if validity.content_validity < 0.6:
                recommendations.append("Improve content coverage of theoretical framework")
            if validity.construct_validity < 0.6:
                recommendations.append("Enhance alignment with theoretical constructs")
            if validity.criterion_validity < 0.6:
                recommendations.append("Improve predictive accuracy against external criteria")
        
        # Reliability recommendations
        if reliability.overall_reliability < 0.7:
            if reliability.internal_consistency < 0.6:
                recommendations.append("Improve internal consistency of benchmark items")
            if reliability.temporal_stability < 0.6:
                recommendations.append("Enhance test-retest reliability over time")
            if reliability.inter_rater_reliability < 0.6:
                recommendations.append("Improve agreement between different raters")
        
        # Discriminatory power recommendations
        if discriminatory_power.discriminatory_power < 0.4:
            recommendations.append("Enhance ability to distinguish between performance levels")
            
        if discriminatory_power.statistical_significance < 0.95:
            recommendations.append("Increase sample size or effect size for statistical power")
        
        return recommendations if recommendations else ["Benchmark shows good overall quality"]
    
    def _create_empty_benchmark_result(self) -> BenchmarkQualityResult:
        """Create empty benchmark result for error cases"""
        return BenchmarkQualityResult(
            validity_result=ValidityResult(0.0, 0.0, 0.0, 0.0, {}, (0.0, 0.0), "error"),
            reliability_result=ReliabilityResult(0.0, 0.0, 0.0, 0.0, 1.0, "error", (0.0, 0.0)),
            discriminatory_power_result=DiscriminatoryPowerResult(0.0, 0.0, 0.0, 0.0, "error", {}),
            overall_benchmark_quality=0.0,
            benchmark_classification="error",
            recommendations=["Benchmark evaluation error"],
            assessment_timestamp=datetime.utcnow()
        )

