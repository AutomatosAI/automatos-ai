

"""
System Integration Evaluator - Coherence, Efficiency & Emergence Analysis
=========================================================================

Implements comprehensive system integration evaluation with:
- Coherence calculation using Coherence(S) = 1 - Σ |Observed - Expected| / N
- Efficiency analysis using Efficiency = Actual / Theoretical  
- Emergent Capability Index using ECI(S) = |System - Σ Components| / |System|
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from scipy import stats
from scipy.stats import pearsonr
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntegrationAspect(Enum):
    """Aspects of system integration"""
    COHERENCE = "coherence"
    EFFICIENCY = "efficiency"
    EMERGENCE = "emergence"
    SYNCHRONIZATION = "synchronization"
    SCALABILITY = "scalability"

class EmergenceType(Enum):
    """Types of emergent behavior"""
    PERFORMANCE_GAIN = "performance_gain"
    CAPABILITY_EXPANSION = "capability_expansion"
    BEHAVIORAL_NOVELTY = "behavioral_novelty"
    ADAPTIVE_LEARNING = "adaptive_learning"

@dataclass
class CoherenceMetrics:
    """Metrics for system coherence analysis"""
    overall_coherence: float
    component_alignment: Dict[str, float]
    communication_coherence: float
    state_coherence: float
    behavioral_consistency: float
    deviation_analysis: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EfficiencyMetrics:
    """Metrics for system efficiency analysis"""
    overall_efficiency: float
    resource_utilization: float
    throughput_efficiency: float
    latency_efficiency: float
    cost_efficiency: float
    bottleneck_analysis: List[str]
    optimization_potential: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass 
class EmergentCapabilityIndex:
    """Emergent capability analysis results"""
    eci_score: float
    emergence_type: EmergenceType
    capability_gain: float
    system_behavior: Dict[str, float]
    component_contributions: Dict[str, float]
    novelty_indicators: List[str]
    prediction_accuracy: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['emergence_type'] = self.emergence_type.value
        return result

@dataclass
class IntegrationEvaluationResult:
    """Complete integration evaluation result"""
    coherence_metrics: CoherenceMetrics
    efficiency_metrics: EfficiencyMetrics
    emergent_capability: EmergentCapabilityIndex
    integration_score: float
    integration_classification: str
    recommendations: List[str]
    evaluation_timestamp: datetime
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['coherence_metrics'] = self.coherence_metrics.to_dict()
        result['efficiency_metrics'] = self.efficiency_metrics.to_dict()
        result['emergent_capability'] = self.emergent_capability.to_dict()
        result['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        return result

class CoherenceAnalyzer:
    """
    System coherence analysis
    
    Implements: Coherence(S) = 1 - Σ |Observed - Expected| / N
    """
    
    def __init__(self):
        self.coherence_weights = {
            "component_alignment": 0.3,
            "communication_coherence": 0.25,
            "state_coherence": 0.25,
            "behavioral_consistency": 0.2
        }
    
    async def analyze_coherence(self, 
                              system_components: List[Dict[str, Any]],
                              interaction_data: List[Dict[str, Any]],
                              expected_behaviors: Dict[str, float]) -> CoherenceMetrics:
        """
        Analyze system coherence using deviation from expected behavior
        """
        try:
            logger.info(f"Analyzing coherence for {len(system_components)} components")
            
            # Component alignment analysis
            component_alignment = await self._analyze_component_alignment(
                system_components, expected_behaviors
            )
            
            # Communication coherence
            communication_coherence = await self._analyze_communication_coherence(
                interaction_data
            )
            
            # State coherence
            state_coherence = await self._analyze_state_coherence(
                system_components
            )
            
            # Behavioral consistency
            behavioral_consistency = await self._analyze_behavioral_consistency(
                interaction_data, expected_behaviors
            )
            
            # Calculate overall coherence
            overall_coherence = (
                self.coherence_weights["component_alignment"] * np.mean(list(component_alignment.values())) +
                self.coherence_weights["communication_coherence"] * communication_coherence +
                self.coherence_weights["state_coherence"] * state_coherence +
                self.coherence_weights["behavioral_consistency"] * behavioral_consistency
            )
            
            # Deviation analysis
            deviation_analysis = await self._calculate_deviation_analysis(
                system_components, expected_behaviors
            )
            
            return CoherenceMetrics(
                overall_coherence=overall_coherence,
                component_alignment=component_alignment,
                communication_coherence=communication_coherence,
                state_coherence=state_coherence,
                behavioral_consistency=behavioral_consistency,
                deviation_analysis=deviation_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in coherence analysis: {e}")
            return CoherenceMetrics(0.0, {}, 0.0, 0.0, 0.0, {})
    
    async def _analyze_component_alignment(self, 
                                         components: List[Dict[str, Any]],
                                         expected: Dict[str, float]) -> Dict[str, float]:
        """Analyze how well components align with expected behavior"""
        alignment_scores = {}
        
        for component in components:
            component_id = component.get('id', 'unknown')
            
            # Performance alignment
            actual_performance = component.get('performance', 0.5)
            expected_performance = expected.get(f"{component_id}_performance", 0.5)
            performance_alignment = 1.0 - abs(actual_performance - expected_performance)
            
            # Behavior alignment
            actual_behavior = component.get('behavior_score', 0.5)
            expected_behavior = expected.get(f"{component_id}_behavior", 0.5)
            behavior_alignment = 1.0 - abs(actual_behavior - expected_behavior)
            
            # Combined alignment
            alignment_scores[component_id] = (performance_alignment + behavior_alignment) / 2.0
        
        return alignment_scores
    
    async def _analyze_communication_coherence(self, 
                                             interactions: List[Dict[str, Any]]) -> float:
        """Analyze coherence in component communications"""
        if not interactions:
            return 0.5
        
        # Message consistency
        message_types = defaultdict(list)
        for interaction in interactions:
            msg_type = interaction.get('type', 'unknown')
            success = interaction.get('success', False)
            message_types[msg_type].append(1.0 if success else 0.0)
        
        # Calculate consistency per message type
        type_consistency = []
        for msg_type, successes in message_types.items():
            if successes:
                consistency = np.mean(successes)
                type_consistency.append(consistency)
        
        return np.mean(type_consistency) if type_consistency else 0.5
    
    async def _analyze_state_coherence(self, 
                                     components: List[Dict[str, Any]]) -> float:
        """Analyze state coherence across components"""
        if not components:
            return 0.5
        
        # State synchronization analysis
        state_values = []
        timestamps = []
        
        for component in components:
            if 'state' in component and 'timestamp' in component:
                state_val = component.get('state', {}).get('value', 0.5)
                timestamp = component.get('timestamp')
                state_values.append(state_val)
                if timestamp:
                    timestamps.append(timestamp)
        
        if not state_values:
            return 0.5
        
        # Calculate state variance (lower variance = higher coherence)
        state_variance = np.var(state_values)
        coherence_score = max(0.0, 1.0 - state_variance)
        
        return coherence_score
    
    async def _analyze_behavioral_consistency(self, 
                                            interactions: List[Dict[str, Any]],
                                            expected: Dict[str, float]) -> float:
        """Analyze consistency of system behavior"""
        if not interactions:
            return 0.5
        
        # Response time consistency
        response_times = [
            interaction.get('response_time', 1.0) 
            for interaction in interactions 
            if 'response_time' in interaction
        ]
        
        if response_times:
            rt_variance = np.var(response_times)
            rt_consistency = max(0.0, 1.0 - (rt_variance / np.mean(response_times)))
        else:
            rt_consistency = 0.5
        
        # Success rate consistency
        successes = [
            1.0 if interaction.get('success', False) else 0.0
            for interaction in interactions
        ]
        
        success_rate = np.mean(successes) if successes else 0.5
        expected_success = expected.get('system_success_rate', 0.8)
        success_consistency = 1.0 - abs(success_rate - expected_success)
        
        return (rt_consistency + success_consistency) / 2.0
    
    async def _calculate_deviation_analysis(self, 
                                          components: List[Dict[str, Any]],
                                          expected: Dict[str, float]) -> Dict[str, float]:
        """Calculate detailed deviation analysis"""
        deviations = {}
        
        for component in components:
            component_id = component.get('id', 'unknown')
            
            # Performance deviation
            actual_perf = component.get('performance', 0.5)
            expected_perf = expected.get(f"{component_id}_performance", 0.5)
            perf_deviation = abs(actual_perf - expected_perf)
            
            deviations[f"{component_id}_performance"] = perf_deviation
        
        # System-level deviations
        all_performances = [c.get('performance', 0.5) for c in components]
        if all_performances:
            system_variance = np.var(all_performances)
            deviations['system_variance'] = system_variance
        
        return deviations

class EfficiencyAnalyzer:
    """
    System efficiency analysis
    
    Implements: Efficiency = Actual / Theoretical
    """
    
    def __init__(self):
        self.efficiency_weights = {
            "resource_utilization": 0.25,
            "throughput_efficiency": 0.3,
            "latency_efficiency": 0.25,
            "cost_efficiency": 0.2
        }
    
    async def analyze_efficiency(self, 
                               system_metrics: Dict[str, Any],
                               theoretical_benchmarks: Dict[str, float],
                               resource_data: Dict[str, Any]) -> EfficiencyMetrics:
        """
        Analyze system efficiency against theoretical benchmarks
        """
        try:
            logger.info("Analyzing system efficiency")
            
            # Resource utilization efficiency
            resource_efficiency = await self._analyze_resource_utilization(
                resource_data, theoretical_benchmarks
            )
            
            # Throughput efficiency  
            throughput_efficiency = await self._analyze_throughput_efficiency(
                system_metrics, theoretical_benchmarks
            )
            
            # Latency efficiency
            latency_efficiency = await self._analyze_latency_efficiency(
                system_metrics, theoretical_benchmarks
            )
            
            # Cost efficiency
            cost_efficiency = await self._analyze_cost_efficiency(
                system_metrics, theoretical_benchmarks
            )
            
            # Calculate overall efficiency
            overall_efficiency = (
                self.efficiency_weights["resource_utilization"] * resource_efficiency +
                self.efficiency_weights["throughput_efficiency"] * throughput_efficiency +
                self.efficiency_weights["latency_efficiency"] * latency_efficiency +
                self.efficiency_weights["cost_efficiency"] * cost_efficiency
            )
            
            # Bottleneck analysis
            bottlenecks = await self._identify_bottlenecks(
                system_metrics, theoretical_benchmarks
            )
            
            # Optimization potential
            optimization_potential = await self._calculate_optimization_potential(
                [resource_efficiency, throughput_efficiency, latency_efficiency, cost_efficiency]
            )
            
            return EfficiencyMetrics(
                overall_efficiency=overall_efficiency,
                resource_utilization=resource_efficiency,
                throughput_efficiency=throughput_efficiency,
                latency_efficiency=latency_efficiency,
                cost_efficiency=cost_efficiency,
                bottleneck_analysis=bottlenecks,
                optimization_potential=optimization_potential
            )
            
        except Exception as e:
            logger.error(f"Error in efficiency analysis: {e}")
            return EfficiencyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, [], 0.0)
    
    async def _analyze_resource_utilization(self, 
                                          resource_data: Dict[str, Any],
                                          benchmarks: Dict[str, float]) -> float:
        """Analyze resource utilization efficiency"""
        cpu_usage = resource_data.get('cpu_usage', 0.5)
        memory_usage = resource_data.get('memory_usage', 0.5)
        
        theoretical_cpu = benchmarks.get('optimal_cpu_usage', 0.7)
        theoretical_memory = benchmarks.get('optimal_memory_usage', 0.6)
        
        cpu_efficiency = min(1.0, cpu_usage / theoretical_cpu) if theoretical_cpu > 0 else 0.5
        memory_efficiency = min(1.0, memory_usage / theoretical_memory) if theoretical_memory > 0 else 0.5
        
        return (cpu_efficiency + memory_efficiency) / 2.0
    
    async def _analyze_throughput_efficiency(self, 
                                           metrics: Dict[str, Any],
                                           benchmarks: Dict[str, float]) -> float:
        """Analyze throughput efficiency"""
        actual_throughput = metrics.get('throughput', 1.0)
        theoretical_throughput = benchmarks.get('max_throughput', 2.0)
        
        return min(1.0, actual_throughput / theoretical_throughput) if theoretical_throughput > 0 else 0.5
    
    async def _analyze_latency_efficiency(self, 
                                        metrics: Dict[str, Any],
                                        benchmarks: Dict[str, float]) -> float:
        """Analyze latency efficiency (lower latency = higher efficiency)"""
        actual_latency = metrics.get('average_latency', 1.0)
        theoretical_latency = benchmarks.get('min_latency', 0.5)
        
        # Efficiency is inverse of latency ratio
        efficiency = theoretical_latency / actual_latency if actual_latency > 0 else 0.5
        return min(1.0, efficiency)
    
    async def _analyze_cost_efficiency(self, 
                                     metrics: Dict[str, Any],
                                     benchmarks: Dict[str, float]) -> float:
        """Analyze cost efficiency"""
        actual_cost = metrics.get('operational_cost', 1.0)
        theoretical_cost = benchmarks.get('optimal_cost', 0.8)
        
        # Lower cost relative to benchmark = higher efficiency
        efficiency = theoretical_cost / actual_cost if actual_cost > 0 else 0.5
        return min(1.0, efficiency)
    
    async def _identify_bottlenecks(self, 
                                  metrics: Dict[str, Any],
                                  benchmarks: Dict[str, float]) -> List[str]:
        """Identify system bottlenecks"""
        bottlenecks = []
        
        # CPU bottleneck
        cpu_usage = metrics.get('cpu_usage', 0.5)
        if cpu_usage > benchmarks.get('cpu_threshold', 0.8):
            bottlenecks.append("High CPU utilization")
        
        # Memory bottleneck
        memory_usage = metrics.get('memory_usage', 0.5)
        if memory_usage > benchmarks.get('memory_threshold', 0.8):
            bottlenecks.append("High memory utilization")
        
        # Latency bottleneck
        latency = metrics.get('average_latency', 1.0)
        if latency > benchmarks.get('latency_threshold', 2.0):
            bottlenecks.append("High response latency")
        
        # Throughput bottleneck
        throughput = metrics.get('throughput', 1.0)
        if throughput < benchmarks.get('throughput_threshold', 0.5):
            bottlenecks.append("Low throughput")
        
        return bottlenecks
    
    async def _calculate_optimization_potential(self, 
                                             efficiency_scores: List[float]) -> float:
        """Calculate optimization potential based on current efficiency"""
        if not efficiency_scores:
            return 0.0
        
        min_efficiency = min(efficiency_scores)
        max_efficiency = max(efficiency_scores)
        
        # Potential is the gap between current performance and ideal
        potential = 1.0 - np.mean(efficiency_scores)
        return max(0.0, min(1.0, potential))

class EmergenceAnalyzer:
    """
    Emergent capability analysis
    
    Implements: ECI(S) = |System - Σ Components| / |System|
    """
    
    def __init__(self):
        self.emergence_thresholds = {
            EmergenceType.PERFORMANCE_GAIN: 0.1,
            EmergenceType.CAPABILITY_EXPANSION: 0.15,
            EmergenceType.BEHAVIORAL_NOVELTY: 0.2,
            EmergenceType.ADAPTIVE_LEARNING: 0.12
        }
    
    async def analyze_emergence(self, 
                              system_performance: Dict[str, float],
                              component_performances: List[Dict[str, float]],
                              historical_data: Optional[List[Dict[str, Any]]] = None) -> EmergentCapabilityIndex:
        """
        Analyze emergent capabilities using ECI formula
        """
        try:
            logger.info("Analyzing emergent capabilities")
            
            # Calculate system capability
            system_capability = system_performance.get('overall_capability', 0.5)
            
            # Calculate sum of component capabilities
            component_sum = sum(
                comp.get('capability', 0.5) for comp in component_performances
            ) / len(component_performances) if component_performances else 0.5
            
            # Calculate ECI score
            if system_capability > 0:
                eci_score = abs(system_capability - component_sum) / system_capability
            else:
                eci_score = 0.0
            
            # Determine emergence type
            emergence_type = await self._classify_emergence_type(
                system_performance, component_performances, eci_score
            )
            
            # Calculate capability gain
            capability_gain = max(0.0, system_capability - component_sum)
            
            # Analyze system behavior patterns
            system_behavior = await self._analyze_system_behavior(
                system_performance, historical_data
            )
            
            # Calculate component contributions
            component_contributions = await self._calculate_component_contributions(
                component_performances
            )
            
            # Identify novelty indicators
            novelty_indicators = await self._identify_novelty_indicators(
                system_behavior, historical_data
            )
            
            # Calculate prediction accuracy
            prediction_accuracy = await self._calculate_prediction_accuracy(
                system_performance, component_performances
            )
            
            return EmergentCapabilityIndex(
                eci_score=eci_score,
                emergence_type=emergence_type,
                capability_gain=capability_gain,
                system_behavior=system_behavior,
                component_contributions=component_contributions,
                novelty_indicators=novelty_indicators,
                prediction_accuracy=prediction_accuracy
            )
            
        except Exception as e:
            logger.error(f"Error in emergence analysis: {e}")
            return EmergentCapabilityIndex(
                eci_score=0.0,
                emergence_type=EmergenceType.PERFORMANCE_GAIN,
                capability_gain=0.0,
                system_behavior={},
                component_contributions={},
                novelty_indicators=[],
                prediction_accuracy=0.0
            )
    
    async def _classify_emergence_type(self, 
                                     system_perf: Dict[str, float],
                                     component_perfs: List[Dict[str, float]],
                                     eci_score: float) -> EmergenceType:
        """Classify the type of emergence based on patterns"""
        
        # Performance gain emergence
        system_capability = system_perf.get('overall_capability', 0.5)
        avg_component_capability = np.mean([
            comp.get('capability', 0.5) for comp in component_perfs
        ]) if component_perfs else 0.5
        
        if system_capability > avg_component_capability * 1.1:
            return EmergenceType.PERFORMANCE_GAIN
        
        # Capability expansion
        system_features = system_perf.get('feature_count', 5)
        component_features = sum(
            comp.get('feature_count', 1) for comp in component_perfs
        ) if component_perfs else 5
        
        if system_features > component_features:
            return EmergenceType.CAPABILITY_EXPANSION
        
        # Behavioral novelty
        if eci_score > self.emergence_thresholds[EmergenceType.BEHAVIORAL_NOVELTY]:
            return EmergenceType.BEHAVIORAL_NOVELTY
        
        # Adaptive learning (default)
        return EmergenceType.ADAPTIVE_LEARNING
    
    async def _analyze_system_behavior(self, 
                                     system_perf: Dict[str, float],
                                     historical_data: Optional[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Analyze overall system behavior patterns"""
        behavior = {}
        
        # Current behavior metrics
        behavior['responsiveness'] = system_perf.get('response_time_score', 0.7)
        behavior['adaptability'] = system_perf.get('adaptability_score', 0.6)
        behavior['stability'] = system_perf.get('stability_score', 0.8)
        behavior['efficiency'] = system_perf.get('efficiency_score', 0.7)
        
        # Historical trend analysis
        if historical_data:
            recent_performance = [
                item.get('performance', 0.5) 
                for item in historical_data[-10:] 
                if 'performance' in item
            ]
            
            if recent_performance:
                behavior['trend_slope'] = self._calculate_trend_slope(recent_performance)
                behavior['performance_variance'] = np.var(recent_performance)
        
        return behavior
    
    async def _calculate_component_contributions(self, 
                                               component_perfs: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate individual component contributions to system capability"""
        contributions = {}
        
        total_capability = sum(comp.get('capability', 0.5) for comp in component_perfs)
        
        for i, comp in enumerate(component_perfs):
            comp_id = comp.get('id', f'component_{i}')
            comp_capability = comp.get('capability', 0.5)
            
            if total_capability > 0:
                contribution = comp_capability / total_capability
            else:
                contribution = 1.0 / len(component_perfs) if component_perfs else 0.0
            
            contributions[comp_id] = contribution
        
        return contributions
    
    async def _identify_novelty_indicators(self, 
                                         system_behavior: Dict[str, float],
                                         historical_data: Optional[List[Dict[str, Any]]]) -> List[str]:
        """Identify indicators of novel emergent behavior"""
        indicators = []
        
        # High adaptability indicates novel behavior
        if system_behavior.get('adaptability', 0.0) > 0.8:
            indicators.append("High adaptability suggests emergent learning")
        
        # Unusual performance patterns
        if system_behavior.get('performance_variance', 0.0) > 0.3:
            indicators.append("High performance variance indicates novel behaviors")
        
        # Positive trend in capabilities
        trend_slope = system_behavior.get('trend_slope', 0.0)
        if trend_slope > 0.1:
            indicators.append("Positive performance trend suggests capability emergence")
        
        # System exceeds expected efficiency
        if system_behavior.get('efficiency', 0.0) > 0.9:
            indicators.append("Exceptional efficiency indicates emergent optimization")
        
        return indicators
    
    async def _calculate_prediction_accuracy(self, 
                                           system_perf: Dict[str, float],
                                           component_perfs: List[Dict[str, float]]) -> float:
        """Calculate how well component performance predicts system performance"""
        if not component_perfs:
            return 0.0
        
        # Simple linear prediction model
        predicted_system_perf = np.mean([comp.get('capability', 0.5) for comp in component_perfs])
        actual_system_perf = system_perf.get('overall_capability', 0.5)
        
        # Accuracy based on prediction error
        prediction_error = abs(predicted_system_perf - actual_system_perf)
        accuracy = max(0.0, 1.0 - prediction_error)
        
        return accuracy
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _, _, _, _ = stats.linregress(x, values)
        return slope

class SystemIntegrationEvaluator:
    """
    Main system integration evaluator
    """
    
    def __init__(self):
        self.coherence_analyzer = CoherenceAnalyzer()
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.emergence_analyzer = EmergenceAnalyzer()
        
        self.integration_weights = {
            "coherence": 0.35,
            "efficiency": 0.35, 
            "emergence": 0.30
        }
    
    async def evaluate_integration(self, 
                                 system_components: List[Dict[str, Any]],
                                 interaction_data: List[Dict[str, Any]],
                                 system_metrics: Dict[str, Any],
                                 expected_behaviors: Dict[str, float],
                                 theoretical_benchmarks: Dict[str, float],
                                 resource_data: Dict[str, Any],
                                 historical_data: Optional[List[Dict[str, Any]]] = None) -> IntegrationEvaluationResult:
        """
        Comprehensive system integration evaluation
        """
        try:
            logger.info("Starting comprehensive system integration evaluation")
            
            # Coherence analysis
            coherence_metrics = await self.coherence_analyzer.analyze_coherence(
                system_components, interaction_data, expected_behaviors
            )
            
            # Efficiency analysis
            efficiency_metrics = await self.efficiency_analyzer.analyze_efficiency(
                system_metrics, theoretical_benchmarks, resource_data
            )
            
            # Emergence analysis
            system_performance = {
                'overall_capability': system_metrics.get('capability', 0.5),
                'response_time_score': 1.0 / max(1.0, system_metrics.get('average_latency', 1.0)),
                'adaptability_score': system_metrics.get('adaptability', 0.7),
                'stability_score': system_metrics.get('stability', 0.8),
                'efficiency_score': efficiency_metrics.overall_efficiency
            }
            
            component_performances = [
                {
                    'id': comp.get('id', 'unknown'),
                    'capability': comp.get('performance', 0.5),
                    'feature_count': comp.get('feature_count', 1)
                }
                for comp in system_components
            ]
            
            emergent_capability = await self.emergence_analyzer.analyze_emergence(
                system_performance, component_performances, historical_data
            )
            
            # Calculate overall integration score
            integration_score = (
                self.integration_weights["coherence"] * coherence_metrics.overall_coherence +
                self.integration_weights["efficiency"] * efficiency_metrics.overall_efficiency +
                self.integration_weights["emergence"] * emergent_capability.eci_score
            )
            
            # Classification
            classification = self._classify_integration_level(integration_score)
            
            # Generate recommendations
            recommendations = await self._generate_integration_recommendations(
                coherence_metrics, efficiency_metrics, emergent_capability
            )
            
            # Calculate confidence level
            confidence = self._calculate_confidence(
                coherence_metrics, efficiency_metrics, emergent_capability
            )
            
            result = IntegrationEvaluationResult(
                coherence_metrics=coherence_metrics,
                efficiency_metrics=efficiency_metrics,
                emergent_capability=emergent_capability,
                integration_score=integration_score,
                integration_classification=classification,
                recommendations=recommendations,
                evaluation_timestamp=datetime.utcnow(),
                confidence_level=confidence
            )
            
            logger.info(f"Integration evaluation completed. Score: {integration_score:.3f}, "
                       f"Classification: {classification}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in integration evaluation: {e}")
            return self._create_error_result()
    
    def _classify_integration_level(self, score: float) -> str:
        """Classify integration level based on score"""
        if score >= 0.9:
            return "exceptional"
        elif score >= 0.8:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.6:
            return "adequate"
        elif score >= 0.4:
            return "needs_improvement"
        else:
            return "poor"
    
    async def _generate_integration_recommendations(self, 
                                                  coherence: CoherenceMetrics,
                                                  efficiency: EfficiencyMetrics,
                                                  emergence: EmergentCapabilityIndex) -> List[str]:
        """Generate recommendations for integration improvement"""
        recommendations = []
        
        # Coherence recommendations
        if coherence.overall_coherence < 0.7:
            recommendations.append("Improve component alignment and communication protocols")
        
        if coherence.behavioral_consistency < 0.6:
            recommendations.append("Standardize behavioral patterns across components")
        
        # Efficiency recommendations
        if efficiency.overall_efficiency < 0.7:
            recommendations.append("Optimize resource utilization and throughput")
        
        for bottleneck in efficiency.bottleneck_analysis:
            recommendations.append(f"Address bottleneck: {bottleneck}")
        
        # Emergence recommendations
        if emergence.eci_score < 0.1:
            recommendations.append("Foster conditions for emergent capabilities")
        
        if emergence.prediction_accuracy < 0.6:
            recommendations.append("Improve predictability of component interactions")
        
        return recommendations if recommendations else ["System shows good integration"]
    
    def _calculate_confidence(self, 
                            coherence: CoherenceMetrics,
                            efficiency: EfficiencyMetrics,
                            emergence: EmergentCapabilityIndex) -> float:
        """Calculate confidence level for evaluation"""
        # Base confidence on consistency of metrics
        coherence_variance = np.var([
            coherence.component_alignment.get(k, 0.5) 
            for k in coherence.component_alignment.keys()
        ]) if coherence.component_alignment else 0.1
        
        efficiency_metrics = [
            efficiency.resource_utilization,
            efficiency.throughput_efficiency,
            efficiency.latency_efficiency,
            efficiency.cost_efficiency
        ]
        efficiency_variance = np.var(efficiency_metrics)
        
        # Lower variance indicates higher confidence
        avg_variance = (coherence_variance + efficiency_variance) / 2.0
        confidence = max(0.5, 1.0 - avg_variance)
        
        return confidence
    
    def _create_error_result(self) -> IntegrationEvaluationResult:
        """Create error result for exception cases"""
        return IntegrationEvaluationResult(
            coherence_metrics=CoherenceMetrics(0.0, {}, 0.0, 0.0, 0.0, {}),
            efficiency_metrics=EfficiencyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, [], 0.0),
            emergent_capability=EmergentCapabilityIndex(
                0.0, EmergenceType.PERFORMANCE_GAIN, 0.0, {}, {}, [], 0.0
            ),
            integration_score=0.0,
            integration_classification="error",
            recommendations=["System evaluation error"],
            evaluation_timestamp=datetime.utcnow(),
            confidence_level=0.0
        )

