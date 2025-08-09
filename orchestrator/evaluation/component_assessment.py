

"""
Component Assessment Framework - Performance, Reliability & Readiness Analysis
==============================================================================

Implements comprehensive component assessment with:
- Performance calculation using P(c, i, e) = f(Capability(c), Input(i), Environment(e))
- Reliability analysis using R(t) = e^(-λt)  
- Interaction Readiness Score using IRS(c) = Σ wᵢ × Scoreᵢ(c)
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of system components"""
    ORCHESTRATOR = "orchestrator"
    AGENT = "agent"
    WORKFLOW = "workflow"
    CONTEXT_MANAGER = "context_manager"
    MEMORY_SYSTEM = "memory_system"
    REASONING_ENGINE = "reasoning_engine"

class ReadinessAspect(Enum):
    """Aspects of interaction readiness"""
    INTERFACE = "interface"
    ERROR_HANDLING = "error_handling"
    STATE_MANAGEMENT = "state_management"
    COMMUNICATION = "communication"
    ADAPTABILITY = "adaptability"

@dataclass
class ComponentMetrics:
    """Metrics for component assessment"""
    component_id: str
    component_type: ComponentType
    performance_score: float
    reliability_score: float
    readiness_score: float
    capability_rating: float
    complexity_index: float
    environment_factor: float
    assessment_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['component_type'] = self.component_type.value
        result['assessment_timestamp'] = self.assessment_timestamp.isoformat()
        return result

@dataclass
class InteractionReadinessScore:
    """Detailed interaction readiness analysis"""
    overall_score: float
    aspect_scores: Dict[str, float]
    confidence_level: float
    readiness_classification: str
    recommendations: List[str]

@dataclass
class PerformanceBreakdown:
    """Detailed performance analysis"""
    capability_score: float
    input_complexity_score: float
    environment_score: float
    combined_performance: float
    bottleneck_factors: List[str]

class PerformanceCalculator:
    """
    Performance calculation engine
    
    Implements: P(c, i, e) = f(Capability(c), Input(i), Environment(e))
    """
    
    def __init__(self):
        self.capability_weights = {
            ComponentType.ORCHESTRATOR: {"processing": 0.4, "coordination": 0.3, "scalability": 0.3},
            ComponentType.AGENT: {"task_execution": 0.5, "communication": 0.3, "learning": 0.2},
            ComponentType.WORKFLOW: {"efficiency": 0.4, "flexibility": 0.3, "robustness": 0.3},
            ComponentType.CONTEXT_MANAGER: {"retrieval": 0.4, "relevance": 0.3, "speed": 0.3},
            ComponentType.MEMORY_SYSTEM: {"storage": 0.3, "retrieval": 0.3, "organization": 0.4},
            ComponentType.REASONING_ENGINE: {"logic": 0.4, "inference": 0.3, "adaptability": 0.3}
        }
    
    async def calculate_performance(self, 
                                  component_type: ComponentType,
                                  task_data: Dict[str, Any],
                                  environment_data: Dict[str, Any]) -> PerformanceBreakdown:
        """
        Calculate component performance using the formula:
        P(c, i, e) = f(Capability(c), Input(i), Environment(e))
        """
        try:
            # Calculate capability score
            capability_score = await self._assess_capability(component_type, task_data)
            
            # Calculate input complexity score  
            input_complexity = await self._assess_input_complexity(task_data)
            
            # Calculate environment score
            environment_score = await self._assess_environment(environment_data)
            
            # Combine scores using weighted formula
            combined_performance = (
                capability_score * 0.5 + 
                input_complexity * 0.3 + 
                environment_score * 0.2
            )
            
            # Identify bottlenecks
            bottlenecks = []
            if capability_score < 0.6:
                bottlenecks.append("Low capability rating")
            if input_complexity > 0.8:
                bottlenecks.append("High input complexity")
            if environment_score < 0.5:
                bottlenecks.append("Suboptimal environment")
            
            return PerformanceBreakdown(
                capability_score=capability_score,
                input_complexity_score=input_complexity,
                environment_score=environment_score,
                combined_performance=combined_performance,
                bottleneck_factors=bottlenecks
            )
            
        except Exception as e:
            logger.error(f"Error calculating performance: {e}")
            return PerformanceBreakdown(0.0, 0.0, 0.0, 0.0, ["Calculation error"])
    
    async def _assess_capability(self, 
                               component_type: ComponentType, 
                               task_data: Dict[str, Any]) -> float:
        """Assess component capability based on type and task requirements"""
        base_capabilities = {
            ComponentType.ORCHESTRATOR: 0.85,
            ComponentType.AGENT: 0.80,
            ComponentType.WORKFLOW: 0.75,
            ComponentType.CONTEXT_MANAGER: 0.82,
            ComponentType.MEMORY_SYSTEM: 0.78,
            ComponentType.REASONING_ENGINE: 0.88
        }
        
        base_score = base_capabilities.get(component_type, 0.7)
        
        # Adjust based on task complexity
        task_complexity = len(str(task_data.get('description', ''))) / 1000.0
        complexity_factor = max(0.5, 1.0 - (task_complexity * 0.2))
        
        return min(1.0, base_score * complexity_factor)
    
    async def _assess_input_complexity(self, task_data: Dict[str, Any]) -> float:
        """Assess input complexity factor"""
        description = str(task_data.get('description', ''))
        
        # Length factor
        length_factor = min(1.0, len(description) / 500.0)
        
        # Keyword complexity
        complex_keywords = ['integrate', 'optimize', 'analyze', 'coordinate', 'synthesize']
        keyword_count = sum(1 for keyword in complex_keywords if keyword in description.lower())
        keyword_factor = min(1.0, keyword_count * 0.2)
        
        # Data structure complexity
        data_complexity = 0.3 if isinstance(task_data, dict) and len(task_data) > 5 else 0.1
        
        return min(1.0, length_factor * 0.5 + keyword_factor * 0.3 + data_complexity * 0.2)
    
    async def _assess_environment(self, environment_data: Dict[str, Any]) -> float:
        """Assess environment favorability"""
        # Resource availability
        cpu_usage = environment_data.get('cpu_usage', 0.5)
        memory_usage = environment_data.get('memory_usage', 0.5)
        resource_score = 1.0 - ((cpu_usage + memory_usage) / 2.0)
        
        # System load
        load_factor = 1.0 - environment_data.get('system_load', 0.3)
        
        # Network conditions
        network_quality = environment_data.get('network_quality', 0.8)
        
        return (resource_score * 0.4 + load_factor * 0.3 + network_quality * 0.3)

class ReliabilityAnalyzer:
    """
    Reliability analysis using exponential decay model
    
    Implements: R(t) = e^(-λt)
    """
    
    def __init__(self, decay_constant: float = 0.1):
        self.decay_constant = decay_constant
        self.component_factors = {
            ComponentType.ORCHESTRATOR: 0.95,
            ComponentType.AGENT: 0.90,
            ComponentType.WORKFLOW: 0.85,
            ComponentType.CONTEXT_MANAGER: 0.88,
            ComponentType.MEMORY_SYSTEM: 0.92,
            ComponentType.REASONING_ENGINE: 0.87
        }
    
    async def calculate_reliability(self, 
                                  component_type: ComponentType,
                                  creation_time: datetime,
                                  operation_history: List[Dict[str, Any]]) -> float:
        """
        Calculate reliability using R(t) = e^(-λt)
        """
        try:
            # Time-based reliability decay
            age_hours = (datetime.utcnow() - creation_time).total_seconds() / 3600.0
            time_reliability = np.exp(-self.decay_constant * age_hours)
            
            # Component type factor
            type_factor = self.component_factors.get(component_type, 0.8)
            
            # Operation history factor
            history_factor = await self._analyze_operation_history(operation_history)
            
            # Combined reliability
            combined_reliability = time_reliability * type_factor * history_factor
            
            return min(1.0, max(0.0, combined_reliability))
            
        except Exception as e:
            logger.error(f"Error calculating reliability: {e}")
            return 0.5
    
    async def _analyze_operation_history(self, history: List[Dict[str, Any]]) -> float:
        """Analyze operation history for reliability factors"""
        if not history:
            return 0.8  # Default for no history
        
        success_count = sum(1 for op in history if op.get('success', False))
        total_operations = len(history)
        
        success_rate = success_count / total_operations if total_operations > 0 else 0.8
        
        # Recent performance weighting
        recent_operations = [op for op in history[-10:]]  # Last 10 operations
        if recent_operations:
            recent_success = sum(1 for op in recent_operations if op.get('success', False))
            recent_rate = recent_success / len(recent_operations)
            # Weight recent performance more heavily
            weighted_reliability = (success_rate * 0.3 + recent_rate * 0.7)
        else:
            weighted_reliability = success_rate
        
        return weighted_reliability

class InteractionReadinessAnalyzer:
    """
    Interaction Readiness Score calculation
    
    Implements: IRS(c) = Σ wᵢ × Scoreᵢ(c)
    """
    
    def __init__(self):
        self.aspect_weights = {
            ReadinessAspect.INTERFACE: 0.25,
            ReadinessAspect.ERROR_HANDLING: 0.20,
            ReadinessAspect.STATE_MANAGEMENT: 0.20,
            ReadinessAspect.COMMUNICATION: 0.20,
            ReadinessAspect.ADAPTABILITY: 0.15
        }
        
        self.readiness_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "adequate": 0.60,
            "needs_improvement": 0.45,
            "poor": 0.0
        }
    
    async def calculate_readiness(self, 
                                component_type: ComponentType,
                                component_data: Dict[str, Any]) -> InteractionReadinessScore:
        """
        Calculate Interaction Readiness Score
        """
        try:
            aspect_scores = {}
            
            # Evaluate each readiness aspect
            aspect_scores[ReadinessAspect.INTERFACE.value] = await self._evaluate_interface(
                component_type, component_data
            )
            
            aspect_scores[ReadinessAspect.ERROR_HANDLING.value] = await self._evaluate_error_handling(
                component_data
            )
            
            aspect_scores[ReadinessAspect.STATE_MANAGEMENT.value] = await self._evaluate_state_management(
                component_data
            )
            
            aspect_scores[ReadinessAspect.COMMUNICATION.value] = await self._evaluate_communication(
                component_data
            )
            
            aspect_scores[ReadinessAspect.ADAPTABILITY.value] = await self._evaluate_adaptability(
                component_data
            )
            
            # Calculate overall readiness score
            overall_score = sum(
                self.aspect_weights[ReadinessAspect(aspect)] * score
                for aspect, score in aspect_scores.items()
            )
            
            # Classify readiness level
            classification = self._classify_readiness(overall_score)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(aspect_scores)
            
            # Calculate confidence
            score_variance = np.var(list(aspect_scores.values()))
            confidence = max(0.5, 1.0 - score_variance)
            
            return InteractionReadinessScore(
                overall_score=overall_score,
                aspect_scores=aspect_scores,
                confidence_level=confidence,
                readiness_classification=classification,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error calculating readiness: {e}")
            return InteractionReadinessScore(
                overall_score=0.0,
                aspect_scores={},
                confidence_level=0.0,
                readiness_classification="error",
                recommendations=["System evaluation error"]
            )
    
    async def _evaluate_interface(self, component_type: ComponentType, data: Dict[str, Any]) -> float:
        """Evaluate interface quality"""
        # API completeness
        api_endpoints = data.get('api_endpoints', [])
        api_completeness = min(1.0, len(api_endpoints) / 10.0)
        
        # Documentation quality
        docs_quality = 0.8 if data.get('has_documentation', False) else 0.4
        
        # Interface consistency
        consistency_score = 0.85  # Placeholder - would analyze actual interface patterns
        
        return (api_completeness * 0.4 + docs_quality * 0.3 + consistency_score * 0.3)
    
    async def _evaluate_error_handling(self, data: Dict[str, Any]) -> float:
        """Evaluate error handling capabilities"""
        error_history = data.get('error_history', [])
        
        if not error_history:
            return 0.7  # Default for no error data
        
        # Error recovery rate
        recovered_errors = sum(1 for err in error_history if err.get('recovered', False))
        total_errors = len(error_history)
        recovery_rate = recovered_errors / total_errors if total_errors > 0 else 0.7
        
        # Error severity handling
        critical_errors = sum(1 for err in error_history if err.get('severity') == 'critical')
        critical_handling = 1.0 - (critical_errors / max(1, total_errors))
        
        return (recovery_rate * 0.6 + critical_handling * 0.4)
    
    async def _evaluate_state_management(self, data: Dict[str, Any]) -> float:
        """Evaluate state management quality"""
        # State persistence
        has_persistence = data.get('has_state_persistence', False)
        persistence_score = 0.9 if has_persistence else 0.5
        
        # State consistency
        consistency_score = 0.8  # Placeholder - would analyze actual state handling
        
        # State recovery
        recovery_score = 0.85  # Placeholder - would analyze recovery capabilities
        
        return (persistence_score * 0.4 + consistency_score * 0.3 + recovery_score * 0.3)
    
    async def _evaluate_communication(self, data: Dict[str, Any]) -> float:
        """Evaluate communication capabilities"""
        # Protocol support
        protocols = data.get('supported_protocols', [])
        protocol_score = min(1.0, len(protocols) / 5.0)
        
        # Message reliability
        message_success_rate = data.get('message_success_rate', 0.8)
        
        # Real-time capabilities
        realtime_support = 0.9 if data.get('supports_realtime', False) else 0.5
        
        return (protocol_score * 0.3 + message_success_rate * 0.4 + realtime_support * 0.3)
    
    async def _evaluate_adaptability(self, data: Dict[str, Any]) -> float:
        """Evaluate adaptability capabilities"""
        # Configuration flexibility
        config_options = data.get('configuration_options', [])
        flexibility_score = min(1.0, len(config_options) / 10.0)
        
        # Learning capability
        learning_score = 0.8 if data.get('supports_learning', False) else 0.4
        
        # Context awareness
        context_awareness = 0.85 if data.get('context_aware', False) else 0.5
        
        return (flexibility_score * 0.3 + learning_score * 0.4 + context_awareness * 0.3)
    
    def _classify_readiness(self, score: float) -> str:
        """Classify readiness level based on score"""
        for level, threshold in self.readiness_thresholds.items():
            if score >= threshold:
                return level
        return "poor"
    
    async def _generate_recommendations(self, aspect_scores: Dict[str, float]) -> List[str]:
        """Generate improvement recommendations based on scores"""
        recommendations = []
        
        for aspect, score in aspect_scores.items():
            if score < 0.6:
                if aspect == ReadinessAspect.INTERFACE.value:
                    recommendations.append("Improve API documentation and interface consistency")
                elif aspect == ReadinessAspect.ERROR_HANDLING.value:
                    recommendations.append("Enhance error recovery mechanisms and logging")
                elif aspect == ReadinessAspect.STATE_MANAGEMENT.value:
                    recommendations.append("Implement better state persistence and recovery")
                elif aspect == ReadinessAspect.COMMUNICATION.value:
                    recommendations.append("Improve message reliability and protocol support")
                elif aspect == ReadinessAspect.ADAPTABILITY.value:
                    recommendations.append("Add more configuration options and learning capabilities")
        
        if not recommendations:
            recommendations.append("System shows good readiness across all aspects")
        
        return recommendations

class ComponentAssessmentFramework:
    """
    Main framework for comprehensive component assessment
    """
    
    def __init__(self):
        self.performance_calculator = PerformanceCalculator()
        self.reliability_analyzer = ReliabilityAnalyzer()
        self.readiness_analyzer = InteractionReadinessAnalyzer()
    
    async def assess_component(self, 
                             component_id: str,
                             component_type: ComponentType,
                             task_data: Dict[str, Any],
                             environment_data: Dict[str, Any],
                             creation_time: Optional[datetime] = None,
                             operation_history: Optional[List[Dict[str, Any]]] = None) -> ComponentMetrics:
        """
        Comprehensive component assessment
        """
        try:
            logger.info(f"Assessing component {component_id} of type {component_type.value}")
            
            creation_time = creation_time or datetime.utcnow()
            operation_history = operation_history or []
            
            # Performance assessment
            performance_breakdown = await self.performance_calculator.calculate_performance(
                component_type, task_data, environment_data
            )
            
            # Reliability assessment
            reliability_score = await self.reliability_analyzer.calculate_reliability(
                component_type, creation_time, operation_history
            )
            
            # Readiness assessment
            component_data = {
                **task_data,
                **environment_data,
                'operation_history': operation_history
            }
            
            readiness_result = await self.readiness_analyzer.calculate_readiness(
                component_type, component_data
            )
            
            # Create comprehensive metrics
            metrics = ComponentMetrics(
                component_id=component_id,
                component_type=component_type,
                performance_score=performance_breakdown.combined_performance,
                reliability_score=reliability_score,
                readiness_score=readiness_result.overall_score,
                capability_rating=performance_breakdown.capability_score,
                complexity_index=performance_breakdown.input_complexity_score,
                environment_factor=performance_breakdown.environment_score,
                assessment_timestamp=datetime.utcnow()
            )
            
            logger.info(f"Component assessment completed. Performance: {metrics.performance_score:.3f}, "
                       f"Reliability: {metrics.reliability_score:.3f}, Readiness: {metrics.readiness_score:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in component assessment: {e}")
            return ComponentMetrics(
                component_id=component_id,
                component_type=component_type,
                performance_score=0.0,
                reliability_score=0.0,
                readiness_score=0.0,
                capability_rating=0.0,
                complexity_index=0.0,
                environment_factor=0.0,
                assessment_timestamp=datetime.utcnow()
            )

