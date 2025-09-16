

"""
Evaluation Methodologies Module
Advanced evaluation framework for Automatos AI system assessment
"""

from .evaluation_engine import (
    EvaluationEngine,
    SystemQualityResult,
    EvaluationMetrics,
    EmergenceDetector
)

from .component_assessment import (
    ComponentAssessmentFramework,
    ComponentMetrics,
    InteractionReadinessScore,
    PerformanceCalculator
)

from .integration_evaluator import (
    SystemIntegrationEvaluator,
    CoherenceMetrics,
    EfficiencyMetrics,
    EmergentCapabilityIndex
)

from .benchmark_design import (
    BenchmarkDesignFramework,
    ValidityCalculator,
    ReliabilityCalculator,
    DiscriminatoryPowerAnalyzer
)

from .evaluation_service import (
    EvaluationService,
    EvaluationResult,
    BenchmarkResult,
    AssessmentReport
)

__all__ = [
    # Core Engine
    'EvaluationEngine',
    'SystemQualityResult', 
    'EvaluationMetrics',
    'EmergenceDetector',
    
    # Component Assessment
    'ComponentAssessmentFramework',
    'ComponentMetrics',
    'InteractionReadinessScore',
    'PerformanceCalculator',
    
    # Integration Evaluation
    'SystemIntegrationEvaluator',
    'CoherenceMetrics',
    'EfficiencyMetrics', 
    'EmergentCapabilityIndex',
    
    # Benchmark Design
    'BenchmarkDesignFramework',
    'ValidityCalculator',
    'ReliabilityCalculator',
    'DiscriminatoryPowerAnalyzer',
    
    # Service Layer
    'EvaluationService',
    'EvaluationResult',
    'BenchmarkResult',
    'AssessmentReport'
]

__version__ = "1.0.0"
__description__ = "Comprehensive evaluation methodologies for AI orchestrator systems"

