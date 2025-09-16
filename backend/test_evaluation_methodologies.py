

"""
Test Script for Evaluation Methodologies
========================================

Comprehensive testing of the evaluation methodologies module including:
- System quality evaluation
- Component assessment
- Integration analysis  
- Benchmark validation
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import evaluation components
from evaluation.evaluation_engine import EvaluationEngine, EvaluationDimension
from evaluation.component_assessment import ComponentAssessmentFramework, ComponentType
from evaluation.integration_evaluator import SystemIntegrationEvaluator
from evaluation.benchmark_design import BenchmarkDesignFramework
from evaluation.evaluation_service import EvaluationService, EvaluationRequest, EvaluationType, EvaluationScope

class MockTask:
    """Mock task object for testing"""
    def __init__(self, task_id: int, description: str, status: str = "completed"):
        self.id = task_id
        self.description = description
        self.status = status
        self.created_at = datetime.utcnow() - timedelta(hours=1)
        self.updated_at = datetime.utcnow()

async def test_evaluation_engine():
    """Test the core evaluation engine"""
    print("\n=== Testing Evaluation Engine ===")
    
    # Create test data
    tasks = [
        MockTask(1, "Create REST API with authentication", "completed"),
        MockTask(2, "Implement user management system", "completed"), 
        MockTask(3, "Add real-time notifications", "in_progress"),
        MockTask(4, "Deploy to production environment", "failed"),
        MockTask(5, "Optimize database performance", "completed")
    ]
    
    # Initialize evaluation engine
    engine = EvaluationEngine()
    
    # Perform evaluation
    result = await engine.evaluate_system(db=None, tasks=tasks)
    
    print(f"Overall Quality: {result.overall_quality:.3f}")
    print("Dimension Scores:")
    for dimension, score in result.dimension_scores.items():
        print(f"  {dimension}: {score:.3f}")
    
    print("\nEmergence Indicators:")
    for indicator, value in result.emergence_indicators.items():
        print(f"  {indicator}: {value:.3f}")
    
    print(f"Confidence Interval: ({result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f})")
    
    return result.overall_quality > 0.5  # Pass if quality > 0.5

async def test_component_assessment():
    """Test the component assessment framework"""
    print("\n=== Testing Component Assessment ===")
    
    # Initialize framework
    framework = ComponentAssessmentFramework()
    
    # Test data
    task_data = {
        "description": "Process complex multi-step workflow with error handling",
        "complexity": "high",
        "requirements": ["authentication", "validation", "logging"]
    }
    
    environment_data = {
        "cpu_usage": 0.65,
        "memory_usage": 0.45,
        "system_load": 0.3,
        "network_quality": 0.85
    }
    
    operation_history = [
        {"success": True, "duration": 1.2, "timestamp": datetime.utcnow()},
        {"success": True, "duration": 0.8, "timestamp": datetime.utcnow()},
        {"success": False, "duration": 2.1, "timestamp": datetime.utcnow()},
        {"success": True, "duration": 1.5, "timestamp": datetime.utcnow()}
    ]
    
    # Perform assessment
    metrics = await framework.assess_component(
        component_id="orchestrator_001",
        component_type=ComponentType.ORCHESTRATOR,
        task_data=task_data,
        environment_data=environment_data,
        operation_history=operation_history
    )
    
    print(f"Component: {metrics.component_id} ({metrics.component_type.value})")
    print(f"Performance Score: {metrics.performance_score:.3f}")
    print(f"Reliability Score: {metrics.reliability_score:.3f}")  
    print(f"Readiness Score: {metrics.readiness_score:.3f}")
    print(f"Capability Rating: {metrics.capability_rating:.3f}")
    
    return metrics.performance_score > 0.5 and metrics.reliability_score > 0.5

async def test_integration_evaluator():
    """Test the system integration evaluator"""
    print("\n=== Testing Integration Evaluator ===")
    
    # Initialize evaluator
    evaluator = SystemIntegrationEvaluator()
    
    # Test data
    system_components = [
        {
            "id": "orchestrator",
            "performance": 0.85,
            "behavior_score": 0.78,
            "state": {"value": 0.82},
            "timestamp": datetime.utcnow()
        },
        {
            "id": "agent_manager", 
            "performance": 0.79,
            "behavior_score": 0.81,
            "state": {"value": 0.85},
            "timestamp": datetime.utcnow()
        },
        {
            "id": "context_engine",
            "performance": 0.88,
            "behavior_score": 0.83,
            "state": {"value": 0.80},
            "timestamp": datetime.utcnow()
        }
    ]
    
    interaction_data = [
        {"type": "api_call", "success": True, "response_time": 0.12},
        {"type": "data_sync", "success": True, "response_time": 0.08},
        {"type": "workflow_exec", "success": False, "response_time": 1.2},
        {"type": "api_call", "success": True, "response_time": 0.15}
    ]
    
    system_metrics = {
        "capability": 0.82,
        "average_latency": 0.5,
        "throughput": 150,
        "operational_cost": 0.8,
        "cpu_usage": 0.6,
        "memory_usage": 0.5
    }
    
    expected_behaviors = {
        "orchestrator_performance": 0.8,
        "agent_manager_performance": 0.75,
        "context_engine_performance": 0.85,
        "system_success_rate": 0.85
    }
    
    theoretical_benchmarks = {
        "optimal_cpu_usage": 0.7,
        "optimal_memory_usage": 0.6,
        "max_throughput": 200,
        "min_latency": 0.1
    }
    
    resource_data = {
        "cpu_usage": 0.6,
        "memory_usage": 0.5
    }
    
    # Perform integration evaluation
    result = await evaluator.evaluate_integration(
        system_components=system_components,
        interaction_data=interaction_data,
        system_metrics=system_metrics,
        expected_behaviors=expected_behaviors,
        theoretical_benchmarks=theoretical_benchmarks,
        resource_data=resource_data
    )
    
    print(f"Integration Score: {result.integration_score:.3f}")
    print(f"Classification: {result.integration_classification}")
    
    print("\nCoherence Metrics:")
    print(f"  Overall Coherence: {result.coherence_metrics.overall_coherence:.3f}")
    print(f"  Communication Coherence: {result.coherence_metrics.communication_coherence:.3f}")
    
    print("\nEfficiency Metrics:")
    print(f"  Overall Efficiency: {result.efficiency_metrics.overall_efficiency:.3f}")
    print(f"  Resource Utilization: {result.efficiency_metrics.resource_utilization:.3f}")
    
    print("\nEmergent Capability:")
    print(f"  ECI Score: {result.emergent_capability.eci_score:.3f}")
    print(f"  Emergence Type: {result.emergent_capability.emergence_type.value}")
    
    return result.integration_score > 0.5

async def test_benchmark_design():
    """Test the benchmark design framework"""
    print("\n=== Testing Benchmark Design Framework ===")
    
    # Initialize framework
    framework = BenchmarkDesignFramework()
    
    # Test data
    benchmark_data = {
        "components": ["performance_test", "reliability_test", "scalability_test"],
        "scores": [0.85, 0.78, 0.92, 0.76, 0.89, 0.83, 0.91, 0.77],
        "predictions": [0.8, 0.75, 0.9, 0.8, 0.85, 0.8, 0.9, 0.75],
        "high_performance_scores": [0.92, 0.89, 0.91, 0.88, 0.94],
        "low_performance_scores": [0.45, 0.52, 0.48, 0.41, 0.49],
        "all_scores": [0.85, 0.78, 0.92, 0.76, 0.89, 0.83, 0.91, 0.77, 0.45, 0.52],
        "type": "performance"
    }
    
    theoretical_framework = {
        "required_capabilities": ["speed", "accuracy", "reliability", "scalability"],
        "predicted_scores": [0.8, 0.75, 0.9, 0.8, 0.85, 0.8, 0.9, 0.75]
    }
    
    validation_data = {
        "actual_outcomes": [True, True, True, True, False, True, True, False]
    }
    
    # Perform benchmark evaluation
    result = await framework.evaluate_benchmark_quality(
        benchmark_data=benchmark_data,
        theoretical_framework=theoretical_framework,
        validation_data=validation_data
    )
    
    print(f"Overall Benchmark Quality: {result.overall_benchmark_quality:.3f}")
    print(f"Classification: {result.benchmark_classification}")
    
    print("\nValidity Results:")
    print(f"  Overall Validity: {result.validity_result.overall_validity:.3f}")
    print(f"  Content Validity: {result.validity_result.content_validity:.3f}")
    print(f"  Construct Validity: {result.validity_result.construct_validity:.3f}")
    
    print("\nReliability Results:")
    print(f"  Overall Reliability: {result.reliability_result.overall_reliability:.3f}")
    print(f"  Internal Consistency: {result.reliability_result.internal_consistency:.3f}")
    
    print("\nDiscriminatory Power:")
    print(f"  Discriminatory Power: {result.discriminatory_power_result.discriminatory_power:.3f}")
    print(f"  Effect Size: {result.discriminatory_power_result.effect_size:.3f}")
    
    return result.overall_benchmark_quality > 0.5

async def test_evaluation_service():
    """Test the comprehensive evaluation service"""
    print("\n=== Testing Evaluation Service ===")
    
    # Initialize service
    service = EvaluationService()
    
    # Test system quality evaluation
    print("\n--- System Quality Evaluation ---")
    system_quality_request = EvaluationRequest(
        evaluation_type=EvaluationType.SYSTEM_QUALITY,
        scope=EvaluationScope.SYSTEM,
        target_id="test_system_001",
        parameters={
            "tasks": [
                {"id": 1, "description": "API Development", "status": "completed"},
                {"id": 2, "description": "Database Design", "status": "completed"},
                {"id": 3, "description": "Frontend Integration", "status": "in_progress"}
            ]
        },
        user_id=1
    )
    
    result = await service.evaluate(system_quality_request)
    print(f"System Quality Score: {result.overall_score:.3f}")
    print(f"Success: {result.success}")
    
    # Test component assessment
    print("\n--- Component Assessment ---")
    component_request = EvaluationRequest(
        evaluation_type=EvaluationType.COMPONENT_ASSESSMENT,
        scope=EvaluationScope.COMPONENT,
        target_id="orchestrator_001",
        parameters={
            "component_type": "orchestrator",
            "task_data": {"description": "Complex workflow processing"},
            "environment_data": {"cpu_usage": 0.6, "memory_usage": 0.4},
            "operation_history": [{"success": True, "duration": 1.2}]
        }
    )
    
    result = await service.evaluate(component_request)
    print(f"Component Assessment Score: {result.overall_score:.3f}")
    print(f"Success: {result.success}")
    
    # Get performance metrics
    print("\n--- Service Performance Metrics ---")
    metrics = service.get_performance_metrics()
    print(f"Total Evaluations: {metrics['total_evaluations']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Average Execution Time: {metrics['average_execution_time']:.3f}s")
    
    return True

async def run_comprehensive_tests():
    """Run all evaluation methodology tests"""
    print("🔬 Starting Comprehensive Evaluation Methodologies Testing")
    print("=" * 60)
    
    test_results = []
    
    try:
        # Test evaluation engine
        result1 = await test_evaluation_engine()
        test_results.append(("Evaluation Engine", result1))
        
        # Test component assessment
        result2 = await test_component_assessment()
        test_results.append(("Component Assessment", result2))
        
        # Test integration evaluator
        result3 = await test_integration_evaluator()
        test_results.append(("Integration Evaluator", result3))
        
        # Test benchmark design
        result4 = await test_benchmark_design()
        test_results.append(("Benchmark Design", result4))
        
        # Test evaluation service
        result5 = await test_evaluation_service()
        test_results.append(("Evaluation Service", result5))
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        test_results.append(("Error", False))
    
    # Print test summary
    print("\n" + "=" * 60)
    print("🎯 Test Results Summary:")
    print("-" * 30)
    
    passed = 0
    total = len(test_results)
    
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if success:
            passed += 1
    
    print("-" * 30)
    print(f"Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All evaluation methodology tests PASSED!")
        return True
    else:
        print("⚠️  Some tests FAILED. Check implementation.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_comprehensive_tests())
    exit(0 if success else 1)

