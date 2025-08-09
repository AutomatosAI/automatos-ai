

"""
Evaluation Service - Comprehensive Evaluation Integration Service
===============================================================

Provides unified service layer for all evaluation methodologies:
- System quality evaluation
- Component assessment 
- Integration analysis
- Benchmark design validation
"""

import asyncio
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from sqlalchemy.orm import Session

# Import evaluation components
from .evaluation_engine import (
    EvaluationEngine, 
    SystemQualityResult,
    EvaluationMetrics
)
from .component_assessment import (
    ComponentAssessmentFramework,
    ComponentMetrics,
    ComponentType
)
from .integration_evaluator import (
    SystemIntegrationEvaluator,
    IntegrationEvaluationResult
)
from .benchmark_design import (
    BenchmarkDesignFramework,
    BenchmarkQualityResult
)

logger = logging.getLogger(__name__)

class EvaluationType(Enum):
    """Types of evaluations"""
    SYSTEM_QUALITY = "system_quality"
    COMPONENT_ASSESSMENT = "component_assessment"
    INTEGRATION_ANALYSIS = "integration_analysis"
    BENCHMARK_VALIDATION = "benchmark_validation"
    COMPREHENSIVE = "comprehensive"

class EvaluationScope(Enum):
    """Scope of evaluation"""
    SINGLE_TASK = "single_task"
    COMPONENT = "component"
    SYSTEM = "system"
    ENTERPRISE = "enterprise"

@dataclass
class EvaluationRequest:
    """Request structure for evaluation"""
    evaluation_type: EvaluationType
    scope: EvaluationScope
    target_id: str
    parameters: Dict[str, Any]
    context_data: Optional[Dict[str, Any]] = None
    user_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['evaluation_type'] = self.evaluation_type.value
        result['scope'] = self.scope.value
        return result

@dataclass
class EvaluationResult:
    """Unified evaluation result"""
    evaluation_id: str
    evaluation_type: EvaluationType
    scope: EvaluationScope
    target_id: str
    system_quality: Optional[SystemQualityResult] = None
    component_metrics: Optional[ComponentMetrics] = None
    integration_analysis: Optional[IntegrationEvaluationResult] = None
    benchmark_quality: Optional[BenchmarkQualityResult] = None
    overall_score: float = 0.0
    evaluation_timestamp: datetime = None
    execution_time_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.evaluation_timestamp is None:
            self.evaluation_timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['evaluation_type'] = self.evaluation_type.value
        result['scope'] = self.scope.value
        result['evaluation_timestamp'] = self.evaluation_timestamp.isoformat()
        
        # Convert nested results to dict
        if self.system_quality:
            result['system_quality'] = self.system_quality.to_dict()
        if self.component_metrics:
            result['component_metrics'] = self.component_metrics.to_dict()
        if self.integration_analysis:
            result['integration_analysis'] = self.integration_analysis.to_dict()
        if self.benchmark_quality:
            result['benchmark_quality'] = self.benchmark_quality.to_dict()
        
        return result

@dataclass
class BenchmarkResult:
    """Benchmark evaluation result"""
    benchmark_id: str
    benchmark_name: str
    quality_assessment: BenchmarkQualityResult
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    certification_level: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['quality_assessment'] = self.quality_assessment.to_dict()
        return result

@dataclass
class AssessmentReport:
    """Comprehensive assessment report"""
    report_id: str
    report_title: str
    evaluation_summary: Dict[str, Any]
    detailed_results: List[EvaluationResult]
    trends_analysis: Dict[str, Any]
    recommendations: List[str]
    report_timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['report_timestamp'] = self.report_timestamp.isoformat()
        result['detailed_results'] = [r.to_dict() for r in self.detailed_results]
        return result

class EvaluationService:
    """
    Comprehensive evaluation service integrating all evaluation methodologies
    """
    
    def __init__(self):
        # Initialize evaluation components
        self.evaluation_engine = EvaluationEngine()
        self.component_framework = ComponentAssessmentFramework()
        self.integration_evaluator = SystemIntegrationEvaluator()
        self.benchmark_framework = BenchmarkDesignFramework()
        
        # Evaluation cache
        self.evaluation_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Performance tracking
        self.evaluation_history = []
        self.performance_metrics = {
            'total_evaluations': 0,
            'average_execution_time': 0.0,
            'success_rate': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("Evaluation service initialized with all methodologies")
    
    async def evaluate(self, 
                      request: EvaluationRequest,
                      db: Optional[Session] = None) -> EvaluationResult:
        """
        Main evaluation method - routes to appropriate evaluation methodology
        """
        start_time = datetime.utcnow()
        evaluation_id = f"eval_{int(start_time.timestamp())}_{request.target_id}"
        
        try:
            logger.info(f"Starting evaluation {evaluation_id} - Type: {request.evaluation_type.value}")
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info(f"Returning cached result for {evaluation_id}")
                self._update_performance_metrics(True, 0.0, True)
                return cached_result
            
            # Route to appropriate evaluation method
            result = None
            
            if request.evaluation_type == EvaluationType.SYSTEM_QUALITY:
                result = await self._evaluate_system_quality(evaluation_id, request, db)
                
            elif request.evaluation_type == EvaluationType.COMPONENT_ASSESSMENT:
                result = await self._evaluate_component(evaluation_id, request)
                
            elif request.evaluation_type == EvaluationType.INTEGRATION_ANALYSIS:
                result = await self._evaluate_integration(evaluation_id, request)
                
            elif request.evaluation_type == EvaluationType.BENCHMARK_VALIDATION:
                result = await self._evaluate_benchmark(evaluation_id, request)
                
            elif request.evaluation_type == EvaluationType.COMPREHENSIVE:
                result = await self._evaluate_comprehensive(evaluation_id, request, db)
                
            else:
                raise ValueError(f"Unknown evaluation type: {request.evaluation_type}")
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result.execution_time_seconds = execution_time
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update performance metrics
            self._update_performance_metrics(True, execution_time, False)
            
            # Store in history
            self.evaluation_history.append(result)
            
            logger.info(f"Evaluation {evaluation_id} completed successfully in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            error_message = f"Error in evaluation {evaluation_id}: {str(e)}"
            logger.error(error_message)
            
            # Create error result
            result = EvaluationResult(
                evaluation_id=evaluation_id,
                evaluation_type=request.evaluation_type,
                scope=request.scope,
                target_id=request.target_id,
                overall_score=0.0,
                execution_time_seconds=execution_time,
                success=False,
                error_message=error_message
            )
            
            # Update performance metrics
            self._update_performance_metrics(False, execution_time, False)
            
            return result
    
    async def _evaluate_system_quality(self, 
                                     evaluation_id: str,
                                     request: EvaluationRequest,
                                     db: Optional[Session]) -> EvaluationResult:
        """Evaluate overall system quality"""
        logger.info(f"Performing system quality evaluation for {evaluation_id}")
        
        # Extract parameters
        tasks = request.parameters.get('tasks', [])
        context_data = request.context_data or {}
        
        # Perform evaluation
        system_quality = await self.evaluation_engine.evaluate_system(
            db, tasks, context_data
        )
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=request.evaluation_type,
            scope=request.scope,
            target_id=request.target_id,
            system_quality=system_quality,
            overall_score=system_quality.overall_quality
        )
    
    async def _evaluate_component(self, 
                                evaluation_id: str,
                                request: EvaluationRequest) -> EvaluationResult:
        """Evaluate individual component"""
        logger.info(f"Performing component assessment for {evaluation_id}")
        
        # Extract parameters
        component_type_str = request.parameters.get('component_type', 'orchestrator')
        component_type = ComponentType(component_type_str)
        
        task_data = request.parameters.get('task_data', {})
        environment_data = request.parameters.get('environment_data', {})
        creation_time = request.parameters.get('creation_time')
        operation_history = request.parameters.get('operation_history', [])
        
        # Perform assessment
        component_metrics = await self.component_framework.assess_component(
            component_id=request.target_id,
            component_type=component_type,
            task_data=task_data,
            environment_data=environment_data,
            creation_time=creation_time,
            operation_history=operation_history
        )
        
        # Calculate overall score
        overall_score = (
            component_metrics.performance_score * 0.4 +
            component_metrics.reliability_score * 0.3 +
            component_metrics.readiness_score * 0.3
        )
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=request.evaluation_type,
            scope=request.scope,
            target_id=request.target_id,
            component_metrics=component_metrics,
            overall_score=overall_score
        )
    
    async def _evaluate_integration(self, 
                                  evaluation_id: str,
                                  request: EvaluationRequest) -> EvaluationResult:
        """Evaluate system integration"""
        logger.info(f"Performing integration analysis for {evaluation_id}")
        
        # Extract parameters
        system_components = request.parameters.get('system_components', [])
        interaction_data = request.parameters.get('interaction_data', [])
        system_metrics = request.parameters.get('system_metrics', {})
        expected_behaviors = request.parameters.get('expected_behaviors', {})
        theoretical_benchmarks = request.parameters.get('theoretical_benchmarks', {})
        resource_data = request.parameters.get('resource_data', {})
        historical_data = request.parameters.get('historical_data')
        
        # Perform evaluation
        integration_analysis = await self.integration_evaluator.evaluate_integration(
            system_components=system_components,
            interaction_data=interaction_data,
            system_metrics=system_metrics,
            expected_behaviors=expected_behaviors,
            theoretical_benchmarks=theoretical_benchmarks,
            resource_data=resource_data,
            historical_data=historical_data
        )
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=request.evaluation_type,
            scope=request.scope,
            target_id=request.target_id,
            integration_analysis=integration_analysis,
            overall_score=integration_analysis.integration_score
        )
    
    async def _evaluate_benchmark(self, 
                                evaluation_id: str,
                                request: EvaluationRequest) -> EvaluationResult:
        """Evaluate benchmark quality"""
        logger.info(f"Performing benchmark validation for {evaluation_id}")
        
        # Extract parameters
        benchmark_data = request.parameters.get('benchmark_data', {})
        theoretical_framework = request.parameters.get('theoretical_framework', {})
        validation_data = request.parameters.get('validation_data')
        
        # Perform evaluation
        benchmark_quality = await self.benchmark_framework.evaluate_benchmark_quality(
            benchmark_data=benchmark_data,
            theoretical_framework=theoretical_framework,
            validation_data=validation_data
        )
        
        return EvaluationResult(
            evaluation_id=evaluation_id,
            evaluation_type=request.evaluation_type,
            scope=request.scope,
            target_id=request.target_id,
            benchmark_quality=benchmark_quality,
            overall_score=benchmark_quality.overall_benchmark_quality
        )
    
    async def _evaluate_comprehensive(self, 
                                    evaluation_id: str,
                                    request: EvaluationRequest,
                                    db: Optional[Session]) -> EvaluationResult:
        """Perform comprehensive evaluation across all methodologies"""
        logger.info(f"Performing comprehensive evaluation for {evaluation_id}")
        
        try:
            # Perform all evaluations in parallel
            tasks = []
            
            # System quality evaluation
            if 'tasks' in request.parameters:
                system_quality_request = EvaluationRequest(
                    evaluation_type=EvaluationType.SYSTEM_QUALITY,
                    scope=request.scope,
                    target_id=request.target_id,
                    parameters={'tasks': request.parameters['tasks']},
                    context_data=request.context_data,
                    user_id=request.user_id
                )
                tasks.append(self._evaluate_system_quality(
                    f"{evaluation_id}_system", system_quality_request, db
                ))
            
            # Component assessment
            if 'component_type' in request.parameters:
                component_request = EvaluationRequest(
                    evaluation_type=EvaluationType.COMPONENT_ASSESSMENT,
                    scope=request.scope,
                    target_id=request.target_id,
                    parameters=request.parameters,
                    context_data=request.context_data,
                    user_id=request.user_id
                )
                tasks.append(self._evaluate_component(
                    f"{evaluation_id}_component", component_request
                ))
            
            # Integration analysis
            if 'system_components' in request.parameters:
                integration_request = EvaluationRequest(
                    evaluation_type=EvaluationType.INTEGRATION_ANALYSIS,
                    scope=request.scope,
                    target_id=request.target_id,
                    parameters=request.parameters,
                    context_data=request.context_data,
                    user_id=request.user_id
                )
                tasks.append(self._evaluate_integration(
                    f"{evaluation_id}_integration", integration_request
                ))
            
            # Benchmark validation
            if 'benchmark_data' in request.parameters:
                benchmark_request = EvaluationRequest(
                    evaluation_type=EvaluationType.BENCHMARK_VALIDATION,
                    scope=request.scope,
                    target_id=request.target_id,
                    parameters=request.parameters,
                    context_data=request.context_data,
                    user_id=request.user_id
                )
                tasks.append(self._evaluate_benchmark(
                    f"{evaluation_id}_benchmark", benchmark_request
                ))
            
            # Execute all tasks
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Combine results
                combined_result = EvaluationResult(
                    evaluation_id=evaluation_id,
                    evaluation_type=request.evaluation_type,
                    scope=request.scope,
                    target_id=request.target_id
                )
                
                scores = []
                for result in results:
                    if isinstance(result, EvaluationResult) and result.success:
                        if result.system_quality:
                            combined_result.system_quality = result.system_quality
                            scores.append(result.overall_score)
                        if result.component_metrics:
                            combined_result.component_metrics = result.component_metrics
                            scores.append(result.overall_score)
                        if result.integration_analysis:
                            combined_result.integration_analysis = result.integration_analysis
                            scores.append(result.overall_score)
                        if result.benchmark_quality:
                            combined_result.benchmark_quality = result.benchmark_quality
                            scores.append(result.overall_score)
                
                # Calculate overall score
                combined_result.overall_score = sum(scores) / len(scores) if scores else 0.0
                
                return combined_result
            
            else:
                # No valid evaluation parameters found
                return EvaluationResult(
                    evaluation_id=evaluation_id,
                    evaluation_type=request.evaluation_type,
                    scope=request.scope,
                    target_id=request.target_id,
                    overall_score=0.0,
                    success=False,
                    error_message="No valid evaluation parameters provided for comprehensive evaluation"
                )
                
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            return EvaluationResult(
                evaluation_id=evaluation_id,
                evaluation_type=request.evaluation_type,
                scope=request.scope,
                target_id=request.target_id,
                overall_score=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def generate_assessment_report(self, 
                                       evaluation_results: List[EvaluationResult],
                                       report_title: str = "System Assessment Report") -> AssessmentReport:
        """Generate comprehensive assessment report"""
        logger.info(f"Generating assessment report with {len(evaluation_results)} evaluations")
        
        report_id = f"report_{int(datetime.utcnow().timestamp())}"
        
        # Calculate summary statistics
        successful_evaluations = [r for r in evaluation_results if r.success]
        total_evaluations = len(evaluation_results)
        success_rate = len(successful_evaluations) / total_evaluations if total_evaluations > 0 else 0.0
        
        scores = [r.overall_score for r in successful_evaluations]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        evaluation_summary = {
            'total_evaluations': total_evaluations,
            'successful_evaluations': len(successful_evaluations),
            'success_rate': success_rate,
            'average_score': avg_score,
            'score_distribution': {
                'min': min(scores) if scores else 0.0,
                'max': max(scores) if scores else 0.0,
                'std': float(np.std(scores)) if scores else 0.0
            },
            'evaluation_types': list(set(r.evaluation_type.value for r in evaluation_results))
        }
        
        # Trends analysis
        trends_analysis = await self._analyze_evaluation_trends(successful_evaluations)
        
        # Generate recommendations
        recommendations = await self._generate_report_recommendations(successful_evaluations)
        
        return AssessmentReport(
            report_id=report_id,
            report_title=report_title,
            evaluation_summary=evaluation_summary,
            detailed_results=evaluation_results,
            trends_analysis=trends_analysis,
            recommendations=recommendations,
            report_timestamp=datetime.utcnow()
        )
    
    async def _analyze_evaluation_trends(self, 
                                       evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze trends in evaluation results"""
        if not evaluation_results:
            return {}
        
        # Time-based trends
        results_by_time = sorted(evaluation_results, key=lambda r: r.evaluation_timestamp)
        if len(results_by_time) > 1:
            early_scores = [r.overall_score for r in results_by_time[:len(results_by_time)//2]]
            late_scores = [r.overall_score for r in results_by_time[len(results_by_time)//2:]]
            
            trend_direction = "improving" if (sum(late_scores)/len(late_scores)) > (sum(early_scores)/len(early_scores)) else "declining"
        else:
            trend_direction = "stable"
        
        # Type-based analysis
        type_performance = {}
        for result in evaluation_results:
            eval_type = result.evaluation_type.value
            if eval_type not in type_performance:
                type_performance[eval_type] = []
            type_performance[eval_type].append(result.overall_score)
        
        # Calculate averages by type
        for eval_type in type_performance:
            scores = type_performance[eval_type]
            type_performance[eval_type] = {
                'average': sum(scores) / len(scores),
                'count': len(scores)
            }
        
        return {
            'trend_direction': trend_direction,
            'performance_by_type': type_performance,
            'total_span_hours': (
                max(r.evaluation_timestamp for r in results_by_time) -
                min(r.evaluation_timestamp for r in results_by_time)
            ).total_seconds() / 3600.0 if len(results_by_time) > 1 else 0.0
        }
    
    async def _generate_report_recommendations(self, 
                                             evaluation_results: List[EvaluationResult]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        if not evaluation_results:
            return ["No evaluation data available for recommendations"]
        
        scores = [r.overall_score for r in evaluation_results]
        avg_score = sum(scores) / len(scores)
        
        # General recommendations based on score levels
        if avg_score < 0.6:
            recommendations.append("Overall system performance is below acceptable levels. Consider comprehensive optimization.")
        elif avg_score < 0.8:
            recommendations.append("System performance is adequate but has room for improvement. Focus on identified weak areas.")
        else:
            recommendations.append("System performance is good. Continue monitoring and maintain current standards.")
        
        # Type-specific recommendations
        system_quality_results = [r for r in evaluation_results if r.system_quality]
        component_results = [r for r in evaluation_results if r.component_metrics]
        integration_results = [r for r in evaluation_results if r.integration_analysis]
        benchmark_results = [r for r in evaluation_results if r.benchmark_quality]
        
        if system_quality_results:
            avg_system_score = sum(r.system_quality.overall_quality for r in system_quality_results) / len(system_quality_results)
            if avg_system_score < 0.7:
                recommendations.append("System quality evaluation shows areas for improvement in performance and efficiency.")
        
        if component_results:
            avg_component_score = sum(r.overall_score for r in component_results) / len(component_results)
            if avg_component_score < 0.7:
                recommendations.append("Component assessments reveal reliability and readiness issues that need attention.")
        
        if integration_results:
            avg_integration_score = sum(r.integration_analysis.integration_score for r in integration_results) / len(integration_results)
            if avg_integration_score < 0.7:
                recommendations.append("System integration analysis suggests coherence and coordination improvements needed.")
        
        if benchmark_results:
            avg_benchmark_score = sum(r.benchmark_quality.overall_benchmark_quality for r in benchmark_results) / len(benchmark_results)
            if avg_benchmark_score < 0.7:
                recommendations.append("Benchmark validation indicates measurement quality improvements are needed.")
        
        return recommendations
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get evaluation service performance metrics"""
        return {
            **self.performance_metrics,
            'cached_evaluations': len(self.evaluation_cache),
            'evaluation_history_size': len(self.evaluation_history),
            'last_evaluation_time': self.evaluation_history[-1].evaluation_timestamp.isoformat() if self.evaluation_history else None
        }
    
    def _generate_cache_key(self, request: EvaluationRequest) -> str:
        """Generate cache key for evaluation request"""
        # Create deterministic hash of request parameters
        key_components = [
            request.evaluation_type.value,
            request.scope.value,
            request.target_id,
            str(sorted(request.parameters.items())) if request.parameters else "",
            str(sorted(request.context_data.items())) if request.context_data else ""
        ]
        return "_".join(key_components)
    
    def _get_cached_result(self, cache_key: str) -> Optional[EvaluationResult]:
        """Get cached evaluation result if valid"""
        if cache_key in self.evaluation_cache:
            cached_entry = self.evaluation_cache[cache_key]
            if datetime.utcnow() - cached_entry['timestamp'] < self.cache_ttl:
                return cached_entry['result']
            else:
                # Remove expired entry
                del self.evaluation_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: EvaluationResult):
        """Cache evaluation result"""
        self.evaluation_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.utcnow()
        }
        
        # Clean up old entries periodically
        if len(self.evaluation_cache) > 100:  # Max cache size
            oldest_key = min(self.evaluation_cache.keys(), 
                           key=lambda k: self.evaluation_cache[k]['timestamp'])
            del self.evaluation_cache[oldest_key]
    
    def _update_performance_metrics(self, success: bool, execution_time: float, cache_hit: bool):
        """Update service performance metrics"""
        self.performance_metrics['total_evaluations'] += 1
        
        # Update average execution time
        prev_avg = self.performance_metrics['average_execution_time']
        prev_count = self.performance_metrics['total_evaluations'] - 1
        new_avg = (prev_avg * prev_count + execution_time) / self.performance_metrics['total_evaluations']
        self.performance_metrics['average_execution_time'] = new_avg
        
        # Update success rate
        successful_evaluations = sum(1 for r in self.evaluation_history if r.success)
        if success:
            successful_evaluations += 1
        self.performance_metrics['success_rate'] = successful_evaluations / self.performance_metrics['total_evaluations']
        
        # Update cache hit rate
        cache_hits = sum(1 for _ in self.evaluation_cache.values())  # Approximation
        self.performance_metrics['cache_hit_rate'] = cache_hits / self.performance_metrics['total_evaluations']

