

"""
Evaluation Methodologies API Endpoints
=====================================

REST API endpoints for comprehensive evaluation methodologies including:
- System quality evaluation
- Component assessment
- Integration analysis  
- Benchmark validation
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

# Database imports
from database import get_db
from models import (
    EvaluationResult as EvaluationResultDB,
    BenchmarkAssessment,
    ComponentMetricsDB,
    IntegrationAnalysisDB,
    Agent,
    Workflow
)

# Evaluation imports
from evaluation.evaluation_service import (
    EvaluationService,
    EvaluationRequest,
    EvaluationResult,
    EvaluationType,
    EvaluationScope,
    BenchmarkResult,
    AssessmentReport
)

logger = logging.getLogger(__name__)

# Initialize evaluation service
evaluation_service = EvaluationService()

# Create router
router = APIRouter(prefix="/api/evaluation", tags=["evaluation"])

# Pydantic models for API
class EvaluationRequestAPI(BaseModel):
    """API model for evaluation requests"""
    evaluation_type: str = Field(..., description="Type of evaluation")
    scope: str = Field(..., description="Scope of evaluation")
    target_id: str = Field(..., description="ID of target entity")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Evaluation parameters")
    context_data: Optional[Dict[str, Any]] = Field(None, description="Additional context")

class EvaluationResultResponse(BaseModel):
    """API response model for evaluation results"""
    evaluation_id: str
    evaluation_type: str
    scope: str
    target_id: str
    overall_score: float
    success: bool
    execution_time_seconds: float
    evaluation_timestamp: str
    detailed_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class BenchmarkValidationRequest(BaseModel):
    """API model for benchmark validation requests"""
    benchmark_name: str
    benchmark_data: Dict[str, Any]
    theoretical_framework: Dict[str, Any]
    validation_data: Optional[Dict[str, Any]] = None

class ComponentAssessmentRequest(BaseModel):
    """API model for component assessment requests"""
    component_id: str
    component_type: str
    task_data: Dict[str, Any] = Field(default_factory=dict)
    environment_data: Dict[str, Any] = Field(default_factory=dict)
    operation_history: List[Dict[str, Any]] = Field(default_factory=list)

class IntegrationAnalysisRequest(BaseModel):
    """API model for integration analysis requests"""
    system_id: str
    system_components: List[Dict[str, Any]]
    interaction_data: List[Dict[str, Any]] = Field(default_factory=list)
    system_metrics: Dict[str, Any] = Field(default_factory=dict)
    expected_behaviors: Dict[str, float] = Field(default_factory=dict)
    theoretical_benchmarks: Dict[str, float] = Field(default_factory=dict)
    resource_data: Dict[str, Any] = Field(default_factory=dict)

class AssessmentReportRequest(BaseModel):
    """API model for assessment report requests"""
    evaluation_ids: List[str]
    report_title: str = "System Assessment Report"
    include_trends: bool = True
    include_recommendations: bool = True

# API Endpoints

@router.post("/evaluate", response_model=EvaluationResultResponse)
async def evaluate_system(
    request: EvaluationRequestAPI,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None, description="User ID for evaluation")
):
    """
    Perform comprehensive evaluation based on request type
    
    Supports:
    - system_quality: Overall system quality assessment
    - component_assessment: Individual component evaluation  
    - integration_analysis: System integration evaluation
    - benchmark_validation: Benchmark quality validation
    - comprehensive: All evaluation types combined
    """
    try:
        # Validate evaluation type
        try:
            evaluation_type = EvaluationType(request.evaluation_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid evaluation type: {request.evaluation_type}"
            )
        
        # Validate scope
        try:
            scope = EvaluationScope(request.scope)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid evaluation scope: {request.scope}"
            )
        
        # Create evaluation request
        eval_request = EvaluationRequest(
            evaluation_type=evaluation_type,
            scope=scope,
            target_id=request.target_id,
            parameters=request.parameters,
            context_data=request.context_data,
            user_id=user_id
        )
        
        # Perform evaluation
        result = await evaluation_service.evaluate(eval_request, db)
        
        # Store result in database
        db_result = EvaluationResultDB(
            evaluation_id=result.evaluation_id,
            evaluation_type=result.evaluation_type.value,
            scope=result.scope.value,
            target_id=result.target_id,
            overall_score=result.overall_score,
            detailed_results=result.to_dict(),
            success=result.success,
            error_message=result.error_message,
            execution_time_seconds=result.execution_time_seconds,
            user_id=user_id
        )
        
        db.add(db_result)
        db.commit()
        
        # Return API response
        return EvaluationResultResponse(
            evaluation_id=result.evaluation_id,
            evaluation_type=result.evaluation_type.value,
            scope=result.scope.value,
            target_id=result.target_id,
            overall_score=result.overall_score,
            success=result.success,
            execution_time_seconds=result.execution_time_seconds,
            evaluation_timestamp=result.evaluation_timestamp.isoformat(),
            detailed_results=result.to_dict() if result.success else None,
            error_message=result.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in evaluation endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/system-quality")
async def evaluate_system_quality(
    target_id: str,
    tasks: List[Dict[str, Any]],
    context_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None)
):
    """
    Evaluate overall system quality using multi-dimensional analysis
    """
    try:
        request = EvaluationRequestAPI(
            evaluation_type="system_quality",
            scope="system",
            target_id=target_id,
            parameters={"tasks": tasks},
            context_data=context_data
        )
        
        return await evaluate_system(request, db, user_id)
        
    except Exception as e:
        logger.error(f"Error in system quality evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/component-assessment")
async def assess_component(
    request: ComponentAssessmentRequest,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None)
):
    """
    Assess individual component performance, reliability, and readiness
    """
    try:
        eval_request = EvaluationRequestAPI(
            evaluation_type="component_assessment",
            scope="component",
            target_id=request.component_id,
            parameters={
                "component_type": request.component_type,
                "task_data": request.task_data,
                "environment_data": request.environment_data,
                "operation_history": request.operation_history
            }
        )
        
        result = await evaluate_system(eval_request, db, user_id)
        
        # Store component-specific metrics
        if result.success and "component_metrics" in result.detailed_results:
            metrics = result.detailed_results["component_metrics"]
            
            db_metrics = ComponentMetricsDB(
                component_id=request.component_id,
                component_type=request.component_type,
                performance_score=metrics.get("performance_score", 0.0),
                reliability_score=metrics.get("reliability_score", 0.0),
                readiness_score=metrics.get("readiness_score", 0.0),
                capability_rating=metrics.get("capability_rating", 0.0),
                complexity_index=metrics.get("complexity_index", 0.0),
                environment_factor=metrics.get("environment_factor", 0.0),
                assessment_details=metrics
            )
            
            db.add(db_metrics)
            db.commit()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in component assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/integration-analysis")
async def analyze_integration(
    request: IntegrationAnalysisRequest,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None)
):
    """
    Analyze system integration including coherence, efficiency, and emergence
    """
    try:
        eval_request = EvaluationRequestAPI(
            evaluation_type="integration_analysis",
            scope="system",
            target_id=request.system_id,
            parameters={
                "system_components": request.system_components,
                "interaction_data": request.interaction_data,
                "system_metrics": request.system_metrics,
                "expected_behaviors": request.expected_behaviors,
                "theoretical_benchmarks": request.theoretical_benchmarks,
                "resource_data": request.resource_data
            }
        )
        
        result = await evaluate_system(eval_request, db, user_id)
        
        # Store integration analysis
        if result.success and "integration_analysis" in result.detailed_results:
            analysis = result.detailed_results["integration_analysis"]
            
            db_analysis = IntegrationAnalysisDB(
                system_id=request.system_id,
                coherence_score=analysis.get("coherence_score", 0.0),
                efficiency_score=analysis.get("efficiency_score", 0.0),
                emergence_score=analysis.get("emergence_score", 0.0),
                integration_score=analysis.get("integration_score", 0.0),
                integration_classification=analysis.get("integration_classification", "unknown"),
                analysis_data=analysis,
                recommendations=analysis.get("recommendations", []),
                confidence_level=analysis.get("confidence_level", 0.0)
            )
            
            db.add(db_analysis)
            db.commit()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in integration analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/benchmark-validation")
async def validate_benchmark(
    request: BenchmarkValidationRequest,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None)
):
    """
    Validate benchmark quality including validity, reliability, and discriminatory power
    """
    try:
        eval_request = EvaluationRequestAPI(
            evaluation_type="benchmark_validation",
            scope="enterprise",
            target_id=request.benchmark_name,
            parameters={
                "benchmark_data": request.benchmark_data,
                "theoretical_framework": request.theoretical_framework,
                "validation_data": request.validation_data
            }
        )
        
        result = await evaluate_system(eval_request, db, user_id)
        
        # Store benchmark assessment
        if result.success and "benchmark_quality" in result.detailed_results:
            quality = result.detailed_results["benchmark_quality"]
            
            db_assessment = BenchmarkAssessment(
                benchmark_id=request.benchmark_name,
                benchmark_name=request.benchmark_name,
                benchmark_type=request.benchmark_data.get("type", "performance"),
                validity_score=quality.get("overall_validity", 0.0),
                reliability_score=quality.get("overall_reliability", 0.0),
                discriminatory_power=quality.get("discriminatory_power", 0.0),
                overall_quality=quality.get("overall_benchmark_quality", 0.0),
                quality_classification=quality.get("benchmark_classification", "unknown"),
                assessment_data=quality,
                recommendations=quality.get("recommendations", [])
            )
            
            db.add(db_assessment)
            db.commit()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in benchmark validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/comprehensive")
async def comprehensive_evaluation(
    target_id: str,
    parameters: Dict[str, Any],
    context_data: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db),
    user_id: Optional[int] = Query(None)
):
    """
    Perform comprehensive evaluation across all methodologies
    """
    try:
        eval_request = EvaluationRequestAPI(
            evaluation_type="comprehensive",
            scope="system",
            target_id=target_id,
            parameters=parameters,
            context_data=context_data
        )
        
        return await evaluate_system(eval_request, db, user_id)
        
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assessment-report")
async def generate_assessment_report(
    request: AssessmentReportRequest,
    db: Session = Depends(get_db)
):
    """
    Generate comprehensive assessment report from multiple evaluations
    """
    try:
        # Retrieve evaluation results
        evaluation_results = []
        
        for eval_id in request.evaluation_ids:
            db_result = db.query(EvaluationResultDB).filter(
                EvaluationResultDB.evaluation_id == eval_id
            ).first()
            
            if db_result:
                # Convert database result to EvaluationResult
                result = EvaluationResult(
                    evaluation_id=db_result.evaluation_id,
                    evaluation_type=EvaluationType(db_result.evaluation_type),
                    scope=EvaluationScope(db_result.scope),
                    target_id=db_result.target_id,
                    overall_score=db_result.overall_score,
                    evaluation_timestamp=db_result.created_at,
                    execution_time_seconds=db_result.execution_time_seconds,
                    success=db_result.success,
                    error_message=db_result.error_message
                )
                evaluation_results.append(result)
        
        if not evaluation_results:
            raise HTTPException(
                status_code=404,
                detail="No evaluation results found for provided IDs"
            )
        
        # Generate assessment report
        report = await evaluation_service.generate_assessment_report(
            evaluation_results, request.report_title
        )
        
        return report.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating assessment report: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results/{evaluation_id}")
async def get_evaluation_result(
    evaluation_id: str,
    db: Session = Depends(get_db)
):
    """
    Retrieve specific evaluation result by ID
    """
    try:
        result = db.query(EvaluationResultDB).filter(
            EvaluationResultDB.evaluation_id == evaluation_id
        ).first()
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Evaluation result not found: {evaluation_id}"
            )
        
        return {
            "evaluation_id": result.evaluation_id,
            "evaluation_type": result.evaluation_type,
            "scope": result.scope,
            "target_id": result.target_id,
            "overall_score": result.overall_score,
            "detailed_results": result.detailed_results,
            "success": result.success,
            "error_message": result.error_message,
            "execution_time_seconds": result.execution_time_seconds,
            "created_at": result.created_at.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving evaluation result: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_evaluation_history(
    target_id: Optional[str] = Query(None, description="Filter by target ID"),
    evaluation_type: Optional[str] = Query(None, description="Filter by evaluation type"),
    limit: int = Query(50, description="Maximum number of results"),
    offset: int = Query(0, description="Offset for pagination"),
    db: Session = Depends(get_db)
):
    """
    Retrieve evaluation history with optional filtering
    """
    try:
        query = db.query(EvaluationResultDB)
        
        if target_id:
            query = query.filter(EvaluationResultDB.target_id == target_id)
        
        if evaluation_type:
            query = query.filter(EvaluationResultDB.evaluation_type == evaluation_type)
        
        # Order by creation date (most recent first)
        query = query.order_by(EvaluationResultDB.created_at.desc())
        
        # Apply pagination
        results = query.offset(offset).limit(limit).all()
        total_count = query.count()
        
        return {
            "results": [
                {
                    "evaluation_id": r.evaluation_id,
                    "evaluation_type": r.evaluation_type,
                    "scope": r.scope,
                    "target_id": r.target_id,
                    "overall_score": r.overall_score,
                    "success": r.success,
                    "execution_time_seconds": r.execution_time_seconds,
                    "created_at": r.created_at.isoformat()
                }
                for r in results
            ],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error retrieving evaluation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance-metrics")
async def get_performance_metrics():
    """
    Get evaluation service performance metrics
    """
    try:
        metrics = evaluation_service.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint for evaluation service
    """
    try:
        metrics = evaluation_service.get_performance_metrics()
        
        return {
            "status": "healthy",
            "service": "evaluation-methodologies",
            "version": "1.0.0",
            "total_evaluations": metrics.get("total_evaluations", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

