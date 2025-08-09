
"""
Field Theory Integration API Endpoints
======================================

REST API endpoints for field theory-based context management including:
- Field representation (scalar, vector, tensor)
- Field propagation and gradient calculations
- Context interaction modeling
- Dynamic field management and optimization
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

# Import database and dependencies
from database import get_db
from services.orchestrator_service import EnhancedOrchestratorService

logger = logging.getLogger(__name__)

# Pydantic Models for API
class FieldUpdateRequest(BaseModel):
    """Request model for updating field context representations"""
    session_id: str = Field(..., min_length=1, description="Unique session identifier for field context")
    context_data: Dict[str, Any] = Field(..., description="Context data to be represented as field")
    field_type: Optional[str] = Field("scalar", description="Field type: scalar (numerical values), vector (directional data), tensor (multi-dimensional)")
    initialization_strategy: Optional[str] = Field("gaussian", description="Field initialization strategy: gaussian, uniform, zeros, custom")
    boundary_conditions: Optional[Dict[str, Any]] = Field(None, description="Field boundary conditions")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "field-session-001",
                "context_data": {
                    "topic": "system_performance",
                    "metrics": {"cpu_usage": 0.75, "memory_usage": 0.68, "response_time": 120},
                    "timestamp": "2024-08-09T10:00:00Z",
                    "source": "monitoring_system"
                },
                "field_type": "scalar",
                "initialization_strategy": "gaussian",
                "boundary_conditions": {"periodic": True, "damping": 0.1}
            }
        }

class FieldPropagationRequest(BaseModel):
    """Request model for propagating field influences using gradient calculations"""
    session_id: str = Field(..., min_length=1, description="Session identifier for field to propagate")
    propagation_steps: Optional[int] = Field(3, ge=1, le=20, description="Number of gradient propagation steps")
    damping_factor: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Damping factor to prevent oscillations")
    convergence_threshold: Optional[float] = Field(0.01, ge=0.001, le=0.1, description="Convergence threshold for stopping")
    propagation_method: Optional[str] = Field("gradient_descent", description="Propagation method: gradient_descent, diffusion, wave")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "field-session-001",
                "propagation_steps": 5,
                "damping_factor": 0.95,
                "convergence_threshold": 0.005,
                "propagation_method": "gradient_descent"
            }
        }

class FieldInteractionRequest(BaseModel):
    """Request model for modeling field interactions between contexts"""
    task_id: int = Field(..., ge=1, description="Task identifier for interaction modeling")
    user_id: int = Field(..., ge=1, description="User identifier requesting interaction analysis")
    similarity_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity threshold for interactions")
    interaction_types: Optional[List[str]] = Field(["semantic", "temporal", "causal"], description="Types of interactions to model")
    max_interactions: Optional[int] = Field(50, ge=1, le=500, description="Maximum number of interactions to return")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": 123,
                "user_id": 456,
                "similarity_threshold": 0.7,
                "interaction_types": ["semantic", "temporal", "causal"],
                "max_interactions": 20
            }
        }

class DynamicFieldRequest(BaseModel):
    """Request model for dynamic field management with real-time updates"""
    session_id: str = Field(..., min_length=1, description="Session identifier for dynamic field")
    alpha: Optional[float] = Field(0.1, ge=0.0, le=1.0, description="Gradient influence factor (∇C component)")
    beta: Optional[float] = Field(0.2, ge=0.0, le=1.0, description="Interaction influence factor (I(x,y) component)")
    update_frequency: Optional[int] = Field(5, ge=1, le=100, description="Update frequency in seconds")
    adaptation_rate: Optional[float] = Field(0.05, ge=0.001, le=0.5, description="Rate of field adaptation")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "dynamic-field-789",
                "alpha": 0.15,
                "beta": 0.25,
                "update_frequency": 3,
                "adaptation_rate": 0.1
            }
        }

class FieldOptimizationRequest(BaseModel):
    """Request model for multi-objective field optimization"""
    task_id: int = Field(..., ge=1, description="Task identifier for optimization")
    user_id: int = Field(..., ge=1, description="User identifier requesting optimization")
    objectives: Optional[Dict[str, float]] = Field(
        {"performance": 0.4, "stability": 0.3, "adaptability": 0.3}, 
        description="Optimization objectives with weights (must sum to 1.0)"
    )
    optimization_algorithm: Optional[str] = Field("multi_objective", description="Algorithm: multi_objective, genetic, simulated_annealing")
    max_iterations: Optional[int] = Field(100, ge=10, le=1000, description="Maximum optimization iterations")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": 123,
                "user_id": 456,
                "objectives": {"performance": 0.5, "stability": 0.3, "adaptability": 0.2},
                "optimization_algorithm": "multi_objective",
                "max_iterations": 150
            }
        }

# Create router
router = APIRouter(
    prefix="/api/field-theory", 
    tags=["🌐 Field Theory Integration"],
    responses={
        404: {"description": "Resource not found", "model": dict},
        422: {"description": "Validation error", "model": dict},
        500: {"description": "Internal server error", "model": dict}
    }
)

# Initialize service
orchestrator_service = EnhancedOrchestratorService()

@router.post("/fields/update", response_model=Dict[str, Any])
async def update_field_context(
    request: FieldUpdateRequest
):
    """
    Update field representation for context data
    
    Creates or updates scalar, vector, or tensor field representations
    based on context data with mathematical field calculations.
    """
    try:
        result = await orchestrator_service.update_field_context(
            session_id=request.session_id,
            context_data=request.context_data,
            field_type=request.field_type
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Field context updated for session {request.session_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Field context update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field update failed: {str(e)}")

@router.post("/fields/propagate", response_model=Dict[str, Any])
async def propagate_field_influence(
    request: FieldPropagationRequest
):
    """
    Propagate field influence using gradient calculations
    
    Implements gradient-based field propagation: ∇C(x) to spread
    influence across the context field space.
    """
    try:
        result = await orchestrator_service.propagate_field_influence(
            session_id=request.session_id,
            propagation_steps=request.propagation_steps
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Field propagation completed with {request.propagation_steps} steps"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Field propagation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field propagation failed: {str(e)}")

@router.post("/fields/interactions", response_model=Dict[str, Any])
async def model_field_interactions(
    request: FieldInteractionRequest,
    db: Session = Depends(get_db)
):
    """
    Model interactions between task contexts using field theory
    
    Analyzes semantic and contextual interactions between tasks
    using field theory principles and vector embeddings.
    """
    try:
        result = await orchestrator_service.model_field_interactions(
            db=db,
            task_id=request.task_id,
            user_id=request.user_id,
            similarity_threshold=request.similarity_threshold
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Field interactions modeled for task {request.task_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Field interaction modeling failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field interaction modeling failed: {str(e)}")

@router.post("/fields/dynamic", response_model=Dict[str, Any])
async def manage_dynamic_fields(
    request: DynamicFieldRequest
):
    """
    Dynamic field management with real-time updates
    
    Implements dynamic field updates using the equation:
    dC/dt = α * ∇C + β * I(x, y) for continuous field evolution.
    """
    try:
        result = await orchestrator_service.manage_dynamic_fields(
            session_id=request.session_id,
            alpha=request.alpha,
            beta=request.beta
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Dynamic field management completed for session {request.session_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Dynamic field management failed: {e}")
        raise HTTPException(status_code=500, detail=f"Dynamic field management failed: {str(e)}")

@router.post("/fields/optimize", response_model=Dict[str, Any])
async def optimize_field_configuration(
    request: FieldOptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Multi-objective field optimization
    
    Optimizes field parameters using the formula:
    O* = arg max_O [Performance(C), Stability(C), Adaptability(C)]
    """
    try:
        result = await orchestrator_service.optimize_field_configuration(
            db=db,
            task_id=request.task_id,
            user_id=request.user_id,
            objectives=request.objectives
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Field optimization completed for task {request.task_id}"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Field optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Field optimization failed: {str(e)}")

@router.get("/fields/context/{session_id}", response_model=Dict[str, Any])
async def get_field_context(
    session_id: str
):
    """Get field context for a specific session"""
    try:
        context = await orchestrator_service.field_manager.get_context(session_id)
        
        if not context:
            raise HTTPException(status_code=404, detail=f"No context found for session {session_id}")
        
        return {
            "status": "success",
            "data": context,
            "message": f"Field context retrieved for session {session_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get field context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fields/statistics", response_model=Dict[str, Any])
async def get_field_statistics():
    """Get comprehensive field theory statistics and performance metrics"""
    try:
        stats = orchestrator_service.field_manager.get_field_statistics()
        
        return {
            "status": "success",
            "data": stats,
            "message": "Field theory statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get field statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fields/states", response_model=Dict[str, Any])
async def get_field_states():
    """Get current field states across all sessions"""
    try:
        field_states = {}
        for field_id, state in orchestrator_service.field_manager.field_states.items():
            field_states[field_id] = {
                "field_type": state.field_type.value,
                "value": state.value,
                "gradient": state.gradient,
                "stability": state.stability,
                "timestamp": state.timestamp.isoformat()
            }
        
        return {
            "status": "success",
            "data": {
                "field_states": field_states,
                "total_states": len(field_states)
            },
            "message": f"Retrieved {len(field_states)} field states"
        }
        
    except Exception as e:
        logger.error(f"Failed to get field states: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fields/interactions", response_model=Dict[str, Any])
async def get_field_interactions():
    """Get current field interactions across all contexts"""
    try:
        interactions = []
        for interaction in orchestrator_service.field_manager.field_interactions:
            interactions.append({
                "source_field": interaction.source_field,
                "target_field": interaction.target_field,
                "interaction_type": interaction.interaction_type,
                "strength": interaction.strength,
                "semantic_similarity": interaction.semantic_similarity,
                "timestamp": interaction.timestamp.isoformat()
            })
        
        return {
            "status": "success",
            "data": {
                "interactions": interactions,
                "total_interactions": len(interactions)
            },
            "message": f"Retrieved {len(interactions)} field interactions"
        }
        
    except Exception as e:
        logger.error(f"Failed to get field interactions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/fields/context/{session_id}", response_model=Dict[str, Any])
async def clear_field_context(
    session_id: str
):
    """Clear field context for a specific session"""
    try:
        if session_id in orchestrator_service.field_manager.contexts:
            del orchestrator_service.field_manager.contexts[session_id]
            
            # Remove associated field states
            field_ids_to_remove = [
                field_id for field_id in orchestrator_service.field_manager.field_states.keys()
                if session_id in field_id
            ]
            
            for field_id in field_ids_to_remove:
                del orchestrator_service.field_manager.field_states[field_id]
            
            return {
                "status": "success",
                "data": {
                    "session_id": session_id,
                    "removed_field_states": len(field_ids_to_remove)
                },
                "message": f"Field context cleared for session {session_id}"
            }
        else:
            raise HTTPException(status_code=404, detail=f"No context found for session {session_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear field context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any])
async def field_theory_health_check():
    """Health check for field theory integration"""
    try:
        # Check system components
        field_manager_status = "healthy" if orchestrator_service.field_manager else "unavailable"
        
        # Check embedding model
        embedding_status = "healthy" if orchestrator_service.field_manager.embedding_model else "unavailable"
        
        # Check field states
        field_states_count = len(orchestrator_service.field_manager.field_states)
        contexts_count = len(orchestrator_service.field_manager.contexts)
        interactions_count = len(orchestrator_service.field_manager.field_interactions)
        
        overall_status = "healthy" if field_manager_status == "healthy" else "degraded"
        
        return {
            "status": overall_status,
            "components": {
                "field_manager": field_manager_status,
                "embedding_model": embedding_status
            },
            "metrics": {
                "field_states": field_states_count,
                "active_contexts": contexts_count,
                "field_interactions": interactions_count
            },
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Field theory integration status: {overall_status}"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Batch operations
@router.post("/fields/batch/update", response_model=Dict[str, Any])
async def batch_update_fields(
    updates: Dict[str, FieldUpdateRequest]
):
    """
    Batch update multiple field contexts
    
    Efficiently update multiple field contexts in a single operation
    for improved performance in high-throughput scenarios.
    """
    try:
        results = {}
        
        for update_id, request in updates.items():
            try:
                result = await orchestrator_service.update_field_context(
                    session_id=request.session_id,
                    context_data=request.context_data,
                    field_type=request.field_type
                )
                results[update_id] = {"status": "success", "data": result}
            except Exception as e:
                results[update_id] = {"status": "error", "error": str(e)}
        
        successful_updates = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "data": results,
            "summary": {
                "total_updates": len(updates),
                "successful_updates": successful_updates,
                "failed_updates": len(updates) - successful_updates
            },
            "message": f"Batch field update completed: {successful_updates}/{len(updates)} successful"
        }
        
    except Exception as e:
        logger.error(f"Batch field update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch field update failed: {str(e)}")

@router.post("/fields/batch/propagate", response_model=Dict[str, Any])
async def batch_propagate_fields(
    propagations: Dict[str, FieldPropagationRequest]
):
    """
    Batch propagate multiple field influences
    
    Efficiently propagate field influences for multiple sessions
    in a single operation for batch processing scenarios.
    """
    try:
        results = {}
        
        for prop_id, request in propagations.items():
            try:
                result = await orchestrator_service.propagate_field_influence(
                    session_id=request.session_id,
                    propagation_steps=request.propagation_steps
                )
                results[prop_id] = {"status": "success", "data": result}
            except Exception as e:
                results[prop_id] = {"status": "error", "error": str(e)}
        
        successful_propagations = sum(1 for r in results.values() if r["status"] == "success")
        
        return {
            "status": "completed",
            "data": results,
            "summary": {
                "total_propagations": len(propagations),
                "successful_propagations": successful_propagations,
                "failed_propagations": len(propagations) - successful_propagations
            },
            "message": f"Batch field propagation completed: {successful_propagations}/{len(propagations)} successful"
        }
        
    except Exception as e:
        logger.error(f"Batch field propagation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch field propagation failed: {str(e)}")
