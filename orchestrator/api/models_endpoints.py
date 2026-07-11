"""
Model Management API Endpoints (PRD-15)
========================================

REST API endpoints for LLM model discovery, recommendations, and cost estimation.

Author: Automatos AI
Date: 2025-10-06
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from core.database.database import get_db
from core.llm.model_registry import ModelRegistry, ModelInfo
import logging
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/", response_model=List[Dict[str, Any]])
async def list_models(
    provider: Optional[str] = Query(None, description="Filter by provider (openai, anthropic, huggingface)"),
    status: str = Query('active', description="Filter by status (active, deprecated, beta)"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    List all available LLM models.
    
    Query Parameters:
    - provider: Optional filter by provider name
    - status: Filter by status (default: active)
    
    Returns:
    - List of model objects with full metadata
    
    Example:
        GET /api/models/?provider=openai&status=active
    """
    try:
        registry = ModelRegistry(db)
        models = registry.get_all_models(provider=provider, status=status)
        
        logger.info(f"Listed {len(models)} models (provider={provider}, status={status})")
        
        return [model.to_dict() for model in models]
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model(
    model_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific model.
    
    Path Parameters:
    - model_id: Unique model identifier (e.g., 'google/gemini-2.5-flash', 'openai/gpt-5.5')
    
    Returns:
    - Model object with full metadata
    
    Example:
        GET /api/models/gpt-4
    """
    try:
        registry = ModelRegistry(db)
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404, 
                detail=f"Model not found: {model_id}"
            )
        
        logger.info(f"Retrieved model: {model_id}")
        return model.to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/providers/", response_model=List[Dict[str, Any]])
async def list_providers(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """
    List all available LLM providers with their models.
    
    Returns:
    - List of providers with model counts and model lists
    
    Example Response:
        [
            {
                "name": "openai",
                "model_count": 3,
                "models": [
                    {"model_id": "google/gemini-2.5-flash", "display_name": "Gemini 2.5 Flash"},
                    {"model_id": "openai/gpt-5.5", "display_name": "GPT-5.5"}
                ]
            }
        ]
    """
    try:
        registry = ModelRegistry(db)
        stats = registry.get_provider_stats()
        
        providers = []
        for provider_name, provider_stats in stats.items():
            providers.append({
                "name": provider_name,
                "model_count": provider_stats['model_count'],
                "avg_input_cost": provider_stats.get('avg_input_cost', 0),
                "avg_output_cost": provider_stats.get('avg_output_cost', 0),
                "models": provider_stats['models']
            })
        
        logger.info(f"Listed {len(providers)} providers")
        return providers
        
    except Exception as e:
        logger.error(f"Error listing providers: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/recommend", response_model=Dict[str, Any], dependencies=[Depends(require_workspace_permission("agents:read"))])
async def recommend_model(
    requirements: Dict[str, Any],
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Recommend the best model based on requirements.
    
    Request Body:
    {
        "task_type": "code_analysis",           // Optional: Type of task
        "max_cost": 0.05,                       // Optional: Max cost per 1K tokens
        "min_context": 8000,                    // Optional: Minimum context window
        "required_capabilities": ["reasoning"], // Optional: Required capabilities
        "prefer_provider": "openai",            // Optional: Preferred provider
        "require_functions": false,             // Optional: Requires function calling
        "require_vision": false                 // Optional: Requires vision support
    }
    
    Returns:
    - Recommended model with reasoning
    
    Example:
        POST /api/models/recommend
        {
            "task_type": "code_analysis",
            "max_cost": 0.05,
            "prefer_provider": "openai"
        }
    """
    try:
        registry = ModelRegistry(db)
        model = registry.find_best_model(requirements)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail="No model found matching requirements"
            )
        
        logger.info(
            f"Recommended model: {model.model_id} for requirements: {requirements}"
        )
        
        return {
            "model_id": model.model_id,
            "display_name": model.display_name,
            "provider": model.provider,
            "reason": "Best match based on requirements",
            "model": model.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recommending model: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/estimate-cost", response_model=Dict[str, Any], dependencies=[Depends(require_workspace_permission("agents:read"))])
async def estimate_cost(
    request: Dict[str, Any],
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Estimate cost for model usage.
    
    Request Body:
    {
        "model_id": "google/gemini-2.5-flash",
        "input_tokens": 1500,
        "output_tokens": 800
    }
    
    Returns:
    - Cost estimation in USD
    
    Example:
        POST /api/models/estimate-cost
        {
            "model_id": "google/gemini-2.5-flash",
            "input_tokens": 1500,
            "output_tokens": 800
        }
    """
    try:
        model_id = request.get("model_id")
        input_tokens = request.get("input_tokens", 0)
        output_tokens = request.get("output_tokens", 0)
        
        if not model_id:
            raise HTTPException(
                status_code=400,
                detail="model_id is required"
            )
        
        registry = ModelRegistry(db)
        model = registry.get_model(model_id)
        
        if not model:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {model_id}"
            )
        
        cost = model.estimate_cost(input_tokens, output_tokens)
        
        logger.info(
            f"Cost estimated for {model_id}: "
            f"{input_tokens} input + {output_tokens} output = ${cost:.6f}"
        )
        
        return {
            "model_id": model_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": (input_tokens / 1000) * model.input_cost_per_1k,
            "output_cost": (output_tokens / 1000) * model.output_cost_per_1k,
            "total_cost": cost,
            "currency": "USD",
            "model_details": {
                "display_name": model.display_name,
                "provider": model.provider,
                "input_cost_per_1k": model.input_cost_per_1k,
                "output_cost_per_1k": model.output_cost_per_1k
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating cost: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/by-task/{task_type}", response_model=List[Dict[str, Any]])
async def get_models_by_task(
    task_type: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get models recommended for a specific task type.
    
    Path Parameters:
    - task_type: Type of task (e.g., 'code_analysis', 'creative_writing')
    
    Returns:
    - List of recommended models sorted by cost
    
    Example:
        GET /api/models/by-task/code_analysis
    """
    try:
        registry = ModelRegistry(db)
        models = registry.get_recommended_models(task_type)
        
        if not models:
            return []
        
        logger.info(f"Found {len(models)} models for task '{task_type}'")
        
        return [model.to_dict() for model in models]
        
    except Exception as e:
        logger.error(f"Error getting models by task: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/", response_model=Dict[str, Any])
async def get_model_stats(ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db)):
    """
    Get overall statistics about available models.
    
    Returns:
    - Statistics by provider and overall totals
    
    Example Response:
        {
            "total_models": 6,
            "total_providers": 2,
            "providers": {
                "openai": {
                    "model_count": 3,
                    "avg_input_cost": 0.01,
                    "avg_output_cost": 0.03
                }
            }
        }
    """
    try:
        registry = ModelRegistry(db)
        provider_stats = registry.get_provider_stats()
        
        total_models = sum(p['model_count'] for p in provider_stats.values())
        
        return {
            "total_models": total_models,
            "total_providers": len(provider_stats),
            "providers": provider_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

