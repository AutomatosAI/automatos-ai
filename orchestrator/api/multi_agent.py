
"""
Multi-Agent Systems API Endpoints
=================================

REST API endpoints for multi-agent system functionality including:
- Collaborative reasoning across multiple agents
- Agent coordination strategies and load balancing
- Emergent behavior monitoring and analysis
- Multi-objective optimization for agent systems
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

# Import database and dependencies
from database.database import get_db
from services.orchestrator_service import EnhancedOrchestratorService
from ..main import require_api_key

logger = logging.getLogger(__name__)

# Pydantic Models for API
class CollaborativeReasoningRequest(BaseModel):
    """Request model for collaborative reasoning between multiple agents"""
    task_id: int = Field(..., ge=1, description="Unique identifier for the task")
    user_id: int = Field(..., ge=1, description="User identifier requesting the reasoning")
    agents: List[str] = Field(..., min_items=2, description="List of agent IDs to participate in reasoning")
    strategy: Optional[str] = Field("majority_vote", description="Reasoning strategy: majority_vote, weighted_consensus, expert_override, iterative_refinement")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for reasoning")
    timeout_seconds: Optional[int] = Field(300, ge=10, le=3600, description="Maximum time allowed for reasoning")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": 123,
                "user_id": 456,
                "agents": ["agent-001", "agent-002", "agent-003"],
                "strategy": "majority_vote",
                "context": {
                    "domain": "system_analysis",
                    "priority": "high",
                    "data_sources": ["metrics", "logs", "alerts"]
                },
                "timeout_seconds": 180
            }
        }

class AgentCoordinationRequest(BaseModel):
    """Request model for coordinating multiple agents"""
    task_id: int = Field(..., ge=1, description="Unique identifier for the task")
    user_id: int = Field(..., ge=1, description="User identifier requesting coordination")
    agents: List[str] = Field(..., min_items=2, description="List of agent IDs to coordinate")
    strategy: Optional[str] = Field("adaptive", description="Coordination strategy: sequential, parallel, hierarchical, mesh, adaptive")
    load_balance: Optional[bool] = Field(True, description="Whether to perform load balancing")
    context: Optional[Dict[str, Any]] = Field(None, description="Coordination context and preferences")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": 123,
                "user_id": 456,
                "agents": ["agent-001", "agent-002", "agent-003"],
                "strategy": "adaptive",
                "load_balance": True,
                "context": {
                    "urgency": "medium",
                    "resource_constraints": {"memory": "2GB", "cpu": "moderate"},
                    "preferred_completion_time": "30min"
                }
            }
        }

class BehaviorMonitoringRequest(BaseModel):
    """Request model for monitoring agent behavior patterns"""
    session_id: str = Field(..., min_length=1, description="Unique session identifier for monitoring")
    agents: List[str] = Field(..., min_items=1, description="List of agent IDs to monitor")
    interactions: List[Dict[str, Any]] = Field(..., description="List of agent interactions to analyze")
    monitoring_duration: Optional[int] = Field(600, ge=60, description="Monitoring duration in seconds")
    anomaly_threshold: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Threshold for anomaly detection")
    context: Optional[Dict[str, Any]] = Field(None, description="Monitoring context and parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "monitor-session-789",
                "agents": ["agent-001", "agent-002"],
                "interactions": [
                    {
                        "timestamp": "2024-08-09T10:00:00Z",
                        "source_agent": "agent-001",
                        "target_agent": "agent-002",
                        "interaction_type": "task_handoff",
                        "data": {"task_id": 123, "completion_status": "partial"}
                    }
                ],
                "monitoring_duration": 300,
                "anomaly_threshold": 0.85,
                "context": {
                    "focus_areas": ["response_time", "error_rate", "collaboration_quality"]
                }
            }
        }

class OptimizationRequest(BaseModel):
    """Request model for multi-agent system optimization"""
    task_id: int = Field(..., ge=1, description="Unique identifier for the optimization task")
    user_id: int = Field(..., ge=1, description="User identifier requesting optimization")
    agents: List[str] = Field(..., min_items=1, description="List of agent IDs to optimize")
    optimization_objectives: List[str] = Field(["performance", "efficiency"], description="Optimization objectives")
    config: Optional[Dict[str, Any]] = Field(None, description="Optimization configuration parameters")
    max_iterations: Optional[int] = Field(50, ge=1, le=1000, description="Maximum optimization iterations")
    convergence_threshold: Optional[float] = Field(0.01, ge=0.001, le=0.1, description="Convergence threshold")
    context: Optional[Dict[str, Any]] = Field(None, description="Optimization context")
    
    class Config:
        schema_extra = {
            "example": {
                "task_id": 123,
                "user_id": 456,
                "agents": ["agent-001", "agent-002", "agent-003"],
                "optimization_objectives": ["performance", "efficiency", "resource_usage"],
                "config": {
                    "algorithm": "bayesian_optimization",
                    "learning_rate": 0.01,
                    "regularization": 0.001
                },
                "max_iterations": 100,
                "convergence_threshold": 0.005,
                "context": {
                    "priority": "high",
                    "deadline": "2024-08-09T18:00:00Z"
                }
            }
        }

class AgentRebalanceRequest(BaseModel):
    """Request model for rebalancing agent loads"""
    agents: List[str] = Field(..., min_items=2, description="List of agent IDs to rebalance")
    target_balance: Optional[float] = Field(0.8, ge=0.0, le=1.0, description="Target balance score (0-1)")
    rebalance_strategy: Optional[str] = Field("adaptive", description="Rebalancing strategy: round_robin, load_based, adaptive")
    preserve_affinity: Optional[bool] = Field(True, description="Whether to preserve agent-task affinity")
    
    class Config:
        schema_extra = {
            "example": {
                "agents": ["agent-001", "agent-002", "agent-003"],
                "target_balance": 0.85,
                "rebalance_strategy": "adaptive",
                "preserve_affinity": True
            }
        }

# Create router
router = APIRouter(
    prefix="/api/multi-agent", 
    tags=["👥 Multi-Agent Systems"],
    responses={
        404: {"description": "Resource not found", "model": dict},
        422: {"description": "Validation error", "model": dict},
        500: {"description": "Internal server error", "model": dict}
    }
)

# Initialize service
orchestrator_service = EnhancedOrchestratorService()

@router.post("/reasoning/collaborative", 
             response_model=Dict[str, Any],
              dependencies=[Depends(require_api_key)],
             summary="🧠 Collaborative Agent Reasoning",
             description="Enable multiple agents to collaborate on complex reasoning tasks with consensus mechanisms",
             response_description="Collaborative reasoning results with consensus score and agent insights",
             responses={
                 200: {
                     "description": "Collaborative reasoning completed successfully",
                     "content": {
                         "application/json": {
                             "example": {
                                 "status": "success",
                                 "data": {
                                     "consensus_score": 0.87,
                                     "reasoning_strategy": "majority_vote",
                                     "agents_participated": 3,
                                     "conclusion": "System health is optimal based on analyzed metrics",
                                     "confidence": 0.87,
                                     "participating_agents": ["agent-001", "agent-002", "agent-003"],
                                     "reasoning_time": 1.24,
                                     "individual_conclusions": [
                                         {"agent": "agent-001", "conclusion": "System healthy", "confidence": 0.9},
                                         {"agent": "agent-002", "conclusion": "System healthy", "confidence": 0.85},
                                         {"agent": "agent-003", "conclusion": "System healthy", "confidence": 0.86}
                                     ]
                                 },
                                 "message": "Collaborative reasoning completed with 3 agents"
                             }
                         }
                     }
                 },
                 404: {"description": "Task or agents not found"},
                 422: {"description": "Invalid request parameters"},
                 500: {"description": "Internal reasoning system error"}
             })
async def perform_collaborative_reasoning(
    request: CollaborativeReasoningRequest,
    db: Session = Depends(get_db)
):
    """
    ## 🧠 Collaborative Agent Reasoning
    
    **Purpose**: Enable multiple agents to work together on complex reasoning tasks, 
    leveraging collective intelligence and consensus mechanisms.
    
    ### 🎯 **Key Features:**
    - **Consensus Building**: Multiple voting and consensus strategies
    - **Conflict Resolution**: Automatic resolution of conflicting agent opinions  
    - **Strategy Selection**: Adaptive strategy selection based on task complexity
    - **Quality Assurance**: Confidence scoring and validation
    
    ### 📊 **Reasoning Strategies:**
    - **majority_vote**: Simple majority wins approach
    - **weighted_consensus**: Agent expertise-weighted decisions
    - **expert_override**: Domain expert agent has final say  
    - **iterative_refinement**: Multiple reasoning rounds for convergence
    
    ### 🔬 **Use Cases:**
    - Complex problem analysis requiring multiple perspectives
    - System diagnostic tasks with uncertain outcomes
    - Decision making in ambiguous situations
    - Quality assurance through agent cross-validation
    
    ### 📈 **Performance Metrics:**
    - **Consensus Score**: Measure of agent agreement (0-1)
    - **Confidence Level**: Overall confidence in the conclusion
    - **Reasoning Time**: Time taken for collaborative process
    - **Participation Rate**: Percentage of agents that contributed
    
    ### 🚨 **Error Handling:**
    - Graceful degradation when agents are unavailable
    - Timeout protection for long-running reasoning tasks
    - Automatic retry with reduced agent set on partial failures
    """
    try:
        result = await orchestrator_service.perform_collaborative_reasoning(
            db=db,
            task_id=request.task_id,
            user_id=request.user_id,
            agents=request.agents,
            context=request.context
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Collaborative reasoning completed with {len(request.agents)} agents"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Collaborative reasoning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Collaborative reasoning failed: {str(e)}")

@router.post("/coordination/coordinate", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def coordinate_agents(
    request: AgentCoordinationRequest,
    db: Session = Depends(get_db)
):
    """
    Coordinate multiple agents for task execution
    
    Implements various coordination strategies including sequential, parallel,
    hierarchical, mesh, and adaptive coordination with load balancing.
    """
    try:
        result = await orchestrator_service.coordinate_multi_agents(
            db=db,
            task_id=request.task_id,
            user_id=request.user_id,
            agents=request.agents,
            strategy=request.strategy,
            context=request.context
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Agent coordination completed using {result.get('strategy_used', 'default')} strategy"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Agent coordination failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent coordination failed: {str(e)}")

@router.post("/behavior/monitor", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def monitor_emergent_behavior(
    request: BehaviorMonitoringRequest
):
    """
    Monitor and analyze emergent behaviors in agent interactions
    
    Detects behavioral patterns, analyzes system stability, and identifies
    anomalies in multi-agent interactions.
    """
    try:
        result = await orchestrator_service.monitor_emergent_behavior(
            session_id=request.session_id,
            agents=request.agents,
            interactions=request.interactions,
            context=request.context
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Behavior monitoring completed for {len(request.agents)} agents"
        }
        
    except Exception as e:
        logger.error(f"Behavior monitoring failed: {e}")
        raise HTTPException(status_code=500, detail=f"Behavior monitoring failed: {str(e)}")

@router.post("/optimization/optimize", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def optimize_multi_agent_system(
    request: OptimizationRequest,
    db: Session = Depends(get_db)
):
    """
    Optimize multi-agent system configuration and performance
    
    Performs multi-objective optimization considering performance, scalability,
    robustness, efficiency, cost, and latency objectives.
    """
    try:
        result = await orchestrator_service.optimize_multi_agent_system(
            db=db,
            task_id=request.task_id,
            user_id=request.user_id,
            agents=request.agents,
            config=request.config,
            context=request.context
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Multi-agent optimization completed for {len(request.agents)} agents"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Multi-agent optimization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Multi-agent optimization failed: {str(e)}")

@router.post("/coordination/rebalance", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def rebalance_agents(
    request: AgentRebalanceRequest,
    db: Session = Depends(get_db)
):
    """
    Rebalance agent loads to achieve target balance score
    
    Redistributes tasks among agents to optimize load distribution
    and improve overall system performance.
    """
    try:
        result = await orchestrator_service.coordination_manager.rebalance_agents(
            db=db,
            agents=request.agents,
            target_balance=request.target_balance
        )
        
        return {
            "status": "success",
            "data": result,
            "message": f"Agent rebalancing completed for {len(request.agents)} agents"
        }
        
    except Exception as e:
        logger.error(f"Agent rebalancing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent rebalancing failed: {str(e)}")

@router.get("/reasoning/statistics", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def get_reasoning_statistics():
    """Get collaborative reasoning statistics and performance metrics"""
    try:
        stats = orchestrator_service.collaborative_reasoning.get_reasoning_statistics()
        
        return {
            "status": "success",
            "data": stats,
            "message": "Reasoning statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get reasoning statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordination/statistics", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def get_coordination_statistics():
    """Get coordination management statistics and performance metrics"""
    try:
        stats = orchestrator_service.coordination_manager.get_coordination_statistics()
        
        return {
            "status": "success",
            "data": stats,
            "message": "Coordination statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get coordination statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/behavior/statistics", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def get_behavior_statistics():
    """Get behavior monitoring statistics and performance metrics"""
    try:
        stats = orchestrator_service.behavior_monitor.get_monitoring_statistics()
        
        return {
            "status": "success",
            "data": stats,
            "message": "Behavior statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get behavior statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/optimization/statistics", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def get_optimization_statistics():
    """Get optimization statistics and performance metrics"""
    try:
        stats = orchestrator_service.multi_agent_optimizer.get_optimization_statistics()
        
        return {
            "status": "success",
            "data": stats,
            "message": "Optimization statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to get optimization statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=Dict[str, Any], dependencies=[Depends(require_api_key)])
async def multi_agent_health_check():
    """Health check for multi-agent systems"""
    try:
        # Check system components
        reasoning_status = "healthy" if orchestrator_service.collaborative_reasoning else "unavailable"
        coordination_status = "healthy" if orchestrator_service.coordination_manager else "unavailable"
        behavior_status = "healthy" if orchestrator_service.behavior_monitor else "unavailable"
        optimization_status = "healthy" if orchestrator_service.multi_agent_optimizer else "unavailable"
        
        overall_status = "healthy" if all(status == "healthy" for status in [
            reasoning_status, coordination_status, behavior_status, optimization_status
        ]) else "degraded"
        
        return {
            "status": overall_status,
            "components": {
                "collaborative_reasoning": reasoning_status,
                "coordination_management": coordination_status,
                "behavior_monitoring": behavior_status,
                "optimization": optimization_status
            },
            "timestamp": datetime.utcnow().isoformat(),
            "message": f"Multi-agent systems status: {overall_status}"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# WebSocket endpoint for real-time behavior monitoring
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio

@router.websocket("/behavior/monitor/realtime")
async def realtime_behavior_monitoring(websocket: WebSocket):
    """
    Real-time behavior monitoring via WebSocket
    
    Streams behavior analysis results in real-time for continuous monitoring
    of multi-agent system interactions.
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive monitoring request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            session_id = request_data.get("session_id")
            agents = request_data.get("agents", [])
            interactions = request_data.get("interactions", [])
            context = request_data.get("context")
            
            if not session_id or not agents:
                await websocket.send_text(json.dumps({
                    "error": "Missing required fields: session_id and agents"
                }))
                continue
            
            try:
                # Perform behavior monitoring
                result = await orchestrator_service.monitor_emergent_behavior(
                    session_id=session_id,
                    agents=agents,
                    interactions=interactions,
                    context=context
                )
                
                # Send results back
                await websocket.send_text(json.dumps({
                    "status": "success",
                    "data": result,
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for real-time behavior monitoring")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "error": f"WebSocket error: {str(e)}"
            }))
        except:
            pass
