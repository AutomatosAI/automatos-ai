
"""
Enhanced Pydantic Models for Swagger Documentation
==================================================

Comprehensive models with examples, validation, and detailed field descriptions
for improved API documentation and developer experience.
"""

from pydantic import BaseModel, Field
from pydantic import field_validator, model_validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum

class StatusEnum(str, Enum):
    """API response status values"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    PENDING = "pending"
    COMPLETED = "completed"

class APIResponse(BaseModel):
    """Standardized API response format"""
    status: StatusEnum = Field(..., description="Response status indicator")
    data: Optional[Dict[str, Any]] = Field(None, description="Response payload data")
    message: str = Field(..., description="Human-readable response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "data": {"result": "Operation completed successfully"},
                "message": "Request processed successfully",
                "timestamp": "2024-08-09T10:00:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Standardized error response format"""
    status: str = Field("error", description="Always 'error' for error responses")
    error_code: Optional[str] = Field(None, description="Machine-readable error code")
    error_message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error_code": "VALIDATION_ERROR",
                "error_message": "Invalid input parameters provided",
                "details": {"field": "agent_id", "issue": "Agent ID not found"},
                "timestamp": "2024-08-09T10:00:00Z"
            }
        }

# Multi-Agent System Models
class CollaborativeReasoningResult(BaseModel):
    """Result of collaborative reasoning operation"""
    consensus_score: float = Field(..., ge=0.0, le=1.0, description="Consensus score (0-1)")
    reasoning_strategy: str = Field(..., description="Strategy used for reasoning")
    agents_participated: int = Field(..., ge=1, description="Number of agents that participated")
    conclusion: str = Field(..., description="Final reasoning conclusion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in conclusion")
    participating_agents: List[str] = Field(..., description="IDs of participating agents")
    reasoning_time: float = Field(..., ge=0.0, description="Time taken in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "consensus_score": 0.87,
                "reasoning_strategy": "majority_vote",
                "agents_participated": 3,
                "conclusion": "System health is optimal based on analyzed metrics",
                "confidence": 0.87,
                "participating_agents": ["agent-001", "agent-002", "agent-003"],
                "reasoning_time": 1.24
            }
        }

class AgentCoordinationResult(BaseModel):
    """Result of agent coordination operation"""
    coordination_strategy: str = Field(..., description="Coordination strategy applied")
    agents_coordinated: int = Field(..., ge=1, description="Number of coordinated agents")
    balance_score: float = Field(..., ge=0.0, le=1.0, description="Load balance score achieved")
    coordination_matrix: List[List[float]] = Field(..., description="Agent coordination matrix")
    efficiency_gain: float = Field(..., description="Efficiency improvement achieved")
    
    class Config:
        schema_extra = {
            "example": {
                "coordination_strategy": "adaptive",
                "agents_coordinated": 3,
                "balance_score": 0.92,
                "coordination_matrix": [[1.0, 0.8, 0.9], [0.8, 1.0, 0.7], [0.9, 0.7, 1.0]],
                "efficiency_gain": 0.15
            }
        }

class BehaviorMonitoringResult(BaseModel):
    """Result of behavior monitoring analysis"""
    monitoring_active: bool = Field(..., description="Whether monitoring is currently active")
    agents_monitored: int = Field(..., ge=1, description="Number of agents being monitored")
    patterns_detected: int = Field(..., ge=0, description="Number of behavior patterns detected")
    anomalies: List[str] = Field(..., description="List of detected anomalies")
    behavior_score: float = Field(..., ge=0.0, le=1.0, description="Overall behavior health score")
    performance_metrics: Dict[str, float] = Field(..., description="Performance metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "monitoring_active": True,
                "agents_monitored": 3,
                "patterns_detected": 7,
                "anomalies": [],
                "behavior_score": 0.88,
                "performance_metrics": {
                    "avg_response_time": 1.2,
                    "success_rate": 0.96,
                    "resource_efficiency": 0.84
                }
            }
        }

# Field Theory Models
class FieldUpdateResult(BaseModel):
    """Result of field context update"""
    session_id: str = Field(..., description="Session identifier")
    field_updated: bool = Field(..., description="Whether field was successfully updated")
    field_type: str = Field(..., description="Type of field (scalar, vector, tensor)")
    stability: float = Field(..., ge=0.0, le=1.0, description="Field stability score")
    influence_radius: float = Field(..., ge=0.0, description="Field influence radius")
    update_time: float = Field(..., ge=0.0, description="Update processing time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "field-session-001",
                "field_updated": True,
                "field_type": "scalar",
                "stability": 0.94,
                "influence_radius": 2.5,
                "update_time": 0.123
            }
        }

class FieldPropagationResult(BaseModel):
    """Result of field propagation operation"""
    session_id: str = Field(..., description="Session identifier")
    propagation_steps: int = Field(..., ge=1, description="Number of propagation steps performed")
    convergence_achieved: bool = Field(..., description="Whether propagation converged")
    final_stability: float = Field(..., ge=0.0, le=1.0, description="Final field stability")
    propagation_time: float = Field(..., ge=0.0, description="Total propagation time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "field-session-001",
                "propagation_steps": 5,
                "convergence_achieved": True,
                "final_stability": 0.96,
                "propagation_time": 0.234
            }
        }

# Agent Management Models
class AgentCreationRequest(BaseModel):
    """Request model for creating a new agent"""
    name: str = Field(..., min_length=1, max_length=255, description="Agent name (required)")
    description: Optional[str] = Field(None, max_length=1000, description="Agent description")
    agent_type: str = Field(..., description="Agent type: custom, system, or specialized")
    configuration: Optional[Dict[str, Any]] = Field({}, description="Agent configuration parameters")
    skill_ids: Optional[List[int]] = Field([], description="List of skill IDs to assign")
    
    @field_validator('agent_type')
    def validate_agent_type(cls, v):
        allowed_types = ['custom', 'system', 'specialized']
        if v not in allowed_types:
            raise ValueError(f'Agent type must be one of: {allowed_types}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Data Analysis Agent",
                "description": "Specialized agent for data analysis and reporting",
                "agent_type": "specialized",
                "configuration": {
                    "max_concurrent_tasks": 5,
                    "timeout_seconds": 300,
                    "memory_limit": "1GB"
                },
                "skill_ids": [1, 2, 5]
            }
        }

class AgentResponse(BaseModel):
    """Response model for agent information"""
    id: int = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: str = Field(..., description="Agent type")
    status: str = Field(..., description="Current agent status")
    configuration: Dict[str, Any] = Field({}, description="Agent configuration")
    performance_metrics: Optional[Dict[str, Any]] = Field(None, description="Performance data")
    skills: List[Dict[str, Any]] = Field([], description="Associated skills")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "name": "Data Analysis Agent",
                "description": "Specialized agent for data analysis and reporting",
                "agent_type": "specialized",
                "status": "active",
                "configuration": {
                    "max_concurrent_tasks": 5,
                    "timeout_seconds": 300
                },
                "performance_metrics": {
                    "tasks_completed": 142,
                    "success_rate": 0.96,
                    "avg_execution_time": 45.2
                },
                "skills": [
                    {"id": 1, "name": "Data Processing", "type": "technical"},
                    {"id": 2, "name": "Report Generation", "type": "communication"}
                ],
                "created_at": "2024-08-09T08:00:00Z",
                "updated_at": "2024-08-09T10:00:00Z"
            }
        }

# System Health Models
class ComponentHealth(BaseModel):
    """Health status of individual system component"""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status: healthy, degraded, unhealthy")
    last_check: datetime = Field(..., description="Last health check timestamp")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Component-specific metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "name": "multi_agent_systems",
                "status": "healthy",
                "last_check": "2024-08-09T10:00:00Z",
                "metrics": {
                    "active_agents": 5,
                    "avg_response_time": "45ms",
                    "success_rate": 0.99
                }
            }
        }

class SystemHealthResponse(BaseModel):
    """Comprehensive system health response"""
    overall_status: str = Field(..., description="Overall system health status")
    components: List[ComponentHealth] = Field(..., description="Individual component health")
    system_metrics: Dict[str, Any] = Field(..., description="System-wide metrics")
    uptime: str = Field(..., description="System uptime")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Health check timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "overall_status": "healthy",
                "components": [
                    {
                        "name": "api_server",
                        "status": "healthy",
                        "last_check": "2024-08-09T10:00:00Z",
                        "metrics": {"response_time": "25ms"}
                    }
                ],
                "system_metrics": {
                    "total_requests": 15420,
                    "active_connections": 12,
                    "memory_usage": "512MB"
                },
                "uptime": "5d 12h 34m",
                "version": "1.0.0",
                "timestamp": "2024-08-09T10:00:00Z"
            }
        }

# Context Engineering Models
class EntropyRequest(BaseModel):
    """Request for entropy calculation"""
    text: str = Field(..., min_length=1, description="Text to analyze for entropy")
    normalize: bool = Field(True, description="Whether to normalize the entropy value")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "The quick brown fox jumps over the lazy dog",
                "normalize": True
            }
        }

class EntropyResponse(BaseModel):
    """Response for entropy calculation"""
    text: str = Field(..., description="Original input text")
    entropy: float = Field(..., description="Calculated Shannon entropy")
    bits: float = Field(..., description="Entropy in bits")
    normalized: bool = Field(..., description="Whether entropy was normalized")
    character_count: int = Field(..., description="Number of characters analyzed")
    unique_characters: int = Field(..., description="Number of unique characters")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "The quick brown fox jumps over the lazy dog",
                "entropy": 4.234,
                "bits": 4.234,
                "normalized": True,
                "character_count": 43,
                "unique_characters": 26
            }
        }

class VectorSimilarityRequest(BaseModel):
    """Request for vector similarity calculation"""
    vector1: List[float] = Field(..., min_items=1, description="First vector for comparison")
    vector2: List[float] = Field(..., min_items=1, description="Second vector for comparison")
    metric: str = Field("cosine", description="Similarity metric: cosine, euclidean, manhattan")
    
    @model_validator(mode='after')
    def validate_vectors(self):
        if len(self.vector1) != len(self.vector2):
            raise ValueError('Vectors must have the same dimensions')
        return self
    
    @field_validator('metric')
    def validate_metric(cls, v):
        allowed_metrics = ['cosine', 'euclidean', 'manhattan', 'dot_product']
        if v not in allowed_metrics:
            raise ValueError(f'Metric must be one of: {allowed_metrics}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "vector1": [1.0, 2.0, 3.0, 4.0],
                "vector2": [2.0, 3.0, 4.0, 5.0],
                "metric": "cosine"
            }
        }

class VectorSimilarityResponse(BaseModel):
    """Response for vector similarity calculation"""
    metric: str = Field(..., description="Similarity metric used")
    similarity: float = Field(..., description="Calculated similarity score")
    distance: float = Field(..., description="Calculated distance value")
    vector_dimensions: int = Field(..., description="Number of dimensions")
    computation_time: float = Field(..., description="Computation time in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "metric": "cosine",
                "similarity": 0.9746,
                "distance": 0.0254,
                "vector_dimensions": 4,
                "computation_time": 0.001
            }
        }
