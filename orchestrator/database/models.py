
"""
Database Models for Automotas AI System
=======================================

Comprehensive data models for agents, skills, workflows, documents, and system configuration.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Table
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Union,  Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from uuid import uuid4

# DEPRECATED: This module is deprecated. Use `automatos-ai/orchestrator/models.py` instead.
# Keeping this file as a shim to avoid breaking external imports during transition.
Base = declarative_base()

# Association tables for many-to-many relationships
agent_skills = Table('agent_skills', Base.metadata,
    Column('agent_id', Integer, ForeignKey('agents.id')),
    Column('skill_id', Integer, ForeignKey('skills.id'))
)

workflow_agents = Table('workflow_agents', Base.metadata,
    Column('workflow_id', Integer, ForeignKey('workflows.id')),
    Column('agent_id', Integer, ForeignKey('agents.id'))
)

# PRD-15: LLM Model Registry
class LLMModel(Base):
    """
    Registry of available LLM models from different providers.
    Stores model metadata, capabilities, costs, and recommended use cases.
    """
    __tablename__ = 'llm_models'
    
    id = Column(Integer, primary_key=True)
    provider = Column(String(50), nullable=False, index=True)  # 'openai', 'anthropic', 'huggingface'
    model_id = Column(String(255), nullable=False, unique=True, index=True)  # 'gpt-4', 'claude-3-opus-20240229', etc.
    display_name = Column(String(255), nullable=False)  # Human-readable name
    model_family = Column(String(100), index=True)  # 'gpt-4', 'claude-3', 'llama-2', etc.
    
    # Capabilities
    capabilities = Column(JSON, default=dict)  # {"reasoning": "high", "coding": "excellent", ...}
    context_window = Column(Integer, nullable=False)  # Max context tokens
    max_output_tokens = Column(Integer, nullable=False)  # Max output tokens
    supports_functions = Column(Boolean, default=False)
    supports_vision = Column(Boolean, default=False)
    supports_streaming = Column(Boolean, default=True)
    
    # Cost information (in USD)
    input_cost_per_1k_tokens = Column(Float)  # Cost per 1K input tokens
    output_cost_per_1k_tokens = Column(Float)  # Cost per 1K output tokens
    
    # Metadata
    description = Column(Text)
    release_date = Column(DateTime)
    deprecation_date = Column(DateTime)
    status = Column(String(50), default='active', index=True)  # 'active', 'deprecated', 'beta'
    recommended_for = Column(JSON, default=list)  # ['code_analysis', 'creative_writing', ...]
    
    # Settings
    default_temperature = Column(Float, default=0.7)
    min_temperature = Column(Float, default=0.0)
    max_temperature = Column(Float, default=2.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Database Models
class Agent(Base):
    __tablename__ = 'agents'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    agent_type = Column(String(100), nullable=False)  # 'custom', 'system', 'specialized'
    status = Column(String(50), default='active')  # 'active', 'inactive', 'training'
    configuration = Column(JSON)  # Agent-specific config
    performance_metrics = Column(JSON)  # Performance data
    
    # NEW FIELDS for UI requirements
    priority_level = Column(String(50), default='medium')  # 'low', 'medium', 'high', 'critical'
    max_concurrent_tasks = Column(Integer, default=5)  # Maximum concurrent tasks
    auto_start = Column(Boolean, default=False)  # Auto-start on system boot
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
    # PRD-15: Multi-Model Configuration
    model_config = Column(JSON, default=lambda: {
        "provider": "openai",
        "model_id": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": None
    })  # Model configuration for this agent
    
    model_usage_stats = Column(JSON, default=lambda: {
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_requests": 0,
        "avg_tokens_per_request": 0,
        "last_used_at": None
    })  # Model usage tracking
    
    # Relationships
    skills = relationship("Skill", secondary=agent_skills, back_populates="agents")
    workflows = relationship("Workflow", secondary=workflow_agents, back_populates="agents")
    executions = relationship("WorkflowExecution", back_populates="agent")
    # Phase 3: MCP Tools relationship - forward ref to avoid circular import
    tool_assignments = relationship("AgentToolAssignment", back_populates="agent", cascade="all, delete-orphan", foreign_keys="[AgentToolAssignment.agent_id]")

# Phase 3: MCP Tools Models
class AgentToolAssignment(Base):
    """Agent-Tool Assignment with permissions"""
    __tablename__ = 'agent_tool_assignments'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'), nullable=False)
    tool_id = Column(Integer, ForeignKey('mcp_tools.id', ondelete='CASCADE'), nullable=False)
    enabled = Column(Boolean, default=True)
    permissions = Column(JSON, default={})
    configuration = Column(JSON, default={})
    assigned_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="tool_assignments")
    tool = relationship("MCPTool", back_populates="tool_assignments")

class MCPTool(Base):
    """MCP Tool Model"""
    __tablename__ = 'mcp_tools'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    mcp_server_url = Column(String(500))
    capabilities = Column(JSON, default={})
    credentials_schema = Column(JSON, default={})
    status = Column(String(50), default='active')
    provider = Column(String(255))
    version = Column(String(50))
    icon = Column(String(100))
    category = Column(String(100))
    tags = Column(JSON)  # Using JSON instead of ARRAY for compatibility
    tool_metadata = Column('metadata', JSON, default={})
    created_by = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool_assignments = relationship("AgentToolAssignment", back_populates="tool", cascade="all, delete-orphan")

class Skill(Base):
    __tablename__ = 'skills'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    skill_type = Column(String(100), nullable=False)  # 'cognitive', 'technical', 'communication'
    
    # NEW FIELD for UI requirements - skill categorization
    category = Column(String(100), nullable=False)  # 'development', 'security', 'infrastructure', 'analytics'
    
    implementation = Column(Text)  # Code or configuration
    parameters = Column(JSON)  # Skill parameters
    performance_data = Column(JSON)  # Usage statistics
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
    # Relationships
    agents = relationship("Agent", secondary=agent_skills, back_populates="skills")

class Pattern(Base):
    __tablename__ = 'patterns'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    pattern_type = Column(String(100), nullable=False)  # 'coordination', 'communication', 'decision'
    pattern_data = Column(JSON)  # Pattern definition
    usage_count = Column(Integer, default=0)
    effectiveness_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))

class Workflow(Base):
    __tablename__ = 'workflows'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    goal = Column(Text)  # High-level objective (overrides description if provided)
    context = Column(Text)  # Additional context for execution (codegraph_project, pr_number, etc.)
    tags = Column(JSON, default=list)  # Tags for categorization and filtering
    workflow_definition = Column(JSON)  # Workflow steps and logic
    status = Column(String(50), default='draft')  # 'draft', 'active', 'archived'
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    last_execution = Column(JSON, default=None)  # Quick access to latest execution summary
    
    # Relationships
    agents = relationship("Agent", secondary=workflow_agents, back_populates="workflows")
    executions = relationship("WorkflowExecution", back_populates="workflow")

class WorkflowExecution(Base):
    __tablename__ = 'workflow_executions'
    
    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey('workflows.id'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'
    input_data = Column(JSON)
    output_data = Column(JSON)
    execution_log = Column(Text)
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    error_message = Column(Text)
    execution_metadata = Column(JSON, default={})  # Analytics and tracking data (renamed from metadata - reserved word)
    models_used = Column(JSON, default=list)  # Model usage tracking
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    agent = relationship("Agent", back_populates="executions")

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255))
    file_type = Column(String(100))
    file_size = Column(Integer)
    file_path = Column(String(500))
    content_hash = Column(String(255))
    status = Column(String(50), default='uploaded')  # 'uploaded', 'processing', 'processed', 'failed'
    chunk_count = Column(Integer, default=0)
    tags = Column(JSON)
    description = Column(Text)
    doc_metadata = Column(JSON)
    upload_date = Column(DateTime, default=func.now())
    processed_date = Column(DateTime)
    created_by = Column(String(255))

class SystemConfiguration(Base):
    __tablename__ = 'system_configurations'
    
    id = Column(Integer, primary_key=True)
    config_key = Column(String(255), unique=True, nullable=False)
    config_value = Column(JSON)
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    updated_by = Column(String(255))

class RAGConfiguration(Base):
    __tablename__ = 'rag_configurations'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    embedding_model = Column(String(255))
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    retrieval_strategy = Column(String(100), default='similarity')
    top_k = Column(Integer, default=5)
    similarity_threshold = Column(Float, default=0.7)
    configuration = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))

# Pydantic Models for API
class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"

class AgentType(str, Enum):
    CODE_ARCHITECT = "code_architect"
    SECURITY_EXPERT = "security_expert"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    DATA_ANALYST = "data_analyst"
    INFRASTRUCTURE_MANAGER = "infrastructure_manager"
    CUSTOM = "custom"
    SYSTEM = "system"
    SPECIALIZED = "specialized"

class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SkillType(str, Enum):
    COGNITIVE = "cognitive"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"

class SkillCategory(str, Enum):
    DEVELOPMENT = "development"
    SECURITY = "security"
    INFRASTRUCTURE = "infrastructure"
    ANALYTICS = "analytics"
    DATA = "data"
    PERFORMANCE = "performance"
    AI = "ai"
    DOCUMENTATION = "documentation"
    SYSTEM = "system"
    MONITORING = "monitoring"

class WorkflowStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"

class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# API Request/Response Models
class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_type: str  # Flexible agent type - no enum restriction
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = []
    tool_ids: Optional[List[int]] = []  # NEW: Support for tool assignment during creation
    priority_level: Optional[PriorityLevel] = PriorityLevel.MEDIUM
    max_concurrent_tasks: Optional[int] = Field(default=5, ge=1, le=100)
    auto_start: Optional[bool] = False

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AgentStatus] = None
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None
    priority_level: Optional[PriorityLevel] = None
    max_concurrent_tasks: Optional[int] = Field(default=None, ge=1, le=100)
    auto_start: Optional[bool] = None

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    agent_type: str
    status: str
    configuration: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    priority_level: str
    max_concurrent_tasks: int
    auto_start: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    skills: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []  # Phase 3: MCP Tools
    agent_model_config: Optional[Dict[str, Any]] = None  # PRD-15: Model configuration (renamed from model_config - Pydantic reserved)

class SkillCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    skill_type: SkillType
    category: SkillCategory
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class SkillUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    category: Optional[SkillCategory] = None
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class SkillResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    skill_type: str
    category: str
    implementation: Optional[str] = ""
    parameters: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = {}
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = ""

class PatternCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    pattern_type: str
    pattern_data: Dict[str, Any]

class PatternUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    pattern_type: Optional[str] = None
    pattern_data: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class PatternResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    pattern_type: str
    pattern_data: Dict[str, Any]
    usage_count: int
    effectiveness_score: float
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    goal: Optional[str] = Field(None, description="High-level objective of the workflow (overrides description if provided)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for workflow execution (e.g., codegraph_project, pr_number, git_url)")
    workflow_definition: Dict[str, Any]
    agent_ids: Optional[List[int]] = []

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_definition: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    agent_ids: Optional[List[int]] = None

class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    workflow_definition: Dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    agents: List[Dict[str, Any]] = []
    last_execution: Optional[Dict[str, Any]] = None

class WorkflowExecutionCreate(BaseModel):
    workflow_id: int
    agent_id: int
    input_data: Optional[Dict[str, Any]] = None

class WorkflowExecutionResponse(BaseModel):
    id: int
    workflow_id: int
    agent_id: int
    status: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    execution_log: Optional[str]
    started_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str

class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: Optional[str]
    file_type: Optional[str]
    file_size: Optional[int]
    status: str
    chunk_count: Optional[int]
    tags: Optional[List[str]]
    description: Optional[str]
    upload_date: datetime
    processed_date: Optional[datetime]
    created_by: Optional[str]

class SystemConfigCreate(BaseModel):
    config_key: str
    config_value: Dict[str, Any]
    description: Optional[str] = None

class SystemConfigResponse(BaseModel):
    id: int
    config_key: str
    config_value: Dict[str, Any]
    description: Optional[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    updated_by: Optional[str]

class RAGConfigCreate(BaseModel):
    name: str
    embedding_model: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    retrieval_strategy: Optional[str] = "similarity"
    top_k: Optional[int] = 5
    similarity_threshold: Optional[float] = 0.7
    configuration: Optional[Dict[str, Any]] = None

class RAGConfigResponse(BaseModel):
    id: int
    name: str
    embedding_model: Optional[str]
    chunk_size: int
    chunk_overlap: int
    retrieval_strategy: str
    top_k: int
    similarity_threshold: float
    configuration: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]
    version: str

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# New API Models for UI Requirements
class AgentTemplate(BaseModel):
    name: str
    description: str
    agent_type: AgentType
    default_skills: List[str]
    default_configuration: Dict[str, Any]
    priority_level: PriorityLevel
    max_concurrent_tasks: int

class AgentStatistics(BaseModel):
    total_agents: int
    active_agents: int
    inactive_agents: int
    agents_by_type: Dict[str, int]
    average_performance: float
    total_executions: int
    successful_executions: int
    failed_executions: int

class SkillsByCategory(BaseModel):
    category: str
    skills: List[SkillResponse]

class SystemMetrics(BaseModel):
    uptime: str
    cpu_usage: float
    memory_usage: float
    active_connections: int
    total_requests: int
    error_rate: float

# Memory-related models for API compatibility
class MemoryItemCreate(BaseModel):
    session_id: str
    content: Union[Dict[str, Any], str]  # Can be dict or JSON string
    memory_type: str = "working_data"
    importance: Optional[float] = 0.5
    tags: Optional[List[str]] = []
    context: Optional[Dict] = None
    
class MemoryItemResponse(BaseModel):
    id: str
    content: str
    context: Optional[Dict] = None
    tags: List[str] = []
    created_at: datetime
    
class ExternalKnowledgeCreate(BaseModel):
    source: str
    content: str
    metadata: Optional[Dict] = None
    
class ExternalKnowledgeResponse(BaseModel):
    id: str
    source: str
    content: str
    metadata: Optional[Dict] = None
    created_at: datetime

# Evaluation-related models
class EvaluationResult(BaseModel):
    score: float
    metrics: Dict[str, Any]
    timestamp: datetime
    
class EvaluationResultCreate(BaseModel):
    score: float
    metrics: Dict[str, Any]
    
class EvaluationResultResponse(BaseModel):
    id: str
    score: float
    metrics: Dict[str, Any]
    timestamp: datetime
    
class EvaluationMetricCreate(BaseModel):
    name: str
    value: float
    
class EvaluationMetricResponse(BaseModel):
    id: str
    name: str
    value: float
    
class EvaluationReportCreate(BaseModel):
    title: str
    results: List[Dict]
    
class EvaluationReportResponse(BaseModel):
    id: str
    title: str
    results: List[Dict]
    created_at: datetime

# Additional evaluation models
class BenchmarkAssessment(BaseModel):
    name: str
    score: float
    details: Dict[str, Any]
    
class ComponentMetricsDB(BaseModel):
    component_id: str
    metrics: Dict[str, Any]
    
class IntegrationAnalysisDB(BaseModel):
    integration_id: str
    analysis: Dict[str, Any]

# Tool-related models for API compatibility
class ToolCredentials(BaseModel):
    tool_name: str
    credentials: Dict[str, Any]
    
class ToolConfig(BaseModel):
    name: str
    config: Dict[str, Any]

# Tool model
class Tool(BaseModel):
    id: Optional[str] = None
    name: str
    description: Optional[str] = None
    type: Optional[str] = None
    config: Optional[Dict] = None
    
class CredentialAuditLog(BaseModel):
    id: Optional[str] = None
    action: str
    user_id: Optional[str] = None
    timestamp: datetime
    details: Optional[Dict] = None

# Additional missing models
class ToolConfiguration(BaseModel):
    id: Optional[str] = None
    tool_id: str
    config: Dict[str, Any]
    
class ToolUsageLog(BaseModel):
    id: Optional[str] = None
    tool_id: str
    user_id: Optional[str] = None
    timestamp: datetime
    result: Optional[Dict] = None

class AgentToolPermission(BaseModel):
    id: Optional[str] = None
    agent_id: str
    tool_id: str
    environment: Optional[str] = None
    is_active: bool = True
    
class PermissionAuditLog(BaseModel):
    id: Optional[str] = None
    action: str
    user_id: Optional[str] = None
    timestamp: datetime
    details: Optional[Dict] = None

# Database models for Evaluation
class EvaluationResultDB(Base):
    __tablename__ = 'evaluation_results'
    
    id = Column(Integer, primary_key=True)
    evaluation_id = Column(String(255), unique=True, nullable=False)
    target_id = Column(String(255), nullable=False)
    evaluation_type = Column(String(100), nullable=False)
    score = Column(Float)
    metrics = Column(JSON)
    recommendations = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


# Database models for Permissions
class AgentToolPermissionDB(Base):
    __tablename__ = 'agent_tool_permissions'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(String(255), nullable=False)
    tool_id = Column(String(255), nullable=False)
    environment = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PermissionAuditLogDB(Base):
    __tablename__ = 'permission_audit_logs'
    
    id = Column(Integer, primary_key=True)
    action = Column(String(100), nullable=False)
    user_id = Column(String(255))
    timestamp = Column(DateTime, default=func.now())
    details = Column(JSON)


# SQLAlchemy Tool model (replacing Pydantic one)
class ToolDB(Base):
    __tablename__ = 'tools'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(Text)
    category = Column(String)
    provider = Column(String)
    version = Column(String)
    icon = Column(String)
    pricing = Column(String)
    tags = Column(JSON, default=list)
    permissions = Column(JSON, default=list)
    required_credentials = Column(JSON, default=list)
    supported_environments = Column(JSON, default=list)
    mcp_config = Column(JSON)
    status = Column(String, default='available')
    is_installed = Column(Boolean, default=False)
    installation_date = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    last_updated = Column(DateTime, onupdate=func.now())

# Rename the old Tool to be clear it's Pydantic
ToolPydantic = Tool
Tool = ToolDB

# SQLAlchemy CredentialAuditLog model
class CredentialAuditLogDB(Base):
    __tablename__ = 'credential_audit_logs'
    
    id = Column(Integer, primary_key=True, index=True)
    credential_id = Column(String, index=True)
    tool_id = Column(String, index=True)
    action = Column(String)  # created, updated, deleted, accessed, tested
    user_id = Column(String)
    timestamp = Column(DateTime, server_default=func.now())
    details = Column(JSON)
    ip_address = Column(String)
    user_agent = Column(String)
    success = Column(Boolean)
    error_message = Column(Text)

CredentialAuditLog = CredentialAuditLogDB

# SQLAlchemy ToolCredentials model
class ToolCredentialsDB(Base):
    __tablename__ = 'tool_credentials'
    
    id = Column(Integer, primary_key=True, index=True)
    tool_id = Column(String, index=True)
    tool_name = Column(String)
    credential_key = Column(String)
    encrypted_value = Column(Text)
    environment = Column(String, default='production')
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())
    created_by = Column(String)
    is_active = Column(Boolean, default=True)
    extra_metadata = Column(JSON)

# Update reference
ToolCredentialsPydantic = ToolCredentials
ToolCredentials = ToolCredentialsDB

# ============================================================================
# PRD-04: Inter-Agent Communication & Collaboration Models
# ============================================================================

class AgentMessage(Base):
    """Tracks messages exchanged between agents"""
    __tablename__ = 'agent_messages'
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid4()))
    from_agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'))
    to_agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'))
    message_type = Column(String(50), nullable=False)
    content = Column(JSON, nullable=False)
    priority = Column(Integer, default=5)
    status = Column(String(50), default='sent')  # sent, delivered, read, acknowledged
    requires_ack = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    delivered_at = Column(DateTime)
    read_at = Column(DateTime)
    acknowledged_at = Column(DateTime)

class SharedContext(Base):
    """Shared memory spaces for agent teams"""
    __tablename__ = 'shared_contexts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255))
    team_id = Column(String(255))
    context_data = Column(JSON, nullable=False)
    version = Column(Integer, default=1)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(Integer, ForeignKey('agents.id'))

class ContextPermission(Base):
    """Permissions for shared contexts"""
    __tablename__ = 'context_permissions'
    
    id = Column(Integer, primary_key=True)
    context_id = Column(String(255), ForeignKey('shared_contexts.id', ondelete='CASCADE'))
    agent_id = Column(Integer, ForeignKey('agents.id', ondelete='CASCADE'))
    permission_level = Column(String(50), default='read')  # read, write, admin
    granted_at = Column(DateTime, default=func.now())
    granted_by = Column(Integer, ForeignKey('agents.id'))

class CollaborationSession(Base):
    """Records of multi-agent collaborative problem solving"""
    __tablename__ = 'collaboration_sessions'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    problem_id = Column(Integer, ForeignKey('tasks.id'))
    problem_description = Column(Text)
    team_agents = Column(JSON)  # Array of agent IDs
    strategy = Column(String(50), default='ensemble')  # ensemble, hierarchical, specialized, consensus
    shared_context_id = Column(UUID(as_uuid=True), ForeignKey('shared_contexts.id'))
    status = Column(String(50), default='pending')  # pending, active, completed, failed
    result = Column(JSON)
    consensus_data = Column(JSON)
    metrics = Column(JSON)  # Performance metrics
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)

class CollaborationProposal(Base):
    """Individual agent proposals during collaboration"""
    __tablename__ = 'collaboration_proposals'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('collaboration_sessions.id', ondelete='CASCADE'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    proposal_type = Column(String(50))  # solution, insight, feedback
    proposal_content = Column(JSON, nullable=False)
    confidence = Column(Float, default=0.5)
    tokens_used = Column(Integer)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=func.now())

class ConsensusVote(Base):
    """Voting records for consensus building"""
    __tablename__ = 'consensus_votes'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('collaboration_sessions.id', ondelete='CASCADE'))
    proposal_id = Column(Integer, ForeignKey('collaboration_proposals.id', ondelete='CASCADE'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    vote_weight = Column(Float, default=1.0)
    vote_value = Column(String(50))  # approve, reject, abstain
    reasoning = Column(Text)
    created_at = Column(DateTime, default=func.now())

class MessageBroadcast(Base):
    """Tracking for team broadcast messages"""
    __tablename__ = 'message_broadcasts'
    
    id = Column(String(255), primary_key=True, default=lambda: str(uuid4()))
    from_agent_id = Column(Integer, ForeignKey('agents.id'))
    team_agents = Column(JSON)  # Array of agent IDs
    message_type = Column(String(50))
    content = Column(JSON)
    priority = Column(Integer, default=5)
    delivered_to = Column(JSON)  # Array of successfully delivered agent IDs
    failed_deliveries = Column(JSON)  # Array of failed agent IDs
    created_at = Column(DateTime, default=func.now())

