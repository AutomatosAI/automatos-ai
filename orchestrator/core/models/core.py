
"""
Database Models for Automotas AI System
=======================================

Comprehensive data models for agents, skills, workflows, documents, and system configuration.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Table, CheckConstraint
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, JSONB, UUID
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from uuid import uuid4
# Define PriorityLevel locally to avoid circular imports
class PriorityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

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

# ===================================================================
# LLM MODEL REGISTRY (PRD-15: Multi-Model Configuration)
# ===================================================================

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
    tags = Column(JSON, default=list)  # Lightweight capability tags
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
    # Evaluation fields for enhanced assessment
    quality_score = Column(Float, nullable=True)  # Quality metric
    emergence_score = Column(Float, nullable=True)  # Emergence metric
    performance = Column(Float, nullable=True)  # Performance score
    reliability = Column(Float, nullable=True)  # Reliability metric
    readiness = Column(Float, nullable=True)  # Interaction readiness score
    coherence = Column(Float, nullable=True)  # Coherence metric
    efficiency = Column(Float, nullable=True)  # Efficiency metric
    eci = Column(Float, nullable=True)  # Emergent capability index
    validity = Column(Float, nullable=True)  # Validity score
    discriminatory_power = Column(Float, nullable=True)  # Discriminatory power
    
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
    tool_assignments = relationship("AgentToolAssignment", foreign_keys="[AgentToolAssignment.agent_id]", cascade="all, delete-orphan")

class Skill(Base):
    __tablename__ = 'skills'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    skill_type = Column(String(100), nullable=False)  # 'cognitive', 'technical', 'communication'
    category = Column(String(100), nullable=True)  # Skill category from schema
    implementation = Column(Text)  # Code or configuration
    parameters = Column(JSON)  # Skill parameters
    performance_data = Column(JSON)  # Usage statistics
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
    # PRD-22: New fields for Git-backed skills
    prompt_template = Column(Text, nullable=True)  # Core skill content (Level 2 progressive disclosure)
    tools_schema = Column(JSONB, nullable=True)  # PRD-22: Executable tool definitions for agents
    skill_version = Column(String(20), nullable=True)  # Version string
    skill_source = Column(String(50), nullable=True)  # Source identifier (builtin-seeds, anthropic-official, etc.)
    git_repo_url = Column(Text, nullable=True)  # Git repository URL
    git_commit_sha = Column(String(40), nullable=True)  # Git commit SHA
    git_branch = Column(String(100), nullable=True)  # Git branch
    filesystem_path = Column(Text, nullable=True)  # Path to skill directory
    tags = Column(JSONB, nullable=True)  # Tags array for categorization
    skill_metadata = Column(JSONB, nullable=True)  # Flexible metadata JSON
    last_sync_at = Column(DateTime(timezone=True), nullable=True)  # Last Git sync timestamp
    
    # Relationships
    agents = relationship("Agent", secondary=agent_skills, back_populates="skills")
    # PRD-22: New relationships
    files = relationship("SkillFile", back_populates="skill", cascade="all, delete-orphan")
    source = relationship("SkillSource", foreign_keys=[skill_source], back_populates="skills", 
                         primaryjoin="Skill.skill_source == SkillSource.source_name")
    versions = relationship("SkillVersion", back_populates="skill", cascade="all, delete-orphan")
    audit_logs = relationship("SkillAuditLog", back_populates="skill")

# ===================================================================
# PRD-22: Git-backed Skills Models
# ===================================================================

class SkillFile(Base):
    """PRD-22: Skill files for progressive disclosure"""
    __tablename__ = 'skill_files'
    
    id = Column(Integer, primary_key=True)
    skill_id = Column(Integer, ForeignKey('skills.id', ondelete='CASCADE'), nullable=False)
    file_path = Column(String(500), nullable=False)  # Relative path within skill directory
    file_type = Column(String(50), nullable=False)   # core, resource, script, example
    content_summary = Column(Text, nullable=True)    # Brief summary for indexing
    file_size_bytes = Column(Integer, nullable=True)
    estimated_tokens = Column(Integer, nullable=True)
    load_level = Column(Integer, nullable=False, default=3)  # 1=metadata, 2=core, 3=resource
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationships
    skill = relationship("Skill", back_populates="files")


class SkillSource(Base):
    """PRD-22: Skill sources (Git repositories, uploads, etc.)"""
    __tablename__ = 'skill_sources'
    
    id = Column(Integer, primary_key=True)
    source_name = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(50), nullable=False)  # git, upload, seed, marketplace
    
    # Git-specific fields
    git_url = Column(Text, nullable=True)
    git_branch = Column(String(100), nullable=True, default='main')
    git_commit_sha = Column(String(40), nullable=True)
    
    # Local storage
    local_cache_path = Column(Text, nullable=False)
    
    # Auto-update configuration
    auto_update = Column(Boolean, default=False)
    update_frequency_hours = Column(Integer, nullable=True, default=24)
    
    # Metadata
    skills_discovered = Column(Integer, default=0)
    status = Column(String(50), default='active')  # active, error, syncing, inactive
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    last_sync_status = Column(String(50), nullable=True)  # success, error, partial
    last_sync_error = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_by = Column(String(255), nullable=True)
    
    # Relationships
    skills = relationship("Skill", foreign_keys="Skill.skill_source", back_populates="source",
                         primaryjoin="SkillSource.source_name == Skill.skill_source")
    audit_logs = relationship("SkillAuditLog", back_populates="source")


class SkillVersion(Base):
    """PRD-22: Skill version history"""
    __tablename__ = 'skill_versions'
    
    id = Column(Integer, primary_key=True)
    skill_id = Column(Integer, ForeignKey('skills.id', ondelete='CASCADE'), nullable=False)
    version = Column(String(20), nullable=False)
    git_commit_sha = Column(String(40), nullable=True)
    changelog = Column(Text, nullable=True)
    is_current = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    created_by = Column(String(255), nullable=True)
    
    # Relationships
    skill = relationship("Skill", back_populates="versions")


class SkillAuditLog(Base):
    """PRD-22: Audit log for skill operations"""
    __tablename__ = 'skill_audit_log'
    
    id = Column(Integer, primary_key=True)
    skill_id = Column(Integer, ForeignKey('skills.id', ondelete='SET NULL'), nullable=True)
    source_id = Column(Integer, ForeignKey('skill_sources.id', ondelete='SET NULL'), nullable=True)
    
    # Action details
    action = Column(String(50), nullable=False)  # create, update, delete, load, execute, sync
    action_details = Column(JSON, nullable=True)
    
    # User context
    user_id = Column(String(255), nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Result
    status = Column(String(50), nullable=False)  # success, error, warning
    error_message = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    skill = relationship("Skill", back_populates="audit_logs")
    source = relationship("SkillSource", back_populates="audit_logs")


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

# ===================================================================
# MCP TOOLS MODELS (Phase 3: Skills & Tools Integration)
# ===================================================================

class MCPTool(Base):
    """MCP Tool Model - represents external tools agents can use"""
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
    tags = Column(PG_ARRAY(String))
    tool_metadata = Column('metadata', JSON, default={})  # Renamed from metadata (reserved in SQLAlchemy)
    created_by = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    tool_assignments = relationship("AgentToolAssignment", back_populates="tool", cascade="all, delete-orphan")
    usage_logs = relationship("ToolUsageLog", back_populates="tool", cascade="all, delete-orphan")

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
    tool = relationship("MCPTool", back_populates="tool_assignments")

class ToolUsageLog(Base):
    """Tool Usage Tracking for MCP Tools"""
    __tablename__ = 'tool_usage_logs'
    __table_args__ = {'extend_existing': True}  # Allow extension by tools.py
    
    id = Column(Integer, primary_key=True)
    execution_id = Column(Integer, ForeignKey('workflow_executions.id'))
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=False)
    tool_id = Column(Integer, ForeignKey('mcp_tools.id'), nullable=False)
    method_called = Column(String(255))
    input_data = Column(JSON)
    output_data = Column(JSON)
    success = Column(Boolean)
    execution_time_ms = Column(Integer)
    error_message = Column(Text)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    tool = relationship("MCPTool", back_populates="usage_logs")

class Workflow(Base):
    __tablename__ = 'workflows'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    goal = Column(Text, nullable=True)  # High-level objective (9-stage enhancement)
    context = Column(Text, nullable=True)  # Additional context (9-stage enhancement)
    workflow_definition = Column(JSON)  # Workflow steps and logic
    status = Column(String(50), default='draft')  # 'draft', 'active', 'archived'
    owner = Column(String(255), nullable=True)
    tags = Column(JSON, nullable=True)
    default_policy_id = Column(String(128), nullable=True)
    last_execution = Column(JSON, nullable=True)  # Latest execution summary (9-stage enhancement)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    priority = Column(String(50), nullable=True)  # Workflow priority (9-stage enhancement)
    expected_duration = Column(Integer, nullable=True)  # Expected duration in seconds (9-stage enhancement)
    complexity_score = Column(Float, nullable=True)  # Workflow complexity (9-stage enhancement)
    success_rate = Column(Float, nullable=True)  # Success rate (9-stage enhancement)
    
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
    
    # PRD-15: Track models used in workflow execution
    models_used = Column(JSON, default=list)  # [{"agent_id": 5, "model_id": "gpt-4", "input_tokens": 1500, "output_tokens": 800, "cost": 0.051}]
    
    # 9-Stage enhancement: Execution metadata for analytics
    execution_metadata = Column(JSON, default=dict, nullable=True)
    
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
    tags = Column(PG_ARRAY(String), default=list)  # Fixed: Use PostgreSQL ARRAY instead of JSON to match DB schema
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
    CUSTOM = "custom"
    SYSTEM = "system"
    SPECIALIZED = "specialized"
    CODE_ARCHITECT = "code_architect"
    SECURITY_EXPERT = "security_expert"
    PERFORMANCE_OPTIMIZER = "performance_optimizer"
    DATA_ANALYST = "data_analyst"
    INFRASTRUCTURE_MANAGER = "infrastructure_manager"

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
    tool_ids: Optional[List[int]] = []  # NEW: Phase 3 - Tools
    tags: Optional[List[str]] = []
    priority_level: Optional[PriorityLevel] = None
    max_concurrent_tasks: Optional[int] = 5
    auto_start: Optional[bool] = False

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AgentStatus] = None
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None
    tags: Optional[List[str]] = None

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    agent_type: str
    status: str
    configuration: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]] = None
    tags: List[str] = []
    priority_level: str
    max_concurrent_tasks: int
    auto_start: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    skills: List[Dict[str, Any]] = []
    tools: List[Dict[str, Any]] = []  # Phase 3: MCP Tools assigned to agent
    agent_model_config: Optional[Dict[str, Any]] = None  # PRD-15: Model configuration (renamed from model_config - Pydantic reserved)

class SkillCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    skill_type: SkillType
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None

class SkillUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    implementation: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

class SkillResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    skill_type: str
    implementation: Optional[str] = ""
    parameters: Optional[Dict[str, Any]] = None
    performance_data: Optional[Dict[str, Any]] = {}
    is_active: bool = True
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = ""

# ===================================================================
# MCP TOOLS PYDANTIC MODELS (Phase 3: Skills & Tools Integration)
# ===================================================================

class MCPToolBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    mcp_server_url: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = {}
    credentials_schema: Optional[Dict[str, Any]] = {}
    status: Optional[str] = 'active'
    provider: Optional[str] = None
    version: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    tool_metadata: Optional[Dict[str, Any]] = Field(default={}, alias='metadata')  # Use alias for API compatibility

class MCPToolCreate(MCPToolBase):
    pass

class MCPToolUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    mcp_server_url: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    credentials_schema: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    provider: Optional[str] = None
    version: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None  # Fixed: should be List[str] not str
    tool_metadata: Optional[Dict[str, Any]] = Field(default=None, alias='metadata')

class MCPToolResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    mcp_server_url: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = {}
    credentials_schema: Optional[Dict[str, Any]] = {}
    status: Optional[str] = 'active'
    provider: Optional[str] = None
    version: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    # Use Field with validation_alias to map from SQLAlchemy attribute
    metadata: Optional[Dict[str, Any]] = Field(default={}, validation_alias='tool_metadata')
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    
    class Config:
        from_attributes = True
        populate_by_name = True

class AgentToolAssignmentCreate(BaseModel):
    tool_id: int
    enabled: bool = True
    permissions: Optional[Dict[str, Any]] = {}
    configuration: Optional[Dict[str, Any]] = {}

class AgentToolAssignmentResponse(BaseModel):
    id: int
    agent_id: int
    tool_id: int
    enabled: bool
    permissions: Dict[str, Any]
    configuration: Dict[str, Any]
    assigned_at: datetime
    tool: MCPToolResponse
    
    class Config:
        from_attributes = True

class PatternCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    pattern_type: str
    pattern_data: Dict[str, Any]

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
    created_by: Optional[str] = None

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    goal: Optional[str] = Field(None, description="High-level objective of the workflow (overrides description if provided)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for workflow execution (e.g., codegraph_project, pr_number, git_url)")
    workflow_definition: Dict[str, Any] = Field(default_factory=dict)
    agent_ids: Optional[List[int]] = []
    owner: Optional[str] = None
    tags: Optional[List[str]] = []
    default_policy_id: Optional[str] = None

class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    workflow_definition: Optional[Dict[str, Any]] = None
    status: Optional[WorkflowStatus] = None
    agent_ids: Optional[List[int]] = None
    owner: Optional[str] = None
    tags: Optional[List[str]] = None
    default_policy_id: Optional[str] = None

class WorkflowResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    goal: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    workflow_definition: Dict[str, Any]
    status: str
    owner: Optional[str]
    tags: Optional[List[str]]
    default_policy_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str] = None
    agents: List[Dict[str, Any]] = []
    last_execution: Optional[Dict[str, Any]] = None  # Latest execution summary

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
    created_by: Optional[str] = None

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
    created_by: Optional[str] = None

class SystemHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    metrics: Dict[str, Any]
    version: str

class AgentTemplate(BaseModel):
    """Agent template for creation wizard"""
    name: str
    description: str
    agent_type: AgentType
    default_skills: List[str]
    default_configuration: Dict[str, Any]
    priority_level: PriorityLevel
    max_concurrent_tasks: int

# New models for advanced memory and tool systems

class Task(Base):
    """Task model with enhanced memory and tool execution support"""
    __tablename__ = 'tasks'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(50), nullable=False, default='pending')
    owner_id = Column(Integer, nullable=False)  # User reference
    
    # Memory system fields (from code review 05_memory_systems)
    immediate_memory = Column(JSON, nullable=True)     # Real-time context
    working_memory = Column(JSON, nullable=True)       # Active task context  
    short_term_memory = Column(JSON, nullable=True)    # Recent context
    long_term_memory = Column(JSON, nullable=True)     # Persistent context
    importance = Column(Float, nullable=True, default=0.5)  # Memory importance weight
    
    # Tool execution fields (from code review 06_tool_integrated_reasoning)
    tools = Column(JSON, nullable=True)                # Selected tools
    tool_scores = Column(JSON, nullable=True)          # Tool ranking scores
    dependencies = Column(JSON, nullable=True)         # Tool dependencies
    execution_status = Column(JSON, nullable=True)     # Tool execution results
    reasoning = Column(JSON, nullable=True)            # Reasoning steps
    
    # Augmentation fields (from code review 05_memory_systems)  
    augmented_memory = Column(JSON, nullable=True)     # External context
    similarity_score = Column(Float, nullable=True)    # Augmentation weight
    
    # Multi-agent system fields (from code review 07_multi_agent_systems)
    consensus_score = Column(Float, nullable=True)  # Consensus metric
    coordination = Column(JSON, nullable=True)  # Agent coordination plan
    optimization = Column(JSON, nullable=True)  # Agent optimization weights
    optimization_config = Column(JSON, nullable=True)  # Applied optimization configuration
    
    # Field theory integration fields (from code review 08_field_theory_integration)
    field_value = Column(Float, nullable=True)  # Scalar field value
    influence_weights = Column(JSON, nullable=True)  # Field weights
    gradient = Column(JSON, nullable=True)  # Propagation gradient
    field_timestamp = Column(DateTime, nullable=True)  # Last field update
    propagation_timestamp = Column(DateTime, nullable=True)  # Last propagation
    interactions = Column(JSON, nullable=True)  # Task-task interactions
    emergent_effect = Column(Float, nullable=True)  # Emergent field effect
    embeddings = Column(JSON, nullable=True)  # Cached embeddings
    stability = Column(Float, nullable=True)  # Field stability score
    prev_field_value = Column(Float, nullable=True)  # Previous field value for stability calc
    
    # Standard timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class User(Base):
    """User model for task ownership"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class MemoryItem(Base):
    """Memory items for the advanced memory system"""
    __tablename__ = 'memory_items'
    
    id = Column(String(255), primary_key=True)
    session_id = Column(String(255), nullable=False)
    content = Column(JSON, nullable=False)
    memory_type = Column(String(50), nullable=False)  # semantic, episodic, procedural, etc.
    memory_level = Column(String(50), nullable=False)  # immediate, working, short_term, long_term, archival
    
    importance = Column(Float, default=0.5)
    access_count = Column(Integer, default=0)
    decay_factor = Column(Float, default=0.1)
    consolidation_score = Column(Float, default=0.0)
    
    tags = Column(JSON, nullable=True)  # List of tags
    
    created_at = Column(DateTime, default=func.now())
    last_access = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class ExternalKnowledge(Base):
    """External knowledge for memory augmentation"""
    __tablename__ = 'external_knowledge'
    
    id = Column(Integer, primary_key=True)
    content = Column(JSON, nullable=False)
    source = Column(String(255), nullable=False, default='external')
    knowledge_metadata = Column(JSON, nullable=True)  # Renamed to avoid SQLAlchemy conflict
    access_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

# Enhanced Pydantic Models for new functionality

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    owner_id: int
    importance: Optional[float] = 0.5

class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    importance: Optional[float] = None
    
class TaskResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    status: str
    owner_id: int
    importance: float
    tools: Optional[List[str]]
    reasoning: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime

class MemoryItemCreate(BaseModel):
    session_id: str
    content: Dict[str, Any]
    memory_type: str = "working_data"
    importance: Optional[float] = 0.5
    tags: Optional[List[str]] = []

class MemoryItemResponse(BaseModel):
    id: str
    session_id: str
    content: Dict[str, Any]
    memory_type: str
    memory_level: str
    importance: float
    access_count: int
    tags: Optional[List[str]]
    created_at: datetime
    last_access: datetime

class ExternalKnowledgeCreate(BaseModel):
    content: Dict[str, Any]
    source: str = "external"
    knowledge_metadata: Optional[Dict[str, Any]] = None

class ExternalKnowledgeResponse(BaseModel):
    id: int
    content: Dict[str, Any]
    source: str
    knowledge_metadata: Optional[Dict[str, Any]]
    access_count: int
    created_at: datetime

class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

# Evaluation-specific tables for enhanced assessment methodologies
class EvaluationResult(Base):
    __tablename__ = 'evaluation_results'
    
    id = Column(Integer, primary_key=True)
    evaluation_id = Column(String(255), nullable=False, unique=True)
    evaluation_type = Column(String(100), nullable=False)  # 'system_quality', 'component_assessment', etc.
    scope = Column(String(100), nullable=False)  # 'single_task', 'component', 'system', 'enterprise'
    target_id = Column(String(255), nullable=False)  # ID of evaluated entity
    overall_score = Column(Float, nullable=False)
    detailed_results = Column(JSON)  # Detailed evaluation data
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    execution_time_seconds = Column(Float, nullable=True)
    user_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=func.now())

class BenchmarkAssessment(Base):
    __tablename__ = 'benchmark_assessments'
    
    id = Column(Integer, primary_key=True)
    benchmark_id = Column(String(255), nullable=False)
    benchmark_name = Column(String(255), nullable=False)
    benchmark_type = Column(String(100), nullable=False)  # 'performance', 'quality', 'efficiency'
    validity_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    discriminatory_power = Column(Float, nullable=True)
    overall_quality = Column(Float, nullable=True)
    quality_classification = Column(String(50), nullable=True)
    assessment_data = Column(JSON)  # Detailed assessment results
    recommendations = Column(JSON)  # List of improvement recommendations
    created_at = Column(DateTime, default=func.now())

class ComponentMetricsDB(Base):
    __tablename__ = 'component_metrics'
    
    id = Column(Integer, primary_key=True)
    component_id = Column(String(255), nullable=False)
    component_type = Column(String(100), nullable=False)  # 'orchestrator', 'agent', 'workflow'
    performance_score = Column(Float, nullable=True)
    reliability_score = Column(Float, nullable=True)
    readiness_score = Column(Float, nullable=True)
    capability_rating = Column(Float, nullable=True)
    complexity_index = Column(Float, nullable=True)
    environment_factor = Column(Float, nullable=True)
    assessment_details = Column(JSON)  # Detailed metrics
    assessment_timestamp = Column(DateTime, default=func.now())

class IntegrationAnalysisDB(Base):
    __tablename__ = 'integration_analyses'
    
    id = Column(Integer, primary_key=True)
    system_id = Column(String(255), nullable=False)
    coherence_score = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)
    emergence_score = Column(Float, nullable=True)
    integration_score = Column(Float, nullable=True)
    integration_classification = Column(String(50), nullable=True)
    analysis_data = Column(JSON)  # Detailed analysis results
    recommendations = Column(JSON)  # Integration improvement recommendations
    confidence_level = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now())


# ===================================================================
# PRD-27: CHAT MODELS
# ===================================================================

class Chat(Base):
    """Chat session with user (PRD-27)"""
    __tablename__ = 'chats'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    updated_at = Column(DateTime, nullable=False, server_default=func.now(), onupdate=func.now())
    visibility = Column(String(20), default='private', nullable=False)
    last_context = Column(JSONB, default=dict, server_default='{}')
    
    # Relationships
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan")
    votes = relationship("Vote", back_populates="chat", cascade="all, delete-orphan")
    user = relationship("User", backref="chats")
    
    __table_args__ = (
        CheckConstraint("visibility IN ('private', 'public')", name='check_chat_visibility'),
        {'extend_existing': True}
    )


class Message(Base):
    """Individual message within a chat (PRD-27)"""
    __tablename__ = 'messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    chat_id = Column(UUID(as_uuid=True), ForeignKey('chats.id', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False)
    parts = Column(JSONB, nullable=False, default=list, server_default='[]')
    attachments = Column(JSONB, default=list, server_default='[]')
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    chat = relationship("Chat", back_populates="messages")
    votes = relationship("Vote", back_populates="message", cascade="all, delete-orphan")
    
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name='check_message_role'),
        {'extend_existing': True}
    )


class Vote(Base):
    """User vote on assistant messages (PRD-27)"""
    __tablename__ = 'votes'
    
    chat_id = Column(UUID(as_uuid=True), ForeignKey('chats.id', ondelete='CASCADE'), primary_key=True)
    message_id = Column(UUID(as_uuid=True), ForeignKey('messages.id', ondelete='CASCADE'), primary_key=True)
    is_upvoted = Column(Boolean, nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    
    # Relationships
    chat = relationship("Chat", back_populates="votes")
    message = relationship("Message", back_populates="votes")


class Artifact(Base):
    """Versioned artifacts (code, documents, images) (PRD-27)"""
    __tablename__ = 'artifacts'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    created_at = Column(DateTime, nullable=False, primary_key=True, server_default=func.now())
    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))
    title = Column(String(255), nullable=False)
    content = Column(Text)
    kind = Column(String(20), nullable=False)
    artifact_metadata = Column('metadata', JSONB, default=dict, server_default='{}')
    
    # Relationships
    user = relationship("User", backref="artifacts")
    
    __table_args__ = (
        CheckConstraint("kind IN ('code', 'text', 'image', 'sheet')", name='check_artifact_kind'),
        {'extend_existing': True}
    )


class WorkflowTemplate(Base):
    """
    Workflow templates that users can use to quickly create workflows.
    Templates include pre-configured steps, agents, and settings.
    """
    __tablename__ = 'workflow_templates'
    
    # Primary identification
    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "ai-code-review"
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    
    # Categorization
    category = Column(String(100), nullable=False, index=True)  # Development, Data Processing, etc.
    tags = Column(JSONB, default=list)  # ["code-review", "security", "automation"]
    difficulty = Column(String(50), default='intermediate')  # beginner, intermediate, advanced
    
    # Template definition
    template_definition = Column(JSONB, nullable=False)  # Full workflow structure
    # Contains: { steps: [], agents: [], config: {}, variables: [] }
    
    # Recommended configuration
    recommended_agents = Column(JSONB, default=list)  # Agent types that work well with this template
    estimated_time = Column(String(50))  # "5-10 minutes"
    required_tools = Column(JSONB, default=list)  # Tools that must be installed
    
    # Usage and popularity
    use_count = Column(Integer, default=0)  # How many workflows created from this template
    success_rate = Column(Float, default=0.0)  # Average success rate of workflows using this template
    popularity = Column(Integer, default=0)  # Popularity score (0-100)
    average_rating = Column(Float, default=0.0)  # User ratings (0-5)
    
    # Visibility and access
    is_public = Column(Boolean, default=True)  # Public templates visible to all users
    is_featured = Column(Boolean, default=False)  # Featured on templates page
    is_system = Column(Boolean, default=False)  # System templates (can't be deleted)
    
    # Metadata
    icon = Column(String(50))  # Emoji or icon identifier
    preview_image = Column(String(500))  # URL to preview image
    documentation_url = Column(String(500))  # Link to detailed documentation
    
    # Versioning
    version = Column(String(50), default='1.0')
    changelog = Column(JSONB, default=list)  # Version history
    
    # Timestamps and ownership
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False)
    last_used_at = Column(DateTime)
    
    def to_dict(self):
        """Convert template to dictionary for API responses"""
        return {
            'id': self.id,
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'tags': self.tags or [],
            'difficulty': self.difficulty,
            'template_definition': self.template_definition or {},
            'recommended_agents': self.recommended_agents or [],
            'estimated_time': self.estimated_time,
            'required_tools': self.required_tools or [],
            'use_count': self.use_count,
            'success_rate': self.success_rate,
            'popularity': self.popularity,
            'average_rating': self.average_rating,
            'is_public': self.is_public,
            'is_featured': self.is_featured,
            'is_system': self.is_system,
            'icon': self.icon,
            'preview_image': self.preview_image,
            'documentation_url': self.documentation_url,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }
