
"""
Database Models for Automotas AI System
=======================================

Comprehensive data models for agents, skills, workflows, documents, and system configuration.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Table, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, JSONB, UUID
# Base moved to core/database/base.py to avoid circular imports
from core.database.base import Base
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

    # PRD-54: Marketplace & multi-tier fields
    tier = Column(String(50), default='direct')  # direct, aggregator, byok_only
    tags = Column(JSON, default=list)
    category = Column(String(100))  # general, coding, creative, fast, premium
    popularity_score = Column(Integer, default=0)
    install_count = Column(Integer, default=0)
    avg_rating = Column(Float)
    is_featured = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)  # included in all workspaces
    requires_plan = Column(String(50))  # NULL (all plans), pro, enterprise
    external_id = Column(String(255))  # OpenRouter model ID
    pricing_updated_at = Column(DateTime)

    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # PRD-37: Workspace isolation (nullable for system-wide models)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=True)

    # Relationships
    workspace_installs = relationship("WorkspaceModel", back_populates="model", cascade="all, delete-orphan")


class WorkspaceModel(Base):
    """Tracks which models a workspace has installed from the marketplace."""
    __tablename__ = 'workspace_models'

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    model_id = Column(Integer, ForeignKey('llm_models.id', ondelete='CASCADE'), nullable=False)
    installed_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    source = Column(String(50), default='marketplace')  # default, marketplace, admin

    model = relationship("LLMModel", back_populates="workspace_installs")

    __table_args__ = (
        # Each workspace can install a model only once
        UniqueConstraint('workspace_id', 'model_id', name='uq_workspace_model'),
    )


class UserApiKey(Base):
    """BYOK (Bring Your Own Key) — user-provided API keys per workspace."""
    __tablename__ = 'user_api_keys'

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
    provider = Column(String(50), nullable=False)  # openai, anthropic, google, openrouter
    encrypted_key = Column(Text, nullable=False)
    display_name = Column(String(255))
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LLMUsage(Base):
    """Per-request LLM usage tracking for analytics and billing."""
    __tablename__ = 'llm_usage'

    id = Column(Integer, primary_key=True)  # will be BIGSERIAL in migration
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=False, index=True)
    model_id = Column(String(255), nullable=False, index=True)
    provider = Column(String(100), nullable=False)
    tier = Column(String(50), nullable=False)  # direct, aggregator, byok

    # Request details
    agent_id = Column(Integer)
    execution_id = Column(String(255))
    request_type = Column(String(50))  # chat, agent, recipe, routing, embedding

    # Token usage
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)

    # Cost
    input_cost = Column(Float, nullable=False)
    output_cost = Column(Float, nullable=False)
    total_cost = Column(Float, nullable=False)
    is_byok = Column(Boolean, default=False)

    # Performance
    latency_ms = Column(Integer)
    status = Column(String(50))  # success, error, timeout, rate_limited
    error_message = Column(Text)

    created_at = Column(DateTime, default=func.now())

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
    
    # Ownership & Visibility (New)
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    is_shared = Column(Boolean, default=True)  # True = visible to workspace, False = private to creator

    # PRD-37: Workspace isolation (nullable for marketplace agents)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=True)

    # Marketplace fields (PRD-48: Single-table architecture)
    owner_type = Column(String(20), nullable=False, default='workspace', server_default='workspace')
    owner_id = Column(String(255), nullable=True)

    cloned_from_id = Column(Integer, ForeignKey('agents.id', ondelete='SET NULL'), nullable=True)
    original_creator_id = Column(Integer, ForeignKey('users.id'), nullable=True)

    install_count = Column(Integer, default=0, server_default='0')
    is_featured = Column(Boolean, default=False, server_default='false')
    is_approved = Column(Boolean, default=False, server_default='false')

    marketplace_category = Column(String(100), nullable=True)
    marketplace_icon = Column(String(500), nullable=True)
    version = Column(String(50), default='1.0.0', server_default='1.0.0')

    # PRD-42: Persona assignment
    persona_id = Column(UUID(as_uuid=True), ForeignKey('personas.id'), nullable=True, index=True)
    custom_persona_prompt = Column(Text, nullable=True)
    use_custom_persona = Column(Boolean, default=False, server_default='false')

    # PRD-67: CTO Agent / System agents
    is_system_agent = Column(Boolean, default=False, server_default='false')  # Global, platform-seeded
    required_role = Column(String(50), nullable=True)  # If set, only visible to users with this system_role
    slug = Column(String(100), nullable=True, unique=True)  # Stable ID for system agents (idempotent seeding)

    # PRD-64: Semantic routing embedding
    semantic_embedding = Column(JSONB, nullable=True)       # 2048-float vector for cosine similarity
    semantic_text_hash = Column(String(64), nullable=True)  # SHA-256 for staleness detection

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
    # Tool/app assignments are managed via `AgentAppAssignment` (Composio cache model).

    # Marketplace relationships
    cloned_from = relationship("Agent", remote_side=[id], foreign_keys=[cloned_from_id], backref="clones")
    original_creator = relationship("User", foreign_keys=[original_creator_id])

    # PRD-42: Persona relationship
    persona = relationship("Persona", foreign_keys=[persona_id])

    # PRD-42: Plugin assignments (marketplace plugins assigned to this agent)
    assigned_plugins = relationship(
        "AgentAssignedPlugin",
        cascade="all, delete-orphan",
        order_by="AgentAssignedPlugin.priority.asc()",
    )

    @property
    def is_marketplace_item(self) -> bool:
        """Check if this agent is a marketplace item."""
        return self.owner_type == 'marketplace'

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

    # PRD-37: Workspace isolation for multi-tenant SaaS
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=False)

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

    # PRD-37: Workspace isolation for multi-tenant SaaS
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=False)

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
    
    # Ownership & Visibility (New)
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    is_shared = Column(Boolean, default=True)

    priority = Column(String(50), nullable=True)  # Workflow priority (9-stage enhancement)
    expected_duration = Column(Integer, nullable=True)  # Expected duration in seconds (9-stage enhancement)
    complexity_score = Column(Float, nullable=True)  # Workflow complexity (9-stage enhancement)
    success_rate = Column(Float, nullable=True)  # Success rate (9-stage enhancement)
    
    # PRD-37: Workspace isolation
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id"), nullable=False)
    
    # Relationships
    agents = relationship("Agent", secondary=workflow_agents, back_populates="workflows")
    executions = relationship("WorkflowExecution", back_populates="workflow")

class WorkflowExecution(Base):
    __tablename__ = 'workflow_executions'
    
    id = Column(Integer, primary_key=True)
    workflow_id = Column(Integer, ForeignKey('workflows.id'))
    agent_id = Column(Integer, ForeignKey('agents.id'))
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=False)
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
    
    # Ownership & Visibility (New)
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    is_shared = Column(Boolean, default=True)
    
    # PRD-37: Workspace isolation for multi-tenant SaaS
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id'), nullable=False)

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
    CANCELLED = "cancelled"

# API Request/Response Models
class AgentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    agent_type: str  # Flexible agent type - no enum restriction
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = []
    tool_ids: Optional[List[int]] = []  # NEW: Phase 3 - Tools
    tags: Optional[List[str]] = []
    marketplace_category: Optional[str] = None

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    agent_type: Optional[str] = None  # Allow updating agent type/category
    status: Optional[AgentStatus] = None
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None
    tool_ids: Optional[List[int]] = None  # NEW: Allow updating tool assignments
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
    tools: List[Dict[str, Any]] = []  # Assigned apps/integrations (Composio)
    plugins: List[Dict[str, Any]] = []  # Assigned marketplace plugins
    agent_model_config: Optional[Dict[str, Any]] = None  # PRD-15: Model configuration (renamed from model_config - Pydantic reserved)
    model_usage_stats: Optional[Dict[str, Any]] = None  # PRD-54: LLM usage stats
    # PRD-67: System agent fields
    is_system_agent: bool = False
    slug: Optional[str] = None
    required_role: Optional[str] = None

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
    embedding_model: Optional[str] = None  # Resolved from system_settings at runtime
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
    """User model with Clerk authentication (PRD-37: SaaS Foundation)"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(255), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    
    # PRD-37: Clerk authentication fields
    clerk_user_id = Column(String(255), unique=True, nullable=True)
    name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    last_sign_in = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
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
    
    # Ownership & Visibility (New)
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    is_shared = Column(Boolean, default=True)

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
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False)
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
        CheckConstraint("kind IN ('code', 'text', 'image', 'sheet', 'document')", name='check_artifact_kind'),
        {'extend_existing': True}
    )


class WorkflowTemplate(Base):
    """
    Workflow templates that users can use to quickly create workflows.
    Templates include pre-configured steps, agents, and settings.

    Uses single-table architecture: both workspace and marketplace recipes
    are stored here, distinguished by owner_type.
    """
    __tablename__ = 'workflow_recipes'  # Renamed from workflow_templates

    # Primary identification
    id = Column(Integer, primary_key=True)
    template_id = Column(String(100), unique=True, nullable=False, index=True)  # e.g., "ai-code-review"
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)

    # Ownership and workspace isolation (single-table architecture)
    workspace_id = Column(UUID, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=True, index=True)
    owner_type = Column(String(20), nullable=False, default='workspace', index=True)  # 'workspace' or 'marketplace'
    owner_id = Column(String(255))  # workspace_id for workspace recipes, 'marketplace' for marketplace

    # Cloning and attribution
    cloned_from_id = Column(Integer, ForeignKey('workflow_recipes.id', ondelete='SET NULL'), nullable=True)
    original_creator_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)
    created_by_user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True)

    # Categorization
    tags = Column(JSONB, default=list)  # ["code-review", "security", "automation"]

    # Template definition
    template_definition = Column(JSONB, nullable=False)  # Full workflow structure
    # Contains: { steps: [], agents: [], config: {}, variables: [] }

    # Workflow definition (new 9-stage architecture)
    steps = Column(JSONB, nullable=True)  # Array of step definitions: [{ step_number, agent_id, prompt_template, pass_to, error_handling }]
    inputs = Column(JSONB, nullable=True)  # Input schema like function signature: { "param": { "type": "string", "required": true } }
    outputs = Column(JSONB, nullable=True)  # Expected output format: { "field": { "type": "string" } }
    execution_config = Column(JSONB, nullable=True)  # Runtime behavior: { mode, max_retries, timeout_per_step, quality_threshold }
    schedule_config = Column(JSONB, nullable=True)  # Scheduling: { type: "manual"|"cron"|"trigger", cron_expression, trigger_config }

    # Self-learning fields (Learn-Quality-Memory stages)
    quality_score = Column(Float, nullable=True)  # Average quality from Stage 7 assessments (0.0 - 1.0)
    learning_data = Column(JSONB, nullable=True)  # Patterns, suggestions, performance metrics from Stage 6

    # Recommended configuration
    recommended_agents = Column(JSONB, default=list)  # Agent types that work well with this template
    required_tools = Column(JSONB, default=list)  # Tools that must be installed

    # Usage and popularity
    use_count = Column(Integer, default=0)  # How many workflows created from this template
    success_rate = Column(Float, default=0.0)  # Average success rate of workflows using this template
    popularity = Column(Integer, default=0)  # Popularity score (0-100)
    average_rating = Column(Float, default=0.0)  # User ratings (0-5)
    install_count = Column(Integer, default=0)  # Marketplace install counter

    # Visibility and access
    is_public = Column(Boolean, default=True)  # Public templates visible to all users
    is_featured = Column(Boolean, default=False)  # Featured on templates page
    is_system = Column(Boolean, default=False)  # System templates (can't be deleted)
    is_approved = Column(Boolean, default=False)  # Approval status for marketplace items

    # Metadata
    preview_image = Column(String(500))  # URL to preview image
    documentation_url = Column(String(500))  # Link to detailed documentation
    marketplace_category = Column(String(100))  # Marketplace-specific category
    marketplace_icon = Column(String(500))  # Icon URL for marketplace

    # Versioning
    version = Column(String(50), default='1.0')
    changelog = Column(JSONB, default=list)  # Version history

    # Timestamps and ownership
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(255), nullable=False)
    last_used_at = Column(DateTime)

    # Relationships
    workspace = relationship('Workspace', foreign_keys=[workspace_id])
    cloned_from = relationship('WorkflowTemplate', remote_side=[id], foreign_keys=[cloned_from_id])
    original_creator = relationship('User', foreign_keys=[original_creator_id])
    created_by_user = relationship('User', foreign_keys=[created_by_user_id])

    @property
    def is_marketplace_item(self):
        """Check if this is a marketplace recipe"""
        return self.owner_type == 'marketplace'

    def validate_steps(self):
        """Validate steps array structure"""
        if not self.steps:
            return True, None

        if not isinstance(self.steps, list):
            return False, "steps must be an array"

        for idx, step in enumerate(self.steps):
            if not isinstance(step, dict):
                return False, f"Step {idx} must be an object"

            required_fields = ['step_id', 'order', 'agent_id', 'prompt_template']
            for field in required_fields:
                if field not in step:
                    return False, f"Step {idx} missing required field: {field}"

        return True, None

    def validate_execution_config(self):
        """Validate execution_config structure"""
        if not self.execution_config:
            return True, None

        if not isinstance(self.execution_config, dict):
            return False, "execution_config must be an object"

        # Validate mode
        if 'mode' in self.execution_config:
            if self.execution_config['mode'] not in ['sequential', 'parallel']:
                return False, "mode must be 'sequential' or 'parallel'"

        # Validate numeric fields
        numeric_fields = ['max_retries', 'timeout_per_step', 'total_timeout', 'parallel_limit']
        for field in numeric_fields:
            if field in self.execution_config:
                if not isinstance(self.execution_config[field], (int, float)) or self.execution_config[field] < 0:
                    return False, f"{field} must be a non-negative number"

        return True, None

    def validate_schedule_config(self):
        """Validate schedule_config structure"""
        if not self.schedule_config:
            return True, None

        if not isinstance(self.schedule_config, dict):
            return False, "schedule_config must be an object"

        # Validate type
        if 'type' not in self.schedule_config:
            return False, "schedule_config must have 'type' field"

        schedule_type = self.schedule_config['type']
        if schedule_type not in ['manual', 'cron', 'trigger']:
            return False, "type must be 'manual', 'cron', or 'trigger'"

        # Validate cron expression if type is cron
        if schedule_type == 'cron' and 'cron_expression' not in self.schedule_config:
            return False, "cron type requires cron_expression field"

        # Validate trigger config if type is trigger
        if schedule_type == 'trigger' and 'trigger_config' not in self.schedule_config:
            return False, "trigger type requires trigger_config field"

        return True, None

    # Relationship to executions
    recipe_executions = relationship('RecipeExecution', back_populates='recipe', cascade='all, delete-orphan')

    def to_dict(self):
        """Convert template to dictionary for API responses"""
        return {
            'id': self.id,
            'template_id': self.template_id,
            'name': self.name,
            'description': self.description,
            'workspace_id': str(self.workspace_id) if self.workspace_id else None,
            'owner_type': self.owner_type,
            'owner_id': self.owner_id,
            'cloned_from_id': self.cloned_from_id,
            'original_creator_id': self.original_creator_id,
            'created_by_user_id': self.created_by_user_id,
            'tags': self.tags or [],
            'template_definition': self.template_definition or {},
            'steps': self.steps,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'execution_config': self.execution_config,
            'schedule_config': self.schedule_config,
            'quality_score': self.quality_score,
            'learning_data': self.learning_data,
            'recommended_agents': self.recommended_agents or [],
            'required_tools': self.required_tools or [],
            'use_count': self.use_count,
            'success_rate': self.success_rate,
            'popularity': self.popularity,
            'average_rating': self.average_rating,
            'install_count': self.install_count,
            'is_public': self.is_public,
            'is_featured': self.is_featured,
            'is_system': self.is_system,
            'is_approved': self.is_approved,
            'is_marketplace_item': self.is_marketplace_item,
            'preview_image': self.preview_image,
            'documentation_url': self.documentation_url,
            'marketplace_category': self.marketplace_category,
            'marketplace_icon': self.marketplace_icon,
            'version': self.version,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'created_by': self.created_by,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
        }


# ===================================================================
# PRD-63: Document Generation Module
# ===================================================================

class DocumentTemplate(Base):
    """Document templates for PDF, DOCX, XLSX generation (PRD-63)"""
    __tablename__ = 'document_templates'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    format = Column(String(20), nullable=False)

    # Template content
    template_content = Column(Text, nullable=True)  # HTML/CSS for PDF templates
    template_file_path = Column(String(500), nullable=True)  # Path to .docx template file

    # Schema definition: what variables the template expects
    data_schema = Column(JSONB, nullable=False, default=dict, server_default='{}')

    # Sample data for preview
    sample_data = Column(JSONB, default=dict, server_default='{}')

    # Metadata
    category = Column(String(100), default='general', server_default='general')
    tags = Column(PG_ARRAY(String), default=list)
    thumbnail_url = Column(String(500), nullable=True)

    # Versioning
    version = Column(Integer, default=1, server_default='1')
    is_active = Column(Boolean, default=True, server_default='true')

    # Audit
    created_by = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=func.now(), server_default=func.now())
    updated_at = Column(DateTime, default=func.now(), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("format IN ('pdf', 'docx', 'xlsx')", name='check_document_template_format'),
        UniqueConstraint('workspace_id', 'name', 'version', name='uq_template_workspace_name_version'),
        {'extend_existing': True}
    )


class RecipeExecution(Base):
    """
    Tracks individual recipe executions.
    Each execution record stores input/output data, step-level results,
    current stage in the 9-stage workflow, and status.
    """
    __tablename__ = 'recipe_executions'

    id = Column(Integer, primary_key=True)
    execution_id = Column(String(255), unique=True, nullable=False, index=True)
    recipe_id = Column(Integer, ForeignKey('workflow_recipes.id', ondelete='CASCADE'), nullable=False, index=True)
    workspace_id = Column(UUID, ForeignKey('workspaces.id', ondelete='CASCADE'), nullable=False, index=True)
    status = Column(String(50), nullable=False, default='pending', server_default='pending')
    input_data = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    step_results = Column(JSONB, nullable=True)
    current_stage = Column(Integer, nullable=True)
    current_step = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)
    triggered_by = Column(String(255), nullable=True)
    execution_metadata = Column(JSONB, nullable=True)

    # Relationships
    recipe = relationship('WorkflowTemplate', back_populates='recipe_executions')

    def to_dict(self):
        """Convert execution to dictionary for API responses"""
        return {
            'id': self.id,
            'execution_id': self.execution_id,
            'recipe_id': self.recipe_id,
            'workspace_id': str(self.workspace_id) if self.workspace_id else None,
            'status': self.status,
            'input_data': self.input_data,
            'output_data': self.output_data,
            'step_results': self.step_results,
            'current_stage': self.current_stage,
            'current_step': self.current_step,
            'error_message': self.error_message,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'triggered_by': self.triggered_by,
            'execution_metadata': self.execution_metadata,
        }
