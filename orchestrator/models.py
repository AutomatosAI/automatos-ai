
"""
Database Models for Automotas AI System
=======================================

Comprehensive data models for agents, skills, workflows, documents, and system configuration.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, Float, JSON, ForeignKey, Table
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

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
    
    # Relationships
    skills = relationship("Skill", secondary=agent_skills, back_populates="agents")
    workflows = relationship("Workflow", secondary=workflow_agents, back_populates="agents")
    executions = relationship("WorkflowExecution", back_populates="agent")

class Skill(Base):
    __tablename__ = 'skills'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    skill_type = Column(String(100), nullable=False)  # 'cognitive', 'technical', 'communication'
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
    workflow_definition = Column(JSON)  # Workflow steps and logic
    status = Column(String(50), default='draft')  # 'draft', 'active', 'archived'
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by = Column(String(255))
    
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
    CUSTOM = "custom"
    SYSTEM = "system"
    SPECIALIZED = "specialized"

class SkillType(str, Enum):
    COGNITIVE = "cognitive"
    TECHNICAL = "technical"
    COMMUNICATION = "communication"

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
    agent_type: AgentType
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = []

class AgentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[AgentStatus] = None
    configuration: Optional[Dict[str, Any]] = None
    skill_ids: Optional[List[int]] = None

class AgentResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    agent_type: str
    status: str
    configuration: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]
    skills: List[Dict[str, Any]] = []

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
    description: Optional[str]
    skill_type: str
    implementation: Optional[str]
    parameters: Optional[Dict[str, Any]]
    performance_data: Optional[Dict[str, Any]]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    created_by: Optional[str]

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
    created_by: Optional[str]

class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
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
    chunk_count: int
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
