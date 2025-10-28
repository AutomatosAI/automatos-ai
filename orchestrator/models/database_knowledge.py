"""
Database Knowledge Source Models
=================================
PRD-21: Database models for text-to-SQL with semantic layer

Multi-tenant isolated database knowledge sources with:
- Encrypted credential references
- Schema caching with Redis
- Semantic layer (metrics/dimensions)
- Query audit trail
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, JSON, ForeignKey, Float, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum

from models import Base


# ============================================================================
# SQLAlchemy Database Models
# ============================================================================

class DatabaseKnowledgeSource(Base):
    """
    Database knowledge source with full multi-tenant isolation.
    Each tenant can only see and query their own database sources.
    """
    __tablename__ = 'database_knowledge_sources'
    __table_args__ = (
        Index('idx_dks_tenant_name', 'tenant_id', 'name'),
        Index('idx_dks_credential', 'credential_id'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, nullable=False, index=True)  # Multi-tenant isolation
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Credential reference (uses existing credential system by name)
    credential_name = Column(String(255), nullable=False)
    credential_environment = Column(String(50), default='production')
    
    # Database configuration
    dialect = Column(String(50), nullable=False)  # postgresql, mysql, snowflake, etc.
    connection_pool_size = Column(Integer, default=5)
    max_rows_limit = Column(Integer, default=10000)
    query_timeout_seconds = Column(Integer, default=30)
    
    # Schema metadata (cached)
    schema_metadata = Column(JSON)  # Tables, columns, types, relationships
    schema_version = Column(Integer, default=1)
    schema_hash = Column(String(64))  # Detect schema changes
    last_introspected = Column(DateTime)
    
    # Semantic layer configuration
    semantic_layer = Column(JSON)  # Metrics, dimensions, relationships
    
    # Caching configuration
    schema_cache_ttl = Column(Integer, default=3600)  # 1 hour default
    query_cache_ttl = Column(Integer, default=300)   # 5 minutes default
    
    # Statistics
    total_queries_executed = Column(Integer, default=0)
    avg_query_time_ms = Column(Float)
    last_successful_query = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)
    status = Column(String(50), default='active')  # active, error, maintenance
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    queries = relationship("DatabaseQueryAudit", back_populates="source")
    relationships = relationship("DatabaseRelationship", back_populates="source")
    templates = relationship("DatabaseQueryTemplate", back_populates="source")


class DatabaseRelationship(Base):
    """
    Table relationships for JOIN optimization and schema understanding
    """
    __tablename__ = 'database_relationships'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('database_knowledge_sources.id'), nullable=False)
    source = relationship("DatabaseKnowledgeSource", back_populates="relationships")
    
    from_table = Column(String(255), nullable=False)
    from_column = Column(String(255), nullable=False)
    to_table = Column(String(255), nullable=False)
    to_column = Column(String(255), nullable=False)
    
    relationship_type = Column(String(50))  # one-to-one, one-to-many, many-to-many
    is_inferred = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=func.now())


class DatabaseQueryAudit(Base):
    """
    Audit trail for all database queries (multi-tenant isolated)
    """
    __tablename__ = 'database_query_audit'
    __table_args__ = (
        Index('idx_dqa_tenant_source', 'tenant_id', 'source_id'),
        Index('idx_dqa_user', 'user_id'),
        Index('idx_dqa_created', 'created_at'),
        {'extend_existing': True}
    )
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, nullable=False, index=True)  # Multi-tenant isolation
    
    source_id = Column(Integer, ForeignKey('database_knowledge_sources.id'), nullable=False)
    source = relationship("DatabaseKnowledgeSource", back_populates="queries")
    
    user_id = Column(Integer, index=True)
    agent_id = Column(String(255))
    session_id = Column(String(255))
    
    # Query details
    natural_language_query = Column(Text, nullable=False)
    generated_sql = Column(Text)
    validated_sql = Column(Text)
    
    # Execution metrics
    execution_time_ms = Column(Integer)
    row_count = Column(Integer)
    bytes_processed = Column(Integer)
    
    # Status
    success = Column(Boolean, nullable=False)
    error_message = Column(Text)
    validation_errors = Column(JSON)
    
    # Caching
    was_cached = Column(Boolean, default=False)
    cache_key = Column(String(64))
    
    # Metadata
    visualization_type = Column(String(50))  # table, bar, line, pie
    confidence_score = Column(Float)
    
    created_at = Column(DateTime, default=func.now())


class SemanticMetric(Base):
    """
    Business metrics defined in semantic layer (PROMINENT UI FEATURE)
    """
    __tablename__ = 'semantic_metrics'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('database_knowledge_sources.id'), nullable=False)
    tenant_id = Column(Integer, nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    category = Column(String(100))  # revenue, users, performance, etc.
    
    sql_expression = Column(Text, nullable=False)
    aggregation = Column(String(50))  # sum, avg, count, min, max
    format = Column(String(50))  # number, currency, percentage
    
    description = Column(Text)
    business_definition = Column(Text)  # Plain English explanation
    
    # Which tables this metric uses
    tables_used = Column(JSON)  # ["orders", "customers"]
    
    # Drill-down dimensions
    drill_down_dimensions = Column(JSON)  # ["time_period", "region", "product"]
    
    # Time grain support
    supports_time_grain = Column(Boolean, default=True)
    default_time_grain = Column(String(50))  # day, week, month, quarter, year
    
    # Usage stats
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    is_featured = Column(Boolean, default=False)  # Show prominently in UI
    is_certified = Column(Boolean, default=False)  # Approved by data team
    
    created_by = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class SemanticDimension(Base):
    """
    Dimensions for grouping and filtering (PROMINENT UI FEATURE)
    """
    __tablename__ = 'semantic_dimensions'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('database_knowledge_sources.id'), nullable=False)
    tenant_id = Column(Integer, nullable=False, index=True)
    
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    category = Column(String(100))  # time, geography, product, customer
    
    sql_expression = Column(Text, nullable=False)
    type = Column(String(50))  # categorical, temporal, geographical, hierarchical
    
    description = Column(Text)
    
    # Hierarchy (e.g., country -> state -> city)
    hierarchy_levels = Column(JSON)  # ["country", "state", "city"]
    parent_dimension_id = Column(Integer, ForeignKey('semantic_dimensions.id'))
    
    # Values (cached)
    cached_values = Column(JSON)  # Top 100 values
    total_unique_values = Column(Integer)
    
    is_featured = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class DatabaseQueryTemplate(Base):
    """
    Pre-built query templates (20+ per database type as requested!)
    """
    __tablename__ = 'database_query_templates'
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('database_knowledge_sources.id'))
    source = relationship("DatabaseKnowledgeSource", back_populates="templates")
    
    # Template can be global (source_id=NULL) or source-specific
    dialect = Column(String(50))  # postgresql, mysql, etc.
    category = Column(String(100))  # analytics, reporting, monitoring, troubleshooting
    
    name = Column(String(255), nullable=False)
    description = Column(Text)
    natural_language = Column(Text, nullable=False)  # "Show top 10 customers by revenue"
    sql_template = Column(Text)  # Pre-validated SQL
    
    # Parameters that can be filled
    parameters = Column(JSON)  # [{"name": "days", "type": "integer", "default": 30}]
    
    # Visualization hint
    visualization_type = Column(String(50))  # table, bar, line, pie
    
    # Usage
    usage_count = Column(Integer, default=0)
    avg_execution_time_ms = Column(Float)
    
    # Tags for discovery
    tags = Column(JSON)  # ["revenue", "customers", "top-n"]
    
    is_featured = Column(Boolean, default=False)
    is_certified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())


# ============================================================================
# Pydantic Models for API
# ============================================================================

class DatabaseKnowledgeSourceCreate(BaseModel):
    """Request to create a database knowledge source"""
    name: str
    description: Optional[str] = None
    credential_id: int
    dialect: str
    connection_pool_size: int = 5
    max_rows_limit: int = 10000
    query_timeout_seconds: int = 30
    schema_cache_ttl: int = 3600
    query_cache_ttl: int = 300


class SemanticMetricCreate(BaseModel):
    """Create a business metric"""
    name: str
    display_name: str
    category: str
    sql_expression: str
    aggregation: str = "sum"
    format: str = "number"
    description: Optional[str] = None
    business_definition: Optional[str] = None
    tables_used: List[str]
    drill_down_dimensions: Optional[List[str]] = []
    supports_time_grain: bool = True
    default_time_grain: str = "month"
    is_featured: bool = False


class SemanticDimensionCreate(BaseModel):
    """Create a dimension for grouping"""
    name: str
    display_name: str
    category: str
    sql_expression: str
    type: str  # categorical, temporal, geographical
    description: Optional[str] = None
    hierarchy_levels: Optional[List[str]] = None
    is_featured: bool = False


class DatabaseQueryRequest(BaseModel):
    """Natural language query request"""
    query: str
    source_id: int
    visualization_hint: Optional[str] = None
    max_rows: int = 1000
    use_cache: bool = True
    include_explanation: bool = True


class QueryTemplateExecute(BaseModel):
    """Execute a template with parameters"""
    template_id: int
    parameters: Dict[str, Any] = {}
    max_rows: int = 1000
