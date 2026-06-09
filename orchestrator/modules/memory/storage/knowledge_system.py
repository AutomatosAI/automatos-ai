"""
PRD-05: Memory & Knowledge Systems
===================================

Hierarchical memory system with real Redis TTL, PostgreSQL storage,
pgvector embeddings, and learning capabilities.

NO MOCK DATA - All operations use real services and produce real results.
"""

import json
import math
import redis
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4, UUID as PyUUID
from enum import Enum

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Text, Boolean, select, func, text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

from contextlib import asynccontextmanager

# Import centralized vector operations
from modules.search.vector_store import EnhancedVectorStore

logger = logging.getLogger(__name__)

# Database models for memory system
Base = declarative_base()


class MemoryLevel(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    COLLECTIVE = "collective"


class MemoryType(str, Enum):
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PATTERN = "pattern"
    FEEDBACK = "feedback"


class MemoryItem(Base):
    """Enhanced memory items with vector embeddings"""
    __tablename__ = 'memory_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(Integer, nullable=True)  # Allow NULL for general conversations without specific agent
    workspace_id = Column(UUID(as_uuid=True), nullable=False)
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False)
    memory_level = Column(String(50), nullable=False, default=MemoryLevel.SHORT_TERM)
    importance = Column(Float, default=0.5)
    embedding = Column(Vector(1024))  # Must match DB schema - update via migration when changing embedding model
    access_count = Column(Integer, default=0)
    last_access = Column(DateTime, default=datetime.now)
    decay_rate = Column(Float, default=0.1)
    associations = Column(ARRAY(String), default=[])  # Related memory IDs
    meta_data = Column('metadata', JSON, default={})
    created_at = Column(DateTime, default=datetime.now)
    
    # Performance tracking
    success_rate = Column(Float, default=0.0)
    usage_in_solutions = Column(Integer, default=0)
    average_retrieval_time = Column(Float, default=0.0)


class KnowledgeNode(Base):
    """Knowledge graph nodes"""
    __tablename__ = 'knowledge_nodes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(Integer, nullable=True)  # Null for collective knowledge
    concept = Column(String(255), nullable=False)
    description = Column(Text)
    node_type = Column(String(50), nullable=False)
    embedding = Column(Vector(1024))
    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)
    meta_data = Column('metadata', JSON, default={})
    created_at = Column(DateTime, default=datetime.now)


class KnowledgeEdge(Base):
    """Knowledge graph edges"""
    __tablename__ = 'knowledge_edges'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    from_node_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_nodes.id'))
    to_node_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_nodes.id'))
    relationship = Column(String(100), nullable=False)
    strength = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)


class LearningOutcome(Base):
    """Tracks learning and performance improvements"""
    __tablename__ = 'learning_outcomes'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, nullable=False)
    task_type = Column(String(255))
    learned_pattern = Column(Text)
    success_rate_before = Column(Float)
    success_rate_after = Column(Float)
    execution_time_before = Column(Float)
    execution_time_after = Column(Float)
    confidence = Column(Float, default=0.5)
    application_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    # --- PRD-142 Wave 4 (W4-S11): HARNESS outcome fields (all nullable/additive) ---
    # Doubles learning_outcomes as HARNESS's structured OUTCOME store — the result
    # of an applied prescription, workspace-scoped and auditable (Role 2, §12.2).
    workspace_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    change_type = Column(String(64), nullable=True)
    risk_score = Column(Integer, nullable=True)
    status = Column(String(32), nullable=True)  # applied | rejected | rolled_back
    applied_at = Column(DateTime, nullable=True)
    rolled_back_at = Column(DateTime, nullable=True)
    current_value_before = Column(JSON, nullable=True)


class HarnessPrescription(Base):
    """PRD-142 Wave 4 (W4-S11): HARNESS's structured PRESCRIPTION store — one row
    per proposed config change per tick (Role 2, §12.2). Workspace-scoped,
    queryable, auditable; the DB counterpart of the flat baseline-JSON prescription
    records (S12 dual-writes here).

    No FKs (workspace_id / target_id are plain columns, mirroring routing_rules) so
    table-creation ordering is never a concern. Learning/operational only — never
    holds a business entity (KNOWLEDGE-GRAPH-CANONICAL §4 boundary).
    """
    __tablename__ = 'harness_prescriptions'

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    prescription_id = Column(String(64), nullable=False, index=True)
    target_type = Column(String(32), nullable=True)
    target_id = Column(Integer, nullable=True)
    target_name = Column(String(255), nullable=True)
    change_type = Column(String(64), nullable=False)
    risk_score = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="proposed")  # proposed|queued|applied|rejected|rolled_back
    proposed_value = Column(JSON, nullable=True)
    current_value_before = Column(JSON, nullable=True)
    rationale = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now)

