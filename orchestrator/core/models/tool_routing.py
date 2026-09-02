"""
PRD-139 US-002: Tool Routing Graph tables.

Three tables for persistent graph edges, routing affinities, and intent clusters
that back the semantic tool routing system.

- tool_routing_edges: directional edges between tool actions (e.g. "used_after")
- tool_routing_affinities: agent/intent preferences for actions
- tool_routing_intent_clusters: embedding-based intent groupings
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID

from core.database.base import Base


class ToolRoutingEdge(Base):
    """Directional edge between two tool actions in the routing graph.

    edge_type values in v1: 'used_after'
    Scoped optionally by workspace and/or agent.
    """

    __tablename__ = "tool_routing_edges"

    id = Column(Integer, primary_key=True)
    from_action = Column(String(255), nullable=False)
    to_action = Column(String(255), nullable=False)
    edge_type = Column(String(50), nullable=False)  # 'used_after' only in v1
    workspace_id = Column(PGUUID(as_uuid=True), nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    weight = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)  # Wilson lower bound
    sample_count = Column(Integer, nullable=False)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index(
            "ix_tre_from_type_scope",
            "from_action",
            "edge_type",
            "workspace_id",
            "agent_id",
        ),
        UniqueConstraint(
            "from_action",
            "to_action",
            "edge_type",
            "workspace_id",
            "agent_id",
            name="uq_tre_full_key",
        ),
        {"extend_existing": True},
    )

    def to_dict(self):
        return {
            "id": self.id,
            "from_action": self.from_action,
            "to_action": self.to_action,
            "edge_type": self.edge_type,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "agent_id": self.agent_id,
            "weight": self.weight,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class ToolRoutingIntentCluster(Base):
    """Embedding-based intent cluster that groups semantically similar queries.

    embedding_model_key follows canonical format: provider:model:dimension
    (e.g. 'openrouter:qwen/qwen3-embedding-8b:2048')
    from ActionSemanticIndex._cache_model_key.
    """

    __tablename__ = "tool_routing_intent_clusters"

    id = Column(Integer, primary_key=True)
    centroid_embedding = Column(JSONB, nullable=False)
    embedding_model_key = Column(String(255), nullable=False)
    sample_query = Column(Text, nullable=False)
    action_names_hot = Column(ARRAY(String), nullable=False)
    sample_count = Column(Integer, nullable=False)
    # PRD-232 US-007: where this cluster came from. 'organic' = learned by the
    # nightly edge_builder from tool_execution_logs; 'seeded' = synthetic-utterance
    # cold-start (seed_tool_routing_graph). The nightly rebuild deletes-and-reinserts
    # ONLY 'organic' rows, so seeded cold-start clusters survive 03:00 UTC and the
    # graph routes day-one. server_default 'organic' backfills every pre-migration row.
    provenance = Column(String(20), nullable=True, default="organic", server_default="organic")
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = ({"extend_existing": True},)

    def to_dict(self):
        return {
            "id": self.id,
            "embedding_model_key": self.embedding_model_key,
            "sample_query": self.sample_query,
            "action_names_hot": self.action_names_hot,
            "sample_count": self.sample_count,
            "provenance": self.provenance,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


class ToolRoutingAffinity(Base):
    """Action affinity record scoped by agent, workspace, and/or intent cluster.

    affinity_type values: 'agent_prefers' | 'succeeds_for_intent' | 'fails_for_intent'
    """

    __tablename__ = "tool_routing_affinities"

    id = Column(Integer, primary_key=True)
    action_name = Column(String(255), nullable=False)
    affinity_type = Column(String(50), nullable=False)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    intent_cluster_id = Column(
        Integer,
        ForeignKey("tool_routing_intent_clusters.id"),
        nullable=True,
    )
    weight = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    sample_count = Column(Integer, nullable=False)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index(
            "ix_tra_action_type_agent",
            "action_name",
            "affinity_type",
            "agent_id",
        ),
        Index(
            "ix_tra_intent_type",
            "intent_cluster_id",
            "affinity_type",
        ),
        UniqueConstraint(
            "action_name",
            "affinity_type",
            "workspace_id",
            "agent_id",
            "intent_cluster_id",
            name="uq_tra_full_key",
        ),
        {"extend_existing": True},
    )

    def to_dict(self):
        return {
            "id": self.id,
            "action_name": self.action_name,
            "affinity_type": self.affinity_type,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "agent_id": self.agent_id,
            "intent_cluster_id": self.intent_cluster_id,
            "weight": self.weight,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
