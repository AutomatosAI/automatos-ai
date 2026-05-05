"""PRD-139 US-002: Tool routing graph tables

Creates 3 tables for the semantic tool routing graph:
- tool_routing_edges: directional action edges (e.g. 'used_after')
- tool_routing_intent_clusters: embedding-based intent groupings
- tool_routing_affinities: agent/intent action preferences

Reversible (drops all 3 tables in downgrade).

Revision ID: prd139_tool_routing_graph
Revises: None (standalone — safe to run anytime)
Create Date: 2026-05-04
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID

revision = "prd139_tool_routing_graph"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Intent clusters first (FK target for affinities)
    op.create_table(
        "tool_routing_intent_clusters",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("centroid_embedding", JSONB, nullable=False),
        sa.Column("embedding_model_key", sa.String(255), nullable=False),
        sa.Column("sample_query", sa.Text(), nullable=False),
        sa.Column("action_names_hot", ARRAY(sa.String), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
    )

    # 2. Edges table
    op.create_table(
        "tool_routing_edges",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("from_action", sa.String(255), nullable=False),
        sa.Column("to_action", sa.String(255), nullable=False),
        sa.Column("edge_type", sa.String(50), nullable=False),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=True),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_tre_from_type_scope",
        "tool_routing_edges",
        ["from_action", "edge_type", "workspace_id", "agent_id"],
    )
    op.create_unique_constraint(
        "uq_tre_full_key",
        "tool_routing_edges",
        ["from_action", "to_action", "edge_type", "workspace_id", "agent_id"],
    )

    # 3. Affinities table (depends on intent_clusters FK)
    op.create_table(
        "tool_routing_affinities",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("action_name", sa.String(255), nullable=False),
        sa.Column("affinity_type", sa.String(50), nullable=False),
        sa.Column("workspace_id", PGUUID(as_uuid=True), nullable=True),
        sa.Column("agent_id", sa.Integer(), sa.ForeignKey("agents.id"), nullable=True),
        sa.Column(
            "intent_cluster_id",
            sa.Integer(),
            sa.ForeignKey("tool_routing_intent_clusters.id"),
            nullable=True,
        ),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("sample_count", sa.Integer(), nullable=False),
        sa.Column("last_updated", sa.DateTime(), nullable=False),
    )
    op.create_index(
        "ix_tra_action_type_agent",
        "tool_routing_affinities",
        ["action_name", "affinity_type", "agent_id"],
    )
    op.create_index(
        "ix_tra_intent_type",
        "tool_routing_affinities",
        ["intent_cluster_id", "affinity_type"],
    )
    op.create_unique_constraint(
        "uq_tra_full_key",
        "tool_routing_affinities",
        ["action_name", "affinity_type", "workspace_id", "agent_id", "intent_cluster_id"],
    )


def downgrade() -> None:
    op.drop_table("tool_routing_affinities")
    op.drop_table("tool_routing_edges")
    op.drop_table("tool_routing_intent_clusters")
