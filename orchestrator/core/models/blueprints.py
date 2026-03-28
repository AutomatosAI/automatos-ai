"""
Agent Blueprints — governance rules for agent readiness and mission budgets.

Each workspace can define blueprints that set minimum standards:
min_tools, require_system_prompt, max_budget_per_run, required_tags, allowed_models.
"""

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from core.database.base import Base


class AgentBlueprint(Base):
    __tablename__ = "agent_blueprints"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Governance rules stored as JSONB for flexibility
    # Schema: {
    #   "min_tools": int,
    #   "require_system_prompt": bool,
    #   "max_budget_per_run": float | null,
    #   "required_tags": ["tag1", "tag2"],
    #   "allowed_models": ["model/name", ...] | null (null = any)
    # }
    rules = Column(JSONB, nullable=False, server_default="{}")

    is_default = Column(Boolean, nullable=False, server_default="false")

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = ({"extend_existing": True},)

    def __repr__(self) -> str:
        return f"<AgentBlueprint id={self.id} name={self.name!r}>"
