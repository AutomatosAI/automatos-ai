"""
Orchestration Models — PRD-82A Sequential Mission Coordinator
=============================================================

SQLAlchemy models for the orchestration subsystem:
- OrchestrationRun: top-level mission execution
- OrchestrationTask, OrchestrationTaskDependency, OrchestrationEvent: added in US-003/004

Source: PRD-82A Sections 4-9, PRD-101 Section 13
"""

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from core.database.base import Base
from core.models.orchestration_enums import RunState, StateType, RUN_STATE_TYPE


class OrchestrationRun(Base):
    """
    Top-level mission execution record.

    Represents a user's goal decomposed into tasks and executed sequentially
    by roster agents. Tracks lifecycle from planning through human review.

    Naming: DB uses 'orchestration_runs', API uses 'missions' (PRD-82A Section 10).
    """

    __tablename__ = "orchestration_runs"

    # Primary key — UUID with DB-generated default
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # Tenant isolation
    workspace_id = Column(
        UUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Mission definition
    goal = Column(Text, nullable=False)
    plan = Column(JSONB, nullable=True)
    config = Column(JSONB, nullable=True, server_default="{}")

    # State machine (PRD-82A Section 4.1)
    state = Column(
        String(30),
        nullable=False,
        default=RunState.PENDING.value,
        server_default=RunState.PENDING.value,
    )
    state_type = Column(
        String(10),
        nullable=False,
        default=StateType.INITIAL.value,
        server_default=StateType.INITIAL.value,
    )

    # Ownership
    created_by = Column(String(255), nullable=False)  # Clerk user ID e.g. 'user_xxx'

    # Coordinator assignment
    assigned_coordinator_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Output & budget tracking (PRD-82A Sections 6, 9)
    output_summary = Column(JSONB, nullable=True)
    token_budget_estimate = Column(Integer, nullable=True)
    tokens_used = Column(Integer, nullable=False, server_default="0")

    # Execution limits
    max_retries = Column(Integer, nullable=False, server_default="3")
    max_concurrent = Column(Integer, nullable=False, server_default="1")

    # Timestamps
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
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

    # Optimistic locking (PRD-82A Section 5, principle 7)
    version_id = Column(Integer, nullable=False, server_default="1")

    __mapper_args__ = {"version_id_col": version_id}

    __table_args__ = (
        CheckConstraint(
            f"state IN ({', '.join(repr(s.value) for s in RunState)})",
            name="ck_orchestration_runs_state",
        ),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestrationRun id={self.id} state={self.state} "
            f"goal={self.goal[:50] if self.goal else None!r}>"
        )
