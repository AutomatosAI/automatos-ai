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
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.sql import func

from core.database.base import Base
from core.models.orchestration_enums import (
    RunState,
    StateType,
    TaskState,
    RUN_STATE_TYPE,
    TASK_STATE_TYPE,
    DONE_TASK_STATES,
)


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
    config = Column(JSONB, nullable=True, server_default=text("'{}'"))

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
    max_concurrent = Column(Integer, nullable=False, server_default="3")

    # Replan tracking (PRD-82B US-005)
    replan_count = Column(Integer, nullable=False, server_default="0")

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
            f"goal={(self.goal[:50] if self.goal else None)!r}>"
        )


class OrchestrationTask(Base):
    """
    Individual task within a mission execution.

    Each task is assigned to a roster agent, executed sequentially, and verified
    before the next task is dispatched. State machine: PRD-82A Section 4.2.

    CRITICAL: `completed` is NOT terminal — only `verified`, `failed`, `skipped`.
    Board `done` status maps ONLY from `verified` (PRD-82A Section 4.3).
    """

    __tablename__ = "orchestration_tasks"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # Parent mission
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Task definition
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    task_type = Column(
        String(30),
        nullable=False,
        default="llm_generation",
        server_default="llm_generation",
    )

    # Ordering within the mission
    sequence_number = Column(Integer, nullable=False)

    # Desired agent role (e.g. 'researcher') — AgentMatcher resolves to actual agent
    agent_role = Column(String(100), nullable=True)

    # State machine (PRD-82A Section 4.1)
    state = Column(
        String(30),
        nullable=False,
        default=TaskState.PENDING.value,
        server_default=TaskState.PENDING.value,
    )
    state_type = Column(
        String(10),
        nullable=False,
        default=StateType.INITIAL.value,
        server_default=StateType.INITIAL.value,
    )

    # Agent assignment (filled by AgentMatcher + dispatcher)
    assigned_agent_id = Column(
        Integer,
        ForeignKey("agents.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Verification criteria — deterministic checks + LLM judge config
    verification_criteria = Column(JSONB, nullable=True)

    # Input/output
    input_context = Column(JSONB, nullable=True)
    output = Column(Text, nullable=True)
    output_metadata = Column(JSONB, nullable=True)

    # Failure tracking (PRD-82A Section 8)
    failure_reason_code = Column(String(50), nullable=True)
    failure_detail = Column(Text, nullable=True)

    # Retry tracking (PRD-82A Section 11)
    attempt_number = Column(Integer, nullable=False, server_default="0")
    max_retries = Column(Integer, nullable=False, server_default="3")

    # Complexity and parallel execution (PRD-82C)
    complexity = Column(String(10), nullable=False, server_default="moderate")
    parallel_group = Column(String(50), nullable=True)
    estimated_tokens = Column(Integer, nullable=False, server_default="4000")

    # Budget tracking (PRD-82A Section 9)
    tokens_used = Column(Integer, nullable=False, server_default="0")

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
        # Composite index for ordering tasks within a run
        Index("ix_orchestration_tasks_run_sequence", "run_id", "sequence_number"),
        # Partial index on active (non-terminal) states for coordinator tick queries
        Index(
            "ix_orchestration_tasks_active",
            "run_id",
            "state",
            postgresql_where=text(
                f"state NOT IN ({', '.join(repr(s.value) for s in DONE_TASK_STATES)})"
            ),
        ),
        CheckConstraint(
            f"state IN ({', '.join(repr(s.value) for s in TaskState)})",
            name="ck_orchestration_tasks_state",
        ),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestrationTask id={self.id} run_id={self.run_id} "
            f"seq={self.sequence_number} state={self.state} "
            f"title={(self.title[:40] if self.title else None)!r}>"
        )


class OrchestrationTaskDependency(Base):
    """
    DAG edge between two orchestration tasks.

    Represents a dependency: `task_id` cannot start until `depends_on_task_id`
    reaches a terminal success state (controlled by `trigger_rule`).

    Used by DependencyResolver to validate the DAG (no cycles) and determine
    which tasks are ready for dispatch.

    Source: PRD-82A Section 4, PRD-101 Section 5.5
    """

    __tablename__ = "orchestration_task_dependencies"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # The downstream task (blocked until dependency met)
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # The upstream task (must complete first)
    depends_on_task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # When the dependency is considered met (PRD-82A: only all_success for v1)
    trigger_rule = Column(
        String(30),
        nullable=False,
        default="all_success",
        server_default="all_success",
    )

    __table_args__ = (
        # Prevent duplicate edges
        UniqueConstraint(
            "task_id",
            "depends_on_task_id",
            name="uq_orchestration_task_dep_pair",
        ),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestrationTaskDependency task={self.task_id} "
            f"depends_on={self.depends_on_task_id} rule={self.trigger_rule}>"
        )


class OrchestrationArchive(Base):
    """
    Archived orchestration runs — full snapshot of terminal missions.

    Terminal runs (completed, failed, cancelled) older than the configured
    retention period are serialized into ``archive_data`` (JSONB) and moved
    here so the active tables stay fast.

    Source: PRD-82B US-009
    """

    __tablename__ = "orchestration_archive"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # Reference to the original run (informational — the run row is deleted after archive)
    original_run_id = Column(UUID(as_uuid=True), nullable=False, unique=True)

    # Denormalized fields for search/filter without unpacking JSONB
    goal = Column(Text, nullable=False)
    state = Column(String(30), nullable=False)
    workspace_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    created_by = Column(String(255), nullable=False)

    # Timestamps from original run
    created_at = Column(DateTime(timezone=True), nullable=False)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Full snapshot: run fields + all tasks + all events + dependencies
    archive_data = Column(JSONB, nullable=False)

    # When this record was archived
    archived_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_orchestration_archive_workspace", "workspace_id"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestrationArchive id={self.id} "
            f"original_run={self.original_run_id} state={self.state}>"
        )


class OrchestrationEvent(Base):
    """
    Append-only audit log for the orchestration subsystem.

    Every state change on runs/tasks produces a corresponding event row in
    the SAME transaction (dual-write pattern, PRD-82A Section 5 principle 2).
    Non-transition events (budget warnings, stall detections) are also logged.

    This table is NEVER updated — only INSERTed. No version_id column.

    Source: PRD-82A Section 5, PRD-101 Section 6.2
    """

    __tablename__ = "orchestration_events"

    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    # Parent mission (always present)
    run_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orchestration_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Associated task (nullable — run-level events have no task)
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("orchestration_tasks.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    # Event classification
    event_type = Column(String(50), nullable=False)

    # Who triggered this event
    actor_type = Column(String(20), nullable=False)
    actor_id = Column(String(255), nullable=True)

    # State transition (nullable for non-transition events)
    old_state = Column(String(30), nullable=True)
    new_state = Column(String(30), nullable=True)

    # Arbitrary event data (verification scores, error details, etc.)
    payload = Column(JSONB, nullable=True)

    # Immutable timestamp
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        # Composite index for event timeline queries per run
        Index("ix_orchestration_events_run_created", "run_id", "created_at"),
        {"extend_existing": True},
    )

    def __repr__(self) -> str:
        return (
            f"<OrchestrationEvent id={self.id} run={self.run_id} "
            f"type={self.event_type} {self.old_state}→{self.new_state}>"
        )
