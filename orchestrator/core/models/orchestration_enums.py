"""
Orchestration Enums — PRD-82A Sequential Mission Coordinator
=============================================================

Canonical StrEnums, state mappings, transition dicts, and terminal frozensets
for the orchestration subsystem.

Source: PRD-82A Sections 4.1-4.3, 8; PRD-101 Section 3.2, 3.10, 6.2
"""

from enum import Enum


# ---------------------------------------------------------------------------
# StateType — coarse-grained categories the coordinator switches on
# ---------------------------------------------------------------------------

class StateType(str, Enum):
    INITIAL = "initial"
    ACTIVE = "active"
    BLOCKED = "blocked"
    TERMINAL = "terminal"


# ---------------------------------------------------------------------------
# RunState — top-level mission states (10 values, PRD-82A Section 4.1)
# ---------------------------------------------------------------------------

class RunState(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    RUNNING = "running"
    PAUSED = "paused"
    REPLANNING = "replanning"
    VERIFYING = "verifying"
    AWAITING_HUMAN = "awaiting_human"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# TaskState — per-task states (11 values, PRD-82A Section 4.1)
# CRITICAL: `completed` is NOT terminal — only verified/failed/skipped are.
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    FAILED = "failed"
    SKIPPED = "skipped"
    STALLED = "stalled"
    RETRYING = "retrying"


# ---------------------------------------------------------------------------
# EventType — typed events for orchestration_events (30+ values)
# Source: PRD-101 Section 6.2
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    # Run lifecycle
    RUN_CREATED = "run_created"
    RUN_PLANNING_STARTED = "run_planning_started"
    RUN_PLAN_READY = "run_plan_ready"
    RUN_APPROVED = "run_approved"
    RUN_REJECTED = "run_rejected"
    RUN_STARTED = "run_started"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    RUN_REPLANNING = "run_replanning"
    RUN_REPLANNED = "run_replanned"
    RUN_BUDGET_WARNING = "run_budget_warning"
    RUN_BUDGET_EXCEEDED = "run_budget_exceeded"
    RUN_BUDGET_INCREASED = "run_budget_increased"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    RUN_VERIFYING = "run_verifying"
    RUN_AWAITING_HUMAN = "run_awaiting_human"

    # Task lifecycle
    TASK_CREATED = "task_created"
    TASK_QUEUED = "task_queued"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_CONTINUING = "task_continuing"
    TASK_RESUMED = "task_resumed"
    TASK_OUTPUT_SUBMITTED = "task_output_submitted"
    TASK_VERIFICATION_STARTED = "task_verification_started"
    TASK_VERIFICATION_PASSED = "task_verification_passed"
    TASK_VERIFICATION_FAILED = "task_verification_failed"
    TASK_VERIFICATION_COMPLETED = "task_verification_completed"
    TASK_HUMAN_REVIEW_REQUESTED = "task_human_review_requested"
    TASK_HUMAN_APPROVED = "task_human_approved"
    TASK_HUMAN_REJECTED = "task_human_rejected"
    TASK_RETRYING = "task_retrying"
    TASK_CRASHED = "task_crashed"
    TASK_FAILED = "task_failed"
    TASK_SKIPPED = "task_skipped"
    TASK_CANCELLED = "task_cancelled"
    TASK_STALLED = "task_stalled"

    # System
    STALL_DETECTED = "stall_detected"
    MODEL_FALLBACK = "model_fallback"
    COST_SNAPSHOT = "cost_snapshot"
    BUDGET_WARNING = "budget_warning"

    # Cross-task consistency (PRD-82B US-006)
    CONSISTENCY_CHECKED = "consistency_checked"


# ---------------------------------------------------------------------------
# ActorType — who triggered an orchestration event
# ---------------------------------------------------------------------------

class ActorType(str, Enum):
    SYSTEM = "system"
    COORDINATOR = "coordinator"
    AGENT = "agent"
    VERIFIER = "verifier"
    HUMAN = "human"
    SCHEDULER = "scheduler"
    RECONCILER = "reconciler"


# ---------------------------------------------------------------------------
# TaskType — classification of orchestration task work
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    LLM_GENERATION = "llm_generation"
    TOOL_EXECUTION = "tool_execution"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    REVIEW = "review"


# ---------------------------------------------------------------------------
# TriggerRule — when a task's dependencies are considered met
# ---------------------------------------------------------------------------

class TriggerRule(str, Enum):
    ALL_SUCCESS = "all_success"
    ANY_SUCCESS = "any_success"
    ALL_COMPLETE = "all_complete"


# ---------------------------------------------------------------------------
# FailureReasonCode — structured failure classification (PRD-82A Section 8)
# ---------------------------------------------------------------------------

class FailureReasonCode(str, Enum):
    AGENT_ERROR = "agent_error"
    AGENT_TIMEOUT = "agent_timeout"
    VERIFICATION_FAIL = "verification_fail"
    VERIFICATION_REJECT = "verification_reject"
    NO_AGENT_AVAILABLE = "no_agent_available"
    DEPENDENCY_FAILED = "dependency_failed"
    CANCELLED = "cancelled"
    MAX_RETRIES_EXHAUSTED = "max_retries_exhausted"


# ===========================================================================
# State → StateType mappings
# ===========================================================================

RUN_STATE_TYPE: dict[RunState, StateType] = {
    RunState.PENDING: StateType.INITIAL,
    RunState.PLANNING: StateType.ACTIVE,
    RunState.AWAITING_APPROVAL: StateType.BLOCKED,
    RunState.RUNNING: StateType.ACTIVE,
    RunState.PAUSED: StateType.BLOCKED,
    RunState.REPLANNING: StateType.ACTIVE,
    RunState.VERIFYING: StateType.ACTIVE,
    RunState.AWAITING_HUMAN: StateType.BLOCKED,
    RunState.COMPLETED: StateType.TERMINAL,
    RunState.FAILED: StateType.TERMINAL,
    RunState.CANCELLED: StateType.TERMINAL,
}

TASK_STATE_TYPE: dict[TaskState, StateType] = {
    TaskState.PENDING: StateType.INITIAL,
    TaskState.QUEUED: StateType.ACTIVE,
    TaskState.ASSIGNED: StateType.ACTIVE,
    TaskState.RUNNING: StateType.ACTIVE,
    TaskState.COMPLETED: StateType.ACTIVE,      # NOT terminal — awaiting verification
    TaskState.VERIFYING: StateType.ACTIVE,
    TaskState.VERIFIED: StateType.BLOCKED,     # done, but human can reject → RETRYING
    TaskState.FAILED: StateType.TERMINAL,
    TaskState.SKIPPED: StateType.TERMINAL,
    TaskState.STALLED: StateType.ACTIVE,
    TaskState.RETRYING: StateType.ACTIVE,
}


# ===========================================================================
# Terminal state frozensets
# ===========================================================================

TERMINAL_RUN_STATES: frozenset[RunState] = frozenset(
    s for s, t in RUN_STATE_TYPE.items() if t == StateType.TERMINAL
)

TERMINAL_TASK_STATES: frozenset[TaskState] = frozenset(
    s for s, t in TASK_STATE_TYPE.items() if t == StateType.TERMINAL
)

# Tasks that count as "done" for dependency resolution and mission completion.
# VERIFIED is not TERMINAL (human can reject it back to RETRYING), but it IS
# a success state for downstream dependency satisfaction.
DONE_TASK_STATES: frozenset[TaskState] = frozenset(
    {TaskState.VERIFIED, TaskState.FAILED, TaskState.SKIPPED}
)


# ===========================================================================
# Allowed state transitions (PRD-82A Section 4.2)
# ===========================================================================

ALLOWED_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.QUEUED, TaskState.SKIPPED}),
    TaskState.QUEUED: frozenset({TaskState.ASSIGNED, TaskState.SKIPPED}),
    TaskState.ASSIGNED: frozenset({TaskState.RUNNING, TaskState.STALLED, TaskState.SKIPPED}),
    TaskState.RUNNING: frozenset({TaskState.COMPLETED, TaskState.STALLED, TaskState.FAILED, TaskState.SKIPPED}),
    TaskState.COMPLETED: frozenset({TaskState.VERIFYING, TaskState.SKIPPED}),
    TaskState.VERIFYING: frozenset({TaskState.VERIFIED, TaskState.RETRYING, TaskState.FAILED, TaskState.SKIPPED}),
    TaskState.VERIFIED: frozenset({TaskState.RETRYING}),  # human rejection re-queues for retry
    TaskState.FAILED: frozenset(),    # terminal
    TaskState.SKIPPED: frozenset(),   # terminal
    TaskState.STALLED: frozenset({TaskState.QUEUED, TaskState.ASSIGNED, TaskState.SKIPPED}),
    TaskState.RETRYING: frozenset({TaskState.ASSIGNED, TaskState.SKIPPED}),
}

ALLOWED_RUN_TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    RunState.PENDING: frozenset({RunState.PLANNING, RunState.CANCELLED}),
    RunState.PLANNING: frozenset({RunState.AWAITING_APPROVAL, RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.AWAITING_APPROVAL: frozenset({RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}),
    RunState.RUNNING: frozenset({RunState.PAUSED, RunState.REPLANNING, RunState.VERIFYING, RunState.FAILED, RunState.CANCELLED}),
    RunState.PAUSED: frozenset({RunState.RUNNING, RunState.CANCELLED}),
    RunState.REPLANNING: frozenset({RunState.RUNNING, RunState.FAILED}),
    RunState.VERIFYING: frozenset({RunState.AWAITING_HUMAN, RunState.FAILED, RunState.CANCELLED}),
    RunState.AWAITING_HUMAN: frozenset({RunState.COMPLETED, RunState.RUNNING, RunState.CANCELLED}),
    RunState.COMPLETED: frozenset(),   # terminal
    RunState.FAILED: frozenset({RunState.REPLANNING}),  # replannable (PRD-82B US-005)
    RunState.CANCELLED: frozenset(),   # terminal
}


# ===========================================================================
# Board status mapping (PRD-82A Section 4.3)
# Maps TaskState → board_tasks.status string for kanban visibility.
# CRITICAL: completed → in_review (NOT done). Only verified → done.
# ===========================================================================

BOARD_STATUS_MAP: dict[TaskState, str] = {
    TaskState.PENDING: "backlog",
    TaskState.QUEUED: "todo",
    TaskState.ASSIGNED: "in_progress",
    TaskState.RUNNING: "in_progress",
    TaskState.COMPLETED: "in_review",
    TaskState.VERIFYING: "in_review",
    TaskState.VERIFIED: "done",
    TaskState.FAILED: "blocked",
    TaskState.STALLED: "blocked",
    TaskState.RETRYING: "in_progress",
    TaskState.SKIPPED: "cancelled",
}
