"""
Watch Enums -- PRD-204 Auto Watcher
===================================

Canonical StrEnums, transition map, and terminal frozensets for the watch
registry. Mirrors the house pattern of ``orchestration_enums`` (PRD-82A):
enums + ALLOWED_*_TRANSITIONS dict + TERMINAL_* frozensets, consumed by the
model's CHECK constraint and by ``WatchService.transition``.

Source: PRD-204 Section 4 (S1/S2).
"""

from enum import Enum


# ---------------------------------------------------------------------------
# WatchStatus -- watch lifecycle states (PRD-204 S1)
# ---------------------------------------------------------------------------

class WatchStatus(str, Enum):
    WATCHING = "watching"                    # supervising the live target
    ACTING = "acting"                        # a corrective action is in flight (S7/S8)
    AWAITING_APPROVAL = "awaiting_approval"  # action parked on an ApprovalGrant
    NEEDS_ATTENTION = "needs_attention"      # hard-stopped (action budget, lost target)
    PASSED = "passed"                        # verdict: target reached a good terminal
    FAILED = "failed"                        # verdict: target failed
    ESCALATED = "escalated"                  # handed to a human; terminal unless renewed
    EXPIRED = "expired"                      # deadline passed without a verdict
    CANCELLED = "cancelled"                  # cancelled by user, or target cancelled


# ---------------------------------------------------------------------------
# WatchType -- what kind of launched unit this watch supervises
# ---------------------------------------------------------------------------

class WatchType(str, Enum):
    MISSION = "mission"
    PLAYBOOK_EXECUTION = "playbook_execution"
    SCHEDULED_PLAYBOOK = "scheduled_playbook"


# ---------------------------------------------------------------------------
# WatchTargetType -- the concrete row the watch currently points at.
# Distinct from WatchType because lineage can repoint a watch (``follow``)
# and terminal hooks also ingest for board tasks.
# ---------------------------------------------------------------------------

class WatchTargetType(str, Enum):
    MISSION = "mission"                      # orchestration_runs.id (UUID string)
    PLAYBOOK_EXECUTION = "playbook_execution"  # recipe_executions.execution_id
    SCHEDULED_PLAYBOOK = "scheduled_playbook"  # workflow_recipes.id (int as string)
    BOARD_TASK = "board_task"                # board_tasks.id (int as string)


# ---------------------------------------------------------------------------
# WatchPolicy -- decision-flow profile (S10 drives the full table; S1-S5 only
# store the value)
# ---------------------------------------------------------------------------

class WatchPolicy(str, Enum):
    RUN_AND_REPORT = "run_and_report"
    SCORE_AND_IMPROVE = "score_and_improve"
    WATCH_CHANGE = "watch_change"
    PERSISTENT = "persistent"


# ---------------------------------------------------------------------------
# WatchEventType -- vocabulary for watch_events.event_type
# ---------------------------------------------------------------------------

class WatchEventType(str, Enum):
    CREATED = "created"                # watch row created
    TERMINAL = "terminal"              # target reached a terminal state
    MISSED_RUN = "missed_run"          # scheduled playbook missed an expected fire
    BENCHED = "benched"                # scheduled playbook skipped on open breaker
    EXPIRED = "expired"                # watch deadline passed
    ACTION = "action"                  # corrective action recorded (S7/S8)
    BUDGET_EXHAUSTED = "budget_exhausted"  # action budget hard-stop
    FOLLOW = "follow"                  # lineage repoint (rerun/replan follows the work)
    STATUS_CHANGE = "status_change"    # explicit watch transition worth recording
    TARGET_MISSING = "target_missing"  # target row no longer exists
    CANCELLED = "cancelled"            # watch cancelled
    SCORED = "scored"                  # run-level verdict written (S6)
    DIAGNOSED = "diagnosed"            # failure/low-score diagnosis recorded (S10)
    CHANGE_REPORT = "change_report"    # before/after comparison (watch_change / persistent)


# ===========================================================================
# Terminal frozensets
# ===========================================================================

# ESCALATED is terminal-unless-renewed: it sits in the terminal set (so a new
# watch on the same target is allowed by the partial unique index) but keeps
# a single renewal transition back to WATCHING -- the same shape as
# RunState.FAILED -> REPLANNING in orchestration_enums.
TERMINAL_WATCH_STATUSES: frozenset[WatchStatus] = frozenset(
    {
        WatchStatus.PASSED,
        WatchStatus.FAILED,
        WatchStatus.ESCALATED,
        WatchStatus.EXPIRED,
        WatchStatus.CANCELLED,
    }
)

LIVE_WATCH_STATUSES: frozenset[WatchStatus] = frozenset(
    s for s in WatchStatus if s not in TERMINAL_WATCH_STATUSES
)

# Only these statuses are claimed by the watcher tick (awaiting_approval and
# needs_attention are parked on a human/grant, not on the clock).
CLAIMABLE_WATCH_STATUSES: frozenset[WatchStatus] = frozenset(
    {WatchStatus.WATCHING, WatchStatus.ACTING}
)


# ===========================================================================
# Allowed transitions (house style of ALLOWED_RUN_TRANSITIONS)
# ===========================================================================

ALLOWED_WATCH_TRANSITIONS: dict[WatchStatus, frozenset[WatchStatus]] = {
    WatchStatus.WATCHING: frozenset(
        {
            WatchStatus.ACTING,
            WatchStatus.AWAITING_APPROVAL,
            WatchStatus.NEEDS_ATTENTION,
            WatchStatus.PASSED,
            WatchStatus.FAILED,
            WatchStatus.ESCALATED,
            WatchStatus.EXPIRED,
            WatchStatus.CANCELLED,
        }
    ),
    WatchStatus.ACTING: frozenset(
        {
            WatchStatus.WATCHING,
            WatchStatus.AWAITING_APPROVAL,
            WatchStatus.NEEDS_ATTENTION,
            WatchStatus.PASSED,
            WatchStatus.FAILED,
            WatchStatus.ESCALATED,
            WatchStatus.EXPIRED,
            WatchStatus.CANCELLED,
        }
    ),
    WatchStatus.AWAITING_APPROVAL: frozenset(
        {
            WatchStatus.WATCHING,
            WatchStatus.ACTING,
            WatchStatus.NEEDS_ATTENTION,
            WatchStatus.PASSED,
            WatchStatus.FAILED,
            WatchStatus.ESCALATED,
            WatchStatus.EXPIRED,
            WatchStatus.CANCELLED,
        }
    ),
    WatchStatus.NEEDS_ATTENTION: frozenset(
        {
            WatchStatus.WATCHING,
            WatchStatus.ACTING,
            WatchStatus.PASSED,
            WatchStatus.FAILED,
            WatchStatus.ESCALATED,
            WatchStatus.EXPIRED,
            WatchStatus.CANCELLED,
        }
    ),
    WatchStatus.PASSED: frozenset(),      # terminal
    WatchStatus.FAILED: frozenset(),      # terminal
    WatchStatus.ESCALATED: frozenset({WatchStatus.WATCHING}),  # renew only
    WatchStatus.EXPIRED: frozenset(),     # terminal
    WatchStatus.CANCELLED: frozenset(),   # terminal
}


# ===========================================================================
# Terminal target state -> watch status (v1, pre-S6 scoring)
# ===========================================================================

# Until the S6 run-level verdict lands, a terminal target closes the watch by
# outcome. S6 inserts scoring between "terminal observed" and "watch closed"
# without changing this vocabulary.
WATCH_STATUS_FOR_TERMINAL_TARGET: dict[str, WatchStatus] = {
    "completed": WatchStatus.PASSED,
    "verified": WatchStatus.PASSED,
    "done": WatchStatus.PASSED,
    "failed": WatchStatus.FAILED,
    "cancelled": WatchStatus.CANCELLED,
}


def terminal_watch_statuses_sql() -> str:
    """The terminal-status set as a SQL IN-list body (stable, sorted).

    Single source for the model's partial unique index; the alembic migration
    carries the same literal snapshot (migrations are frozen by design).
    """
    return ", ".join(repr(s.value) for s in sorted(TERMINAL_WATCH_STATUSES))
