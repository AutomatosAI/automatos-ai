"""PRD-228 — Fleet State read-model.

One read-model over existing floor state so "Auto, how's the team doing?" gets a
grounded answer and the Agents surface can show what each agent is doing *right
now*. Every input already exists, scattered across board leases, mission-task
state, watches, pending asks, and usage telemetry; this composes them, per
active agent, into a deterministic shape.

Design invariants (PRD-228 §5, binding rules):

* **READ-ONLY.** This service performs zero writes — no session mutation calls
  anywhere. A grep test enforces it; do not introduce persistence here.
* **NO N+1.** A bounded query set (one query per source, six total) regardless
  of agent count — never a per-agent loop hitting the database. Attribution of
  watches/asks/tasks to agents happens in memory from the bulk-loaded rows. A
  query-count assertion test guards this.
* **Fail-soft per source.** An unavailable source omits its fields rather than
  raising: the cost source failing drops ``cost_24h`` from every agent while the
  rest of the response stays intact (same posture as the channels sender).
* **No rival "busy" derivation.** Mission-task busyness reuses the canonical
  :data:`core.models.orchestration_enums.BUSY_TASK_STATES` — the same constant
  the dispatcher's matcher uses, so there is exactly one definition of busy
  across dispatch and reporting.

Cost source (pinned during US-001): the **canonical** per-workspace token/cost
lane is the ``llm_usage`` table (:class:`core.models.core.LLMUsage`) — the same
source the credit/usage/billing surfaces read (``api/llm_analytics.py`` and
``api/kpi_api.py`` both aggregate ``func.sum(LLMUsage.total_cost)`` filtered by
``workspace_id`` + ``created_at``). ``agent.model_usage_stats`` is an explicit
*fallback* written by the same tracker, not a rival store; ``trial_ledger`` is a
platform-wide daily counter, not per-workspace. We reuse ``llm_usage`` directly;
no new cost store is introduced.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from core.models.approval_grants import (
    KIND_QUESTION,
    SUBJECT_BOARD_TASK,
    ApprovalGrant,
    GrantStatus,
)
from core.models.core import Agent, BoardTask, LLMUsage
from core.models.orchestration import OrchestrationTask
# Single source of truth for mission-task busyness (no rival derivation): the
# same constant the dispatcher's matcher uses to decide who is busy.
from core.models.orchestration_enums import BUSY_TASK_STATES, TaskState
from core.models.watch_enums import LIVE_WATCH_STATUSES, WatchTargetType
from core.models.watches import Watch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Read-model version — bump when the emitted shape changes.
FLEET_STATE_VERSION: int = 1

#: The pinned canonical cost source (see module docstring).
COST_SOURCE: str = "llm_usage"

#: Rolling window for cost/token aggregation, in hours (PRD-228 baked decision:
#: rolling 24h, labelled as such in the UI).
COST_WINDOW_HOURS: int = 24

#: Board statuses that count as "on the board" (not archived/terminal). A task
#: with any of these — or with ``blocked_at`` set — is loaded for the fleet.
LIVE_BOARD_STATUSES: tuple[str, ...] = ("inbox", "assigned", "in_progress", "review")

#: Board status meaning "currently executing" (claimed under a lease).
BOARD_STATUS_IN_PROGRESS: str = "in_progress"

#: Board status meaning "assigned but not yet started" (queue depth).
BOARD_STATUS_ASSIGNED: str = "assigned"

#: Live (non-terminal) watch status values, as strings for the IN-filter.
_LIVE_WATCH_STATUS_VALUES: tuple[str, ...] = tuple(
    sorted(s.value for s in LIVE_WATCH_STATUSES)
)

#: Sort floor for nullable ``started_at`` (tz-aware, so comparisons never mix
#: naive/aware datetimes).
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Small helpers (pure)
# ---------------------------------------------------------------------------

def _iso(dt: Optional[datetime]) -> Optional[str]:
    """ISO-8601 string for a datetime, or ``None``."""
    return dt.isoformat() if dt is not None else None


def _max_dt(candidates: Iterable[Optional[datetime]]) -> Optional[datetime]:
    """Latest non-null datetime among candidates, or ``None``.

    Mixed tz-aware/naive datetimes cannot be compared; naive values (only
    ``LLMUsage.created_at`` is naive, and it is not fed here) are skipped
    defensively so a stray naive column can never raise.
    """
    aware = [d for d in candidates if d is not None and d.tzinfo is not None]
    return max(aware) if aware else None


# ---------------------------------------------------------------------------
# Cost source (pinned: llm_usage) — isolated so fail-soft can wrap it
# ---------------------------------------------------------------------------

def _cost_by_agent(
    db: Session,
    workspace_id: UUID,
    agent_ids: Sequence[int],
    since: datetime,
) -> Dict[int, Dict[str, Any]]:
    """Per-agent {tokens, usd} over ``since..now`` from the canonical source.

    One grouped query (mirrors ``api/llm_analytics.py`` ``get_summary``): sums
    ``total_tokens`` and ``total_cost`` on ``llm_usage`` filtered by workspace +
    ``created_at`` window, grouped by ``agent_id``. No per-agent round-trips.
    """
    rows = (
        db.query(
            LLMUsage.agent_id.label("agent_id"),
            func.coalesce(func.sum(LLMUsage.total_tokens), 0).label("tokens"),
            func.coalesce(func.sum(LLMUsage.total_cost), 0.0).label("usd"),
        )
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.agent_id.in_(agent_ids),
            LLMUsage.created_at >= since,
        )
        .group_by(LLMUsage.agent_id)
        .all()
    )
    return {
        row.agent_id: {
            "tokens": int(row.tokens or 0),
            "usd": round(float(row.usd or 0.0), 6),
        }
        for row in rows
        if row.agent_id is not None
    }


def _safe_cost(
    db: Session,
    workspace_id: UUID,
    agent_ids: Sequence[int],
    since: datetime,
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Fail-soft wrapper: ``None`` if the cost source is unavailable.

    A ``None`` return signals the assembler to omit ``cost_24h`` from every
    agent — the rest of the fleet response is unaffected.
    """
    try:
        return _cost_by_agent(db, workspace_id, agent_ids, since)
    except Exception:  # noqa: BLE001 — any cost-source failure degrades softly
        logger.warning(
            "[FleetState] cost source (%s) unavailable; omitting cost fields",
            COST_SOURCE,
            exc_info=True,
        )
        return None


# ---------------------------------------------------------------------------
# Assembly (pure) — no database access; testable with plain row objects
# ---------------------------------------------------------------------------

def _current_from_board(board_tasks: List[Any]) -> Optional[Dict[str, Any]]:
    """The agent's current board work item, or ``None``.

    Current = the most recently started ``in_progress`` board task (a claim held
    under a lease). Deterministic tie-break on id.
    """
    running = [t for t in board_tasks if t.status == BOARD_STATUS_IN_PROGRESS]
    if not running:
        return None
    running.sort(key=lambda t: (t.started_at or _EPOCH, t.id), reverse=True)
    task = running[0]
    return {
        "kind": "board_task",
        "id": task.id,
        "title": task.title,
        "since": _iso(task.started_at),
    }


def _current_from_mission(orch_tasks: List[Any]) -> Optional[Dict[str, Any]]:
    """The agent's current mission task, or ``None``.

    Current = the most recently started ``running`` orchestration task (the
    matcher's busy derivation; ``assigned`` counts as busy but is not yet the
    live work item).
    """
    running = [t for t in orch_tasks if t.state == TaskState.RUNNING.value]
    if not running:
        return None
    running.sort(key=lambda t: (t.started_at or _EPOCH, str(t.id)), reverse=True)
    task = running[0]
    return {
        "kind": "mission_task",
        "id": str(task.id),
        "title": task.title,
        "since": _iso(task.started_at),
    }


def _assemble_fleet(
    agents: List[Any],
    board_tasks: List[Any],
    orch_tasks: List[Any],
    watches: List[Any],
    asks: List[Any],
    costs: Optional[Dict[int, Dict[str, Any]]],
    *,
    generated_at: datetime,
) -> Dict[str, Any]:
    """Compose the deterministic fleet shape from bulk-loaded rows.

    Pure: takes already-loaded rows and returns the response dict. All
    per-agent attribution (tasks, watches, asks) is done here in memory, so the
    caller issues a fixed number of queries irrespective of agent count.

    ``costs`` semantics: a dict (possibly empty) means the cost source is
    available and every agent carries ``cost_24h``; ``None`` means the source
    was unavailable and ``cost_24h`` is omitted everywhere (fail-soft).
    """
    # Bucket tasks by agent (in-memory group-by, no DB).
    board_by_agent: Dict[int, List[Any]] = {}
    for t in board_tasks:
        board_by_agent.setdefault(t.assigned_agent_id, []).append(t)
    orch_by_agent: Dict[int, List[Any]] = {}
    for t in orch_tasks:
        orch_by_agent.setdefault(t.assigned_agent_id, []).append(t)

    cost_available = costs is not None
    cost_map = costs or {}

    fleet: List[Dict[str, Any]] = []
    for agent in agents:
        my_board = board_by_agent.get(agent.id, [])
        my_orch = orch_by_agent.get(agent.id, [])
        my_board_id_strs = {str(t.id) for t in my_board}

        # Current work: board task (leased/in_progress) wins; else running mission.
        current = _current_from_board(my_board) or _current_from_mission(my_orch)

        # Queue depth: assigned-not-started board tasks.
        queue_depth = sum(1 for t in my_board if t.status == BOARD_STATUS_ASSIGNED)

        # Blocked: board tasks flagged blocked + open (pending, question-kind)
        # asks raised against the agent or one of its board tasks.
        blocked_count = sum(1 for t in my_board if t.blocked_at is not None)
        open_asks = _asks_for_agent(agent.id, my_board_id_strs, asks)

        # Active watches touching the agent (owned by it, or targeting one of
        # its board tasks).
        watch_count = _watches_for_agent(agent.id, my_board_id_strs, watches)

        # Last activity: latest task timestamp we already hold (no new tracking).
        last_activity = _max_dt(
            [t.updated_at for t in my_board]
            + [t.started_at for t in my_board]
            + [t.completed_at for t in my_board]
            + [t.updated_at for t in my_orch]
            + [t.started_at for t in my_orch]
            + [t.completed_at for t in my_orch]
        )

        entry: Dict[str, Any] = {
            "agent_id": agent.id,
            "name": agent.name,
            "current": current,
            "queue_depth": queue_depth,
            "blocked": {"count": blocked_count, "open_asks": open_asks},
            "watches": {"active": watch_count},
            "last_activity_at": _iso(last_activity),
        }
        if cost_available:
            entry["cost_24h"] = cost_map.get(agent.id, {"tokens": 0, "usd": 0.0})
        fleet.append(entry)

    return {
        "version": FLEET_STATE_VERSION,
        "generated_at": _iso(generated_at),
        "window_hours": COST_WINDOW_HOURS,
        "cost_available": cost_available,
        "cost_source": COST_SOURCE if cost_available else None,
        "agents": fleet,
    }


def _asks_for_agent(
    agent_id: int, board_id_strs: set, asks: List[Any]
) -> List[Any]:
    """Ids of open asks attributed to this agent (in-memory).

    An ask is the agent's when it names the agent directly (``agent_id`` or
    ``asked_by_agent_id``) or targets one of the agent's board tasks.
    """
    out: List[Any] = []
    for grant in asks:
        if (
            grant.agent_id == agent_id
            or grant.asked_by_agent_id == agent_id
            or (
                grant.subject_type == SUBJECT_BOARD_TASK
                and str(grant.subject_id) in board_id_strs
            )
        ):
            out.append(grant.id)
    return out


def _watches_for_agent(
    agent_id: int, board_id_strs: set, watches: List[Any]
) -> int:
    """Count of live watches touching this agent (in-memory).

    A watch touches the agent when it is owned by it (``owner_agent_id``) or its
    current target is one of the agent's board tasks. Deduped by watch id so the
    two conditions never double-count.
    """
    matched = {
        w.id
        for w in watches
        if w.owner_agent_id == agent_id
        or (
            w.target_type == WatchTargetType.BOARD_TASK.value
            and str(w.target_id) in board_id_strs
        )
    }
    return len(matched)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_fleet_state(db: Session, workspace_id: UUID) -> Dict[str, Any]:
    """Compose the fleet read-model for one workspace.

    Bounded query set (six reads, independent of agent count):
      1. active agents in the workspace
      2. their live board tasks (on-board statuses or flagged blocked)
      3. their busy mission tasks (assigned/running — the matcher's derivation)
      4. live watches in the workspace
      5. pending question-kind asks in the workspace
      6. per-agent 24h cost from the canonical source (fail-soft)
    """
    generated_at = datetime.now(timezone.utc)

    agents = (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id)
        .order_by(Agent.name, Agent.id)
        .all()
    )
    agent_ids = [a.id for a in agents]
    if not agent_ids:
        return _assemble_fleet([], [], [], [], [], {}, generated_at=generated_at)

    board_tasks = (
        db.query(BoardTask)
        .filter(
            BoardTask.workspace_id == workspace_id,
            BoardTask.assigned_agent_id.in_(agent_ids),
            or_(
                BoardTask.status.in_(LIVE_BOARD_STATUSES),
                BoardTask.blocked_at.isnot(None),
            ),
        )
        .all()
    )

    # Mission tasks: scoped by assigned_agent_id (already workspace-bounded via
    # agent_ids), matching the matcher's busy filter exactly.
    orch_tasks = (
        db.query(OrchestrationTask)
        .filter(
            OrchestrationTask.assigned_agent_id.in_(agent_ids),
            OrchestrationTask.state.in_(BUSY_TASK_STATES),
        )
        .all()
    )

    watches = (
        db.query(Watch)
        .filter(
            Watch.workspace_id == workspace_id,
            Watch.status.in_(_LIVE_WATCH_STATUS_VALUES),
        )
        .all()
    )

    asks = (
        db.query(ApprovalGrant)
        .filter(
            ApprovalGrant.workspace_id == workspace_id,
            ApprovalGrant.status == GrantStatus.PENDING.value,
            ApprovalGrant.kind == KIND_QUESTION,
        )
        .all()
    )

    # ``llm_usage.created_at`` is a naive UTC column; match the analytics
    # convention (``datetime.utcnow() - window``) so the window aligns with the
    # credit/usage surfaces.
    cost_since = datetime.utcnow() - timedelta(hours=COST_WINDOW_HOURS)
    costs = _safe_cost(db, workspace_id, agent_ids, cost_since)

    return _assemble_fleet(
        agents, board_tasks, orch_tasks, watches, asks, costs,
        generated_at=generated_at,
    )
