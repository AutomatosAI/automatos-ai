"""PRD-222 W1S1 — server-side onboarding state machine (Mission Zero v2).

The Wave-1 funnel record. Every workspace carries a single ``onboarding`` JSONB
document (added by the ``prd222_w1s1_onboarding_jsonb`` migration); this service
is the ONLY writer of that document's stage machine. There is no second table
and there will be no second migration on this branch — the $5 trial ledger
(W1S9), the segment answers, and the per-stage funnel timestamps all live inside
this one JSONB blob.

Two hard rules are encoded here:

1. **Rebuild, never mutate.** Every write deep-copies the current document,
   edits the copy, and reassigns ``workspace.onboarding`` to a NEW object. An
   in-place ``dict`` mutation is invisible to SQLAlchemy's JSONB change
   detection (the PRD-220 silent-loss bug class), so it is made structurally
   impossible here — ``get_onboarding`` hands out copies and ``_persist``
   assigns a fresh object.
2. **Monotonic forward.** Stage transitions only move forward through
   ``STAGE_ORDER``; ``skipped`` is reachable from any non-terminal stage;
   ``completed`` and ``skipped`` are terminal. Backward / same-stage / unknown /
   from-terminal moves raise :class:`InvalidStageTransition`.

Document shape::

    {
      "stage": "not_started",              # current stage (one of ALL_STAGES)
      "stages": {"questions": "<iso>"},    # per-stage funnel timestamps (W1S1)
      "segment": {"business", "goal", "comfort"},
      "started_at": "<iso>",               # first advance away from not_started
      "updated_at": "<iso>",               # every write
      "completed_at": "<iso>",             # set when reaching completed/skipped
      # "trial": {...}                     # added by W1S9 (US-004/005)
    }
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Optional

# Ordered spine (app-level enum, NOT a Postgres enum). ``skipped`` is a terminal
# branch reachable from any non-terminal stage, so it is deliberately NOT part
# of the linear order used for the monotonic-forward index check.
STAGE_ORDER: tuple[str, ...] = (
    "not_started",
    "questions",
    "teach",
    "proposal",
    "building",
    "boom",
    "powerup",
    "completed",
)
SKIPPED = "skipped"
INITIAL_STAGE = "not_started"
ALL_STAGES: frozenset[str] = frozenset(STAGE_ORDER) | {SKIPPED}
TERMINAL_STAGES: frozenset[str] = frozenset({"completed", SKIPPED})
SEGMENT_KEYS: tuple[str, ...] = ("business", "goal", "comfort")


class InvalidStageTransition(ValueError):
    """Raised on a backward, same-stage, unknown, or from-terminal transition."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_onboarding() -> dict[str, Any]:
    return {"stage": INITIAL_STAGE, "stages": {}, "segment": {}}


def get_onboarding(workspace: Any) -> dict[str, Any]:
    """Return the workspace's onboarding document as a defaulted COPY.

    Null-safe: a workspace whose column is NULL / absent / empty reads as a
    fresh ``not_started`` document. The returned dict is a deep copy — callers
    must go through :func:`advance_onboarding_stage` / :func:`set_segment` to
    persist, never mutate the return value expecting it to save.
    """
    raw = getattr(workspace, "onboarding", None)
    if not raw:
        return _default_onboarding()
    doc = copy.deepcopy(raw)
    doc.setdefault("stage", INITIAL_STAGE)
    doc.setdefault("stages", {})
    doc.setdefault("segment", {})
    return doc


def current_stage(workspace: Any) -> str:
    """The workspace's current onboarding stage (``not_started`` when unset)."""
    return get_onboarding(workspace).get("stage", INITIAL_STAGE)


def is_onboarding_active(workspace: Any) -> bool:
    """True while the spine should run — stage NOT IN (completed, skipped)."""
    return current_stage(workspace) not in TERMINAL_STAGES


def public_snapshot(workspace: Any) -> dict[str, Any]:
    """The client-facing onboarding view: ``{stage, trial}``.

    Shared by ``GET /api/workspaces/current`` (W1S2/US-002) and the
    ``platform_update_onboarding`` tool (W1S3/US-003) so both speak one shape.
    ``trial`` is ``None`` until W1S9 grants it; only the three client-safe trial
    fields are exposed — never internal ledger bookkeeping.
    """
    doc = get_onboarding(workspace)
    trial = doc.get("trial")
    trial_out = None
    if trial:
        trial_out = {
            "granted_usd": trial.get("granted_usd"),
            "spent_usd": trial.get("spent_usd", 0),
            "state": trial.get("state"),
        }
    return {"stage": doc.get("stage", INITIAL_STAGE), "trial": trial_out}


def _stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError as exc:  # pragma: no cover - guarded by caller
        raise InvalidStageTransition(f"unknown onboarding stage: {stage!r}") from exc


def _validate_transition(current: str, target: str) -> None:
    if target not in ALL_STAGES:
        raise InvalidStageTransition(f"unknown onboarding stage: {target!r}")
    if current in TERMINAL_STAGES:
        raise InvalidStageTransition(
            f"cannot advance from terminal stage {current!r} to {target!r}"
        )
    if target == SKIPPED:
        return  # reachable from any non-terminal stage
    if _stage_index(target) <= _stage_index(current):
        raise InvalidStageTransition(
            f"non-forward onboarding transition {current!r} -> {target!r}"
        )


def _clean_segment(segment: Optional[dict]) -> dict[str, Any]:
    """Keep only the three known segment keys carrying a non-None value."""
    if not segment:
        return {}
    return {k: segment[k] for k in SEGMENT_KEYS if segment.get(k) is not None}


def _persist(db: Any, workspace: Any, new_doc: dict[str, Any]) -> dict[str, Any]:
    """Assign a NEW dict to the column (never mutate) and commit.

    ``db is None`` runs the assignment only — the escape hatch for pure logic
    tests that verify the rebuild contract without a session.
    """
    workspace.onboarding = new_doc
    if db is not None:
        db.add(workspace)
        db.commit()
    return new_doc


def advance_onboarding_stage(
    db: Any,
    workspace: Any,
    to_stage: str,
    *,
    segment: Optional[dict] = None,
) -> dict[str, Any]:
    """Advance the workspace to ``to_stage``, stamping the funnel timestamp.

    Rebuilds the whole document and reassigns it (rebuild-don't-mutate).
    Optionally merges segment answers in the same write. Raises
    :class:`InvalidStageTransition` on a backward / same-stage / unknown /
    from-terminal move.
    """
    doc = get_onboarding(workspace)  # already a deep copy
    current = doc.get("stage", INITIAL_STAGE)
    _validate_transition(current, to_stage)

    # W1S2/US-002 funnel decision — the JSONB per-stage timestamps below ARE the
    # Wave-1 funnel record. There is no generic analytics/funnel event sink to
    # emit an ``onboarding_stage_changed`` event into: grepping
    # ``orchestrator/`` for record_event/emit_event/track_event/funnel_event
    # turns up only ``services/orchestration_state.emit_event``, which is
    # hard-bound to ``OrchestrationEvent`` and REQUIRES an OrchestrationRun
    # ``run_id`` (mission audit trail, not onboarding); the other *_events tables
    # (widget_event_log, substrate_metric_events, unrouted_events, watch_events,
    # error_events) are each domain-specific. Per the PRD-222 US-002 contract
    # ("do NOT invent a new table or plane"), the ``stages[<stage>]`` ISO stamps
    # stand as the funnel record until a real analytics plane exists (W2+).

    now = _now_iso()
    doc["stage"] = to_stage
    stamped = dict(doc.get("stages") or {})
    stamped[to_stage] = now
    doc["stages"] = stamped
    doc.setdefault("started_at", now)
    doc["updated_at"] = now
    if to_stage in TERMINAL_STAGES:
        doc["completed_at"] = now
    if segment:
        merged = dict(doc.get("segment") or {})
        merged.update(_clean_segment(segment))
        doc["segment"] = merged

    return _persist(db, workspace, doc)


def set_segment(db: Any, workspace: Any, segment: dict) -> dict[str, Any]:
    """Merge segment answers (business/goal/comfort) without advancing the stage.

    Rebuild-don't-mutate. Raises ``ValueError`` if no recognised segment key is
    supplied (so a no-op write can never masquerade as a saved answer).
    """
    cleaned = _clean_segment(segment)
    if not cleaned:
        raise ValueError(
            "set_segment requires at least one of business/goal/comfort"
        )
    doc = get_onboarding(workspace)
    merged = dict(doc.get("segment") or {})
    merged.update(cleaned)
    doc["segment"] = merged
    doc["updated_at"] = _now_iso()
    return _persist(db, workspace, doc)
