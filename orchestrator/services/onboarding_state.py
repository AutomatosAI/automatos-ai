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
      "segment": {"business", "goal", "comfort", "team_size"},
      "started_at": "<iso>",               # first advance away from not_started
      "updated_at": "<iso>",               # every write
      "completed_at": "<iso>",             # set when reaching completed/skipped
      # "trial": {...}                     # added by W1S9 (US-004/005)
    }
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

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
SEGMENT_KEYS: tuple[str, ...] = ("business", "goal", "comfort", "team_size")


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
    """Keep only the three known segment keys carrying a non-None value.

    Boundary-hardened (live-test 2026-08-29): ``segment`` is LLM-supplied through
    the ``platform_update_onboarding`` tool, and the model sometimes passes it as
    a bare string (a free-text business summary) instead of the
    ``{business, goal, comfort}`` object the schema asks for. A non-dict value
    used to reach ``segment.get(k)`` and raise ``'str' object has no attribute
    'get'`` — which failed the WHOLE advance-to-proposal call and stalled every
    onboarding at ``teach``. A non-dict segment now cleans to ``{}`` (ignored:
    the real answers were already captured on the question turns), so the stage
    advance still lands.
    """
    if not isinstance(segment, dict):
        return {}
    return {k: segment[k] for k in SEGMENT_KEYS if segment.get(k) is not None}


def _persist(
    db: Any, workspace: Any, new_doc: dict[str, Any], *, commit: bool = True
) -> dict[str, Any]:
    """Assign a NEW dict to the column (never mutate) and commit.

    ``db is None`` runs the assignment only — the escape hatch for pure logic
    tests that verify the rebuild contract without a session. ``commit=False``
    flushes but leaves the transaction OPEN, so a caller making several writes
    (e.g. a stage advance + a plan funnel stamp) commits them ATOMICALLY in one
    transaction (FR-4 — see ``handlers_onboarding.update_onboarding``).
    """
    workspace.onboarding = new_doc
    if db is not None:
        db.add(workspace)
        if commit:
            db.commit()
        else:
            db.flush()
    return new_doc


def advance_onboarding_stage(
    db: Any,
    workspace: Any,
    to_stage: str,
    *,
    segment: Optional[dict] = None,
    commit: bool = True,
) -> dict[str, Any]:
    """Advance the workspace to ``to_stage``, stamping the funnel timestamp.

    Rebuilds the whole document and reassigns it (rebuild-don't-mutate).
    Optionally merges segment answers in the same write. ``commit=False`` defers
    the commit to an atomic caller (see ``_persist``). Raises
    :class:`InvalidStageTransition` on a backward / same-stage / unknown /
    from-terminal move.
    """
    doc = get_onboarding(workspace)  # already a deep copy
    current = doc.get("stage", INITIAL_STAGE)
    _validate_transition(current, to_stage)

    # Honesty gate: the payoff stage needs a build to show. With no session
    # (pure-document logic path) there is nothing to count against — the
    # ordering validator alone applies, as before.
    if (
        to_stage in BUILD_EVIDENCE_STAGES
        and _stage_index(current) < _stage_index(BUILD_EVIDENCE_STAGE)
        and db is not None
    ):
        if not build_evidence(db, workspace)["any"]:
            raise InvalidStageTransition(
                f"{to_stage} needs a build first: nothing is registered to this workspace "
                "yet (no package installed, no agents created, no mission). Install "
                "the matched package or create the agents, then advance to boom."
            )

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

    return _persist(db, workspace, doc, commit=commit)


def set_segment(
    db: Any, workspace: Any, segment: dict, *, commit: bool = True
) -> dict[str, Any]:
    """Merge segment answers (business/goal/comfort) without advancing the stage.

    Rebuild-don't-mutate. ``commit=False`` defers the commit to an atomic caller.
    Raises ``ValueError`` if no recognised segment key is supplied (so a no-op
    write can never masquerade as a saved answer).
    """
    cleaned = _clean_segment(segment)
    if not cleaned:
        raise ValueError(
            "set_segment requires at least one of business/goal/comfort"
        )
    doc = get_onboarding(workspace)
    merged = dict(doc.get("segment") or {})
    merged.update(cleaned)
    implied = implied_stage(doc.get("stage", INITIAL_STAGE), merged)
    if implied:
        # The recorded answers prove the stage (see implied_stage) — one write
        # advances and merges, exactly as an explicit advance_to would.
        return advance_onboarding_stage(db, workspace, implied, segment=cleaned, commit=commit)
    doc["segment"] = merged
    doc["updated_at"] = _now_iso()
    return _persist(db, workspace, doc, commit=commit)


# The three answers the questions stage exists to collect.
QUESTION_KEYS: tuple[str, ...] = ("business", "goal", "comfort")


def implied_stage(current: str, segment: dict) -> Optional[str]:
    """The stage the RECORDED answers prove, or None.

    Live test 2026-09-02 (prod, Gemini 2.5 Flash): the model saved answers
    through the tool — ``platform_update_onboarding success=True`` twice — but
    never sent ``advance_to``, so the stage sat at ``not_started`` while Auto
    recited the questions; the user saw ordinary chat. The facts are in the
    document: an answer being recorded means the questions are underway; all
    three recorded means they are done. Inferring from those facts is honest,
    deterministic, and still tool-driven (nothing moves without the call).
    Explicit ``advance_to`` targets are never overridden — this only applies
    to segment-only writes.
    """
    if current == INITIAL_STAGE and any(segment.get(k) is not None for k in QUESTION_KEYS):
        implied = "questions"
    else:
        implied = None
    if current in (INITIAL_STAGE, "questions") and all(
        segment.get(k) is not None for k in QUESTION_KEYS
    ):
        implied = "teach"
    return implied


# The funnel-timestamp key stamped by the first integration a workspace connects
# (PRD-222 W2·S3 / US-019). Lives in the same onboarding doc as the per-stage
# timestamps — the Wave-1 funnel record — never exposed by ``public_snapshot``.
FIRST_INTEGRATION_KEY = "first_integration_connected_at"


def record_first_integration_connected(db: Any, workspace: Any) -> bool:
    """Stamp the ``first_integration_connected`` funnel event — once per workspace.

    The Wave-1 funnel has no generic event sink (see ``advance_onboarding_stage``'s
    note): a "funnel event" is a single ISO timestamp in this onboarding JSONB doc
    plus a log line. This mirrors the trial_* funnel events (US-004/005) — an
    idempotent stamp guarded by presence so it fires EXACTLY ONCE per workspace,
    ever, even if the workspace later disconnects every app and reconnects.

    Rebuild-don't-mutate: reassigns a NEW onboarding dict (PRD-220-safe). The
    caller decides WHEN (the active-connection count crossing 0 → 1); this decides
    only whether it has already been recorded. Returns ``True`` when it stamped
    (the first time), ``False`` on the idempotent no-op.
    """
    doc = get_onboarding(workspace)  # deep copy
    if doc.get(FIRST_INTEGRATION_KEY):
        return False  # already recorded — exactly once per workspace
    now = _now_iso()
    doc[FIRST_INTEGRATION_KEY] = now
    doc["updated_at"] = now
    _persist(db, workspace, doc)
    logger.info(
        "Funnel: first_integration_connected for workspace %s",
        getattr(workspace, "id", None),
    )
    return True


# PRD-222 W2·S2 (US-025) — plan funnel events. Same mechanism as the trial_* and
# first_integration events: a named entry (plan + ISO timestamp) in this
# onboarding JSONB doc, the Wave-1 funnel record — there is no generic analytics
# sink to emit into (see ``advance_onboarding_stage``'s note).
PLAN_FUNNEL_EVENTS = ("plan_recommended", "plan_accepted")


def record_plan_event(
    db: Any, workspace: Any, event: str, plan: str, *, commit: bool = True
) -> dict[str, Any]:
    """Stamp a plan funnel event (``plan_recommended`` / ``plan_accepted``).

    Records ``{plan, at}`` under ``onboarding.funnel[event]`` (rebuild-don't-mutate,
    PRD-220-safe), overwriting so it reflects the latest recommendation/acceptance.
    ``commit=False`` defers the commit so the stamp lands ATOMICALLY with the
    plan/stage write that precedes it (FR-4 — see ``update_onboarding``). Raises
    ``ValueError`` for an unknown event name — the funnel keys are a fixed,
    auditable set. Returns the new onboarding doc.
    """
    if event not in PLAN_FUNNEL_EVENTS:
        raise ValueError(f"unknown plan funnel event: {event!r}")
    doc = get_onboarding(workspace)  # deep copy
    now = _now_iso()
    funnel = dict(doc.get("funnel") or {})
    funnel[event] = {"plan": plan, "at": now}
    doc["funnel"] = funnel
    doc["updated_at"] = now
    _persist(db, workspace, doc, commit=commit)
    logger.info(
        "Funnel: %s plan=%s for workspace %s", event, plan, getattr(workspace, "id", None)
    )
    return doc


# PRD-230 US-006/US-009 — package funnel events. Same mechanism as the plan/trial
# events: a named entry ({slug, at}) under ``onboarding.funnel[event]`` in this
# onboarding JSONB doc, the Wave-1 funnel record. ``package_installed`` also gates
# the D6 one-package-during-onboarding restriction (read by the install tool).
PACKAGE_FUNNEL_EVENTS = ("package_offered", "package_accepted", "package_installed")


def record_package_event(
    db: Any, workspace: Any, event: str, slug: str, *, commit: bool = True
) -> dict[str, Any]:
    """Stamp a package funnel event (``package_offered`` / ``package_accepted`` /
    ``package_installed``). Records ``{slug, at}`` under ``onboarding.funnel[event]``
    (rebuild-don't-mutate, PRD-220-safe). ``commit=False`` defers so the stamp
    lands atomically with a preceding write. Raises ``ValueError`` for an unknown
    event. Returns the new onboarding doc."""
    if event not in PACKAGE_FUNNEL_EVENTS:
        raise ValueError(f"unknown package funnel event: {event!r}")
    doc = get_onboarding(workspace)  # deep copy
    now = _now_iso()
    funnel = dict(doc.get("funnel") or {})
    funnel[event] = {"slug": slug, "at": now}
    doc["funnel"] = funnel
    doc["updated_at"] = now
    _persist(db, workspace, doc, commit=commit)
    logger.info(
        "Funnel: %s slug=%s for workspace %s", event, slug, getattr(workspace, "id", None)
    )
    return doc


def onboarding_package_installed(workspace: Any) -> bool:
    """True once a package has been installed during THIS onboarding (D6 gate)."""
    doc = get_onboarding(workspace) or {}
    return bool((doc.get("funnel") or {}).get("package_installed"))


# =========================================================================== #
# Build evidence — the honesty gate on ``boom`` (live test 2026-08-29)
# =========================================================================== #
#
# Two personas (Saffron, Waggle) reached the payoff stage having built NOTHING:
# ``_validate_transition`` checks ordering only, so ``advance_to="boom"`` from
# ``building`` always succeeded and Auto presented a team that did not exist.
# ``boom`` is "here is your team" — it is reachable only once the workspace
# actually holds a build. "Built" reuses the purge's definition
# (``services.workspace_purge._AGENT_SURVIVOR_SQL``, inverted): a workspace-owned
# agent that is neither a platform system agent nor a hidden onboarding-role
# agent. The package funnel stamp and a mission (the larger-build path, which
# awaits approval) count too. Read-only — no new table, no new stamp: only the
# rows onboarding already writes.

BUILD_EVIDENCE_STAGE = "boom"
# Reaching ANY stage at or past the payoff from before it needs the build — the
# validator allows forward skips, so ``advance_to="completed"`` from ``building``
# would otherwise walk around the boom gate (found 2026-09-02).
BUILD_EVIDENCE_STAGES: frozenset[str] = frozenset({"boom", "powerup", "completed"})


def build_evidence(db: Any, workspace: Any) -> dict[str, Any]:
    """What this workspace holds that onboarding could have built.

    Returns ``{package_installed, agents_built, missions, any}``. With ``db``
    None (the pure-document logic path) the live counts are 0 and only the
    funnel stamp can supply evidence.
    """
    package_installed = onboarding_package_installed(workspace)
    agents_built = 0
    missions = 0
    workspace_id = getattr(workspace, "id", None)
    if db is not None and workspace_id is not None:
        from sqlalchemy import or_

        from core.models.core import Agent
        from core.models.orchestration import OrchestrationRun

        agents_built = int(
            db.query(Agent)
            .filter(
                Agent.workspace_id == workspace_id,
                Agent.is_system_agent.isnot(True),
                or_(Agent.required_role.is_(None), Agent.required_role != "onboarding"),
            )
            .count()
            or 0
        )
        missions = int(
            db.query(OrchestrationRun)
            .filter(OrchestrationRun.workspace_id == workspace_id)
            .count()
            or 0
        )
    return {
        "package_installed": bool(package_installed),
        "agents_built": agents_built,
        "missions": missions,
        "any": bool(package_installed or agents_built or missions),
    }


# =========================================================================== #
# PRD-222 W2·S4 (US-020) — the post-setup "run & learn" CHECKLIST.
#
# 3–5 outcome-framed next steps that survive across sessions. Only two dismissal
# flags are STORED in ``onboarding.checklist`` ({dismissed, academy_done}); every
# item's completion is RE-DERIVED from live workspace counts on each read, so a
# tick can never drift from reality. Server is the record (D8) — no localStorage.
# =========================================================================== #

CHECKLIST_KEY = "checklist"

# The Academy lives in a sibling repo at academy.automatos.app. This is the
# static comfort → course mapping the PRD asks for (novice → ABF "AI for
# business", technical → APA "platform"); the referral-parameter deep links are
# W3·S1's job, so these are the plain course entry points.
ACADEMY_BASE_URL = "https://academy.automatos.app"


def academy_url_for_comfort(comfort: Optional[str]) -> str:
    """Map the stored ``segment.comfort`` to a static Academy course URL.

    ``technical`` (or an explicit APA/advanced signal) → APA; everything else,
    including ``novice`` / "brand new" / unset, → ABF (the owner track).
    """
    c = (comfort or "").strip().lower()
    course = "apa" if ("technical" in c or c in ("apa", "advanced", "expert")) else "abf"
    return f"{ACADEMY_BASE_URL}/{course}"


def build_checklist(
    *,
    connections_count: int,
    missions_count: int,
    members_count: int,
    plan_seats: int,
    comfort: Optional[str] = None,
    stored: Optional[dict] = None,
) -> dict[str, Any]:
    """Compute the post-setup checklist from LIVE counts + the stored flags.

    Completion is DERIVED where the platform already records the outcome — never a
    manual tick:

      * ``connect_second_app`` → ``connections_count >= 2`` (a *second* app)
      * ``run_first_mission``  → ``missions_count >= 1``
      * ``invite_teammate``    → ``members_count >= 2`` — and the item is OMITTED
        entirely on single-seat plans (``plan_seats <= 1``)
      * ``take_course``        → the ONE manual exception: no cross-repo completion
        signal exists (the Academy is a sibling repo), so it is checked on dismiss
        and tracked in ``stored.academy_done``.

    ``stored`` is the persisted ``onboarding.checklist`` doc
    (``{dismissed, academy_done}``). Returns the client-facing view.
    """
    s = stored or {}
    items: list[dict[str, Any]] = [
        {
            "id": "connect_second_app",
            "label": "Connect a second app",
            "done": connections_count >= 2,
        },
        {
            "id": "run_first_mission",
            "label": "Run your first mission",
            "done": missions_count >= 1,
        },
    ]
    if plan_seats and plan_seats > 1:
        items.append(
            {
                "id": "invite_teammate",
                "label": "Invite a teammate",
                "done": members_count >= 2,
            }
        )
    items.append(
        {
            "id": "take_course",
            "label": "Take the matched Academy course",
            # Manual: no completion signal crosses repos — checked on dismiss.
            "done": bool(s.get("academy_done")),
            "href": academy_url_for_comfort(comfort),
            "manual": True,
        }
    )
    return {
        "items": items,
        "dismissed": bool(s.get("dismissed")),
        "completed_count": sum(1 for i in items if i["done"]),
        "total_count": len(items),
    }


def get_checklist_state(workspace: Any) -> dict[str, Any]:
    """Read the STORED checklist flags (``{dismissed, academy_done}``) as a copy."""
    return dict(get_onboarding(workspace).get(CHECKLIST_KEY) or {})


def update_checklist(
    db: Any,
    workspace: Any,
    *,
    dismissed: Optional[bool] = None,
    academy_done: Optional[bool] = None,
) -> dict[str, Any]:
    """Persist the checklist dismissal flags — full-JSONB-reassignment (PRD-220-safe).

    Only ``dismissed`` and ``academy_done`` are stored; the derived item
    completion is recomputed from live counts on every read, never persisted.
    Rebuild-don't-mutate: a NEW ``onboarding`` dict is reassigned (same style as
    ``reset_onboarding`` / ``_write_trial``). Returns the new stored flags.
    """
    doc = get_onboarding(workspace)  # deep copy
    stored = dict(doc.get(CHECKLIST_KEY) or {})
    if dismissed is not None:
        stored["dismissed"] = bool(dismissed)
    if academy_done is not None:
        stored["academy_done"] = bool(academy_done)
    doc[CHECKLIST_KEY] = stored
    doc["updated_at"] = _now_iso()
    _persist(db, workspace, doc)
    return stored


# =========================================================================== #
# PRD-222 W1·S10 (D9) — the dev onboarding RESET.
#
# This is the ONLY sanctioned BACKWARD writer of the onboarding document. It
# rewinds ONE workspace to a fresh ``not_started`` so the operator can re-run
# onboarding with a single alias account, instead of provisioning and hard-
# deleting a workspace per attempt. It does so by REPLACING the whole document
# (rebuild-don't-mutate), NOT by driving ``advance_onboarding_stage`` — the
# monotonic/terminal validator above stays strict and untouched.
# =========================================================================== #


def _regrant_trial(db: Any, workspace: Any) -> tuple[Optional[dict[str, Any]], str]:
    """Re-grant the one-time trial after ``reset_trial`` stripped it.

    REUSES the provisioning grant (``grant_trial_at_provisioning``) — never a
    second grant implementation. The workspace's onboarding doc must already be
    trial-less AND flushed to the DB when this runs, so the grant's one-per-user
    check sees the strip. A kill-switch / daily-cap / already-held decline is a
    reported PAUSE, not an error: the grant returns ``None`` and we surface it.
    """
    from services.trial_ledger import grant_trial_at_provisioning

    trial = grant_trial_at_provisioning(
        db, getattr(workspace, "id", None), owner_id=getattr(workspace, "owner_id", None)
    )
    if trial is None:
        return None, "paused (trial disabled, daily cap reached, or already held)"
    return trial, "granted"


def reset_onboarding(
    db: Any,
    workspace: Any,
    *,
    reset_trial: bool = False,
    wipe_built: bool = False,
    wipe_credentials: bool = False,
) -> dict[str, Any]:
    """Rewind ONE workspace's onboarding to a fresh ``not_started``.

    Full-document JSONB REASSIGNMENT (a brand-new dict is assigned to
    ``workspace.onboarding`` — never an in-place edit), matching ``_write_trial``'s
    whole-value style so the PRD-220 silent-loss bug class cannot occur. Stamps an
    incrementing ``resets`` counter and ``last_reset_at`` inside the doc.

    Flags (all default off):
      * ``reset_trial`` — strip the trial, then re-grant a fresh $0 active one via
        the provisioning grant (a decline is a reported pause, not an error).
      * ``wipe_built`` — delete what onboarding built (non-system agents + deps,
        missions + orchestration tasks, reports/Deliverables, intake documents +
        graphs, and the S3 document prefix), sparing identity/access/credential/
        system-agent survivors — reusing ``services.workspace_purge`` machinery.
      * ``wipe_credentials`` — delete THIS workspace's credential rows only.

    Returns a report dict of everything reset/wiped. Commits when ``db`` is a real
    session; ``db is None`` runs the pure document rebuild (logic tests).
    """
    prev = get_onboarding(workspace)  # deep copy
    now = _now_iso()
    resets = int(prev.get("resets") or 0) + 1
    workspace_id = getattr(workspace, "id", None)

    report: dict[str, Any] = {
        "stage": INITIAL_STAGE,
        "resets": resets,
        "last_reset_at": now,
        "reset_trial": reset_trial,
        "wipe_built": wipe_built,
        "wipe_credentials": wipe_credentials,
        "built": None,
        "credentials": None,
        "trial": None,
        "trial_note": None,
    }

    # 1) Destructive wipes first, inside the same transaction, so a failure
    #    aborts before the document is rewritten (all-or-nothing).
    if wipe_built and db is not None and workspace_id is not None:
        from services.workspace_purge import purge_built_artifacts

        report["built"] = purge_built_artifacts(db, workspace_id)
    if wipe_credentials and db is not None and workspace_id is not None:
        from services.workspace_purge import purge_workspace_credentials

        report["credentials"] = purge_workspace_credentials(db, workspace_id)

    # 2) Rebuild the onboarding document (full reassignment — never mutate).
    preserved_trial = None if reset_trial else prev.get("trial")
    new_doc: dict[str, Any] = {
        "stage": INITIAL_STAGE,
        "stages": {},
        "segment": {},
        "resets": resets,
        "last_reset_at": now,
        "updated_at": now,
    }
    if preserved_trial is not None:
        new_doc["trial"] = copy.deepcopy(preserved_trial)
        report["trial"] = preserved_trial
        report["trial_note"] = "preserved"
    workspace.onboarding = new_doc
    if db is not None:
        db.add(workspace)
        # Flush the trial-less row so the raw-SQL re-grant below sees the strip
        # (the grant's one-per-user check reads onboarding.trial from the DB).
        db.flush()

    # 3) Re-grant the trial AFTER the document is trial-less and flushed.
    if reset_trial:
        report["trial"], report["trial_note"] = _regrant_trial(db, workspace)

    if db is not None:
        db.commit()
        # The re-grant wrote onboarding.trial straight to the row via jsonb_set;
        # refresh so a caller reading workspace.onboarding sees committed truth.
        try:
            db.refresh(workspace)
        except Exception:  # pragma: no cover - convenience refresh only
            pass

    return report
