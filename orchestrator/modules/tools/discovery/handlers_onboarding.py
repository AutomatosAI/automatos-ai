"""Onboarding handlers for PlatformActionExecutor (PRD-222 W1S3).

Delegates all state changes to ``services.onboarding_state`` — the single writer
of the stage machine — and returns the client-safe ``{stage, trial}`` snapshot.
Invalid transitions are returned as clear errors, never raised as crashes.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.workspaces import Workspace
from services.onboarding_state import (
    ALL_STAGES,
    QUESTION_KEYS,
    SEGMENT_KEYS,
    InvalidStageTransition,
    advance_onboarding_stage,
    current_stage,
    get_onboarding,
    public_snapshot,
    record_plan_event,
    set_segment,
)

logger = logging.getLogger(__name__)


# What each stage needs before the NEXT advance — returned on a same-stage
# re-assert so a no-op never reads as progress (short: this rides every such
# tool result).
_SAME_STAGE_HINTS = {
    "questions": "Record the answers via segment; the third answer moves the stage to teach.",
    "teach": "Call platform_search_packages, then advance_to proposal with what it returned.",
    "proposal": "Wait for the user's explicit yes, then advance_to building.",
    "building": (
        "Nothing is built yet: install the matched package (platform_install_package) or "
        "create the agents (platform_create_agent / platform_install_marketplace_agent); "
        "advance_to boom only after that succeeds."
    ),
    "boom": "Show the value on their business, then advance_to powerup.",
}


def _usable_segment(segment: Any) -> Optional[Dict[str, Any]]:
    """The segment as a dict carrying at least one recognised key, else None.

    A JSON-encoded string is parsed first: Gemini 2.5 Flash on prod stringifies
    the nested ``segment`` object often enough to matter (live-test 2026-09-02).
    """
    if isinstance(segment, str):
        try:
            segment = json.loads(segment)
        except ValueError:
            return None
    if isinstance(segment, dict) and any(segment.get(k) is not None for k in SEGMENT_KEYS):
        return segment
    return None


def _normalise_params(params: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Coerce the LLM-supplied argument shapes seen in prod into the schema's.

    Returns a NEW dict (the input is never mutated) plus notes of what was
    coerced or dropped — types and keys only, never user text — for the
    handler to log. The shapes, all observed on prod 2026-09-02 (Gemini 2.5
    Flash) while a workspace sat frozen at ``not_started``:

    * ``segment`` as a JSON-encoded string          -> parsed;
    * the answers flat at top level
      (business/goal/comfort/team_size)            -> folded into ``segment``;
    * ``value`` carrying a stage name instead of
      ``advance_to`` (the enum under the wrong key;
      INFERRED from ``keys=['value', 'segment']``)  -> ``advance_to``, only when
      it names a real stage;
    * ``advance_to`` / ``plan`` with stray case or
      whitespace ('Teach', ' Basic ')               -> lower-cased.

    Anything still unusable is dropped with a note; the honest "at least one
    is required" error then applies (2026-08-29 rule: a junk segment must never
    fail a call that also carries a valid advance).
    """
    src = params if isinstance(params, dict) else {}
    out: Dict[str, Any] = dict(src)
    notes: List[str] = []

    value = out.pop("value", None)
    if not out.get("advance_to") and isinstance(value, str) and value.strip().lower() in ALL_STAGES:
        out["advance_to"] = value.strip().lower()
        notes.append("advance_to taken from 'value'")
    elif value is not None and not isinstance(value, str):
        notes.append(f"'value' ignored (type={type(value).__name__})")

    advance_to = out.get("advance_to")
    if isinstance(advance_to, str) and advance_to != advance_to.strip().lower():
        out["advance_to"] = advance_to.strip().lower()
        notes.append("advance_to case/whitespace normalised")

    raw_segment = out.get("segment")
    usable = _usable_segment(raw_segment)
    if isinstance(raw_segment, str) and usable is not None:
        notes.append("segment parsed from a JSON string")
    # Bare text (live-test 2026-09-02, local + prod: the model sends the user's
    # ANSWER as the segment string, sometimes with the question's key name under
    # 'value' — e.g. segment="Fairly comfortable…", value="comfort"). Keep the
    # text aside for the handler, which knows which question is still unanswered.
    bare_text = None
    bare_key = None
    strings = {k: v.strip() for k, v in ((("segment", raw_segment), ("value", value))) if isinstance(v, str) and v.strip()}
    if usable is None and strings:
        keyed = {k: v for k, v in strings.items() if v.lower() in SEGMENT_KEYS}
        texts = {k: v for k, v in strings.items() if v.lower() not in SEGMENT_KEYS and v.lower() not in ALL_STAGES}
        if keyed and texts:
            bare_key = next(iter(keyed.values())).lower()
            bare_text = next(iter(texts.values()))
        elif texts:
            bare_text = next(iter(texts.values()))
    if bare_text:
        out["_bare_answer"] = (bare_key, bare_text)
    flat = {k: out.pop(k) for k in SEGMENT_KEYS if out.get(k) is not None}
    if flat:
        notes.append(f"segment keys arrived flat: {sorted(flat)}")
        usable = {**(usable or {}), **flat}
    if bare_text:
        notes.append("bare-text answer kept for the handler to map onto a question")
    elif raw_segment is not None and usable is None:
        shape = f"type={type(raw_segment).__name__}"
        if isinstance(raw_segment, dict):
            shape += f", keys={sorted(map(str, raw_segment))[:6]}"
        notes.append(f"segment carries nothing usable ({shape}) — dropped")
    out["segment"] = usable

    plan = out.get("plan")
    if isinstance(plan, str) and plan != plan.strip().lower():
        out["plan"] = plan.strip().lower()
        notes.append("plan case/whitespace normalised")
    return out, notes


async def update_onboarding(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Advance the onboarding spine, record segment answers, and/or set the plan.

    Params: ``advance_to`` (next stage), ``segment`` ({business, goal, comfort}),
    and/or ``plan`` (the accepted tier — basic/pro/business only). At least one is
    required. Setting a plan writes plan + plan_limits through the US-023 helper
    (the single writer) and stamps the ``plan_accepted`` funnel event; advancing
    to the proposal stamps ``plan_recommended``. Returns ``{success, data:
    {stage, trial}}``.
    """
    # Argument shapes are external data (LLM-supplied) — coerced at this
    # boundary, never assumed. 2026-08-29: a bare-string segment used to fail
    # the whole call ("'str' object has no attribute 'get'", every onboarding
    # stuck at teach); an unusable one is dropped, a valid advance proceeds.
    # 2026-09-02: the shapes below froze a workspace at not_started — the model
    # sent the stage under 'value' with a stringified segment, was told "at least
    # one is required", retried identically and narrated on with nothing
    # recorded. Every coercion or drop is a WARNING naming the shape (types and
    # keys only, never the user's text) so the next variant names itself.
    params, notes = _normalise_params(params)
    if notes:
        logger.warning("[update_onboarding] argument shape coerced: %s", "; ".join(notes))
    advance_to = params.get("advance_to")
    segment = params.get("segment")
    plan = params.get("plan")

    if not advance_to and not segment and not plan and not params.get("_bare_answer"):
        return {
            "success": False,
            "error": "Provide advance_to, segment, or plan — at least one is required.",
        }

    workspace = (
        db.query(Workspace).filter(Workspace.id == workspace_id).first()
    )
    if not workspace:
        return {"success": False, "error": "workspace not found"}

    # A bare-text answer becomes the first question still unanswered (or the
    # key the model named). The questions are asked in QUESTION order, so the
    # document says which one this is; dropping the text was the freeze.
    bare = params.get("_bare_answer")
    if bare and not segment and not advance_to:
        key, text = bare
        answered = get_onboarding(workspace).get("segment") or {}
        if not key:
            key = next((k for k in QUESTION_KEYS if answered.get(k) is None), None)
        if key:
            segment = {key: text}
            logger.warning("[update_onboarding] bare-text answer recorded as '%s' (%s)", key,
                           "key named by the model" if bare[0] else "first unanswered question")
    if not advance_to and not segment and not plan:
        return {
            "success": False,
            "error": "Provide advance_to, segment, or plan — at least one is required.",
        }

    # Idempotent same-stage advance (live-test 2026-08-29): the LLM routinely
    # re-asserts the stage it is already in — e.g. calling advance_to="building"
    # while building — and the strict monotonic validator raised
    # "non-forward transition 'building' -> 'building'", which the tool surfaced
    # as an error. Auto then apologised to the user for a "hiccup" and looped.
    # Re-asserting the current stage is a benign no-op, not an error: drop the
    # redundant advance (segment/plan writes below still run). BACKWARD and
    # UNKNOWN targets are untouched — they still validate-and-error.
    if advance_to and advance_to == current_stage(workspace):
        logger.info(
            "[update_onboarding] advance_to=%s equals current stage — idempotent no-op",
            advance_to,
        )
        advance_to = None
        if not segment and not plan:
            # Honest no-op (local test 2026-09-02): Auto re-asserted 'building'
            # twice while SAYING "I'm proceeding with the installation" and
            # installing nothing — a bare success read as progress. Say what
            # changed (nothing) and what the stage actually needs.
            snap = public_snapshot(workspace)
            hint = _SAME_STAGE_HINTS.get(snap.get("stage"), "")
            return {
                "success": True,
                "data": snap,
                "noop": True,
                "message": f"Already at '{snap.get('stage')}' — nothing changed. {hint}".strip(),
            }

    # Reject a non-assignable plan BEFORE any write — honest coming-soon copy.
    if plan is not None:
        from services.plan_tiers import is_assignable

        if not is_assignable(plan):
            # Honest copy: only enterprise is "coming soon"; anything else is not
            # a tier at all (live-test 2026-09-02: 'Basic' was answered with
            # "Enterprise is coming soon" — case is forgiven above, so this
            # branch now only sees genuine non-tiers).
            return {
                "success": False,
                "error": (
                    "'enterprise' can't be assigned yet — Enterprise is coming soon. "
                    "Choose basic, pro, or business."
                    if plan == "enterprise"
                    else f"'{plan}' isn't a plan tier — choose basic, pro, or business."
                ),
            }

    # ATOMICITY (FR-4, RVW-2): every writer below runs with commit=False and the
    # WHOLE tool call is committed ONCE at the end. A state change (plan +
    # plan_limits, or a stage advance) and its funnel audit stamp therefore land
    # together or not at all — a durable change can never be reported success:False
    # (nothing committed on failure) nor left without its recorded audit event.
    try:
        if advance_to:
            # A single write advances the stage and merges any segment answers.
            advance_onboarding_stage(db, workspace, advance_to, segment=segment, commit=False)
        elif segment:
            set_segment(db, workspace, segment, commit=False)

        # Funnel: reaching the proposal records the plan Auto will recommend
        # (derived from the stored segment). Auditable, FR-4.
        if advance_to == "proposal":
            from services.plan_tiers import recommend_plan

            rec_plan, _reason = recommend_plan(get_onboarding(workspace).get("segment") or {})
            record_plan_event(db, workspace, "plan_recommended", rec_plan, commit=False)

        # Accepting a plan writes plan + plan_limits through the US-023 helper and
        # stamps plan_accepted — state changes only through this tool (FR-4).
        if plan is not None:
            from services.plan_tiers import assign_plan

            assign_plan(db, workspace, plan, commit=False)
            record_plan_event(db, workspace, "plan_accepted", plan, commit=False)

        # The single commit — all deferred writes land as one transaction.
        if db is not None:
            db.commit()
    except InvalidStageTransition as exc:
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}
    except ValueError as exc:
        # e.g. set_segment called with no recognised keys.
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - surface a clean tool error, never crash
        logger.error("[update_onboarding] failed: %s", exc, exc_info=True)
        if db is not None:
            db.rollback()
        return {"success": False, "error": str(exc)}

    return {"success": True, "data": public_snapshot(workspace)}
