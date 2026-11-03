"""PRD-268 §6.11.4, US-C20a, US-C24 — the ``ApprovalView@1`` a view's id names (``GET /api/v1/approval-grants/{id}/view``).

``view_of`` reads the id the view carries as its ``grant_id``: a number is an approval grant's
row; a UUID is a Mission plan's id (a plan has no grant row, so its view carries the Mission's
id). Anything else names nothing.

A Mission plan's view is read from the Mission in the caller's workspace: its goal, its card's
number and its plan's steps (the first ``PLAN_STEPS_SHOWN``, as the card shows them). While the
Mission awaits approval it names the policy holding it (``mission_approval_rule.policy_holding``,
the card's reader: a policy that can't be read names none, logged, and the view still stands) and
offers Refuse and Allow once, answered on ``POST /api/missions/{id}/reject`` and ``/approve``. A
plan no longer waiting offers no answers and reads the plain sentence: the policy now is not the
one that held it then.

The grant is read in the caller's workspace only, and only an approval (a question is answered,
never allowed: it has no view). Who asked and the ticket come from ``grant_owners`` (the agent
by name, the ticket by number and title, in the workspace). The target and the preview are read
from the stored call (``details.params``) through the card's digest
(``card_digest.params_digest``), never from the raw params, each ticket parameter by its board
number as the card shows it (``card_ticket_numbers.numbers_on_the_card``). A grant no longer
waiting (decided, revoked, or past ``expires_at``) is still shown, with no answers.

The rule that asked:

* a gated call (``tool_call``): the confirmation gate's ``confirm.<risk class>``, the class it
  was issued at (``risk_tier``);
* a board ticket or a playbook run: ``approval_policy.<policy>``, the policy its reason names
  (the rule sentence US-C20b writes, or the raw key ``under '<policy>' policy`` written before
  it), else the workspace's policy now (a reason that names none: the policy may have changed
  since).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.approval_grants import KIND_APPROVAL, SUBJECT_TOOL_CALL, ApprovalGrant, GrantStatus
from core.models.core import BoardTask
from core.models.orchestration import OrchestrationRun
from core.models.orchestration_enums import RunState
from core.services.approval_policy import VALID_POLICIES, load_approval_policy
from modules.policy.rule_sentences import FALLBACK, approval_policy_rule, confirm_rule, rule_sentence
from modules.proactive.locale import ENGLISH
from modules.tools.discovery.card_numbers import MISSION_CARD
from modules.tools.discovery.card_ticket_numbers import numbers_on_the_card
from modules.tools.discovery.mission_approval_rule import policy_holding
from modules.tools.formatting.approval_view import (ANSWERS, RUN, ViewParts, ViewTarget, build_view,
                                                    mission_plan_parts, shown, target_of, verb_for)
from modules.tools.formatting.card_digest import params_digest
from services.grant_owners import grant_owners
from services.ticket_numbers import format_number

__all__ = ["PLAN_STEPS_SHOWN", "grant_view", "mission_plan_view", "view_of"]

# An approval grant's id: ASCII digits, within the grants table's integer column.
_GRANT_ID = re.compile(r"[0-9]{1,10}")
MAX_GRANT_ID = 2**31 - 1
# The plan's steps a view lists: as many as the Mission card carries (handlers_missions._plan_task_summary).
PLAN_STEPS_SHOWN = 10


def view_of(db: Session, workspace_id: Any, view_id: str, language: str) -> Optional[Dict[str, Any]]:
    """The view ``view_id`` names in ``language``: an approval grant's (a number) or a Mission plan's (a
    UUID); None when the workspace holds neither."""
    said = str(view_id or "").strip()
    if _GRANT_ID.fullmatch(said):
        grant_id = int(said)
        return grant_view(db, workspace_id, grant_id, language) if grant_id <= MAX_GRANT_ID else None
    try:
        mission_id = UUID(said)
    except ValueError:
        return None
    return mission_plan_view(db, workspace_id, mission_id, language)


def mission_plan_view(db: Session, workspace_id: Any, mission_id: UUID, language: str) -> Optional[Dict[str, Any]]:
    """The Mission plan's view in ``language``, its ``grant_id`` the Mission's id; None when the workspace
    holds no such Mission. The holding policy, Refuse and Allow once only while it awaits approval."""
    run = (db.query(OrchestrationRun)
           .filter(OrchestrationRun.id == mission_id, OrchestrationRun.workspace_id == workspace_id)
           .first())
    if run is None:
        return None
    tasks = run.plan.get("tasks") if isinstance(run.plan, dict) else None
    steps = [task.get("title") for task in (tasks if isinstance(tasks, list) else [])[:PLAN_STEPS_SHOWN]
             if isinstance(task, dict)]
    waiting = run.state == RunState.AWAITING_APPROVAL.value
    policy = policy_holding(db, workspace_id, run.id) if waiting else None
    parts = mission_plan_parts(str(run.id), run.goal, steps, _mission_number(db, workspace_id, run.id),
                               policy, ANSWERS if waiting else ())
    return build_view(parts, language)


def _mission_number(db: Session, workspace_id: Any, mission_id: UUID) -> Optional[str]:
    """The Mission card's board number (#0282), or None before the card is made."""
    seq = (db.query(BoardTask.workspace_seq)
           .filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == MISSION_CARD,
                   BoardTask.orchestration_run_id == mission_id)
           .limit(1).scalar())
    return format_number(seq)


def grant_view(db: Session, workspace_id: Any, grant_id: int, language: str) -> Optional[Dict[str, Any]]:
    """The approval grant's view in ``language``; None when the workspace holds no such approval."""
    grant = (db.query(ApprovalGrant)
             .filter(ApprovalGrant.id == grant_id, ApprovalGrant.workspace_id == workspace_id,
                     ApprovalGrant.kind == KIND_APPROVAL)
             .first())
    if grant is None:
        return None
    owner = grant_owners(db, workspace_id, [grant]).get(grant.id) or {}
    return build_view(_parts(db, workspace_id, grant, owner), language)


def _parts(db: Session, workspace_id: Any, grant: ApprovalGrant, owner: Dict[str, Any]) -> ViewParts:
    """What the grant's view is made of: the stored call's digest, else its ticket, and its rule."""
    details = grant.details if isinstance(grant.details, dict) else {}
    agent, ticket = owner.get("agent") or {}, owner.get("ticket") or {}
    numbers = numbers_on_the_card(db, workspace_id, grant.tool_name or "", details.get("params"))
    digest = {key: numbers.get(key, value) for key, value in params_digest(details.get("params")).items()}
    gated_call = grant.subject_type == SUBJECT_TOOL_CALL
    summary = grant.question_md or details.get("action_description") or ticket.get("title") or grant.tool_name
    return ViewParts(
        grant_id=grant.id, agent_name=agent.get("name"), verb=verb_for(grant.risk_tier) if gated_call else RUN,
        summary=str(summary or ""), digest=digest,
        rule_id=_gate_rule(grant.risk_tier) if gated_call else approval_policy_rule(
            _policy_of(db, workspace_id, grant.reason)),
        target=target_of(digest) or _ticket_target(ticket),
        expires_at=grant.expires_at.isoformat() if grant.expires_at else None,
        answers=_answers(grant),
    )


def _answers(grant: ApprovalGrant) -> Tuple[str, ...]:
    """Refuse and Allow once while the grant waits; none once it is decided or past its expiry."""
    lapsed = grant.expires_at is not None and grant.expires_at <= datetime.now(timezone.utc)
    return ANSWERS if grant.status == GrantStatus.PENDING.value and not lapsed else ()


def _gate_rule(risk_class: Optional[str]) -> str:
    """The confirmation gate's rule for the class the call was issued at (FALLBACK for an unknown class)."""
    return confirm_rule(risk_class) if risk_class else FALLBACK


def _policy_of(db: Session, workspace_id: Any, reason: Optional[str]) -> str:
    """The approval policy a board or playbook grant's reason names, else the workspace's now."""
    said = str(reason or "")
    for policy in sorted(VALID_POLICIES):
        if f"'{policy}'" in said or said == rule_sentence(approval_policy_rule(policy), ENGLISH):  # raw key, sentence
            return policy
    return load_approval_policy(db, workspace_id)["policy"]


def _ticket_target(ticket: Dict[str, Any]) -> Optional[ViewTarget]:
    """The ticket the grant holds: its number, with its title beside it, else its title alone."""
    title = shown(ticket.get("title")) or None
    number = ticket.get("number")
    if number:
        return ViewTarget(label=shown(number), detail=title)
    return ViewTarget(label=title) if title else None
