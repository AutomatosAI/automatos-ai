"""PRD-256 FX-009 (night 12, B3, F388): an approval card for a mission tool is raised only
on the mission the call will act on.

Night 12, the owner's card was built from the raw ``mission_id`` the model guessed: card
1942 asked about mission_id '220', a task card's number, and card 1965 about '1065', the
mission's ticket. The owner clicked, and the resumed call failed ("#0220 … is a task card,
not a mission"). ``subject_targets.resolve_targets`` now reads ``mission_id`` the way the
tool will (``mission_refs.on_its_mission``): a mission's id, its card's number, a step's
number or its title, in this workspace. A mission it names is named on the card by its
title and number; anything else fails the call back to the model with the tool's own
refusal, and nothing is asked. The call the card asks about names the mission by its own id
(``bound_to_the_mission``), so the click runs on the mission the card showed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.execution.subject_targets import Target

PARAM = "mission_id"
MISSION, STEP = "mission", "mission step"
NOUN = "mission"


def mission_targets(db: Session, workspace_id: Any, params: Any,
                    action: Optional[str]) -> Tuple[List[Target], List[Target]]:
    """``(found, missing)`` for a mission tool's ``mission_id`` in this workspace; nothing
    for a call that names no mission, a tool that takes none, or a public widget turn
    (whose call reaches the tool as sent, F155)."""
    from core.security.surface import widget_turn
    from modules.tools.discovery.mission_refs import does_of, is_uuid

    said = params.get(PARAM) if isinstance(params, dict) else None
    if said in (None, ""):
        return [], []
    does = does_of(action or "")
    if does is None or widget_turn():
        return [], []
    if is_uuid(said):
        return _by_its_id(db, workspace_id, said)
    return _by_a_card(db, workspace_id, said, str(action), does)


def bound_to_the_mission(db: Session, workspace_id: Any, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """The call with ``mission_id`` (and, F308, the step it decides) as the mission's own id,
    so the owner's click runs on the mission its card showed: a title or a number read
    again at the click could name another. The call as it is when it names no mission
    the tool would act on (its refusal comes from ``mission_targets``)."""
    from modules.tools.discovery.mission_refs import card_named, does_of, is_uuid, on_its_mission

    said = params.get(PARAM)
    does = does_of(action) if said not in (None, "") else None
    if does is None or is_uuid(said):
        return params
    card = card_named(db, workspace_id, said)[0]
    named = on_its_mission(db, card, action, does) if card is not None else None
    if named is None or named.run_id is None:
        return params
    return {**params, **named.params}


def _by_its_id(db: Session, workspace_id: Any, said: Any) -> Tuple[List[Target], List[Target]]:
    """A mission named by its own id: found when it is this workspace's."""
    from core.models.orchestration import OrchestrationRun

    run_id = said if isinstance(said, UUID) else UUID(str(said))
    run = db.query(OrchestrationRun).filter(OrchestrationRun.id == run_id,
                                            OrchestrationRun.workspace_id == workspace_id).first()
    if run is None:
        return [], [Target(PARAM, str(said), NOUN, label=f"{NOUN} {said}")]
    card = _mission_card(db, workspace_id, run.id)
    name = card.title if card is not None else run.goal
    return [_target(db, run.id, card, name, MISSION)], []


def _by_a_card(db: Session, workspace_id: Any, said: Any, action: str,
               does: str) -> Tuple[List[Target], List[Target]]:
    """A mission named by a card's number or its title: found when the tool acts on its
    mission, missing with the tool's own refusal otherwise."""
    from modules.tools.discovery.mission_refs import STEP_CARD, card_named, on_its_mission

    card, refusal = card_named(db, workspace_id, said)
    named = on_its_mission(db, card, action, does) if card is not None else None
    if named is None or named.refusal:
        why = refusal if named is None else named.refusal
        return [], [Target(PARAM, str(said), NOUN, label=f"{NOUN} {said}", why=why)]
    if named.run_id is None:  # a read of a card that is no mission answers with the card
        return [], []
    kind = STEP if card.source_type == STEP_CARD else MISSION
    return [_target(db, named.run_id, card, card.title, kind)], []


def _target(db: Session, run_id: Any, card: Any, name: Optional[str], kind: str) -> Target:
    """The mission as its card names it: "'Mission: Plan the spring menu' (mission #0992)"."""
    from modules.tools.execution.subject_targets import NAME_CHARS
    from services.ticket_numbers import ticket_number

    number = ticket_number(db, card) if card is not None else None
    label = f"{kind} {number}" if number else f"{NOUN} {run_id}"
    return Target(PARAM, str(run_id), NOUN, str(name)[:NAME_CHARS] if name else None, label)


def _mission_card(db: Session, workspace_id: Any, run_id: Any) -> Optional[Any]:
    from core.models.core import BoardTask
    from modules.tools.discovery.mission_refs import MISSION_CARD

    return db.query(BoardTask).filter(BoardTask.orchestration_run_id == run_id, BoardTask.workspace_id == workspace_id,
                                      BoardTask.source_type == MISSION_CARD).first()


__all__ = ["bound_to_the_mission", "mission_targets"]
