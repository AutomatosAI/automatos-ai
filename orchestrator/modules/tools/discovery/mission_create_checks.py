"""When platform_create_mission makes nothing (night 8: F288, F289).

- F288: "Run my New Cafe Onboarding playbook for …" became a mission 9 times of 9, even
  after the owner said "playbook" (#0214, #0217, #0252, #0283, #0336, #0365, #0384,
  #0413, #0439). Each was cancelled before the playbook ran. A request that names one
  of the workspace's playbooks, or asks to run a playbook, runs it.
- F289: "Yes, go ahead." made the mission again and approved the copy, leaving #0393
  waiting; "cancel #0433 and start it again" made an unrelated mission, "Research our
  top 5 competitors…" (#0437): the example goal in Auto's own skill, not the owner's.
  A mission with the goal of one made in the last hour that still waits for the owner
  is not made twice, and a goal none of whose words the owner said is not made.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

# The owner asking for a mission outright, which no playbook's name overrules.
ASKS_FOR_A_MISSION = re.compile(r"\b(?:start|create|make|set up|launch|begin|new)\s+(?:a\s+|another\s+|the\s+)?"
                                r"mission\b", re.I)
# Night 8: "Don't make a mission this time. Run my saved playbook…" is no ask for one.
NOT_A_MISSION = re.compile(r"\b(?:don'?t|do not|never|no need to|not)\s+(?:\w+\s+){0,2}?(?:start|create|make|set up|"
                           r"launch|begin|new)\s+(?:a\s+|another\s+|the\s+)?mission\b", re.I)
RUNS_A_PLAYBOOK = re.compile(r"\b(?:run|use|start|kick off|do)\b[^.!?\n]{0,40}\bplaybook\b", re.I)
MIN_PLAYBOOK_NAME = 4
WAITING = ("pending", "planning", "awaiting_approval")
RECENT = timedelta(hours=1)
SAME_GOAL, OWNERS_GOAL = 0.8, 0.3
COMMON_WORDS = frozenset({"the", "and", "for", "with", "this", "that", "our", "your", "you", "are", "from",
                          "into", "them", "then", "than", "just", "please", "mission", "start", "get", "make"})
PLAYBOOK_ASKED = ("The owner asked for their playbook '{name}', so no mission was made. Run it: "
                  "platform_execute_playbook {{playbook_name: \"{name}\", input_data: {{the details they gave, "
                  "under the playbook's own input names (platform_get_playbook lists them)}}}}. When two playbooks "
                  "share the name, the one with an agent on every step is the one that runs.")
A_PLAYBOOK_ASKED = ("The owner asked to run one of their playbooks, so no mission was made: find it with "
                    "platform_list_playbooks and run it with platform_execute_playbook.")
ALREADY_WAITING = ("Mission {number} '{goal}' was made {minutes} min ago with this goal and is waiting for the "
                   "owner's approval, so no copy was made. If the owner said go ahead, approve it: "
                   "platform_approve_mission {{mission_id: \"{number}\"}}.")
NOT_THEIR_GOAL = ("The goal isn't what the owner asked for (none of '{goal}' is in their words), so no mission was "
                  "made. Use their words for the goal.")


def refusal_for_mission(db: Session, workspace_id: Any, params: Dict[str, Any], said: Sequence[str]) -> Optional[str]:
    """Why this mission is not made, or None. Outside a chat the owner drives (no
    words of theirs), only the copy check applies."""
    goal = str((params or {}).get("goal") or "").strip()
    return (_a_playbook(db, workspace_id, goal, said) or _already_waiting(db, workspace_id, goal)
            or _not_their_goal(goal, said))


def _a_playbook(db: Session, workspace_id: Any, goal: str, said: Sequence[str]) -> Optional[str]:
    latest = said[0] if said else ""
    if not latest or ASKS_FOR_A_MISSION.search(NOT_A_MISSION.sub(" ", latest)):
        return None
    from core.models.core import WorkflowTemplate

    names = {str(name).strip() for (name,) in db.query(WorkflowTemplate.name)
             .filter(WorkflowTemplate.workspace_id == workspace_id).all() if name}
    named = sorted((n for n in names if len(n) >= MIN_PLAYBOOK_NAME
                    and re.search(rf"\b{re.escape(n)}\b", f"{latest}\n{goal}", re.I)), key=len, reverse=True)
    if named:
        return PLAYBOOK_ASKED.format(name=named[0])
    return A_PLAYBOOK_ASKED if RUNS_A_PLAYBOOK.search(latest) else None


def _already_waiting(db: Session, workspace_id: Any, goal: str) -> Optional[str]:
    from core.models.orchestration import OrchestrationRun

    if not goal:
        return None
    since = datetime.now(timezone.utc) - RECENT
    for run in (db.query(OrchestrationRun).filter(OrchestrationRun.workspace_id == workspace_id,
                                                  OrchestrationRun.state.in_(WAITING),
                                                  OrchestrationRun.created_at >= since)
                .order_by(OrchestrationRun.created_at.desc()).all()):
        if share_of_words(goal, [run.goal or ""]) >= SAME_GOAL:
            minutes = max(1, int((datetime.now(timezone.utc) - _aware(run.created_at)).total_seconds() // 60))
            return ALREADY_WAITING.format(number=_card_number(db, workspace_id, run) or str(run.id),
                                          goal=_short(run.goal), minutes=minutes)
    return None


def _not_their_goal(goal: str, said: Sequence[str]) -> Optional[str]:
    if not goal or not said or share_of_words(goal, said) >= OWNERS_GOAL:
        return None
    return NOT_THEIR_GOAL.format(goal=_short(goal))


def _card_number(db: Session, workspace_id: Any, run: Any) -> Optional[str]:
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_number

    card = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == "orchestration",
                                      BoardTask.orchestration_run_id == run.id).first()
    return ticket_number(db, card) if card is not None else None


def share_of_words(text: str, sources: Sequence[str]) -> float:
    """How much of ``text`` (its words of three letters or more) is in ``sources``."""
    said = _words(text)
    if not said:
        return 1.0
    pool = set().union(*(_words(source) for source in sources)) if sources else set()
    return len(said & pool) / len(said)


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9£$%']{3,}", (text or "").lower()) if w not in COMMON_WORDS}


def _aware(when: Any) -> datetime:
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _short(text: Any, limit: int = 80) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


__all__: List[str] = ["refusal_for_mission", "share_of_words"]
