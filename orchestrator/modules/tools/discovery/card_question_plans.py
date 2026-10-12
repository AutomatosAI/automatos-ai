"""P256-FIX-N13-7 (F414, night 13): a mission's plan changed from the chat waits for the owner's click.

Night 13, platform_update_mission_plan ran on #0936 with no card (ask 404): a mission's steps
and its checks changed on Auto's word alone, as an agent's settings did before FX-010. A plan
edit from a person's chat that changes the steps (``task_edits``) or whether each step waits
for the owner (``check_each_step``) is owner-only (``owner_only.PLAN_EDITS``), read from its
params as FX-010 reads an agent-setting change. Its card names the mission, each step it edits
with what it sets, and the check.
"""
from __future__ import annotations

from typing import Any, Dict, List

from modules.tools.discovery.card_question_text import said_line, value_line

EDITS = "task_edits"
CHECKS = "check_each_step"
# What names a step in an edit, in the order the tool reads it.
STEP_KEYS = ("task_id", "sequence_number", "temp_id")
# What an edit sets on its step, as the owner calls it.
SET_FIELDS = (("agent_id", "agent"), ("agent_role", "agent"), ("title", "title"), ("description", "brief"))
STEP_FIELD = "step {step} {field}"
UNNAMED_STEP = "?"
EACH_STEP_WAITS = "every step waits in Review for your check before the next starts"
NO_STEP_WAITS = "the steps run without your check"
AGENT_NAMED = "{name} (agent #{id})"


def changes_the_plan(params: Any) -> bool:
    """Whether a plan edit changes the mission's steps or its checks."""
    if not isinstance(params, dict):
        return False
    edits = params.get(EDITS)
    return (isinstance(edits, list) and len(edits) > 0) or params.get(CHECKS) is not None


def plan_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The card's lines for a plan edit: the mission, each step's change, the check."""
    from modules.tools.discovery.card_question_rows import mission_lines

    lines = list(mission_lines(db, workspace_id, params))
    edits = params.get(EDITS) if isinstance(params.get(EDITS), list) else []
    for edit in edits:
        if isinstance(edit, dict):
            lines.extend(_edit_lines(db, workspace_id, edit))
    if params.get(CHECKS) is not None:
        lines.append(value_line(EACH_STEP_WAITS if params.get(CHECKS) else NO_STEP_WAITS))
    return lines


def _edit_lines(db: Any, workspace_id: Any, edit: Dict[str, Any]) -> List[str]:
    step = next((edit[key] for key in STEP_KEYS if edit.get(key) not in (None, "")), UNNAMED_STEP)
    lines = []
    for key, field in SET_FIELDS:
        if edit.get(key) in (None, ""):
            continue
        value = _agent_named(db, workspace_id, edit[key]) if key == "agent_id" else edit[key]
        lines.append(said_line(STEP_FIELD.format(step=step, field=field), value))
    return lines


def _agent_named(db: Any, workspace_id: Any, agent_id: Any) -> Any:
    """'CLUB DESK (agent #12)' for an agent of this workspace; the id as given otherwise."""
    from core.models import Agent

    try:
        row = db.query(Agent.id, Agent.name).filter(Agent.id == int(agent_id),
                                                    Agent.workspace_id == workspace_id).first()
    except (TypeError, ValueError):
        return agent_id
    return AGENT_NAMED.format(name=row.name, id=row.id) if row is not None else agent_id


__all__ = ["changes_the_plan", "plan_lines"]
