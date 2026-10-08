"""PRD-256 FX-008 (B2, F388): an approval card says what it approves.

Night 12, every owner-only card said a verb and no more ('change an agent', 'start a
mission'), and ``question_md`` was empty on every one: CLUB DESK's description was wiped
through a card that said 'change an agent' (A462). The card's question now carries the
subject and the change (``owner_only._ask`` stores it on the grant and in the chat card):
- an agent, ticket, tool or setting change: each changed field, 'field: from → to', the
  current value read from the workspace's row, the new one from the call;
- a new mission: its goal and the steps the owner gave;
- approving or cancelling a mission: its title and its ticket number;
- a Composio send: the recipient, the subject and the body's first line;
- FX-010: a heartbeat's fields, an agent's skills or plugins before and after, a new
  playbook's name and purpose, a playbook's timer, or what a delete takes for good;
- P256-FIX-RVW-14: a timer set through a playbook update, and a plugin or skill taken
  from (or, edited, moved for) every agent of the workspace, naming them.

The question is what the owner reads before the click; the click still runs the exact
call it was asked about (the grant's params hash). A question that cannot be read in
full falls back to its first line: the ask itself always stands.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from modules.tools.discovery import card_question_agents as agents
from modules.tools.discovery import card_question_playbooks as playbooks
from modules.tools.discovery import card_question_rows as rows
from modules.tools.discovery import card_question_skills as skills
from modules.tools.discovery.card_question_text import question, said_line, shown, value_line

logger = logging.getLogger(__name__)

Lines = Callable[[Any, Any, str, Dict[str, Any]], List[str]]

STEPS = "steps:"
STEP = "  {n}. {step}"
NO_STEPS = "steps: none given; the mission's planner drafts them"
STAFFED = "{agent} does: {does}"
# A Composio send's fields, by the names its actions use, in the order the card reads them.
RECIPIENT_KEYS = ("recipient_email", "to", "to_email", "recipient", "recipients", "channel", "channel_id",
                  "chat_id", "phone_number", "email")
SUBJECT_KEYS = ("subject", "title")
BODY_KEYS = ("body", "message", "text", "content", "commentary", "markdown_text", "caption")
SEND_FIELDS = (("to", RECIPIENT_KEYS), ("subject", SUBJECT_KEYS), ("first line", BODY_KEYS))


def platform_question(db: Any, workspace_id: Any, action: str, params: Dict[str, Any], act: str) -> str:
    """The card's question for a platform action: ``act``, then what the call changes."""
    lines: List[str] = []
    reader = READERS.get(action)
    if reader is not None:
        try:
            lines = reader(db, workspace_id, action, params)
        except Exception:  # noqa: BLE001 — the ask stands on its first line
            logger.exception("[card_question] could not read what %s changes", action)
    return question(act, lines)


def send_question(act: str, params: Any) -> str:
    """The card's question for a Composio send: ``act``, then to whom, about what, and
    the body's first line."""
    params = params if isinstance(params, dict) else {}
    lines = []
    for label, keys in SEND_FIELDS:
        said = next((params[key] for key in keys if params.get(key) not in (None, "", [])), None)
        if said is not None:
            lines.append(said_line(label, _first_line(said) if label == "first line" else said))
    return question(act, lines)


def mission_create_lines(params: Dict[str, Any]) -> List[str]:
    """A new mission's goal, the owner's steps in order, and who does what."""
    lines = [said_line("goal", params.get("goal"))]
    steps = [step for step in _listed(params.get("steps")) if str(step).strip()]
    if steps:
        lines.append(value_line(STEPS))
        lines.extend(STEP.format(n=number, step=shown(step)) for number, step in enumerate(steps, 1))
    else:
        lines.append(value_line(NO_STEPS))
    for entry in _listed(params.get("staffing")):
        if isinstance(entry, dict) and entry.get("agent"):
            lines.append(value_line(STAFFED.format(agent=shown(entry["agent"]), does=shown(entry.get("does")))))
    return lines


def _first_line(value: Any) -> str:
    text = str(value).strip()
    return text.splitlines()[0] if text else ""


def _listed(value: Any) -> List[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


READERS: Dict[str, Lines] = {
    "platform_update_agent": lambda db, ws, action, params: rows.agent_lines(db, ws, params),
    "platform_assign_tool_to_agent": rows.tool_lines,
    "platform_unassign_tool_from_agent": rows.tool_lines,
    "platform_update_task": lambda db, ws, action, params: rows.task_lines(db, ws, params),
    "platform_update_task_status": lambda db, ws, action, params: rows.task_lines(db, ws, params),
    "platform_update_system_setting": lambda db, ws, action, params: rows.setting_lines(db, params),
    "platform_create_mission": lambda db, ws, action, params: mission_create_lines(params),
    "platform_approve_mission": lambda db, ws, action, params: rows.mission_lines(db, ws, params),
    "platform_cancel_mission": lambda db, ws, action, params: rows.mission_lines(db, ws, params),
    # FX-010: every agent-setting change, and a playbook made, timed or deleted (Decision D1, amended).
    "platform_configure_agent_heartbeat": agents.heartbeat_lines,
    "platform_delete_agent": agents.delete_agent_lines,
    "platform_assign_skill_to_agent": agents.skill_lines,
    "platform_unassign_skill_from_agent": agents.skill_lines,
    "platform_assign_plugin_to_agent": agents.plugin_lines,
    "platform_create_playbook": playbooks.create_lines,
    "platform_schedule_playbook": playbooks.schedule_lines,
    "platform_delete_playbook": playbooks.delete_lines,
    # P256-FIX-RVW-14: a timer through an update; a plugin or skill taken from every agent.
    "platform_update_playbook": playbooks.update_lines,
    "platform_uninstall_plugin": skills.uninstall_plugin_lines,
    "platform_delete_workspace_skill": skills.delete_skill_lines,
    "platform_update_skill": skills.update_skill_lines,
}


__all__ = ["READERS", "mission_create_lines", "platform_question", "send_question"]
