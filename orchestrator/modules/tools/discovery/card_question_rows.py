"""PRD-256 FX-008: what an approval card's change lines read from the workspace's rows.

Each reader returns the card's lines for one kind of call: the current value comes from
the row the call will change, looked up in the caller's workspace only (an agent, a
ticket or a mission of another workspace is never read), and the new one from the call.
A row that is not there gives no lines: the ask names what it can, and the handler
refuses the call itself.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from modules.tools.discovery.card_question_text import change_line, said_line, shown, value_line

# platform_update_agent: param → (what the owner calls it, how the agent row holds it now).
AGENT_FIELDS: Tuple[Tuple[str, str, Callable[[Any], Any]], ...] = (
    ("new_name", "name", lambda agent: agent.name),
    ("description", "description", lambda agent: agent.description),
    ("status", "status", lambda agent: agent.status),
    ("model_id", "model", lambda agent: (agent.model_config or {}).get("model_id")),
    ("temperature", "temperature", lambda agent: (agent.model_config or {}).get("temperature")),
    ("system_prompt", "system prompt", lambda agent: agent.custom_persona_prompt),
    ("team", "team", lambda agent: agent.team),
    ("job_title", "job title", lambda agent: agent.job_title),
    ("reports_to_id", "reports to", lambda agent: agent.reports_to_id),
    ("tags", "tags", lambda agent: agent.tags),
)
# platform_update_task / platform_update_task_status: param → the ticket's column.
TASK_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("title", "title"), ("description", "description"), ("priority", "priority"),
    ("review_mode", "review_mode"), ("tags", "tags"), ("status", "status"),
)
NOTE_FIELD = "note"
SEND_BACK = "send back to its agent for a redo"
TOOLS_FIELD = "tools"
TOOL_PARAMS = ("app_name", "tool_name")
ASSIGN_TOOL = "platform_assign_tool_to_agent"
MASKED = "****"
MISSION_LINE = "mission: '{title}' ({number})"
MISSION_NO_NUMBER = "mission: '{title}'"
MISSION_CARD = "orchestration"
MAX_CARDS_LISTED = 5


def agent_lines(db: Any, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """Each field the call changes on the agent, 'field: from → to'."""
    from modules.tools.discovery.handlers_assignments import resolve_agent

    agent, _error = resolve_agent(db, workspace_id, params)
    if agent is None:
        return []
    return [change_line(label, read(agent), params[param])
            for param, label, read in AGENT_FIELDS if params.get(param) is not None]


def tool_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The agent's tools before and after the call: 'tools: GMAIL → GMAIL, SLACK'."""
    from core.models.composio_cache import AgentAppAssignment
    from modules.tools.discovery.handlers_assignments import resolve_agent

    app = next((str(params[key]).upper() for key in TOOL_PARAMS if params.get(key)), None)
    agent, _error = resolve_agent(db, workspace_id, params)
    if agent is None or app is None:
        return []
    rows = db.query(AgentAppAssignment.app_name).filter(
        AgentAppAssignment.agent_id == agent.id, AgentAppAssignment.is_active.is_(True)).all()
    now = sorted({str(row.app_name).upper() for row in rows})
    after = sorted({*now, app}) if action == ASSIGN_TOOL else [name for name in now if name != app]
    return [change_line(TOOLS_FIELD, now, after)]


def task_lines(db: Any, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """Each field the call changes on its ticket; a bulk move, one line per ticket."""
    refs = [params["task_id"]] if params.get("task_id") not in (None, "") else []
    listed = params.get("task_ids") if isinstance(params.get("task_ids"), list) else []
    refs = [*refs, *listed][:MAX_CARDS_LISTED]
    tickets = [ticket for ticket in (_ticket(db, workspace_id, ref) for ref in refs) if ticket is not None]
    if len(tickets) == 1:
        return _ticket_changes(tickets[0], params)
    return [f"{_number(db, ticket)} {line}" for ticket in tickets for line in _ticket_changes(ticket, params)]


def _ticket_changes(ticket: Any, params: Dict[str, Any]) -> List[str]:
    lines = [change_line(column, getattr(ticket, column), _new_value(param, params[param]))
             for param, column in TASK_FIELDS if params.get(param) is not None]
    if params.get("send_back"):
        lines.append(SEND_BACK)
    if params.get(NOTE_FIELD):
        lines.append(said_line(NOTE_FIELD, params[NOTE_FIELD]))
    return lines


def _new_value(param: str, value: Any) -> Any:
    """A status as the board will set it ("approved" is Done, call_effects.STATUS_WORDS)."""
    if param != "status":
        return value
    from modules.tools.execution.call_effects import STATUS_WORDS

    said = str(value).strip().lower()
    return STATUS_WORDS.get(said, said)


def _ticket(db: Any, workspace_id: Any, ref: Any) -> Optional[Any]:
    from core.models.core import BoardTask
    from services.ticket_refs import ticket_id_named

    task_id = ticket_id_named(db, workspace_id, ref)[0]
    if task_id is None:
        return None
    return db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()


def _number(db: Any, ticket: Any) -> str:
    from services.ticket_numbers import ticket_number

    return ticket_number(db, ticket) or f"ticket {ticket.id}"


def setting_lines(db: Any, params: Dict[str, Any]) -> List[str]:
    """'category.key: from → to', a sensitive setting's values masked both ways."""
    from core.models.system_settings import SystemSetting

    category, key = params.get("category"), params.get("key")
    if not category or not key or params.get("value") is None:
        return []
    setting = db.query(SystemSetting).filter(SystemSetting.category == category, SystemSetting.key == key).first()
    if setting is None:
        return []
    if setting.is_sensitive:
        return [change_line(f"{category}.{key}", MASKED, MASKED)]
    return [change_line(f"{category}.{key}", setting.value, params["value"])]


def mission_lines(db: Any, workspace_id: Any, params: Dict[str, Any]) -> List[str]:
    """The mission's title (its card's) and its ticket number."""
    run, card = _mission(db, workspace_id, params.get("mission_id"))
    if run is None:
        return []
    title = shown(card.title if card is not None else run.goal)
    number = _number(db, card) if card is not None else None
    line = MISSION_LINE.format(title=title, number=number) if number else MISSION_NO_NUMBER.format(title=title)
    return [value_line(line)]


def _mission(db: Any, workspace_id: Any, said: Any) -> Tuple[Optional[Any], Optional[Any]]:
    """(run, its mission card) for a mission id or a card's number, in this workspace."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun

    run_id = _run_id(db, workspace_id, said)
    if run_id is None:
        return None, None
    run = db.query(OrchestrationRun).filter(
        OrchestrationRun.id == run_id, OrchestrationRun.workspace_id == workspace_id).first()
    if run is None:
        return None, None
    card = db.query(BoardTask).filter(BoardTask.orchestration_run_id == run.id, BoardTask.workspace_id == workspace_id,
                                      BoardTask.source_type == MISSION_CARD).first()
    return run, card


def _run_id(db: Any, workspace_id: Any, said: Any) -> Optional[UUID]:
    """A mission's id as said, or the mission of the card a number (mission_refs, F241) or
    a mission's title (FX-009) names."""
    from modules.tools.discovery.mission_refs import card_named, mission_of_card

    if said in (None, ""):
        return None
    try:
        return said if isinstance(said, UUID) else UUID(str(said))
    except (ValueError, TypeError, AttributeError):
        ticket = card_named(db, workspace_id, said)[0]
        return mission_of_card(db, ticket) if ticket is not None else None


__all__ = ["agent_lines", "mission_lines", "setting_lines", "task_lines", "tool_lines"]
