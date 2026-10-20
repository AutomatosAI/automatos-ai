"""#942: what the workspace has that this agent's sessions cannot reach.

``session_tool_gaps`` used to read only the agent's skill text, so an agent whose
workspace had a connected shop database and whose sessions had no database tool
reported no gap at all (F329: #1995 answered "There is no database query tool
here"). This checks the workspace for each capability a tool group serves (a
connected database, a built Knowledge Graph, document templates, playbooks,
reports, missions, Socials switched on) and, for each one present whose group is
NOT among the groups given, returns an entry the agent page shows with its
one-click fix.

PRD-256 F430: ``assignment_warning`` is the same truth at the moment a ticket is
given to a CLI agent: the platform tools its brief names that the agent's sessions
cannot call, said to whoever assigned it, so the owner hears it then, not from a
refused run.

Each check is one small, workspace-scoped read, made only for a group that is off.
A check that fails is logged and left out: a form's warning is never worth a 500.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterable, List

from sqlalchemy import text

logger = logging.getLogger(__name__)

GAP_KIND = "workspace"
# F430: what an assignment to a CLI agent says when its sessions cannot call what the brief names.
CANNOT_CALL = "{name} runs as a CLI session and cannot call {tools}."
GROUP_FIX = " Its Model tab turns the tool group on."
NO_GROUP_FIX = " No session has that tool: an agent on the API runtime does."


@dataclass(frozen=True)
class Capability:
    """Something the workspace may have, and the group whose tools reach it."""

    capability: str
    group: str
    message: str
    present: Callable[[Any, Any], Awaitable[bool]]


async def _has_database(db: Any, workspace_id: Any) -> bool:
    from core.models.database_knowledge import DatabaseKnowledgeSource as Source

    row = (db.query(Source.id)
           .filter(Source.workspace_id == str(workspace_id), Source.is_active.is_(True)).first())
    return row is not None


async def _has_graph(db: Any, workspace_id: Any) -> bool:
    from modules.knowledge.graph_service import get_graph_service

    return await get_graph_service().get_meta(str(workspace_id)) is not None


async def _has_document_templates(db: Any, workspace_id: Any) -> bool:
    from core.models.core import DocumentTemplate

    return db.query(DocumentTemplate.id).filter(DocumentTemplate.workspace_id == workspace_id).first() is not None


async def _has_playbooks(db: Any, workspace_id: Any) -> bool:
    from core.models.core import WorkflowTemplate

    return db.query(WorkflowTemplate.id).filter(WorkflowTemplate.workspace_id == workspace_id).first() is not None


async def _has_reports(db: Any, workspace_id: Any) -> bool:
    row = db.execute(text("SELECT 1 FROM agent_reports WHERE workspace_id = :ws LIMIT 1"),
                     {"ws": str(workspace_id)}).first()
    return row is not None


async def _has_missions(db: Any, workspace_id: Any) -> bool:
    from core.models.orchestration import OrchestrationRun

    return db.query(OrchestrationRun.id).filter(OrchestrationRun.workspace_id == workspace_id).first() is not None


async def _socials_on(db: Any, workspace_id: Any) -> bool:
    """Socials is on for the workspace (both switches, D1): its agents' posts are wanted."""
    from core.models.workspaces import Workspace
    from modules.socials.settings import socials_off_reason

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if workspace is None:
        return False
    return socials_off_reason(workspace) is None


CAPABILITIES: tuple[Capability, ...] = (
    Capability("database", "data",
               "This workspace has a connected database, and this agent's sessions cannot query it. "
               "Turn on Data so it can answer from the real figures.", _has_database),
    Capability("graph", "graph",
               "This workspace has a built Knowledge Graph, and this agent's sessions cannot read it. "
               "Turn on Knowledge Graph.", _has_graph),
    Capability("documents", "documents",
               "This workspace has document templates, and this agent's sessions cannot fill them. "
               "Turn on Documents.", _has_document_templates),
    Capability("playbooks", "playbooks",
               "This workspace has playbooks, and this agent's sessions cannot read or run them. "
               "Turn on Playbooks.", _has_playbooks),
    Capability("reports", "reports",
               "Agents in this workspace file reports, and this agent's sessions cannot read them. "
               "Turn on Reports.", _has_reports),
    Capability("missions", "missions",
               "This workspace runs missions, and this agent's sessions cannot read them or what their "
               "agents found. Turn on Missions.", _has_missions),
    Capability("socials", "socials",
               "Socials is on for this workspace, and this agent's sessions cannot draft posts. "
               "Turn on Socials.", _socials_on),
)


def gap_entry(cap: Capability) -> Dict[str, Any]:
    """The entry the agent page renders, with the switch that closes it."""
    return {"kind": GAP_KIND, "capability": cap.capability, "group": cap.group,
            "message": cap.message, "fix": {"enable_group": cap.group}}


def _rolled_back(db: Any, what: str) -> None:
    """After a failed read on the caller's session: an aborted read must not fail the next one."""
    try:
        db.rollback()
    except Exception:  # noqa: BLE001
        logger.exception("[session-gaps] rollback after the %s check failed", what)


async def _present(cap: Capability, db: Any, workspace_id: Any) -> bool:
    """Whether the workspace has it; a failed check is logged and counts as no."""
    try:
        return bool(await cap.present(db, workspace_id))
    except Exception:  # noqa: BLE001 — logged; the gap is left out, the page still renders
        logger.exception("[session-gaps] could not check %s for workspace %s", cap.capability, workspace_id)
        _rolled_back(db, cap.capability)
        return False


async def workspace_gaps(db: Any, workspace_id: Any, groups: Iterable[str]) -> List[Dict[str, Any]]:
    """One entry per capability the workspace has whose group is not in ``groups``."""
    enabled = set(groups)
    gaps: List[Dict[str, Any]] = []
    for cap in CAPABILITIES:
        if cap.group not in enabled and await _present(cap, db, workspace_id):
            gaps.append(gap_entry(cap))
    return gaps


def _cannot_call_sentence(agent: Any, names: List[str]) -> str:
    """The warning: who cannot call what, and the fix (a tool group to turn on, or none)."""
    from services.cli_session_prompt import named_gap
    from services.session_tool_groups import group_of
    from services.session_tools import equivalent_of

    name = getattr(agent, "name", None) or "This agent"
    grouped = any(group_of(equivalent_of(tool) or tool) is not None for tool in names)
    tools = ", ".join(named_gap(tool) for tool in names)
    return CANNOT_CALL.format(name=name, tools=tools) + (GROUP_FIX if grouped else NO_GROUP_FIX)


def assignment_warning(agent: Any, title: Any, description: Any) -> Dict[str, Any]:
    """F430: ``{"cannot_call": [names], "warning": sentence}`` for the platform tools a
    ticket's title and brief name that ``agent``'s sessions cannot call; ``{}`` when it
    can call them all (always, for an agent on the API runtime). Never raises: the
    warning is said beside an assignment, never instead of one."""
    try:
        from services.cli_session_prompt import calls_it_cannot_make

        names = calls_it_cannot_make(agent, f"{title or ''}\n{description or ''}")
        return {"cannot_call": names, "warning": _cannot_call_sentence(agent, names)} if names else {}
    except Exception:  # noqa: BLE001 — logged; the assignment stands without the warning
        logger.exception("[F430] could not read what agent %s can call", getattr(agent, "id", "?"))
        return {}


def card_assignment_warning(db: Any, task: Any) -> Dict[str, Any]:
    """``assignment_warning`` for a board card as it stands: its agent (read in the card's
    workspace), its title and its brief; ``{}`` for a card with no agent."""
    agent_id = getattr(task, "assigned_agent_id", None)
    if not agent_id:
        return {}
    try:
        from core.models import Agent

        agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == task.workspace_id).first()
    except Exception:  # noqa: BLE001 — logged; the assignment stands without the warning
        logger.exception("[F430] could not read agent %s of card %s", agent_id, getattr(task, "id", "?"))
        _rolled_back(db, "assignment")
        return {}
    return assignment_warning(agent, task.title, task.description) if agent is not None else {}


def agent_session_view(agent: Any) -> Dict[str, Any]:
    """F430: for an agent on the CLI runtime, the tools its sessions are offered and what its
    skills name that they cannot call (``session_tool_gaps``), so "why did it fail" is
    answered from data; ``{}`` for an API agent, which has every platform tool. Pure."""
    from core.cli_runtime import is_cli_agent
    from services.cli_session_prompt import agent_session_tools, session_tool_gaps

    if not is_cli_agent(getattr(agent, "configuration", None)):
        return {}
    available = list(agent_session_tools(agent))
    return {"session_tools": available, "session_tool_gaps": session_tool_gaps(agent, available)}
