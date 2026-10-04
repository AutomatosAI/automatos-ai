"""#942: what the workspace has that this agent's sessions cannot reach.

``session_tool_gaps`` used to read only the agent's skill text, so an agent whose
workspace had a connected shop database and whose sessions had no database tool
reported no gap at all (F329: #1995 answered "There is no database query tool
here"). This checks the workspace for each capability a tool group serves (a
connected database, a built Knowledge Graph, document templates, playbooks,
reports, missions) and, for each one present whose group is NOT among the groups
given, returns an entry the agent page shows with its one-click fix.

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
)


def gap_entry(cap: Capability) -> Dict[str, Any]:
    """The entry the agent page renders, with the switch that closes it."""
    return {"kind": GAP_KIND, "capability": cap.capability, "group": cap.group,
            "message": cap.message, "fix": {"enable_group": cap.group}}


async def _present(cap: Capability, db: Any, workspace_id: Any) -> bool:
    """Whether the workspace has it; a failed check is logged and counts as no."""
    try:
        return bool(await cap.present(db, workspace_id))
    except Exception:  # noqa: BLE001 — logged; the gap is left out, the page still renders
        logger.exception("[session-gaps] could not check %s for workspace %s", cap.capability, workspace_id)
        try:
            db.rollback()  # the request's session: an aborted read must not fail the next one
        except Exception:  # noqa: BLE001
            logger.exception("[session-gaps] rollback after the %s check failed", cap.capability)
        return False


async def workspace_gaps(db: Any, workspace_id: Any, groups: Iterable[str]) -> List[Dict[str, Any]]:
    """One entry per capability the workspace has whose group is not in ``groups``."""
    enabled = set(groups)
    gaps: List[Dict[str, Any]] = []
    for cap in CAPABILITIES:
        if cap.group not in enabled and await _present(cap, db, workspace_id):
            gaps.append(gap_entry(cap))
    return gaps
