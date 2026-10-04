"""F305 (night 9, Gerard's decision): an agent's or a mission's output stays out of
the owner's documents and the Knowledge Graph unless the workspace opted in.

Night 9 (build 13, MORNING-REPORT.md L90/L97/L134): 28 task reports and 2 mission
outputs became documents #1527–#1555; Auto cited #1547 (ticket #1851's own answer)
for café payment terms, a wrong "140 kg now" answer was among them, and the graph
took 'brazil_cerrado' from a playbook report. The owner's rule: nothing goes into
the knowledge base automatically; the owner adds files.

``services.knowledge_flywheel.flywheel_enabled`` is now off unless the workspace says
``knowledge_flywheel_enabled: true``. Three seams honour it here:

- ``ingests_only_when_opted_in`` wraps the coordinator's mission-output step: off, a
  finished mission still gets its Deliverable (its render, or its final step's
  output) and is marked so the sweep moves on, but no document is made. Before, the
  opt-out returned before the Deliverable, which an off default would have lost.
- ``graph_skips_agent_outputs`` wraps the graph's source collection: a document an
  agent wrote (``source_type`` agent_output, the ones already filed included) is not
  extracted, on a full rebuild or an incremental one.
- ``graph_pending_without_agent_outputs`` wraps the incremental build: a report,
  mission-synthesis or generated-document pending is dropped.

Existing agent_output documents are left as they are (AGENTS.md: ask before removing
user data); search already leaves them out (services/agents_writing.py, F269).
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# The run.config marker the flywheel sweep reads as "handled" (coordinator_service).
SKIPPED_MARKER = "skipped_opt_out"
FAILED_MARKER = "output_ingest_failed"
AGENT_OUTPUT_PENDING_TYPES = ("report", "mission_synthesis", "generated_document")


def _opted_in(db: Any, workspace_id: Any) -> bool:
    from services.knowledge_flywheel import flywheel_enabled

    return flywheel_enabled(db, workspace_id)


def _verified_tasks(db: Any, run: Any) -> List[Any]:
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState

    return (db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id, OrchestrationTask.state == TaskState.VERIFIED.value)
            .order_by(OrchestrationTask.sequence_number).all())


def mission_markdown(run: Any, tasks: List[Any]) -> str:
    """The mission's assembled output, as the coordinator assembles it."""
    parts = [f"# Mission: {run.goal}\n"]
    for t in tasks:
        parts.append(f"## {t.sequence_number}. {t.title}\n")
        if t.output:
            parts.append(str(t.output))
        parts.append("")
    return "\n".join(parts)


def _mark(db: Any, run: Any, key: str, value: str) -> None:
    run.config = {**(run.config or {}), key: value}
    db.flush()


async def deliver_without_ingest(coordinator: Any, db: Any, run: Any) -> None:
    """A finished mission's Deliverable without a document: its rendered document when
    the run asked for one, else its final step's output. Marked handled either way."""
    try:
        tasks = _verified_tasks(db, run)
        if tasks and not await coordinator._emit_mission_document(db, run, mission_markdown(run, tasks)):
            await coordinator._register_final_output_deliverable(db, run, tasks)
        _mark(db, run, "output_ingest", SKIPPED_MARKER)
        logger.info("[F305] mission %s delivered; its output is not filed as a document", run.id)
    except Exception:
        logger.exception("[F305] mission %s: its Deliverable could not be made", run.id)
        _mark(db, run, FAILED_MARKER, datetime.now(timezone.utc).isoformat())


SaveOutput = Callable[[Any, Any, Any], Awaitable[Optional[int]]]


def ingests_only_when_opted_in(save: SaveOutput) -> SaveOutput:
    """Wrap ``CoordinatorService._save_mission_output_as_document``: it runs as it was
    for a workspace that opted in; otherwise the mission is delivered, never filed."""
    @functools.wraps(save)
    async def wrapped(self: Any, db: Any, run: Any) -> Optional[int]:
        if _opted_in(db, run.workspace_id):
            return await save(self, db, run)
        await deliver_without_ingest(self, db, run)
        return None
    return wrapped


def _agents_document_ids(db: Any, workspace_id: Any, ids: List[Any]) -> set:
    from uuid import UUID

    from core.models.core import Document
    from services.knowledge_flywheel import AGENT_OUTPUT_SOURCE_TYPE

    if not ids:
        return set()
    rows = db.query(Document.id).filter(Document.workspace_id == UUID(str(workspace_id)), Document.id.in_(ids),
                                        Document.source_type == AGENT_OUTPUT_SOURCE_TYPE).all()
    return {row.id for row in rows}


def owners_sources(workspace_id: Any, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The graph's sources without the documents an agent wrote, unless the workspace
    opted in. A list that cannot be checked is not extracted (``graph_skips_agent_outputs``)."""
    from core.database.database import get_db_session

    doc_ids = [s.get("id") for s in sources if isinstance(s, dict) and s.get("type") == "document"]
    if not doc_ids:
        return sources
    with get_db_session() as db:
        if _opted_in(db, workspace_id):
            return sources
        agents = _agents_document_ids(db, workspace_id, doc_ids)
    if agents:
        logger.info("[F305] graph: %d document(s) an agent wrote are not extracted", len(agents))
    return [s for s in sources if not (s.get("type") == "document" and s.get("id") in agents)]


def graph_skips_agent_outputs(collect: Callable[..., Awaitable[List[Dict[str, Any]]]]) -> Callable[..., Any]:
    """Wrap ``GraphifyService._collect_sources``: see ``owners_sources``."""
    @functools.wraps(collect)
    async def wrapped(self: Any, workspace_id: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        sources = await collect(self, workspace_id, *args, **kwargs)
        try:
            return owners_sources(workspace_id, sources)
        except Exception:
            logger.exception("[F305] graph sources for %s could not be checked; none are extracted", workspace_id)
            return []
    return wrapped


def owners_pending(workspace_id: Any, pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The incremental build's pendings without an agent output's (report, mission
    synthesis, generated document), unless the workspace opted in."""
    from core.database.database import get_db_session

    if not any(isinstance(p, dict) and p.get("type") in AGENT_OUTPUT_PENDING_TYPES for p in pending or []):
        return pending
    with get_db_session() as db:
        if _opted_in(db, workspace_id):
            return pending
    return [p for p in pending if not (isinstance(p, dict) and p.get("type") in AGENT_OUTPUT_PENDING_TYPES)]


def graph_pending_without_agent_outputs(build: Callable[..., Awaitable[Dict[str, Any]]]) -> Callable[..., Any]:
    """Wrap ``GraphifyService._incremental_build``: see ``owners_pending``."""
    @functools.wraps(build)
    async def wrapped(self: Any, workspace_id: str, existing_graph: Any, pending: List[Dict[str, Any]]
                      ) -> Dict[str, Any]:
        try:
            kept = owners_pending(workspace_id, pending)
        except Exception:
            logger.exception("[F305] graph pendings for %s could not be checked; agent outputs dropped", workspace_id)
            kept = [p for p in pending or [] if not (isinstance(p, dict) and p.get("type") in AGENT_OUTPUT_PENDING_TYPES)]
        return await build(self, workspace_id, existing_graph, kept)
    return wrapped


__all__ = [
    "AGENT_OUTPUT_PENDING_TYPES", "SKIPPED_MARKER", "deliver_without_ingest", "graph_pending_without_agent_outputs",
    "graph_skips_agent_outputs", "ingests_only_when_opted_in", "mission_markdown", "owners_pending", "owners_sources",
]
