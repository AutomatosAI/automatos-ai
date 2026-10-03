"""F248 (night 7): a mission step starts from the results and documents it works from.

- #0126.3, the mission's summary, said "[Number]" and "[Your Name/Company Name]". The
  results of the steps it built on reached it as the field digest trimmed to 1,200
  tokens (PRD-164 S4), so their figures were not in front of it.
- #0139.1 listed 24 of the owner's 27 cafés and £452.40 too little. Its agent could not
  find the invoice sheet the goal named ("I cannot find this file in the workspace or
  through document search") and worked from search snippets. The same job on a plain
  card read the sheet and was exact.

So:
- the mission's last step (one that no other step builds on) gets the results of the
  steps it builds on whole, as a synthesis step does
  (``CoordinatorService._collect_upstream_outputs``: up to 8,000 characters each, 30,000
  in all), and is told to take every figure, name and date from them. The steps before
  it keep PRD-164's budgeted digest;
- a step whose brief or mission goal names a file in the owner's Documents is told that
  document's id, and to read it whole.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.exc import UnmappedInstanceError

logger = logging.getLogger(__name__)

RESULTS_KEY = "upstream_results"
RESULTS_HEADING = "## Results of the steps this work builds on"
RESULTS_RULE = ("This is the mission's last step: build your answer from these results. Take every figure, "
                "name and date from them exactly as they are. Never put a placeholder where one of them "
                "belongs; if a result lacks something you need, say what is missing.")
DOCUMENTS_HEADING = "## Documents this work names"
DOCUMENTS_RULE = ("They are in the owner's Documents. Read each one whole with platform_read_document "
                  "(its document_id) before you use it: a search returns only parts of it.")
DOCUMENTS_NAMED = 5
# A file named in a brief: a name with no spaces and a document's extension, not part of
# a path (a path is a workspace file the step may be told to write).
_FILE_NAME = re.compile(r"(?<![\w./-])([\w][\w.-]{0,150}\.(?:csv|tsv|xlsx|xlsm|xls|ods|pdf|docx|doc|txt|md|json))\b",
                        re.IGNORECASE)


def builds_on_whole_results(attach: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService._attach_field_digest``. For the mission's last step,
    the results it builds on are pinned whole beside the digest instead of inside it."""
    @functools.wraps(attach)
    async def wrapped(self: Any, db: Session, run: Any, task: Any, field_id: Optional[str], agent_id: int,
                      upstream_rows: Optional[List[Dict[str, Any]]] = None) -> None:
        if upstream_rows and _is_last_step(db, task):
            from services.coordinator_service import CoordinatorService

            results = [r for r in CoordinatorService._collect_upstream_outputs(db, task) if r.get("output")]
            task.input_context = {**(task.input_context or {}), RESULTS_KEY: results}
            upstream_rows = []
        return await attach(self, db, run, task, field_id, agent_id, upstream_rows=upstream_rows)
    return wrapped


def _is_last_step(db: Session, task: Any) -> bool:
    from core.models.orchestration import OrchestrationTaskDependency

    return db.query(OrchestrationTaskDependency.id).filter(
        OrchestrationTaskDependency.depends_on_task_id == task.id).first() is None


def with_its_inputs(build: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``MissionDispatcher.build_task_prompt``: the step's prompt ends with the
    results it builds on (the last step's) and the documents it names."""
    @functools.wraps(build)
    def wrapped(task: Any, goal: Optional[str] = None) -> str:
        prompt = build(task, goal=goal)
        blocks = [block for block in (results_block(task), documents_block(task, goal)) if block]
        return "\n\n".join([prompt, *blocks])
    return wrapped


def results_block(task: Any) -> str:
    context = getattr(task, "input_context", None)
    results = context.get(RESULTS_KEY) if isinstance(context, dict) else None
    if not results:
        return ""
    parts = [RESULTS_HEADING, RESULTS_RULE]
    parts.extend(f"### {r.get('title') or 'A step'}\n{r.get('output')}" for r in results)
    return "\n\n".join(parts)


def documents_block(task: Any, goal: Optional[str]) -> str:
    """The files the step's brief or the mission's goal names that are in the owner's Documents."""
    text = "\n".join(str(t or "") for t in (getattr(task, "title", ""), getattr(task, "description", ""), goal))
    names = list(dict.fromkeys(m.group(1) for m in _FILE_NAME.finditer(text)))[:DOCUMENTS_NAMED]
    db = _session_of(task)
    if not names or db is None:
        return ""
    found = _documents_named(db, getattr(task, "run_id", None), names)
    lines = [f"- {name}: document_id {found[name.lower()]}" for name in names if name.lower() in found]
    return "\n".join([DOCUMENTS_HEADING, DOCUMENTS_RULE, *lines]) if lines else ""


def _documents_named(db: Session, run_id: Any, names: List[str]) -> Dict[str, int]:
    """Each named file's document id in the mission's workspace, by its lowercased name."""
    from core.models.core import Document
    from core.models.orchestration import OrchestrationRun

    workspace_id = db.query(OrchestrationRun.workspace_id).filter(OrchestrationRun.id == run_id).scalar()
    if workspace_id is None:
        return {}
    wanted = [n.lower() for n in names]
    rows = db.query(Document.id, Document.filename, Document.original_filename).filter(
        Document.workspace_id == workspace_id,
        or_(func.lower(Document.filename).in_(wanted), func.lower(Document.original_filename).in_(wanted)),
    ).order_by(Document.id.desc()).all()
    found: Dict[str, int] = {}
    for row in rows:
        for name in (row.filename, row.original_filename):
            if name and name.lower() in wanted:
                found.setdefault(name.lower(), row.id)
    return found


def _session_of(task: Any) -> Optional[Session]:
    try:
        db = object_session(task)
    except UnmappedInstanceError:     # a plain object standing in for a step
        return None
    return db if isinstance(db, Session) else None


__all__ = ["RESULTS_KEY", "builds_on_whole_results", "documents_block", "results_block", "with_its_inputs"]
