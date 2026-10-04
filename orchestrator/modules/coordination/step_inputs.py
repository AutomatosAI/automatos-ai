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

F286 (night 8): a summary built from the wrong pieces.
- #0383.3, the summary step, could not see its own mission's approved steps: "I cannot
  locate the specific approved content from cards #0383.1 and #0383.2". It declared no
  steps it built on, and its redo's prompt (a synthesis step's revision) carried only
  its last answer and the owner's words. A step that pulls the mission together (a
  synthesis step, or one whose title or brief asks to summarise, combine, compile or
  pull together) now gets the verified results of the steps before it as well as the
  ones it declares (``with_the_missions_earlier_steps`` around
  ``CoordinatorService._collect_upstream_outputs``, ``earlier_results_block`` in its
  prompt), each under its card number, within a synthesis step's limits (8,000
  characters each, 30,000 in all); and a synthesis step's redo is given them again
  (``a_summary_keeps_its_approved_inputs`` around ``CoordinatorService._prepare_task``).
- #0446.4 quoted the drafts the owner had sent back. Every step's answer goes into the
  mission's shared field when it finishes, before it is checked, under the step's
  title; the last step's results are pinned whole, so its digest no longer carried the
  steps it builds on, and the field's copies, rejected drafts among them, came back in.
  Those keys now carry a pointer to the approved result in the prompt instead
  (``builds_on_whole_results``), and a summary is told that only these results are the
  approved ones. Only verified steps count: a step being redone is not one.
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
EARLIER_HEADING = "## Approved results of this mission's earlier steps"
EARLIER_RULE = ("These are the steps of this mission before yours, as they were approved. Use them where your brief "
                "needs them, and take every figure, name and date from them exactly as they are. They are here: "
                "don't look for them elsewhere or ask for them.")
APPROVED_HEADING = "## The approved results you pull together"
APPROVED_RULE = ("These are the approved versions of the steps you pull together. Drafts that were sent back may "
                 "still be in the shared field or in earlier answers: never use one in place of these.")
# What the field digest shows, under a step's own key, for a result pinned whole in the prompt.
PINNED_BELOW = "Its approved result is in full in this prompt: use that, never an earlier draft of it."
SYNTHESIS = "synthesis"
# The limits CoordinatorService._collect_upstream_outputs keeps on a step's inputs.
RESULT_CHARS = 8000
ALL_RESULTS_CHARS = 30_000
DESCRIPTION_CHARS = 500
TRUNCATED = "\n\n... (truncated)"
# A brief that asks to summarise, synthesise, combine, pull (put, bring) together or compile.
_PULLS_TOGETHER = re.compile(
    r"\b(?:summari[sz](?:e[sd]?|ing)|summary|synthesi[sz](?:e[sd]?|ing)|synthesis"
    r"|combin(?:e[sd]?|ing)|compil(?:e[sd]?|ing)"
    r"|(?:pull|put|bring)(?:s|ing)?\s+(?:[\w'-]+,?\s+){0,8}?together)\b",
    re.IGNORECASE,
)
# A file named in a brief: a name with no spaces and a document's extension, not part of
# a path (a path is a workspace file the step may be told to write).
_FILE_NAME = re.compile(r"(?<![\w./-])([\w][\w.-]{0,150}\.(?:csv|tsv|xlsx|xlsm|xls|ods|pdf|docx|doc|txt|md|json))\b",
                        re.IGNORECASE)


def builds_on_whole_results(attach: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService._attach_field_digest``. For the mission's last step,
    the results it builds on are pinned whole beside the digest instead of inside it.
    F286: under a field, the digest's rows for results pinned in the prompt point to
    them, so the field's copies (rejected drafts among them) stay out."""
    @functools.wraps(attach)
    async def wrapped(self: Any, db: Session, run: Any, task: Any, field_id: Optional[str], agent_id: int,
                      upstream_rows: Optional[List[Dict[str, Any]]] = None) -> None:
        pinned: List[Dict[str, Any]] = []
        if upstream_rows and _is_last_step(db, task):
            pinned = _whole_results(db, task)
            task.input_context = {**(task.input_context or {}), RESULTS_KEY: pinned}
            upstream_rows = []
        elif field_id and not upstream_rows and builds_on_earlier_steps(db, task):
            pinned = _whole_results(db, task)
        if field_id and pinned:
            upstream_rows = [*(upstream_rows or []), *pointers_to(pinned)]
        return await attach(self, db, run, task, field_id, agent_id, upstream_rows=upstream_rows)
    return wrapped


def _whole_results(db: Session, task: Any) -> List[Dict[str, Any]]:
    from services.coordinator_service import CoordinatorService

    return [r for r in CoordinatorService._collect_upstream_outputs(db, task) if r.get("output")]


def pointers_to(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """F286: a digest row under each pinned result's own key (its step's title, the key
    its answers went into the field under) that points to the result in the prompt."""
    keys = dict.fromkeys(str(r.get("key") or r.get("title") or "").strip() for r in results)
    return [{"key": key, "value": PINNED_BELOW} for key in keys if key]


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
    """The results the step builds on: whole for the mission's last step
    (``RESULTS_KEY``), or else its mission's earlier steps' approved results (F286)."""
    context = getattr(task, "input_context", None)
    results = context.get(RESULTS_KEY) if isinstance(context, dict) else None
    if not results:
        return earlier_results_block(task)
    return _block(RESULTS_HEADING, RESULTS_RULE, results)


def earlier_results_block(task: Any) -> str:
    """F286: the verified results of the mission's earlier steps, for a step that builds
    on them (``builds_on_earlier_steps``)."""
    db = _session_of(task)
    if db is None or not builds_on_earlier_steps(db, task):
        return ""
    from services.coordinator_service import CoordinatorService

    results = [r for r in CoordinatorService._collect_upstream_outputs(db, task) if r.get("output")]
    return _block(EARLIER_HEADING, EARLIER_RULE, results) if results else ""


def _block(heading: str, rule: str, results: List[Dict[str, Any]]) -> str:
    parts = [heading, rule]
    parts.extend(f"### {r.get('title') or 'A step'}\n{r.get('output')}" for r in results)
    return "\n\n".join(parts)


def asks_to_summarise(task: Any) -> bool:
    """A step whose title or brief asks to summarise, combine, pull together or compile."""
    text = f"{getattr(task, 'title', '') or ''}\n{getattr(task, 'description', '') or ''}"
    return _PULLS_TOGETHER.search(text) is not None


def pulls_the_mission_together(task: Any) -> bool:
    """A synthesis step, or one whose title or brief asks to summarise, combine,
    compile or pull together (``asks_to_summarise``)."""
    return getattr(task, "task_type", None) == SYNTHESIS or asks_to_summarise(task)


def builds_on_earlier_steps(db: Session, task: Any) -> bool:
    """F286: a step that pulls its mission together and comes after other steps
    builds on their approved results."""
    return pulls_the_mission_together(task) and _comes_after_others(db, task)


def _comes_after_others(db: Session, task: Any) -> bool:
    from core.models.orchestration import OrchestrationTask

    return db.query(OrchestrationTask.id).filter(
        OrchestrationTask.run_id == task.run_id, OrchestrationTask.sequence_number < task.sequence_number,
    ).first() is not None


def with_the_missions_earlier_steps(collect: Callable[..., List[Dict[str, Any]]]) -> Callable[..., List[Dict[str, Any]]]:
    """Wrap ``CoordinatorService._collect_upstream_outputs``: a step that builds on its
    mission's earlier steps also gets their verified results, after the results of the
    steps it declares, within what room the limits leave (F286)."""
    @functools.wraps(collect)
    def wrapped(db: Session, task: Any) -> List[Dict[str, Any]]:
        results = collect(db, task)
        if not builds_on_earlier_steps(db, task):
            return results
        room = ALL_RESULTS_CHARS - sum(len(str(r.get("output") or "")) for r in results)
        return results + earlier_results(db, task, room=room)
    return wrapped


def a_summary_keeps_its_approved_inputs(prepare: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``CoordinatorService._prepare_task``: a synthesis step's prompt says its
    inputs are the approved versions, and its redo (a revision prompt, which carried
    only its last answer and the owner's words, #0383.3) is given them again (F286)."""
    @functools.wraps(prepare)
    async def wrapped(self: Any, db: Session, run: Any, task: Any, agent_id: Any) -> Any:
        prep = await prepare(self, db, run, task, agent_id)
        if not isinstance(prep, dict) or getattr(task, "task_type", None) != SYNTHESIS:
            return prep
        return {**prep, "prompt": f"{prep.get('prompt') or ''}\n\n{_approved_block(db, task)}"}
    return wrapped


def _approved_block(db: Session, task: Any) -> str:
    """The approval rule, and on a redo the approved results themselves."""
    context = getattr(task, "input_context", None)
    redo = isinstance(context, dict) and bool(context.get("previous_output") and context.get("verification_feedback"))
    if not redo:
        return f"{APPROVED_HEADING}\n\n{APPROVED_RULE}"
    return _block(APPROVED_HEADING, APPROVED_RULE, _whole_results(db, task))


def earlier_results(db: Session, task: Any, *, room: int) -> List[Dict[str, Any]]:
    """The verified results of the mission's steps before ``task`` that it does not
    declare, in sequence, each under its card number, within ``room`` characters
    (``RESULT_CHARS`` each at most). A verified step is a live one: a re-plan replaces
    only steps that had not passed."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask, OrchestrationTaskDependency
    from core.models.orchestration_enums import TaskState
    from modules.coordination.mission_ends import step_numbers
    from services.coordinator_service import _sanitize_for_field

    declared = {row[0] for row in db.query(OrchestrationTaskDependency.depends_on_task_id).filter(
        OrchestrationTaskDependency.task_id == task.id).all()}
    earlier = [step for step in db.query(OrchestrationTask).filter(
        OrchestrationTask.run_id == task.run_id, OrchestrationTask.sequence_number < task.sequence_number,
        OrchestrationTask.state == TaskState.VERIFIED.value,
    ).order_by(OrchestrationTask.sequence_number).all() if step.id not in declared and str(step.output or "").strip()]
    run = db.get(OrchestrationRun, task.run_id) if earlier else None
    numbers = step_numbers(db, run, earlier) if run is not None else {}
    results: List[Dict[str, Any]] = []
    for step in earlier:
        if room <= 0:
            break
        output = _within(_sanitize_for_field(str(step.output)), min(RESULT_CHARS, room))
        room -= len(output)
        title = f"{numbers[step.id]} {step.title}" if step.id in numbers else step.title
        results.append({"title": title, "key": step.title, "output": output,
                        "description": (step.description or "")[:DESCRIPTION_CHARS]})
    return results


def _within(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + TRUNCATED


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


__all__ = ["APPROVED_HEADING", "EARLIER_HEADING", "PINNED_BELOW", "RESULTS_KEY", "a_summary_keeps_its_approved_inputs",
           "asks_to_summarise", "builds_on_earlier_steps", "builds_on_whole_results", "documents_block",
           "earlier_results", "pointers_to", "pulls_the_mission_together", "results_block", "with_its_inputs",
           "with_the_missions_earlier_steps"]
