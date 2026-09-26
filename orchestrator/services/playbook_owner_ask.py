"""F140 (night 4, the persona's fix-first #2): a playbook step that needs the
owner stops its run and asks them.

A step could not ask the owner and wait. ``platform_ask_human`` wanted a subject
(a board task id, a run id) that no step is ever given; B15/B23's call with
subject_type 'playbook_run' was refused, and five of night 4's six calls came
as ``platform_execute`` with params={}. Two of those runs then drafted orders
with quantities nobody gave them (500 kg, 100 kg). Steps that ended on a
question never reached the owner either: B4, three steps of exec-dcb82c78525b
ended "Could you please provide…", and the run said complete.

Gerard's call (25 Sep) is the small version. A step that asks ends the run
needing the owner. The run keeps the existing ``failed`` status with the reason
first ("Needs you: …"); its card goes ``blocked`` with the question in the
owner's Questions; the answer runs the playbook again from step 1 (PRD-204's
rerun, retry_of) on the same card, with every answer so far in every step's
prompt. Nothing parks mid-run: a resume would need the steps' whole answers,
and the run keeps 200-character previews.

A step needs the owner when
  (a) it called platform_ask_human. In a playbook step the executor answers the
      call itself: the run is the subject, whatever the model typed;
  (b) its whole answer is short and ends by asking for what it needs. On all
      189 of c1's step logs this flags the 12 asks (at most 402 characters) and
      nothing else; a draft that ends on a question is not an ask;
  (c) it wrote a line starting ``NEEDS YOU:``.

The tester's rules (25 Sep): one bell (the question's), and no failure side
effects: no playbook_failed bell, report, memory, agent failure count or
heartbeat. The watch neither scores the stop nor waits past its deadline for it;
it follows the rerun. A question the owner dismisses closes the run as their no
(``cancelled``) and its card ``done``, both saying "Dismissed by the owner: …";
the watch closes unscored, as F143 does for a rejected plan. JEV's join keys on
the two prefixes.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ASK_ACTION = "platform_ask_human"
# What the run, its card and the watch say (JEV keys on the prefixes).
NEEDS_YOU = "Needs you:"
DISMISSED = "Dismissed by the owner:"
# The line a step writes to stop and ask; the playbook step context names it.
NEEDS_YOU_LINE = "NEEDS YOU:"
RERUN_WARNING = "Answering runs the whole playbook again from step 1."
# The question's details, the stopped run's metadata, the rerun's metadata.
ASK_MARKER = "playbook_ask"
STOPPED_KEY = "needs_owner"
ANSWERS_KEY = "owner_answers"
# How the watch ticker reads a run that stopped for the owner: not terminal yet.
WAITING_FOR_OWNER = "waiting_for_owner"

ASK_TAKEN = ("Your question is with the owner. This run stops after this step, and their answer "
             "runs the playbook again from step 1. End your answer now.")
ASK_WITHOUT_QUESTION = ("platform_ask_human needs your question: params={\"question\": \"<what you need "
                        "from the owner>\"}. In a playbook step the run is the subject; name none.")
NOT_RUN_AFTER_ASK = "Not run: this step stopped to ask the owner."

# (c) The line, allowing markdown around it.
_NEEDS_YOU_AT = re.compile(r"^[ \t>*_#-]*NEEDS YOU:[*_ \t]*", re.MULTILINE)
# (b) A draft addressed to someone else: an email's subject line or greeting. (Not
# a quote or a code block: an ask may quote the error that stopped it.)
_DRAFT = re.compile(r"^\s*(?:subject\s*:|(?:dear|hi|hello|hey)\b[^\n]{0,60},\s*$)", re.IGNORECASE | re.MULTILINE)
# (b) The step says what it lacks...
_NEED = re.compile(r"\bI(?:'m| am| was)? (?:need|unable|not able)\b"
                   r"|\bI (?:can't|cannot|can not|couldn't|could not|don't have|do not have"
                   r"|didn't find|did not find|wasn't able|was not able)\b", re.IGNORECASE)
# ...or asks the reader for it. On a step whose job is writing to someone, only
# the first kind counts: "Could you confirm Thursday?" may be the deliverable.
_REQUEST = re.compile(r"\b(?:please|could you|can you|would you|will you)\b[^?\n]{0,80}?"
                      r"\b(?:provide|confirm|specify|clarify|share|send|upload|tell|give|point)\b"
                      r"|\blet me know\b|\btell me\b", re.IGNORECASE)
_DRAFTING_STEP = re.compile(r"\b(?:draft|drafts|drafted|compose|reply|caption|post|tweet|message|email|emails"
                            r"|e-mail|sms|letter|newsletter|announcement)\b", re.IGNORECASE)
# (b') F183: ...or says its work waits for the answer. Night 6's #1097 listed four
# questions mid-answer, then "Without this information, I can only create a very
# generic draft. Once I have a better understanding …, I will draft the newsletter".
_DEFERRED = re.compile(r"\b(?:once|as soon as|when) (?:I|you)\b[^.\n]{0,120}?\bI(?:'ll| will)\b"
                       r"|\bwithout (?:this|that|these|those|the|your|more|further) "
                       r"(?:information|info|details?|input|answers?|context)\b", re.IGNORECASE)

# A call's name without one of these verbs changed something; the scratchpad,
# the ask and pre_exec belong to the run itself.
_READ_VERBS = frozenset({"get", "list", "fetch", "search", "read", "retrieve", "lookup", "find", "query",
                         "download", "describe", "view", "check", "count", "export", "recommend"})
_RUN_OWN = frozenset({"scratchpad_write", "scratchpad_read", ASK_ACTION, "pre_exec"})
_SUMMARY = re.compile(r"^(?P<action>.+) \((?P<how>success|error)\)$")


# ---------------------------------------------------------------------------
# (a) platform_ask_human inside a step
# ---------------------------------------------------------------------------

def ask_call_params(tool_name: str, tool_args: Any) -> Optional[Dict[str, Any]]:
    """The params of a platform_ask_human call however it was made (the tool, or
    platform_execute's action), else None."""
    if not isinstance(tool_args, dict):
        return None
    if tool_name == ASK_ACTION:
        params: Any = tool_args
    elif tool_args.get("action") == ASK_ACTION:
        params = tool_args.get("params")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                params = {}
    else:
        return None
    return params if isinstance(params, dict) else {}


def take_ask(params: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """(the ask, or None when it names no question; the tool result the model reads)."""
    question = str(params.get("question") or "").strip()
    if not question:
        return None, ASK_WITHOUT_QUESTION
    options = params.get("options")
    options = [str(o).strip() for o in options if str(o).strip()] if isinstance(options, list) else []
    return {"question": question, "options": options or None}, ASK_TAKEN


# ---------------------------------------------------------------------------
# Did this step ask the owner?
# ---------------------------------------------------------------------------

def owner_question(output: Any, result: Dict[str, Any], prompt_template: str) -> Optional[Dict[str, Any]]:
    """``{question, options}`` when this step put a question to the owner, else None."""
    ask = result.get("owner_ask")
    if isinstance(ask, dict) and ask.get("question"):
        return {"question": str(ask["question"]), "options": ask.get("options")}
    text = str(output or "").strip()
    marked = _NEEDS_YOU_AT.search(text)
    if marked and text[marked.end():].strip():
        return {"question": text[marked.end():].strip(), "options": None}
    if _asks_for_what_it_needs(text, prompt_template):
        return {"question": text, "options": None}
    return None


def _asks_for_what_it_needs(text: str, prompt_template: str) -> bool:
    from config import config

    text = text.replace("’", "'")
    if not text or _DRAFT.search(text):
        return False
    if _defers_to_the_owner(text):
        return True
    if len(text) > config.PLAYBOOK_OWNER_ASK_MAX_CHARS:
        return False
    if not _is_question(text.splitlines()[-1]):
        return False
    if _NEED.search(text):
        return True
    return bool(_REQUEST.search(text)) and not _DRAFTING_STEP.search(prompt_template or "")


def _is_question(line: str) -> bool:
    return line.strip().rstrip("*_) ").endswith("?")


def _defers_to_the_owner(text: str) -> bool:
    """F183: says what it lacks, asks for it, and puts the work off until it has
    it. The questions may sit mid-answer, the promise last."""
    from config import config

    return (len(text) <= config.OWNER_ASK_DEFERRED_MAX_CHARS and bool(_NEED.search(text))
            and bool(_DEFERRED.search(text)) and any(_is_question(line) for line in text.splitlines()))


# ---------------------------------------------------------------------------
# The question the owner reads
# ---------------------------------------------------------------------------

def _changes_something(action: str) -> bool:
    if not action or action in _RUN_OWN:
        return False
    try:
        from modules.tools.discovery.action_registry import get_action_registry

        definition = get_action_registry().get(action)
    except Exception:  # noqa: BLE001 -- no registry: judge by the name
        definition = None
    if definition is not None:
        return definition.permission_level in ("write", "destructive")
    words = set(re.split(r"[^a-z0-9]+", action.lower()))
    return not (words & _READ_VERBS)


def changes_so_far(db: Session, step_results: List[dict], step_order: Any,
                   step_calls: List[dict]) -> Tuple[List[Tuple[Any, str]], List[Any]]:
    """What this run already changed outside itself: ``[(step, action)]`` from the
    finished steps and the asking step's own calls, and the session steps, whose
    calls the run never sees."""
    from services.cli_ticket_lane import is_cli_agent

    changes: List[Tuple[Any, str]] = []
    sessions: List[Any] = []
    for step in step_results:
        if step.get("status") != "completed":
            continue
        try:
            session_step = is_cli_agent(db, step.get("agent_id"))
        except Exception:  # noqa: BLE001
            session_step = False
        if session_step:
            sessions.append(step.get("order"))
        for line in step.get("tool_calls_summary") or []:
            found = _SUMMARY.match(str(line))
            if found and found.group("how") == "success" and _changes_something(found.group("action")):
                changes.append((step.get("order"), found.group("action")))
    for call in step_calls:
        if call.get("success") is True and _changes_something(str(call.get("action") or "")):
            changes.append((step_order, str(call["action"])))
    return changes, sessions


def question_for_owner(question: str, changes: List[Tuple[Any, str]], sessions: List[Any]) -> str:
    """The step's question, then what answering does: the rerun from step 1, and
    every change the run already made, since the rerun makes it again."""
    lines = [question.strip(), "", RERUN_WARNING]
    if changes:
        done = "; ".join(f"step {step}: {action}" for step, action in changes)
        lines.append(f"This run already made these changes, and the rerun makes them again: {done}.")
    if sessions:
        which = ", ".join(str(s) for s in sessions)
        lines.append(f"Step {which} ran as a Claude Code session; what it changed is on its ticket, "
                     "and the rerun runs it again.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The stop
# ---------------------------------------------------------------------------

def _run_card(db: Session, recipe: Any, execution: Any) -> Any:
    from core.models.core import BoardTask
    from services.board_task_bridge import create_recipe_board_task

    def _find():
        return (db.query(BoardTask)
                .filter(BoardTask.workspace_id == execution.workspace_id, BoardTask.source_type == "recipe",
                        BoardTask.source_id == execution.execution_id)
                .first())

    card = _find()
    if card is None:  # its creation is best-effort; the question needs it
        create_recipe_board_task(db, recipe, execution)
        card = _find()
    return card


async def stop_for_owner(db: Session, *, execution: Any, recipe: Any, step_order: Any, agent_id: Any,
                         agent_name: Optional[str], ask: Dict[str, Any], step_results: List[dict],
                         step_calls: List[dict], inputs: Optional[List[str]] = None) -> bool:
    """End the run needing the owner and put the question in their Questions,
    on the run's card. False when this run may not ask (a run a website visitor
    started never reaches the owner's Questions) or when nothing of the stop was
    saved; the caller then fails the run as before. ``inputs`` names the inputs
    the run stopped for (F182), which the answer's rerun is given."""
    from core.security.surface import widget_turn
    from modules.tools.discovery.handlers_asks import stage_question
    from services.cli_host_service import MAX_ASK_QUESTION_KEPT

    if widget_turn():
        return False
    question = ask["question"].strip()
    if len(question) > MAX_ASK_QUESTION_KEPT:
        question = question[:MAX_ASK_QUESTION_KEPT].rstrip() + "…"
    changes, sessions = changes_so_far(db, step_results, step_order, step_calls)
    try:
        card = _run_card(db, recipe, execution)
        if card is None:
            return False
        execution.status = "failed"
        execution.error_message = f"{NEEDS_YOU} {question}"
        execution.completed_at = datetime.now(timezone.utc)
        execution.step_results = list(step_results)
        # The marker rides the question's own first commit: the run, its card and
        # the question are saved together or not at all.
        execution.execution_metadata = {
            **(execution.execution_metadata or {}),
            STOPPED_KEY: {"step": step_order, "card_id": card.id},
        }
        staged = await stage_question(
            db, execution.workspace_id,
            subject_type="board_task", subject_id=str(card.id),
            question=question_for_owner(question, changes, sessions), options=ask.get("options"),
            asked_by_agent_id=int(agent_id) if agent_id else None, agent_name=agent_name, park=card,
            details={ASK_MARKER: {"execution_id": execution.execution_id, "recipe_id": recipe.id,
                                  "step": step_order, "question": question,
                                  **({"inputs": list(inputs)} if inputs else {})}},
        )
        card.blocked_reason = f"{NEEDS_YOU} {question} (ask #{staged['ask_id']})"
        execution.execution_metadata = {
            **(execution.execution_metadata or {}),
            STOPPED_KEY: {"ask_id": staged["ask_id"], "step": step_order, "card_id": card.id},
        }
        db.commit()
    except Exception:  # noqa: BLE001 -- the run still ends, one way or the other
        logger.error("[F140] asking the owner failed part-way for %s", execution.execution_id, exc_info=True)
        db.rollback()
        # stage_question commits twice. After its first commit the run, its card
        # and the question are saved and the owner is asked: the stop stands, or
        # the caller would fail the run with every side effect the stop avoids.
        return _stop_saved(db, execution)
    logger.info("[F140] %s stopped at step %s to ask the owner (ask #%s)",
                execution.execution_id, step_order, staged["ask_id"])
    return True


def _stop_saved(db: Session, execution: Any) -> bool:
    try:
        db.refresh(execution)
    except Exception:  # noqa: BLE001 -- unreadable: the caller fails the run as before
        return False
    return waiting_for_owner(execution.status, execution.execution_metadata)


def ended_with_an_outcome(execution_model: Any) -> Any:
    """SQL: the runs that ended with an outcome. A run that stopped to ask the
    owner has none (the answer's rerun has it), so the repeated-failure breaker
    and a scheduled playbook's watch skip it, as they skip a cancelled run: each
    scheduled fire asks its own question and runs as normal."""
    from sqlalchemy import or_

    metadata = execution_model.execution_metadata
    return or_(metadata.is_(None), ~metadata.has_key(STOPPED_KEY))


def waiting_for_owner(status: Any, execution_metadata: Any) -> bool:
    """A run that stopped to ask the owner and has not been answered or dismissed."""
    stopped = (execution_metadata or {}).get(STOPPED_KEY) if isinstance(execution_metadata, dict) else None
    return (status == "failed" and isinstance(stopped, dict)
            and not stopped.get("rerun") and not stopped.get("dismissed_at"))


# ---------------------------------------------------------------------------
# The answer, and a dismissal
# ---------------------------------------------------------------------------

def ask_marker(grant: Any) -> Optional[Dict[str, Any]]:
    """``{execution_id, recipe_id, step, question}`` when this question stopped a run."""
    details = getattr(grant, "details", None)
    marker = details.get(ASK_MARKER) if isinstance(details, dict) else None
    return marker if isinstance(marker, dict) and marker.get("execution_id") else None


def _stopped_run(db: Session, grant: Any) -> Tuple[Any, Any]:
    """(the question's card, the run it stopped), each None when gone."""
    from core.models.core import BoardTask, RecipeExecution

    try:
        card_id = int(grant.subject_id)
    except (TypeError, ValueError):
        card_id = None
    card = (db.query(BoardTask).filter(BoardTask.id == card_id, BoardTask.workspace_id == grant.workspace_id)
            .first() if card_id is not None else None)
    execution = (db.query(RecipeExecution)
                 .filter(RecipeExecution.execution_id == ask_marker(grant)["execution_id"],
                         RecipeExecution.workspace_id == grant.workspace_id)
                 .first())
    return card, execution


def _live_watch(db: Session, execution: Any) -> Any:
    try:
        from services.watch_service import WatchService

        return WatchService.find_live_watch(db, workspace_id=execution.workspace_id,
                                            target_type="playbook_execution",
                                            target_id=execution.execution_id)
    except Exception:  # noqa: BLE001 -- a watch is an observer; the run goes on without it
        logger.warning("[F140] watch lookup failed for %s", execution.execution_id, exc_info=True)
        return None


def owner_answers_block(execution_metadata: Any) -> str:
    """Every answer the owner gave this playbook's stopped runs, for every step's prompt."""
    answers = (execution_metadata or {}).get(ANSWERS_KEY) if isinstance(execution_metadata, dict) else None
    if not answers:
        return ""
    lines = ["## The owner's answer",
             "An earlier run of this playbook stopped to ask the owner, and they answered. "
             "Use the answer; do not ask it again."]
    for entry in answers:
        lines += ["", f"**Step {entry.get('step')} asked:** {entry.get('question')}",
                  f"**The owner answered:** {entry.get('answer')}"]
    return "\n".join(lines)


def rerun_after_answer(db: Session, grant: Any) -> bool:
    """The owner answered: run the playbook again from step 1 with the answer, on
    the same card, and the watch follows. The answer is the owner's go for the
    rerun (the question said what answering does), so no second approval is asked.
    True when the rerun launched."""
    from core.models.core import WorkflowTemplate
    from services.cli_host_service import MAX_ASK_ANSWER_KEPT, _kept
    from services.operator_stop import operator_stop
    from services.watch_rerun import TRIGGERED_BY_HUMAN, create_rerun_execution, launch_execution

    marker = ask_marker(grant)
    card, original = _stopped_run(db, grant)
    if card is None or original is None or card.status != "blocked" or operator_stop(card):
        return False
    recipe = (db.query(WorkflowTemplate)
              .filter(WorkflowTemplate.id == original.recipe_id, WorkflowTemplate.workspace_id == grant.workspace_id)
              .first())
    if recipe is None:
        return False

    answered = {"step": marker.get("step"), "question": marker.get("question"), "ask_id": grant.id,
                "answer": _kept(grant.answer_text or "", MAX_ASK_ANSWER_KEPT, what="The owner's answer is",
                                grant_id=grant.id)}
    earlier = (original.execution_metadata or {}).get(ANSWERS_KEY) or []
    rerun = create_rerun_execution(db, recipe, original, triggered_by=TRIGGERED_BY_HUMAN)
    rerun.execution_metadata = {**(rerun.execution_metadata or {}), ANSWERS_KEY: [*earlier, answered]}
    if marker.get("inputs"):  # F182: the run stopped for inputs, and the answer gives them
        from core.services.playbook_inputs import contract_of, inputs_from_answer

        given = inputs_from_answer(grant.answer_text or "", list(marker["inputs"]), contract_of(recipe))
        rerun.input_data = {**(rerun.input_data or {}), **given}

    stopped = dict((original.execution_metadata or {}).get(STOPPED_KEY) or {})
    original.execution_metadata = {**(original.execution_metadata or {}),
                                   STOPPED_KEY: {**stopped, "rerun": rerun.execution_id}}
    # The card follows the work: one card for the whole job.
    card.source_id = rerun.execution_id
    card.status = "in_progress"
    card.blocked_at = None
    card.blocked_reason = None
    card.completed_at = None
    card.error_message = None
    card.result = f"Re-run after your answer: the playbook runs again from step 1 ({rerun.execution_id})."
    card.planning_data = {**(card.planning_data or {}), "execution_id": rerun.execution_id,
                          "rerun_after_answer": {"from": original.execution_id, "ask_id": grant.id}}
    watch = _live_watch(db, original)
    if watch is not None:
        from services.watch_service import WatchService

        WatchService.follow(db, watch, new_target_type="playbook_execution", new_target_id=rerun.execution_id,
                            reason=f"re-run after the owner's answer to ask #{grant.id}")
    db.commit()  # the engine's task opens its own session
    launch_execution(rerun)
    logger.info("[F140] ask #%s answered: %s runs again as %s", grant.id, original.execution_id,
                rerun.execution_id)
    return True


def dismiss_stopped_run(db: Session, grant: Any) -> None:
    """The owner dismissed the question: the run closes as their no (``cancelled``),
    its card ``done``, both "Dismissed by the owner: <question>", and the watch
    closes unscored. Never failed, never completed."""
    marker = ask_marker(grant)
    card, execution = _stopped_run(db, grant)
    reason = f"{DISMISSED} {marker.get('question') or grant.question_md}"
    now = datetime.now(timezone.utc)
    if card is not None and card.status == "blocked":
        card.status = "done"
        card.completed_at = now
        card.error_message = reason
        card.blocked_at = None
        card.blocked_reason = None
    if execution is None or execution.status != "failed":
        return
    execution.status = "cancelled"
    execution.error_message = reason
    stopped = dict((execution.execution_metadata or {}).get(STOPPED_KEY) or {})
    execution.execution_metadata = {**(execution.execution_metadata or {}),
                                    STOPPED_KEY: {**stopped, "dismissed_at": now.isoformat()}}
    from services.watch_hooks import watch_ingest_terminal

    watch_ingest_terminal(db, workspace_id=execution.workspace_id, target_type="playbook_execution",
                          target_id=execution.execution_id, terminal_state="cancelled", summary=reason)
