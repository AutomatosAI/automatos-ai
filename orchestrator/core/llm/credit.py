"""F197 (night 6): running out of model credit is said once, in plain words, and
a scheduled report that failed on it runs again when credit is back.

From 05:42Z the provider refused calls for credit (F196). The raw 402 went onto
the owner's surfaces, carrying the account's internal user id: the board
(tickets 1160, 1162-1165: "Task execution failed after 2 attempts: Error code:
402 - {'error': ..."), Reports, the bell (17 notices in 14 minutes) and Auto's
sentences. The failed Monday Stock Report never said so in plain words, and
never ran again when credit returned.

Now:
- A run that fails on credit says so in one sentence; the raw text stays in the
  log.
- The bell gets one notice per workspace per outage.
- A scheduled run that failed on credit is marked. The next successful model
  call for its workspace runs it again, once.

The outage is tracked in this process. The marks are in the database, so a
restart loses no run.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

OUT_OF_CREDIT = "out_of_credit"
SCHEDULED_TRIGGERS = frozenset({"cron_scheduler"})
MARK_KEY = "out_of_credit"          # recipe_executions.execution_metadata

OUT_OF_CREDIT_TEXT = ("The AI provider's account ran out of credit, so this stopped before it finished. "
                      "Top up the provider account, then run it again.")
OUT_OF_CREDIT_SCHEDULED_TEXT = ("The AI provider's account ran out of credit, so this stopped before it finished. "
                                "It runs again by itself once credit is back.")
OUTAGE_TITLE = "Out of AI credit"
OUTAGE_NOTICE = ("The AI provider's account ran out of credit, so tasks and playbooks stop until it is topped up. "
                 "This is the only notice until credit is back. Scheduled reports that stopped run again by "
                 "themselves then.")

# The provider's words (OpenRouter's 402 bodies) and our own sentence above.
_CREDIT = re.compile(r"Error code: 402\b|\b402 Payment Required\b|requires more credits|exceed your available "
                     r"credits|insufficient (?:credit|balance|funds)|ran out of credit", re.I)
_PROVIDER_STATUS = re.compile(r"Error code: (\d{3}) - \{")
_ACCOUNT_ID = re.compile(r"\buser_[A-Za-z0-9]{8,}\b")
PROVIDER_REFUSED_TEXT = ("The AI provider refused the request (HTTP {status}), so this stopped. "
                         "The details are in the server log.")
# The bell events a credit failure reaches: a failed task or playbook, and the
# report filed for the failed task.
CREDIT_EVENTS = frozenset({"task_failed", "playbook_failed", "playbook_step_failed", "mission_failed",
                           "report_submitted"})

_lock = threading.Lock()
_outage: Dict[str, bool] = {}
_seen_since_start: Set[str] = set()
_running: Set["asyncio.Task[None]"] = set()


def is_out_of_credit(error: Any) -> bool:
    return bool(_CREDIT.search(str(error or "")))


def plain_failure(error: Any, *, scheduled: bool = False) -> str:
    """What the owner reads for a run that failed on ``error``. A credit failure
    is one sentence. Another provider refusal gives its status, not its payload.
    Anything else keeps its words, without an account id."""
    text = str(error or "")
    if is_out_of_credit(text):
        return OUT_OF_CREDIT_SCHEDULED_TEXT if scheduled else OUT_OF_CREDIT_TEXT
    refused = _PROVIDER_STATUS.search(text)
    if refused:
        return PROVIDER_REFUSED_TEXT.format(status=refused.group(1))
    return _ACCOUNT_ID.sub("user_…", text)


def first_notice_of_outage(workspace_id: Any) -> bool:
    """True for the first credit failure a workspace's bell hears about."""
    key = str(workspace_id)
    with _lock:
        if _outage.get(key):
            return False
        _outage[key] = True
        return True


def _credit_is_back(workspace_id: str) -> bool:
    """True when this success ends an outage, or is the workspace's first since
    the process started (marks from before a restart are looked for then)."""
    with _lock:
        ended = _outage.pop(workspace_id, False)
        first = workspace_id not in _seen_since_start
        _seen_since_start.add(workspace_id)
    return ended or first


def mark_for_rerun(execution: Any) -> bool:
    """Mark a scheduled run that failed on credit; True when it was marked."""
    if getattr(execution, "triggered_by", None) not in SCHEDULED_TRIGGERS:
        return False
    execution.execution_metadata = {**(execution.execution_metadata or {}), MARK_KEY: True}
    return True


def _stage_reruns(workspace_id: str) -> List[Any]:
    """The marked runs of a workspace, each copied into a new execution (sync:
    call it off the loop). The marks are cleared, so each runs again once."""
    from sqlalchemy.orm.attributes import flag_modified

    from core.database.database import SessionLocal
    from core.models import RecipeExecution, WorkflowTemplate
    from services.watch_rerun import create_rerun_execution

    staged: List[Any] = []
    db = SessionLocal()
    try:
        marked = (db.query(RecipeExecution)
                  .filter(RecipeExecution.workspace_id == workspace_id,
                          RecipeExecution.status == "failed",
                          RecipeExecution.execution_metadata[MARK_KEY].astext == "true")
                  .all())
        for original in marked:
            recipe = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == original.recipe_id).first()
            if recipe is None:
                continue
            rerun = create_rerun_execution(db, recipe, original, triggered_by="credit_back")
            original.execution_metadata = {**(original.execution_metadata or {}), MARK_KEY: False,
                                           "rerun_on_credit": rerun.execution_id}
            flag_modified(original, "execution_metadata")
            staged.append(rerun)
        db.commit()
        for rerun in staged:
            db.refresh(rerun)
            db.expunge(rerun)
        return staged
    except Exception:
        db.rollback()
        logger.warning("[F197] could not stage the reruns for %s", workspace_id, exc_info=True)
        return []
    finally:
        db.close()


async def _rerun_marked(workspace_id: str) -> None:
    from services.watch_rerun import launch_execution

    for rerun in await asyncio.to_thread(_stage_reruns, workspace_id):
        logger.info(f"[F197] credit is back: {rerun.retry_of} runs again as {rerun.execution_id}")
        try:
            launch_execution(rerun)
        except Exception:
            logger.warning("[F197] could not launch rerun %s", rerun.execution_id, exc_info=True)


def note_model_success(workspace_id: Any) -> None:
    """A model call for ``workspace_id`` just succeeded. When that ends an outage
    (or is the first since start), the workspace's marked runs go again. Only
    dictionary work happens here; the database work runs off the loop. With no
    loop running nothing is consumed, so the next call on a loop still sees it."""
    if not workspace_id:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    key = str(workspace_id)
    if not _credit_is_back(key):
        return
    task = loop.create_task(_rerun_marked(key))
    _running.add(task)
    task.add_done_callback(_running.discard)


def reset() -> None:
    """Tests: forget every outage and every workspace seen."""
    with _lock:
        _outage.clear()
        _seen_since_start.clear()
