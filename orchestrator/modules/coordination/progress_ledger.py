"""Progress ledger + joiner decision — PRD-164 S4 bounded replanning.

Magentic-One outer-loop pattern: after every coordinator tick the run's task
states are reduced to a small snapshot and compared against the ledger stored
on ``run.config['progress_ledger']``:

* **progress** — done/verified counts moved forward → stall streak resets;
* **churn**    — attempts climbed (or a verified task regressed) while nothing
  moved forward → the loop signal → streak increments;
* **idle**     — nothing changed at all (e.g. waiting out a stall threshold
  between 5s ticks) → streak unchanged, and the SAME ledger object is returned
  so the caller can skip the JSONB write.

When the streak reaches ``stall_limit`` the LLMCompiler-style joiner verdict
is REPLAN while ``replan_count`` is under ``max_replans``, else HALT. This
module is pure (no DB, no clock injection beyond ``datetime.now`` for audit
timestamps, no I/O) — the coordinator owns persistence and side effects.

Deliberately NO new planner algorithm lives here (PRD-164 non-goal): the
joiner only decides; replanning goes through the one existing
``CoordinatorService.replan_mission`` engine.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple

from core.models.orchestration_enums import DONE_TASK_STATES, TaskState

# Bounded audit trail kept on the mission config — enough to reconstruct why
# the joiner intervened without growing the JSONB forever.
HISTORY_LIMIT = 20


class JoinerDecision(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    HALT = "halt"


def snapshot_tasks(tasks: Iterable[Any]) -> Dict[str, int]:
    """Reduce a run's tasks to the four counters the ledger compares.

    ``tasks`` only needs ``.state`` (TaskState value string) and
    ``.attempt_number`` — ORM rows and test stubs both qualify.
    """
    total = done = verified = attempts = 0
    for task in tasks:
        total += 1
        state = TaskState(task.state)
        if state in DONE_TASK_STATES:
            done += 1
        if state == TaskState.VERIFIED:
            verified += 1
        attempts += int(task.attempt_number or 0)
    return {"total": total, "done": done, "verified": verified,
            "attempts": attempts}


def _entry(observation: str, streak: int, decision: JoinerDecision,
           snapshot: Optional[Dict[str, int]]) -> Dict[str, Any]:
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "observation": observation,
        "stall_streak": streak,
        "decision": decision.value,
        "snapshot": dict(snapshot) if snapshot else None,
    }


def advance(
    ledger: Optional[Dict[str, Any]],
    snapshot: Dict[str, int],
    *,
    stall_limit: int,
    replan_count: int,
    max_replans: int,
) -> Tuple[Dict[str, Any], JoinerDecision]:
    """Advance the ledger with a fresh snapshot and return the joiner verdict.

    Never mutates its inputs. On an idle observation below the limit the
    ORIGINAL ledger object is returned unchanged (callers use identity to
    skip persisting a no-op).
    """
    previous: Optional[Dict[str, Any]] = (ledger or {}).get("snapshot")
    prior_streak = int((ledger or {}).get("stall_streak", 0))

    if previous is None:
        observation, streak = "baseline", 0
    else:
        progressed = (
            snapshot["done"] > int(previous.get("done", 0))
            or snapshot["verified"] > int(previous.get("verified", 0))
        )
        churned = (
            snapshot["attempts"] > int(previous.get("attempts", 0))
            or snapshot["verified"] < int(previous.get("verified", 0))
        )
        if progressed:
            observation, streak = "progress", 0
        elif churned:
            observation, streak = "churn", prior_streak + 1
        else:
            observation, streak = "idle", prior_streak

    if streak >= stall_limit:
        decision = (
            JoinerDecision.REPLAN
            if replan_count < max_replans
            else JoinerDecision.HALT
        )
    else:
        decision = JoinerDecision.CONTINUE

    if (
        ledger is not None
        and observation == "idle"
        and decision is JoinerDecision.CONTINUE
    ):
        return ledger, decision

    history: List[Dict[str, Any]] = list((ledger or {}).get("history", []))
    history = history[-(HISTORY_LIMIT - 1):] + [
        _entry(observation, streak, decision, snapshot)
    ]
    new_ledger = {
        "snapshot": dict(snapshot),
        "stall_streak": streak,
        "history": history,
        "last_decision": decision.value,
        "updated_at": history[-1]["at"],
    }
    return new_ledger, decision


def reset_after_replan(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Rebaseline after a successful joiner replan: streak back to zero and
    the stored snapshot cleared so the fresh plan's first tick reads as a
    baseline, not as churn against pre-replan counts. History is preserved
    (audit trail) with an explicit reset marker appended."""
    history = list(ledger.get("history", []))
    history = history[-(HISTORY_LIMIT - 1):] + [
        _entry("replan_reset", 0, JoinerDecision.CONTINUE, None)
    ]
    return {
        "snapshot": None,
        "stall_streak": 0,
        "history": history,
        "last_decision": JoinerDecision.CONTINUE.value,
        "updated_at": history[-1]["at"],
    }
