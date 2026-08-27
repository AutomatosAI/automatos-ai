"""
Watch decision step -- PRD-204 S10
==================================

The deterministic brain between "terminal observed" and "watch closed".
``WatchService.ingest_terminal`` records the terminal and pulls the next
check forward; the ticker claims the watch and hands it here. Policy is a
DATA TABLE (golden-file tested), not scattered ifs:

    POLICY_TABLE[policy] = {
        scores:             score the run output (S6 RunVerdictService)
        acts_on_low_score:  below-threshold completed run -> diagnose + act
        acts_on_failure:    failed run -> diagnose + act
        compares:           build a before/after change report from lineage
        recurring:          never closes on target terminal (persistent)
    }

Decision spine per terminal:
    cancelled/unknown   -- handled at ingest (S2 semantics unchanged)
    persistent          -- record meaningful change, notify on flip, stay
    score (if scores)   -- S6 verdict -> final_score/final_verdict; a failed
                           judge degrades to close-by-outcome (fail-soft)
    pass                -- close PASSED + watch_verdict notification
    low score / failure -- if the policy acts AND no improve cycle was spent
                           (ONE tweak+rerun, then rescore -> final):
                           diagnose (LLM job a) -> choose allowed action ->
                           playbook: record_action + S7 request_rerun
                           mission:  S8 run_mission_action (own budget+gate)
                           else close FAILED + watch_verdict
    action budget       -- record_action / run_mission_action hard-stop ->
                           escalate (board card + watch_escalation)

LLM is used for EXACTLY two bounded jobs (single calls, cost-attributed
``request_type='watch'`` / ``execution_id='watch-<id>'`` like S6):
(a) failure/low-score diagnosis -> one-paragraph cause + proposed action +
optional step_overrides draft; (b) tweak drafting when a tweak was proposed
without overrides. Never loops -- ``action_budget`` is the hard rail.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# The policy table (S10) -- golden-file locked in tests/golden/
# ---------------------------------------------------------------------------

POLICY_TABLE: Dict[str, Dict[str, bool]] = {
    "run_and_report": {
        "scores": True,
        "acts_on_low_score": False,
        "acts_on_failure": False,
        "compares": False,
        "recurring": False,
    },
    "score_and_improve": {
        "scores": True,
        "acts_on_low_score": True,
        "acts_on_failure": True,
        "compares": False,
        "recurring": False,
    },
    "watch_change": {
        "scores": True,
        "acts_on_low_score": False,
        "acts_on_failure": False,
        "compares": True,
        "recurring": False,
    },
    "persistent": {
        "scores": False,
        "acts_on_low_score": False,
        "acts_on_failure": False,
        "compares": False,
        "recurring": True,
    },
}

# Default corrective vocabulary per watch shape (watch.allowed_actions
# overrides). tweak_rerun collapses onto the S7 rerun with step_overrides.
DEFAULT_ALLOWED_ACTIONS: Dict[str, List[str]] = {
    "mission": ["replan", "reassign", "spawn_agent", "escalate"],
    "playbook_execution": ["rerun", "tweak_rerun", "escalate"],
    "board_task": ["escalate"],
    "scheduled_playbook": ["escalate"],
}

_COMPLETED_STATES = frozenset({"completed", "verified", "done"})

# Decision labels (returned for logging/tests)
DECIDED_PASSED = "closed_passed"
DECIDED_FAILED = "closed_failed"
DECIDED_ACTED = "acted"
DECIDED_PARKED = "parked_on_grant"
DECIDED_ESCALATED = "escalated"
DECIDED_RECORDED = "recorded_change"
DECIDED_NOOP = "noop"


# ---------------------------------------------------------------------------
# Diagnosis (LLM job a) + tweak drafting (LLM job b)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnosis:
    """One bounded LLM read of why the run came up short."""

    cause: str
    proposed_action: str
    step_overrides: Optional[Dict[str, Dict[str, str]]] = None
    confidence: float = 0.5


_DIAGNOSIS_SYSTEM_PROMPT = """\
You are the diagnosis step of a work supervisor. A launched unit of work
finished badly (failed, or scored below its quality bar). Read the evidence
and answer with ONE root cause and ONE proposed next action.

Rules:
- cause: one paragraph, concrete, grounded in the evidence given.
- proposed_action: exactly one of the allowed actions you are given.
- step_overrides: ONLY when you propose tweak_rerun and can write a better
  prompt for a specific step id; otherwise null.
- Return ONLY a single JSON object.
"""

_TWEAK_SYSTEM_PROMPT = """\
You are drafting a one-shot prompt tweak for a playbook step that produced
weak output. Rewrite the step prompt to fix the diagnosed problem while
keeping the step's original job. Return ONLY a JSON object:
{"step_overrides": {"<step_id>": {"prompt_template": "..."}}}
"""


def _default_llm_factory(watch, model: Optional[str] = None):
    """Same cost attribution as S6: request_type='watch', watch-scoped
    execution_id."""
    from core.llm import create_llm_manager

    llm = create_llm_manager(
        service_name="watch_decider",
        model=model,
        workspace_id=getattr(watch, "workspace_id", None),
        request_type="watch",
    )
    if hasattr(llm, "_tracking_ctx"):
        llm._tracking_ctx["request_type"] = "watch"
        llm._tracking_ctx["execution_id"] = f"watch-{getattr(watch, 'id', 'unknown')}"
    return llm


class WatchDiagnoser:
    """The two bounded LLM jobs. Injectable factory; always stubbed in CI."""

    def __init__(self, llm_factory: Optional[Callable[..., Any]] = None):
        self._llm_factory = llm_factory or _default_llm_factory

    async def diagnose(
        self,
        db: Session,
        watch,
        *,
        terminal_state: str,
        verdict_reasoning: Optional[str],
        allowed_actions: List[str],
    ) -> Optional[Diagnosis]:
        """Single diagnosis call. None on any failure (caller falls back to
        the deterministic default action)."""
        evidence = self._collect_evidence(db, watch, terminal_state)
        prompt = f"""\
## What was asked
{getattr(watch, 'success_criteria', None) or getattr(watch, 'title', '')}

## What happened
Terminal state: {terminal_state}
{f"Verdict reasoning: {verdict_reasoning}" if verdict_reasoning else ""}

## Evidence
{evidence}

## Allowed actions
{allowed_actions}

## Required JSON output
{{
  "cause": "one-paragraph root cause",
  "proposed_action": "one of {allowed_actions}",
  "step_overrides": {{"<step_id>": {{"prompt_template": "..."}}}} or null,
  "confidence": 0.0-1.0
}}
"""
        raw = await self._call_json(
            watch,
            [
                {"role": "system", "content": _DIAGNOSIS_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if raw is None:
            return None

        proposed = str(raw.get("proposed_action") or "").strip()
        if proposed not in allowed_actions:
            proposed = ""
        overrides = raw.get("step_overrides")
        if not isinstance(overrides, dict) or not overrides:
            overrides = None
        confidence = raw.get("confidence", 0.5)
        confidence = (
            max(0.0, min(1.0, float(confidence)))
            if isinstance(confidence, (int, float))
            else 0.5
        )
        return Diagnosis(
            cause=str(raw.get("cause") or "").strip()[:2000],
            proposed_action=proposed,
            step_overrides=overrides,
            confidence=confidence,
        )

    async def draft_tweak(
        self, db: Session, watch, *, diagnosis: Diagnosis
    ) -> Optional[Dict[str, Dict[str, str]]]:
        """LLM job b: draft step_overrides for a proposed tweak that came
        without them. Single call; None on failure (plain rerun instead)."""
        steps_text = self._collect_recipe_steps(db, watch)
        if not steps_text:
            return None
        prompt = f"""\
## Diagnosed problem
{diagnosis.cause}

## Playbook steps (id + current prompt)
{steps_text}

Return the JSON tweak now.
"""
        raw = await self._call_json(
            watch,
            [
                {"role": "system", "content": _TWEAK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if raw is None:
            return None
        overrides = raw.get("step_overrides")
        return overrides if isinstance(overrides, dict) and overrides else None

    # -- internals -----------------------------------------------------

    async def _call_json(self, watch, messages) -> Optional[Dict[str, Any]]:
        from modules.coordination.verification import _extract_judge_json

        try:
            llm = self._llm_factory(watch)
            response = await llm.generate_response(messages)
            return _extract_judge_json(getattr(response, "content", None))
        except Exception:
            logger.error(
                "[WatchDecider] LLM call failed for watch %s",
                getattr(watch, "id", "?"),
                exc_info=True,
            )
            return None

    @staticmethod
    def _collect_evidence(db: Session, watch, terminal_state: str) -> str:
        """Compact, deterministic evidence block (no LLM)."""
        try:
            if watch.target_type == "playbook_execution":
                from core.models.core import RecipeExecution

                execution = (
                    db.query(RecipeExecution)
                    .filter(RecipeExecution.execution_id == watch.target_id)
                    .first()
                )
                if execution is None:
                    return "(target execution missing)"
                steps = [
                    {
                        "step_id": sr.get("step_id"),
                        "status": sr.get("status"),
                        "error": (str(sr.get("error"))[:200] if sr.get("error") else None),
                        "retries": sr.get("retries", 0),
                    }
                    for sr in (execution.step_results or [])
                    if isinstance(sr, dict)
                ]
                return json.dumps(
                    {
                        "error_message": (execution.error_message or "")[:500],
                        "step_results": steps,
                    },
                    default=str,
                )[:4000]

            if watch.target_type == "mission":
                from core.models.orchestration import OrchestrationTask

                failed = (
                    db.query(OrchestrationTask)
                    .filter(
                        OrchestrationTask.run_id == watch.target_id,
                        OrchestrationTask.state == "failed",
                    )
                    .order_by(OrchestrationTask.sequence_number)
                    .all()
                )
                return json.dumps(
                    {
                        "failed_tasks": [
                            {
                                "title": t.title,
                                "agent_role": t.agent_role,
                                "failure": (t.failure_detail or t.failure_reason_code or "")[:300],
                            }
                            for t in failed
                        ]
                    },
                    default=str,
                )[:4000]
        except Exception:
            logger.warning("[WatchDecider] evidence collection failed", exc_info=True)
        return "(no structured evidence available)"

    @staticmethod
    def _collect_recipe_steps(db: Session, watch) -> Optional[str]:
        try:
            from core.models.core import RecipeExecution, WorkflowTemplate

            execution = (
                db.query(RecipeExecution)
                .filter(RecipeExecution.execution_id == watch.target_id)
                .first()
            )
            if execution is None:
                return None
            recipe = (
                db.query(WorkflowTemplate)
                .filter(WorkflowTemplate.id == execution.recipe_id)
                .first()
            )
            if recipe is None:
                return None
            lines = [
                f"- {s.get('step_id')}: {str(s.get('prompt_template') or '')[:300]}"
                for s in (recipe.steps or [])
                if isinstance(s, dict)
            ]
            return "\n".join(lines)[:4000] or None
        except Exception:
            logger.warning("[WatchDecider] step collection failed", exc_info=True)
            return None


# ---------------------------------------------------------------------------
# The decider
# ---------------------------------------------------------------------------


class WatchDecider:
    """Deterministic policy spine. Stateless besides injected seams."""

    def __init__(self, verdict_service=None, diagnoser: Optional[WatchDiagnoser] = None):
        self._verdict_service = verdict_service
        self._diagnoser = diagnoser or WatchDiagnoser()

    # -- policy helpers -------------------------------------------------

    @staticmethod
    def policy_flags(policy: str) -> Dict[str, bool]:
        return POLICY_TABLE.get(policy, POLICY_TABLE["run_and_report"])

    @staticmethod
    def allowed_actions(watch) -> List[str]:
        configured = getattr(watch, "allowed_actions", None)
        if isinstance(configured, list) and configured:
            return [str(a) for a in configured]
        return list(
            DEFAULT_ALLOWED_ACTIONS.get(watch.target_type, ["escalate"])
        )

    def _verdicts(self):
        if self._verdict_service is None:
            from modules.coordination.run_verdict import RunVerdictService

            self._verdict_service = RunVerdictService()
        return self._verdict_service

    # -- entry point (ticker calls this on a claimed, live watch) -------

    async def decide_terminal(
        self, db: Session, watch, terminal_state: str, now
    ) -> str:
        """Run the policy table against a terminal target. Returns a
        decision label (logging/tests). Never raises into the tick."""
        from core.models.watch_enums import CLAIMABLE_WATCH_STATUSES, WatchStatus

        if WatchStatus(watch.status) not in CLAIMABLE_WATCH_STATUSES:
            return DECIDED_NOOP

        flags = self.policy_flags(watch.policy)

        if flags["recurring"]:
            return await self._observe_persistent(db, watch, terminal_state)

        # --- score (S6) ---
        verdict = None
        if flags["scores"]:
            verdict = await self._score(db, watch)

        completed = terminal_state in _COMPLETED_STATES
        scored = verdict is not None and verdict.score is not None
        passed = bool(
            completed
            and (
                not flags["scores"]
                or not scored  # judge unavailable -> close by outcome
                or verdict.passes(watch.quality_threshold or 0.8)
            )
        )

        if flags["compares"]:
            self._record_change_report(db, watch, verdict)

        if passed:
            return await self._close(
                db,
                watch,
                passed=True,
                terminal_state=terminal_state,
                explanation=(verdict.reasoning if verdict else None)
                or "The watched work completed.",
            )

        # --- below threshold or failed ---
        # PRD-224 US-003: a board ticket re-runs through the board's run-now
        # machinery, budget-railed -- rerun while budget remains, escalate when
        # exhausted. It is a plain re-dispatch: no diagnose LLM and no
        # one-improve-cycle cap (the action_budget is the sole limiter).
        if watch.target_type == "board_task":
            return await self._act_board_task(
                db, watch, terminal_state=terminal_state, verdict=verdict,
                completed=completed,
            )

        acts = flags["acts_on_failure"] if not completed else flags["acts_on_low_score"]
        improve_spent = (watch.actions_taken or 0) > 0
        budget_left = (watch.actions_taken or 0) < (watch.action_budget or 0)
        if acts and not improve_spent:
            if not budget_left:
                # Runaway guard: action_budget=0 -> straight to escalate,
                # no diagnosis LLM spend when no action is possible.
                from services.watch_actions import escalate_watch_now

                await escalate_watch_now(
                    db,
                    watch,
                    reason=(
                        f"Run ended '{terminal_state}' below the bar and the "
                        f"action budget is 0 -- handing to a human."
                    ),
                )
                return DECIDED_ESCALATED
            return await self._diagnose_and_act(
                db, watch, terminal_state=terminal_state, verdict=verdict
            )

        explanation = self._failure_explanation(watch, terminal_state, verdict, improve_spent)
        watch.final_verdict = explanation
        return await self._close(
            db,
            watch,
            passed=False,
            terminal_state=terminal_state,
            explanation=explanation,
        )

    # -- persistent -----------------------------------------------------

    async def _observe_persistent(self, db: Session, watch, terminal_state: str) -> str:
        """Recurring watch on a run-shaped target: record the observation,
        notify on a meaningful CHANGE, never close on terminal (deadline
        expiry is the ticker's job)."""
        from core.models.watch_enums import WatchEventType
        from services.watch_notifications import dispatch_watch_notification
        from services.watch_service import WatchService

        event = WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.CHANGE_REPORT.value,
            event_key=f"observed:{watch.target_type}:{watch.target_id}:{terminal_state}",
            summary=f"Observed terminal state '{terminal_state}'",
            snapshot={"terminal_state": terminal_state},
        )
        if event is None:
            return DECIDED_NOOP  # same observation already recorded

        previous = self._previous_observation(db, watch, before_event_id=event.id)
        if previous is not None and previous != terminal_state:
            degraded = terminal_state == "failed"
            await dispatch_watch_notification(
                db,
                watch,
                event_type="watch_escalation" if degraded else "watch_verdict",
                title=(
                    f"Watched work {'degraded' if degraded else 'recovered'}: "
                    f"{(watch.title or '')[:90]}"
                ),
                message=(
                    f"State changed '{previous}' -> '{terminal_state}' on "
                    f"{watch.target_type} {watch.target_id}."
                ),
                status="error" if degraded else "ok",
            )
        return DECIDED_RECORDED

    @staticmethod
    def _previous_observation(db: Session, watch, *, before_event_id) -> Optional[str]:
        from core.models.watch_enums import WatchEventType
        from core.models.watches import WatchEvent

        row = (
            db.query(WatchEvent)
            .filter(
                WatchEvent.watch_id == watch.id,
                WatchEvent.event_type == WatchEventType.CHANGE_REPORT.value,
                WatchEvent.id != before_event_id,
            )
            .order_by(WatchEvent.created_at.desc())
            .first()
        )
        if row is None or not isinstance(row.snapshot, dict):
            return None
        return row.snapshot.get("terminal_state")

    async def observe_scheduled(self, db: Session, watch, playbook, now) -> str:
        """Persistent flip detection for scheduled-playbook watches (called
        from the ticker's scheduled branch): compare the two most recent
        terminal executions; notify on a status flip."""
        from core.models.core import RecipeExecution
        from core.models.watch_enums import WatchEventType
        from services.watch_notifications import dispatch_watch_notification
        from services.watch_service import WatchService

        latest_two = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.recipe_id == playbook.id,
                RecipeExecution.status.in_(("completed", "failed")),
            )
            .order_by(RecipeExecution.started_at.desc())
            .limit(2)
            .all()
        )
        if len(latest_two) < 2:
            return DECIDED_NOOP
        latest, previous = latest_two[0], latest_two[1]
        if latest.status == previous.status:
            return DECIDED_NOOP

        event = WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.CHANGE_REPORT.value,
            event_key=f"flip:{latest.execution_id}",
            summary=(
                f"Run outcome flipped '{previous.status}' -> '{latest.status}' "
                f"on playbook '{playbook.name}'"
            ),
            snapshot={
                "previous": {"execution_id": previous.execution_id, "status": previous.status},
                "latest": {"execution_id": latest.execution_id, "status": latest.status},
            },
            requires_attention=(latest.status == "failed"),
        )
        if event is None:
            return DECIDED_NOOP

        degraded = latest.status == "failed"
        await dispatch_watch_notification(
            db,
            watch,
            event_type="watch_escalation" if degraded else "watch_verdict",
            title=(
                f"Scheduled playbook {'started failing' if degraded else 'recovered'}: "
                f"{playbook.name[:80]}"
            ),
            message=(
                f"Latest run {latest.execution_id} {latest.status} "
                f"(previous run {previous.status})."
            ),
            status="error" if degraded else "ok",
        )
        return DECIDED_RECORDED

    # -- scoring / closing ----------------------------------------------

    async def _score(self, db: Session, watch):
        try:
            verdicts = self._verdicts()
            verdict = await verdicts.score_run(db, watch)
            if verdict is not None:
                verdicts.apply_verdict(db, watch, verdict)
            return verdict
        except Exception:
            logger.error(
                "[WatchDecider] scoring failed for watch %s",
                getattr(watch, "id", "?"),
                exc_info=True,
            )
            return None

    async def _close(
        self, db: Session, watch, *, passed: bool, terminal_state: str,
        explanation: Optional[str],
    ) -> str:
        from core.models.watch_enums import WatchStatus
        from services.watch_notifications import notify_watch_verdict
        from services.watch_service import WatchService

        if not passed and not watch.final_verdict:
            watch.final_verdict = explanation

        WatchService.transition(
            db,
            watch,
            WatchStatus.PASSED if passed else WatchStatus.FAILED,
            reason=f"decision step: terminal '{terminal_state}'",
        )
        await notify_watch_verdict(
            db,
            watch,
            score=watch.final_score,
            explanation=explanation,
            passed=passed,
            terminal_state=terminal_state,
        )
        return DECIDED_PASSED if passed else DECIDED_FAILED

    @staticmethod
    def _failure_explanation(watch, terminal_state, verdict, improve_spent) -> str:
        bits = []
        if terminal_state in _COMPLETED_STATES:
            score = getattr(verdict, "score", None) if verdict else None
            bar = watch.quality_threshold or 0.8
            if score is not None:
                bits.append(
                    f"The run completed but scored {score:.2f} against a bar of {bar:.2f}."
                )
            else:
                bits.append("The run completed but could not be scored.")
        else:
            bits.append(f"The run reached terminal state '{terminal_state}'.")
        if improve_spent:
            bits.append(
                "A corrective attempt was already made; closing with the final result."
            )
        if verdict is not None and verdict.reasoning:
            bits.append(verdict.reasoning.strip())
        return " ".join(bits)

    # -- change report (watch_change) ------------------------------------

    def _record_change_report(self, db: Session, watch, verdict) -> None:
        """Before/after from lineage: prior target's SCORED event vs the
        current verdict. Purely deterministic -- no LLM comparison."""
        from core.models.watch_enums import WatchEventType
        from core.models.watches import WatchEvent
        from services.watch_service import WatchService

        lineage = watch.lineage or []
        if len(lineage) < 2:
            return
        prior_target_id = lineage[-2].get("target_id")

        prior_scored = (
            db.query(WatchEvent)
            .filter(
                WatchEvent.watch_id == watch.id,
                WatchEvent.event_type == WatchEventType.SCORED.value,
                WatchEvent.event_key.like(f"scored:%:{prior_target_id}:%"),
            )
            .order_by(WatchEvent.created_at.desc())
            .first()
        )
        before = prior_scored.score if prior_scored is not None else None
        after = getattr(verdict, "score", None) if verdict else None
        delta = (
            round(after - before, 4)
            if isinstance(before, (int, float)) and isinstance(after, (int, float))
            else None
        )
        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.CHANGE_REPORT.value,
            event_key=f"change:{watch.target_type}:{watch.target_id}",
            summary=(
                f"Before/after: {before if before is not None else 'unscored'} -> "
                f"{after if after is not None else 'unscored'}"
                + (f" (delta {delta:+.4f})" if delta is not None else "")
            ),
            snapshot={
                "before_target_id": prior_target_id,
                "before_score": before,
                "after_target_id": watch.target_id,
                "after_score": after,
                "delta": delta,
            },
            score=after,
        )

    # -- diagnose + act ---------------------------------------------------

    async def _diagnose_and_act(
        self, db: Session, watch, *, terminal_state: str, verdict
    ) -> str:
        from services.watch_actions import (
            ACTION_ESCALATE,
            MISSION_ACTIONS,
            escalate_watch_now,
            run_mission_action,
        )
        from core.models.watch_enums import WatchEventType
        from services.watch_service import WatchService

        allowed = self.allowed_actions(watch)
        diagnosis = await self._diagnoser.diagnose(
            db,
            watch,
            terminal_state=terminal_state,
            verdict_reasoning=getattr(verdict, "reasoning", None) if verdict else None,
            allowed_actions=allowed,
        )
        action = self._choose_action(watch, diagnosis, allowed)
        cause = (
            diagnosis.cause
            if diagnosis is not None and diagnosis.cause
            else f"Run ended '{terminal_state}' below the bar (diagnosis unavailable)."
        )

        WatchService.ingest(
            db,
            watch,
            event_type=WatchEventType.DIAGNOSED.value,
            event_key=f"diagnosed:{watch.target_id}:{watch.actions_taken or 0}",
            summary=cause[:500],
            snapshot={
                "proposed_action": action,
                "confidence": getattr(diagnosis, "confidence", None),
                "has_step_overrides": bool(getattr(diagnosis, "step_overrides", None)),
            },
        )

        if action == ACTION_ESCALATE:
            await escalate_watch_now(db, watch, reason=cause)
            return DECIDED_ESCALATED

        if action in MISSION_ACTIONS:
            outcome = await run_mission_action(db, watch, action, diagnosis=cause)
            if outcome.escalated:
                return DECIDED_ESCALATED
            if outcome.parked:
                return DECIDED_PARKED
            if outcome.executed:
                await self._notify_action(db, watch, action, cause)
                return DECIDED_ACTED
            await escalate_watch_now(
                db, watch, reason=f"'{action}' failed: {outcome.error}"
            )
            return DECIDED_ESCALATED

        # playbook rerun / tweak_rerun
        return await self._act_rerun(db, watch, action, diagnosis, cause)

    @staticmethod
    def _choose_action(watch, diagnosis: Optional[Diagnosis], allowed: List[str]) -> str:
        """Deterministic choice: the diagnosis proposal when allowed, else
        the first sensible allowed default, else escalate."""
        proposed = getattr(diagnosis, "proposed_action", "") if diagnosis else ""
        if proposed in allowed:
            return proposed
        # No usable diagnosis: plain rerun before a tweak label (there is
        # nothing to tweak FROM), then the mission defaults.
        for fallback in ("rerun", "tweak_rerun", "replan", "reassign"):
            if fallback in allowed:
                return fallback
        return "escalate"

    async def _act_rerun(
        self, db: Session, watch, action: str, diagnosis: Optional[Diagnosis], cause: str
    ) -> str:
        from core.models.core import RecipeExecution, WorkflowTemplate
        from services.watch_actions import escalate_watch_now
        from services.watch_rerun import (
            TRIGGERED_BY_WATCH,
            request_rerun,
            validate_step_overrides,
        )
        from services.watch_service import WatchService

        original = (
            db.query(RecipeExecution)
            .filter(
                RecipeExecution.execution_id == watch.target_id,
                RecipeExecution.workspace_id == watch.workspace_id,
            )
            .first()
        )
        recipe = (
            db.query(WorkflowTemplate)
            .filter(WorkflowTemplate.id == original.recipe_id)
            .first()
            if original is not None
            else None
        )
        if original is None or recipe is None:
            await escalate_watch_now(
                db, watch, reason="rerun impossible: execution or playbook missing"
            )
            return DECIDED_ESCALATED

        # Tweak drafting (LLM job b) only when a tweak was chosen without
        # overrides from the diagnosis.
        overrides = getattr(diagnosis, "step_overrides", None) if diagnosis else None
        if action == "tweak_rerun" and not overrides and diagnosis is not None:
            overrides = await self._diagnoser.draft_tweak(db, watch, diagnosis=diagnosis)
        validated, err = validate_step_overrides(recipe, overrides)
        if err:
            logger.warning(
                "[WatchDecider] dropping invalid drafted overrides for watch %s: %s",
                watch.id,
                err,
            )
            validated = None

        _, allowed_budget = WatchService.record_action(
            db,
            watch,
            action=action,
            summary=cause,
            snapshot={"step_overrides": validated},
        )
        if not allowed_budget:
            await escalate_watch_now(
                db,
                watch,
                reason=(
                    f"Action budget exhausted "
                    f"({watch.actions_taken}/{watch.action_budget}) -- refused '{action}'"
                ),
            )
            return DECIDED_ESCALATED

        outcome = await request_rerun(
            db,
            workspace_id=watch.workspace_id,
            recipe=recipe,
            original=original,
            step_overrides=validated,
            triggered_by=TRIGGERED_BY_WATCH,
            watch=watch,
        )
        if outcome.launched:
            await self._notify_action(db, watch, action, cause)
            return DECIDED_ACTED
        return DECIDED_PARKED

    async def _act_board_task(
        self, db: Session, watch, *, terminal_state: str, verdict, completed: bool
    ) -> str:
        """PRD-224 US-003: corrective flow for a board-ticket watch.

        A policy that acts (score_and_improve) re-runs the ticket through the
        run-now machinery (``watch_actions.run_board_task_action``) -- budget-
        railed, escalating the moment the budget is exhausted. A non-acting
        policy (run_and_report) reports the failure verdict and closes. No
        diagnose LLM: a board re-run is a plain replay of authorised work, so
        the deterministic failure explanation is the narration.
        """
        from services.watch_actions import escalate_watch_now, run_board_task_action

        flags = self.policy_flags(watch.policy)
        acts = flags["acts_on_failure"] if not completed else flags["acts_on_low_score"]
        improve_spent = (watch.actions_taken or 0) > 0
        explanation = self._failure_explanation(watch, terminal_state, verdict, improve_spent)

        if not acts:
            watch.final_verdict = explanation
            return await self._close(
                db, watch, passed=False, terminal_state=terminal_state,
                explanation=explanation,
            )

        outcome = await run_board_task_action(db, watch, "rerun", diagnosis=explanation)
        if outcome.escalated:
            return DECIDED_ESCALATED
        if outcome.executed:
            await self._notify_action(db, watch, "rerun", explanation)
            return DECIDED_ACTED
        # run_board_task_action escalates on any failure; belt-and-braces for an
        # unexpected non-escalated error outcome.
        await escalate_watch_now(db, watch, reason=f"re-run unavailable: {outcome.error}")
        return DECIDED_ESCALATED

    @staticmethod
    async def _notify_action(db: Session, watch, action: str, cause: str) -> None:
        from services.watch_notifications import dispatch_watch_notification

        await dispatch_watch_notification(
            db,
            watch,
            event_type="watch_action",
            title=f"Watcher acting ({action}): {(watch.title or '')[:90]}",
            message=cause,
            status="warning",
        )


# ---------------------------------------------------------------------------
# Singleton (house pattern)
# ---------------------------------------------------------------------------

_watch_decider: Optional[WatchDecider] = None


def get_watch_decider() -> WatchDecider:
    global _watch_decider
    if _watch_decider is None:
        _watch_decider = WatchDecider()
    return _watch_decider
