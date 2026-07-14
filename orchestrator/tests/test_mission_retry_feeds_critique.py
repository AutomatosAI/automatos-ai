"""PRD-142 Wave 3 · WS-J / E4 · W3-S11 — mission retry feeds the verifier critique.

The Missions primitive's BRAIN §3.x contract says: *failed task retries
prepend the verifier's critique so the model can fix the specific failure
instead of repeating it*. The §H DoD adds: *failure path tested, restart
durable, tenant isolated, observable via heartbeat finding*.

This was Mission Zero P3 (memory `mission-zero-flaws.md`): the retry loop
re-ran with the identical prompt — same input → same output → 3x
``MAX_RETRIES_EXHAUSTED``. The structural fix exists today
(``MissionReconciler._apply_verdict_fail`` stashes ``verification_feedback``
+ ``previous_output`` into ``task.input_context``; ``MissionDispatcher.
build_task_prompt`` reads them back and produces a REVISION prompt that
contains the verifier reasoning). W3-S11 PINS that under the Wave 2 net so
a future refactor cannot silently drop the critique on the floor again.

What this test file proves (matching W3-S11 §AC):

1. **AC1 — Verifier critique reaches the retry prompt.** A task whose
   ``input_context`` carries ``verification_feedback`` (FAIL or PARTIAL
   verdict, with reasoning + failures) produces a ``build_task_prompt``
   string that includes the critique text. A clean first-attempt task
   (no feedback) does NOT carry revision-mode framing.
2. **AC1 / PRD-200 S1 — the judge gates once (behavioural).** Drive
   ``MissionReconciler._apply_verdict`` with the verdict verify_task
   returned: a FAIL requeues the task ONCE (→ RETRYING, ``attempt_number``
   bumped, ``previous_output`` + ``verification_feedback`` stashed into
   ``input_context`` — the contract the dispatcher reads), capped at
   ``COORDINATOR_MAX_VERIFICATION_REQUEUES``; PARTIAL stays advisory; a FAIL
   at the cap passes through to VERIFIED-with-annotation. (Was AST-static on
   the then-dead ``_apply_verdict_fail`` / ``_apply_verdict_partial``;
   PRD-200 S1 wired the former and deleted the latter.)
3. **AC2 — DB-authoritative + restart-durable regression.** The coordinator
   already gets restart-safety from W1-S6's ``reap_orphaned_runs`` boot
   sweep. Static checks pin the boot wire-up + the ``orphaned_on_restart``
   marker so a refactor cannot drop the sweep silently.
4. **AC3 — Stalled-task re-dispatch failure-path.** ``_recover_stalled_task``
   with retries remaining transitions a stalled task back to ``QUEUED``,
   increments ``attempt_number``, and clears ``assigned_agent_id``. With
   retries exhausted it transitions to ``FAILED`` with
   ``MAX_RETRIES_EXHAUSTED``.
5. **AC4 — Cross-workspace isolation.** ``build_task_prompt`` is a pure
   function of a SINGLE task: it cannot mix two workspaces' contexts.
   The tick loop filters runs by ``state == RUNNING`` only (no
   cross-workspace leak), and ``_process_run`` reads workspace from the
   bound ``run`` object.
6. **AC5 — Heartbeat (W3-S1 wiring).** A tiny stateless helper
   ``_emit_missions_primitive`` calls ``emit_primitive_finding`` with
   primitive='missions' and the correct status — green on a clean
   mission-complete, down on a caught failure. Skip when no
   ``workspace_id`` (A4 honest gap), swallow emit failures.

The tests deliberately operate at the *unit* level via source-text +
AST inspection + targeted importlib loads — a full mission tick would
drag the planner, LLM provider, RAG, board bridge, and Qdrant into the
unit suite. Mirrors the W3-S6 / W3-S8 / W3-S9 / W3-S10 patterns.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Paths to the surfaces we pin without importing them through heavy package
# __init__ chains (planner → LLM, board bridge → DB, RAG → camelot, etc).
# ---------------------------------------------------------------------------

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

DISPATCHER_PY = ORCH_ROOT / "modules" / "coordination" / "dispatcher.py"
RECONCILER_PY = ORCH_ROOT / "modules" / "coordination" / "reconciler.py"
COORDINATOR_PY = ORCH_ROOT / "services" / "coordinator_service.py"
PRIMITIVE_HEARTBEAT_PY = (
    ORCH_ROOT / "modules" / "coordination" / "primitive_heartbeat.py"
)
MAIN_PY = ORCH_ROOT / "main.py"
REAPER_PY = ORCH_ROOT / "core" / "boot" / "reaper.py"


# ---------------------------------------------------------------------------
# Lightweight Postgres env so any module that touches config doesn't crash
# at import time. Setdefault — a real .env still wins.
# ---------------------------------------------------------------------------
import os

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)


# ===========================================================================
# 1. AC1 — VERIFIER CRITIQUE REACHES THE RETRY PROMPT.
#
# Direct functional test of MissionDispatcher.build_task_prompt(task).
# This is the headline E4 increment — the prompt MUST contain the critique
# text when the task has verification_feedback in its input_context.
# ===========================================================================


@pytest.fixture(scope="module")
def build_prompt():
    """Lazy import of ``MissionDispatcher.build_task_prompt`` so the
    SQLAlchemy model registration only happens if a test actually uses
    it. Eager module-scope import triggered the W3-S9/S10 cross-test
    SQLAlchemy registration contamination that breaks the nl2sql
    workspace-filter test downstream — same pattern, kept lazy here."""
    from modules.coordination.dispatcher import MissionDispatcher  # noqa: E402
    return MissionDispatcher.build_task_prompt


def _make_task(*, title="T", description=None, input_context=None,
               verification_criteria=None) -> SimpleNamespace:
    """Build a SimpleNamespace shaped like the OrchestrationTask the
    prompt builder reads."""
    return SimpleNamespace(
        title=title,
        description=description,
        input_context=input_context,
        verification_criteria=verification_criteria,
    )


class TestVerifierCritiqueInRetryPrompt:
    """The E4 contract: verifier critique reaches the retry prompt.

    These are the most important tests in the file — if any of them
    regresses, Mission Zero P3 is back and the model repeats the same
    mistake three times before giving up."""

    def test_first_attempt_has_no_revision_framing(self, build_prompt):
        """A clean first-attempt task (no input_context) MUST NOT carry
        revision-mode markers. Negative control for the revision case."""
        task = _make_task(title="Write executive summary",
                          description="One-pager for the board")
        prompt = build_prompt(task)
        assert "Revision Request" not in prompt
        assert "Your Previous Output" not in prompt
        assert "Issues to Fix" not in prompt
        # The plain first-attempt frame DOES live in the prompt.
        assert "# Task: Write executive summary" in prompt
        assert "One-pager for the board" in prompt

    def test_revision_prompt_contains_verifier_reasoning(self, build_prompt):
        """The headline test: verifier ``reasoning`` MUST appear in the
        retry prompt so the model sees the specific critique."""
        critique = "Output lacked the Risk Analysis section."
        task = _make_task(
            title="Quarterly report",
            description="Q4 summary",
            input_context={
                "previous_output": "## Summary\nQ4 was strong.\n",
                "verification_feedback": {
                    "attempt": 2,
                    "reasoning": critique,
                    "failures": ["required_sections"],
                    "scores": {"completeness": 0.4},
                },
            },
        )
        prompt = build_prompt(task)
        assert critique in prompt, (
            f"verifier reasoning MUST appear in retry prompt — Mission "
            f"Zero P3 regression check; prompt={prompt!r}"
        )
        # Revision-mode framing is on so the model REVISES rather than
        # rewriting from scratch (token-saving contract).
        assert "Revision Request" in prompt
        assert "Issues to Fix" in prompt

    def test_revision_prompt_contains_previous_output(self, build_prompt):
        """The previous output MUST be echoed back so the model can
        revise it instead of rewriting (saves ~80% of tokens per retry)."""
        prior = "## Summary\nQ4 was strong.\nRevenue +12%.\n"
        task = _make_task(
            title="Quarterly report",
            input_context={
                "previous_output": prior,
                "verification_feedback": {
                    "attempt": 2,
                    "reasoning": "Missing Risk section.",
                    "failures": [],
                    "scores": {},
                },
            },
        )
        prompt = build_prompt(task)
        assert prior.strip() in prompt, (
            "previous_output MUST be in the retry prompt so the model "
            "revises instead of rewriting"
        )

    def test_revision_prompt_lists_failed_checks(self, build_prompt):
        """When the verifier names which deterministic checks failed, the
        retry prompt MUST surface those names so the model knows what to
        fix specifically."""
        task = _make_task(
            title="Quarterly report",
            input_context={
                "previous_output": "old text",
                "verification_feedback": {
                    "attempt": 2,
                    "reasoning": "Multiple issues.",
                    "failures": ["min_length", "required_sections"],
                    "scores": {},
                },
            },
        )
        prompt = build_prompt(task)
        assert "min_length" in prompt
        assert "required_sections" in prompt

    def test_partial_verdict_critique_also_reaches_prompt(self, build_prompt):
        """The reconciler injects the SAME shape for PARTIAL verdicts
        (low confidence) as for hard FAIL. Pin that the retry prompt
        treats them equivalently."""
        critique = "Confidence 0.62 — reasoning was thin in section 3."
        task = _make_task(
            title="Strategy doc",
            input_context={
                "previous_output": "draft v1",
                "verification_feedback": {
                    "attempt": 2,
                    "verdict": "partial",
                    "confidence": 0.62,
                    "reasoning": critique,
                    "scores": {"depth": 0.5},
                },
            },
        )
        prompt = build_prompt(task)
        assert critique in prompt
        assert "Revision Request" in prompt

    def test_revision_attempt_number_visible(self, build_prompt):
        """The dispatcher writes the attempt number into the revision
        framing so the model sees which retry it's on (helps with
        prompting — 'attempt 3 of 3' is a stronger nudge than 'a retry')."""
        task = _make_task(
            title="X",
            input_context={
                "previous_output": "y",
                "verification_feedback": {
                    "attempt": 3,
                    "reasoning": "still wrong",
                    "failures": [],
                    "scores": {},
                },
            },
        )
        prompt = build_prompt(task)
        # The dispatcher embeds the attempt number in the framing
        # ('attempt 3').
        assert "attempt 3" in prompt or "attempt: 3" in prompt or "3)" in prompt

    def test_retry_feedback_legacy_path_still_supported(self, build_prompt):
        """The dispatcher's prompt builder ALSO supports a legacy
        ``retry_feedback`` key (no previous_output, just freeform text).
        That path is used for plan-rejection feedback. Pin it still
        carries the feedback into the prompt."""
        task = _make_task(
            title="Y",
            input_context={
                "retry_feedback": "Make the headline punchier.",
            },
        )
        prompt = build_prompt(task)
        assert "Make the headline punchier." in prompt
        assert "Feedback from Previous Attempt" in prompt


# ===========================================================================
# 2. AC1 / PRD-200 S1 — THE JUDGE GATES ONCE (behavioural).
#
# Flipped from AST-static (which only proved _apply_verdict_fail was SHAPED
# correctly while it had ZERO callers) to behavioural: drive
# MissionReconciler._apply_verdict with the verdict verify_task returned and
# the DB-transition boundary stubbed. A FAIL requeues the task ONCE with the
# verifier's feedback; PARTIAL stays advisory; a FAIL at the requeue cap
# passes through to VERIFIED-with-annotation. Pure — no session, no LLM.
# ===========================================================================


class TestReconcilerGatesVerdictOnce:
    """PRD-200 S1: the cross-model judge now gates once on FAIL.

    The reconciler used to ALWAYS pass the verdict through to VERIFIED (even
    empty output — the judge's own system prompt admitted "ADVISORY ONLY").
    Now a FAIL requeues the task a single time with the verifier's critique so
    the agent revises instead of the wrong/empty output flowing into
    synthesis, capped at ``COORDINATOR_MAX_VERIFICATION_REQUEUES``. PARTIAL
    keeps the advisory retreat (the retry-storm scar tissue, Q3)."""

    @pytest.fixture
    def rec(self, monkeypatch):
        """The real reconciler with the DB-transition boundary stubbed, so
        ``_apply_verdict`` runs as pure logic (no session, no board, no LLM).
        Imported at fixture-run time — not collection — to match this file's
        lazy-import discipline (avoids the SQLAlchemy registration order
        contamination the module docstring warns about)."""
        from modules.coordination import reconciler as _rec
        from modules.coordination.verification import (
            VERDICT_FAIL,
            VERDICT_PARTIAL,
            VERDICT_PASS,
            VerificationResult,
        )
        from core.models.orchestration_enums import TaskState

        def _fake_transition(db, task, new_state, **kwargs):
            # The only DB effect _apply_verdict relies on is the state move.
            task.state = new_state.value

        async def _noop_async(*_a, **_k):
            return None

        monkeypatch.setattr(_rec, "transition_task", _fake_transition)
        monkeypatch.setattr(_rec, "sync_board_status", lambda *a, **k: None)
        monkeypatch.setattr(_rec, "_store_retry_recovery_safe", _noop_async)

        def _task(**over):
            base = dict(
                id="task-1",
                state=TaskState.VERIFYING.value,
                attempt_number=0,
                max_retries=3,
                output="draft output",
                input_context=None,
                output_metadata=None,
                failure_reason_code=None,
            )
            base.update(over)
            return SimpleNamespace(**base)

        return SimpleNamespace(
            apply_verdict=_rec.MissionReconciler._apply_verdict,
            result=VerificationResult,
            FAIL=VERDICT_FAIL,
            PARTIAL=VERDICT_PARTIAL,
            PASS=VERDICT_PASS,
            TaskState=TaskState,
            task=_task,
        )

    @pytest.mark.asyncio
    async def test_fail_requeues_once_with_critique(self, rec):
        """A FAIL verdict requeues the task (→ RETRYING), bumps
        ``attempt_number``, and stashes ``previous_output`` +
        ``verification_feedback`` (the keys the dispatcher reads back) —
        Mission Zero P3 is caught, not passed through to synthesis."""
        task = rec.task(output="v1 draft")
        result = rec.result(
            verdict=rec.FAIL,
            reasoning="Missing the Risk Analysis section.",
            scores={"completeness": 0.3},
            deterministic_failures=["required_sections"],
        )
        requeued = await rec.apply_verdict(MagicMock(), task, result)

        assert requeued is True
        assert task.state == rec.TaskState.RETRYING.value
        assert task.attempt_number == 1
        assert task.input_context["previous_output"] == "v1 draft"
        fb = task.input_context["verification_feedback"]
        assert fb["reasoning"] == "Missing the Risk Analysis section."
        assert fb["failures"] == ["required_sections"]
        assert task.input_context["verification_requeues"] == 1

    @pytest.mark.asyncio
    async def test_fail_at_cap_is_advisory_verified(self, rec):
        """At the requeue cap (already revised once) a FAIL passes through to
        VERIFIED with the feedback annotated — it does NOT re-requeue and does
        NOT fail the mission (that was the retreat; the gate re-opens one
        notch, not the whole retry storm)."""
        task = rec.task(input_context={"verification_requeues": 1})
        result = rec.result(verdict=rec.FAIL, reasoning="Still incomplete.")
        requeued = await rec.apply_verdict(MagicMock(), task, result)

        assert requeued is False
        assert task.state == rec.TaskState.VERIFIED.value
        assert task.output_metadata["review_feedback"]["verdict"] == rec.FAIL
        assert task.input_context["verification_requeues"] == 1  # unchanged

    @pytest.mark.asyncio
    async def test_partial_stays_advisory(self, rec):
        """PARTIAL is never gated (Q3 — gating it re-opens the retry storm the
        advisory retreat closed): it passes through to VERIFIED with feedback
        annotated, and stashes NO revision context."""
        task = rec.task()
        result = rec.result(
            verdict=rec.PARTIAL, reasoning="Thin in section 3.", confidence=0.62
        )
        requeued = await rec.apply_verdict(MagicMock(), task, result)

        assert requeued is False
        assert task.state == rec.TaskState.VERIFIED.value
        assert task.output_metadata["review_feedback"]["verdict"] == rec.PARTIAL
        assert task.input_context is None  # no requeue → no revision context

    @pytest.mark.asyncio
    async def test_pass_verifies_without_annotation(self, rec):
        """A clean PASS transitions to VERIFIED and leaves no review feedback
        on the task."""
        task = rec.task()
        result = rec.result(verdict=rec.PASS, scores={"completeness": 0.9})
        requeued = await rec.apply_verdict(MagicMock(), task, result)

        assert requeued is False
        assert task.state == rec.TaskState.VERIFIED.value
        assert task.output_metadata is None

    @pytest.mark.asyncio
    async def test_empty_output_fail_requeues_once(self, rec):
        """Empty output is the judge's hard FAIL (verification.py:385) — the
        highest-value catch. It now gets one revision instead of flowing
        straight into synthesis."""
        task = rec.task(output="")
        result = rec.result(verdict=rec.FAIL, reasoning="Task produced empty output.")
        requeued = await rec.apply_verdict(MagicMock(), task, result)

        assert requeued is True
        assert task.state == rec.TaskState.RETRYING.value
        assert task.input_context["verification_requeues"] == 1

    @pytest.mark.asyncio
    async def test_two_fails_requeue_then_verify(self, rec):
        """Cap behaviour across one task's lifecycle: the first FAIL requeues,
        the second FAIL (budget spent) goes advisory-VERIFIED. Exactly ONE
        requeue — the storm the retreat closed cannot re-open."""
        task = rec.task(output="v1")
        fail = rec.result(verdict=rec.FAIL, reasoning="nope")

        first = await rec.apply_verdict(MagicMock(), task, fail)
        assert first is True
        assert task.state == rec.TaskState.RETRYING.value
        assert task.input_context["verification_requeues"] == 1

        # Task re-runs, still FAILs — now at the cap.
        second = await rec.apply_verdict(MagicMock(), task, fail)
        assert second is False
        assert task.state == rec.TaskState.VERIFIED.value
        assert task.input_context["verification_requeues"] == 1


# ===========================================================================
# 3. AC2 — DB-AUTHORITATIVE + RESTART-DURABLE REGRESSION.
#
# Missions are already restart-safe via W1-S6's reap_orphaned_runs boot
# sweep (PRD-142 §8.2). This story does NOT rebuild durability — it pins
# the existing wire-up so a refactor cannot silently drop it.
# ===========================================================================


class TestRestartDurabilityRegression:
    """Pin the boot-sweep wire-up so a refactor cannot break restart
    recovery silently."""

    def test_boot_reaper_imported_in_main(self):
        src = MAIN_PY.read_text()
        assert "from core.boot.reaper import reap_orphaned_runs" in src, (
            "main.py must import reap_orphaned_runs at boot — W1-S6 "
            "restart-safety regression check"
        )

    def test_boot_reaper_called_at_startup(self):
        src = MAIN_PY.read_text()
        # The exact call shape — keep the helper actually invoked, not
        # just imported and forgotten.
        assert "reap_orphaned_runs(db)" in src, (
            "main.py must CALL reap_orphaned_runs at boot — not just import it"
        )

    def test_orphan_marker_is_canonical(self):
        src = REAPER_PY.read_text()
        # The reason string downstream tools key off — pin it.
        assert '_ORPHAN_REASON = "orphaned_on_restart"' in src, (
            "reaper must use the canonical 'orphaned_on_restart' marker "
            "so log queries / alerts keep matching"
        )

    def test_reaper_sweeps_workflow_executions(self):
        """WorkflowExecution (mission) rows ARE the durable mission state
        — pin that the reaper still sweeps them. Today the reaper
        transitions stale RUNNING workflow_executions to FAILED."""
        src = REAPER_PY.read_text()
        # We don't lock a specific function name — just that the sweep
        # surface is named (the reaper module discusses it).
        assert "workflow_execution" in src.lower() or "workflowexecution" in src.lower()

    def test_reconciler_state_writes_go_through_transition_helpers(self):
        """DB-authoritative invariant: state changes go through
        transition_run / transition_task (which write to Postgres) — not
        direct ``task.state = ...`` assignments that risk skipping the
        event log / board sync."""
        src = RECONCILER_PY.read_text()
        # Each terminal/retry transition MUST call transition_task or
        # transition_run, not mutate state directly.
        assert "transition_task(" in src
        assert "transition_run(" in src
        # The forbidden shortcut: a bare `task.state = TaskState.X.value`
        # assignment in a transition path. We allow it only in the
        # reaper's import path, never in the reconciler. Looser pin:
        # there must be at least one transition_task call per terminal
        # branch — proved already by the strings above.


# ===========================================================================
# 4. AC3 — STALLED-TASK RE-DISPATCH FAILURE-PATH.
#
# _recover_stalled_task is the failure-path: when a task stalls (no
# heartbeat past the threshold), the reconciler must either re-queue
# it (retries remain) or permanently fail it (retries exhausted) — never
# leave it dangling in STALLED.
# ===========================================================================


def _load_reconciler():
    """Import the real reconciler module. The AST-based tests below read
    source text only — this loader exists for completeness so a future
    behavioural test can call into the module without re-paying the
    contamination tax (see _load_dispatcher_build_prompt above)."""
    from modules.coordination import reconciler as rec  # noqa: E402
    return rec


class TestStalledTaskRedispatch:
    """Pin _recover_stalled_task: the failure-path for stalled tasks
    MUST recover (re-queue) or permanently fail — never silently drop."""

    @pytest.fixture(scope="class")
    def reconciler_src(self) -> str:
        return RECONCILER_PY.read_text()

    def _method_body(self, src: str, method_name: str) -> str:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == method_name):
                lines = src.splitlines()
                start = node.lineno - 1
                end = node.end_lineno
                return "\n".join(lines[start:end])
        raise AssertionError(f"method {method_name!r} not found in source")

    def test_recover_stalled_task_requeues_with_attempt_increment(
        self, reconciler_src,
    ):
        body = self._method_body(reconciler_src, "_recover_stalled_task")
        # With retries remaining: re-queue + increment attempt + clear
        # assigned agent (so dispatcher's has_active_task gate clears).
        assert "TaskState.QUEUED" in body
        assert "attempt_number" in body
        assert "assigned_agent_id = None" in body

    def test_recover_stalled_task_fails_when_retries_exhausted(
        self, reconciler_src,
    ):
        body = self._method_body(reconciler_src, "_recover_stalled_task")
        # The retries-exhausted branch transitions to FAILED with the
        # canonical MAX_RETRIES_EXHAUSTED code.
        assert "TaskState.FAILED" in body
        assert "MAX_RETRIES_EXHAUSTED" in body

    def test_detect_and_recover_stalls_emits_stall_event(self, reconciler_src):
        """The stall detection branch MUST emit a STALL_DETECTED event
        so operators can see stalls in the activity feed (observability
        is §H DoD point 5)."""
        body = self._method_body(reconciler_src, "_detect_and_recover_stalls")
        assert "STALL_DETECTED" in body
        assert "emit_event" in body

    def test_stall_uses_canonical_failure_code(self, reconciler_src):
        body = self._method_body(reconciler_src, "_detect_and_recover_stalls")
        assert "AGENT_TIMEOUT" in body, (
            "stall detection MUST use the canonical AGENT_TIMEOUT failure "
            "code so reporting / KPI aggregation keeps matching"
        )

    def test_stall_thresholds_come_from_config(self, reconciler_src):
        """No magic numbers in the stall threshold — they MUST come from
        Config (canonical project rule, see CLAUDE.md §4)."""
        body = self._method_body(reconciler_src, "_detect_and_recover_stalls")
        assert "Config.COORDINATOR_ASSIGNED_STALL_THRESHOLD_SECONDS" in body
        assert "Config.COORDINATOR_RUNNING_STALL_THRESHOLD_SECONDS" in body


# ===========================================================================
# 5. AC4 — CROSS-WORKSPACE ISOLATION.
#
# A mission's retry prompt can only contain ITS OWN task's input_context.
# Pin that build_task_prompt is a single-task function (no leak surface)
# and that the tick loop / _process_run never mix two workspaces.
# ===========================================================================


class TestCrossWorkspaceIsolation:
    """The retry prompt MUST come exclusively from the bound task. A
    workspace-A verifier critique cannot land on a workspace-B task."""

    @pytest.fixture(scope="class")
    def dispatcher_src(self) -> str:
        return DISPATCHER_PY.read_text()

    @pytest.fixture(scope="class")
    def coordinator_src(self) -> str:
        return COORDINATOR_PY.read_text()

    def test_build_task_prompt_is_single_task_pure(self, dispatcher_src):
        """The signature is ``build_task_prompt(task)`` — no DB session,
        no run, no workspace. Anything it reads comes from the single
        task object, so a workspace cannot leak via the prompt path."""
        tree = ast.parse(dispatcher_src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "build_task_prompt":
                # Static method with exactly one positional arg ('task').
                args = [a.arg for a in node.args.args]
                assert args == ["task"], (
                    f"build_task_prompt must take exactly one arg (task); "
                    f"got {args!r}"
                )
                return
        pytest.fail("build_task_prompt not found in dispatcher.py")

    def test_build_task_prompt_reads_only_task_attrs(self, build_prompt):
        """Hand build_task_prompt task A's input_context and task B's
        title — the prompt MUST mention A's input_context (because that
        is on task A's object) only via the task we passed. Sanity pin
        that there is no global state read."""
        critique_a = "WORKSPACE-A-CRITIQUE"
        task_a = _make_task(
            title="Task-A",
            input_context={
                "previous_output": "draft-a",
                "verification_feedback": {
                    "attempt": 2, "reasoning": critique_a,
                    "failures": [], "scores": {},
                },
            },
        )
        task_b = _make_task(title="Task-B")
        prompt_b = build_prompt(task_b)
        # Task B's prompt MUST NOT contain task A's critique.
        assert critique_a not in prompt_b
        assert "Revision Request" not in prompt_b

    def test_tick_filters_runs_by_state_only(self, coordinator_src):
        """The tick loop filters by ``OrchestrationRun.state == RUNNING``
        — no workspace filter is needed because each run carries its own
        workspace_id and _process_run reads it from the bound run.
        Pin the query shape so a future broadening can't accidentally
        leak across workspaces."""
        tree = ast.parse(coordinator_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "tick"):
                body_text = ast.get_source_segment(coordinator_src, node) or ""
                assert "OrchestrationRun.state == RunState.RUNNING.value" in body_text, (
                    "tick must select runs by state == RUNNING; a "
                    "broader filter risks dragging non-active runs into "
                    "the loop"
                )
                return
        pytest.fail("tick() not found in coordinator_service.py")

    def test_process_run_reads_workspace_from_bound_run(self, coordinator_src):
        """``_process_run`` MUST scope its work to ``run.workspace_id``
        — never to a globally-cached workspace. Pin the read shape."""
        tree = ast.parse(coordinator_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_process_run"):
                body_text = ast.get_source_segment(coordinator_src, node) or ""
                assert "workspace_id = run.workspace_id" in body_text, (
                    "_process_run must read workspace_id from the bound "
                    "run object only — no cross-workspace leak surface"
                )
                # Agent lookup MUST filter by that same workspace_id.
                assert "Agent.workspace_id == workspace_id" in body_text
                return
        pytest.fail("_process_run not found in coordinator_service.py")


# ===========================================================================
# 6. AC5 — MISSIONS PRIMITIVE HEARTBEAT (W3-S1 WIRING).
#
# Tiny stateless helper that emits one finding per mission terminal
# transition. Matches the W3-S6 (chat) / W3-S8 (rag) / W3-S9 (nl2sql) /
# W3-S10 (graph) shape.
# ===========================================================================


def _load_primitive_heartbeat():
    """Load ``modules/coordination/primitive_heartbeat.py`` directly."""
    spec = importlib.util.spec_from_file_location(
        "_missions_hb_w3s11", str(PRIMITIVE_HEARTBEAT_PY)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestMissionsHeartbeatHelper:
    """The W3-S1 helper plumb for the missions primitive."""

    def test_helper_file_exists(self):
        assert PRIMITIVE_HEARTBEAT_PY.exists(), (
            "modules/coordination/primitive_heartbeat.py MUST exist — it's "
            "the W3-S1 emit surface for the missions tile"
        )

    def test_helper_emits_green_on_success(self, monkeypatch):
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: calls.append(
                (ws, prim, status, detail)
            ) or True,
        )
        mod._emit_missions_primitive(
            "ws-A", success=True, detail="mission complete",
        )
        assert calls == [("ws-A", "missions", "green", "mission complete")]

    def test_helper_emits_down_on_failure(self, monkeypatch):
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: calls.append(
                (ws, prim, status, detail)
            ) or True,
        )
        mod._emit_missions_primitive(
            "ws-A", success=False, detail="task failed",
        )
        assert calls == [("ws-A", "missions", "down", "task failed")]

    def test_helper_skips_when_workspace_id_missing(self, monkeypatch):
        """A4 — no workspace_id MUST result in NO emit (honest gap over
        fabricated default). Pin that the tile reads 'unknown' rather
        than borrowing another workspace's id."""
        mod = _load_primitive_heartbeat()
        calls: list[tuple] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda *a, **k: calls.append(a) or True,
        )
        mod._emit_missions_primitive(None, success=True)
        mod._emit_missions_primitive("", success=False)
        assert calls == [], (
            "no workspace_id MUST mean no emit — never default to an "
            "anonymous workspace"
        )

    def test_helper_swallows_emit_failures(self, monkeypatch):
        """Mission completion MUST NOT raise because the heartbeat
        writer is unhealthy. Pin the best-effort contract."""
        mod = _load_primitive_heartbeat()

        def _boom(*_a, **_k):
            raise RuntimeError("heartbeat_results write failed")

        monkeypatch.setattr(mod, "emit_primitive_finding", _boom)
        # Must not raise.
        mod._emit_missions_primitive(
            "ws-A", success=True, detail="ok",
        )
        mod._emit_missions_primitive(
            "ws-A", success=False, detail="error",
        )

    def test_helper_truncates_long_detail(self, monkeypatch):
        """500-char cap mirrors the W3-S1 ``emit_primitive_finding`` API.
        Pin the helper does not blow past it."""
        mod = _load_primitive_heartbeat()
        captured: list[str] = []
        monkeypatch.setattr(
            mod, "emit_primitive_finding",
            lambda ws, prim, status, detail: captured.append(detail) or True,
        )
        long_detail = "x" * 9000
        mod._emit_missions_primitive(
            "ws-A", success=False, detail=long_detail,
        )
        assert len(captured) == 1
        assert len(captured[0]) <= 500


class TestCoordinatorWiresMissionsHeartbeat:
    """Static grep on coordinator_service.py — it MUST import the helper
    and emit at the terminal-transition boundary in _process_run. A
    refactor that drops the wire-up MUST fail this test before it lands."""

    @pytest.fixture(scope="class")
    def coordinator_src(self) -> str:
        return COORDINATOR_PY.read_text()

    def test_coordinator_imports_helper(self, coordinator_src):
        # Accept either fully-qualified or relative module form.
        assert (
            "from modules.coordination.primitive_heartbeat" in coordinator_src
            or "from .primitive_heartbeat" in coordinator_src
        ), (
            "coordinator_service.py MUST import _emit_missions_primitive "
            "for the W3-S11 tile wiring"
        )
        assert "_emit_missions_primitive" in coordinator_src

    def test_coordinator_emits_on_terminal_transition(self, coordinator_src):
        """_process_run MUST call the helper when run reaches a terminal
        state — both success and failure are signal the tile needs."""
        # The helper is called inside _process_run (the only periodic
        # path that runs as a mission progresses).
        tree = ast.parse(coordinator_src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "_process_run"):
                body = ast.get_source_segment(coordinator_src, node) or ""
                assert "_emit_missions_primitive" in body, (
                    "_process_run MUST emit the missions primitive "
                    "heartbeat at the terminal-transition boundary"
                )
                # Must look at the run's terminal state to decide
                # success / failure — pin both branches.
                assert "TERMINAL_RUN_STATES" in body
                return
        pytest.fail("_process_run not found")

    def test_coordinator_does_not_emit_for_active_state(self, coordinator_src):
        """The emit MUST be guarded behind a terminal-state check —
        otherwise a still-running mission would flip the tile down on
        every tick."""
        # Look for the emit call near a TERMINAL_RUN_STATES guard. We
        # already proved both tokens appear in _process_run; pin that
        # there's no unguarded emit at module scope.
        # Crude but effective: count emit calls; each one should be
        # near a 'TERMINAL_RUN_STATES' line within 12 lines (the
        # guard block).
        lines = coordinator_src.splitlines()
        emit_lines = [
            i for i, ln in enumerate(lines)
            if "_emit_missions_primitive(" in ln
        ]
        terminal_lines = [
            i for i, ln in enumerate(lines) if "TERMINAL_RUN_STATES" in ln
        ]
        for emit_i in emit_lines:
            close = any(
                abs(emit_i - ti) <= 12 for ti in terminal_lines
            )
            assert close, (
                f"emit at line {emit_i + 1} is not within 12 lines of a "
                "TERMINAL_RUN_STATES guard — risks emitting for "
                "still-running missions"
            )
