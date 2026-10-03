"""F248 (night 7): a mission must not pass results that are wrong.

Step #0126.3 was marked "verified" with "[Number]" and "[Your Name/Company Name]"
in it. Mission verification never looked for a template's slots (only submit_report
and the Watch verdict did, F202), and a FAIL after the step's one revision passed
through to VERIFIED anyway. Now a step's output with slots left in fails its checks
first, whatever its criteria. If the slots are still there after its revision, the
step fails, naming them.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState

WINBACK = ("Hi there,\n\nWe noticed [Number] of our coffee club members paused last month, so we're offering "
           "a free bag on their next order.\n\nWarm regards,\n[Your Name/Company Name]")
CHECKLIST = "**Date:** [Current Monday's Date]\n\n1. Verify roasts\n2. See [the order sheet](https://example.com)"


def test_name_and_figure_slots_are_placeholders_and_a_dated_blank_is_not():
    from core.services.placeholders import template_placeholders

    assert template_placeholders(WINBACK) == ["[Number]", "[Your Name/Company Name]"]
    assert template_placeholders("Dear [Member Name], your total is [Amount].") == ["[Member Name]", "[Amount]"]
    assert template_placeholders(CHECKLIST) == []      # F202's printable checklist leaves the date to Tom


def test_a_step_with_slots_left_in_fails_its_checks_whatever_its_criteria():
    from core.services.placeholders import UNFINISHED
    from modules.coordination.deterministic_checks import DeterministicChecker

    checker = DeterministicChecker()
    for criteria in (None, [{"type": "min_length", "value": 10, "must_pass": False}]):
        result = checker.check(WINBACK, criteria)
        assert (result.passed, result.short_circuited) == (False, True)
        assert result.failures[0].description == f"{UNFINISHED}[Number], [Your Name/Company Name]."
    assert checker.check(CHECKLIST, None).passed is True


@pytest.fixture
def verifying(db_session, seed_workspace):
    """#0126.3's shape: the mission's last step, back from its one revision, being verified."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Churn analysis and a win-back note", state=RunState.RUNNING.value,
                           created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()

    def task(requeues):
        step = OrchestrationTask(run_id=run.id, title="Churn analysis and win-back note for approval",
                                 description="Write it.", sequence_number=3, state=TaskState.VERIFYING.value,
                                 state_type="active", output=WINBACK, input_context={"verification_requeues": requeues})
        db_session.add(step)
        db_session.flush()
        return step
    return NS(db=db_session, task=task)


def _verdict(verifying, task, failures):
    from modules.coordination.reconciler import MissionReconciler
    from modules.coordination.verification import VERDICT_FAIL, VerificationResult

    result = VerificationResult(verdict=VERDICT_FAIL, reasoning="Deterministic must_pass check failed",
                                deterministic_passed=False, deterministic_failures=failures)
    return asyncio.run(MissionReconciler._apply_verdict(verifying.db, task, result))


def test_slots_still_in_after_the_revision_fail_the_step_and_name_them(verifying):
    from core.services.placeholders import UNFINISHED

    step = verifying.task(requeues=1)                # its one revision is spent
    failure = f"{UNFINISHED}[Number], [Your Name/Company Name]."

    assert _verdict(verifying, step, [failure]) is True
    assert step.state == TaskState.FAILED.value and step.failure_detail == failure


def test_the_first_time_it_is_revised_as_before(verifying):
    from core.services.placeholders import UNFINISHED

    step = verifying.task(requeues=0)
    assert _verdict(verifying, step, [f"{UNFINISHED}[Number]."]) is True
    assert step.state == TaskState.RETRYING.value


def test_any_other_fail_after_the_revision_stays_advisory(verifying):
    step = verifying.task(requeues=1)
    assert _verdict(verifying, step, ["Output is 12 words; at least 50 required"]) is False
    assert step.state == TaskState.VERIFIED.value
