"""F248 (night 7): a mission must not pass results that are wrong.

Step #0126.3 was marked "verified" with "[Number]" and "[Your Name/Company Name]"
in it. Mission verification never looked for a template's slots (only submit_report
and the Watch verdict did, F202), and a FAIL after the step's one revision passed
through to VERIFIED anyway. Now a step's output with slots left in fails its checks
first, whatever its criteria. If the slots are still there after its revision, the
step fails, naming them.

The slots were there because the summary never had the figures: the results of the
steps it built on reached it trimmed to 1,200 tokens. The mission's last step now gets
them whole. And #0139.1 listed 24 of 27 cafés because its agent could not find the
sheet the goal named; a step is now told the document id of a file it names.
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


# ── the summary is built from the steps' results; a step is told the sheet it names ──

SHEET = "harbourline-wholesale-invoices-2026-09-26.csv"
OWED = "\n".join(f"| HL-23{n:02d} | Cafe {n} | contact{n}@example.com | £{100 + n}.40 | due 2026-08-{n % 28 + 1:02d} |"
                 for n in range(1, 28))                      # 27 cafés, as the owner's sheet has
LETTER = ("Hi all,\n\nFrom 2 November our wholesale prices change: " + "House Espresso £24.50 a kilo, " * 123
          + "and delivery stays as it is.\n\nThanks, Gerard")    # with OWED, over the digest's 1,200 tokens


@pytest.fixture
def mission(db_session, seed_workspace):
    """#0139's shape: a list from the invoice sheet, a letter, and a note built on both."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask, OrchestrationTaskDependency

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal=f"Get my wholesale cafes ready for the price change, from {SHEET}",
                           state=RunState.RUNNING.value, created_by="user_test", config={})
    db_session.add(run)
    db_session.flush()

    def step(n, title, output=None):
        task = OrchestrationTask(run_id=run.id, title=title, description="Do it.", sequence_number=n,
                                 state=TaskState.VERIFIED.value if output else TaskState.RUNNING.value,
                                 state_type="active", output=output, input_context={})
        db_session.add(task)
        db_session.flush()
        return task

    owed, letter = step(1, "List every cafe and what it owes", OWED), step(2, "Write the price-change letter", LETTER)
    note = step(3, "A one-page note: who gets the letter, and who to chase first")
    db_session.add_all([OrchestrationTaskDependency(task_id=note.id, depends_on_task_id=owed.id),
                        OrchestrationTaskDependency(task_id=note.id, depends_on_task_id=letter.id)])
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, owed=owed, letter=letter, note=note)


def _dispatched(mission, task):
    """What _prepare_task does for a step: the digest is attached, then its prompt is built."""
    from modules.coordination.dispatcher import MissionDispatcher
    from services.coordinator_service import CoordinatorService

    svc = CoordinatorService.__new__(CoordinatorService)
    svc._field, svc._get_field = None, lambda: None
    rows = CoordinatorService._collect_upstream_digest_rows(mission.db, task)
    asyncio.run(svc._attach_field_digest(mission.db, mission.run, task, None, 1, upstream_rows=rows))
    return MissionDispatcher.build_task_prompt(task, goal=mission.run.goal)


def test_the_last_step_is_given_the_results_it_builds_on_whole(mission):
    from modules.coordination.step_inputs import RESULTS_HEADING

    prompt = _dispatched(mission, mission.note)

    assert RESULTS_HEADING in prompt and "Take every figure, name and date from them exactly" in prompt
    assert OWED in prompt and LETTER in prompt      # the 1,200-token digest kept only the first of them
    assert "field_digest" not in mission.note.input_context


def test_a_step_another_builds_on_keeps_the_budgeted_digest(mission):
    from core.models.orchestration import OrchestrationTaskDependency
    from modules.coordination.step_inputs import RESULTS_HEADING

    mission.db.add(OrchestrationTaskDependency(task_id=mission.letter.id, depends_on_task_id=mission.owed.id))
    mission.db.flush()
    prompt = _dispatched(mission, mission.letter)

    assert RESULTS_HEADING not in prompt
    assert "Cafe 1 " in mission.letter.input_context["field_digest"]


def test_a_step_is_told_the_document_its_mission_names(mission):
    from core.models.core import Document
    from modules.coordination.dispatcher import MissionDispatcher

    sheet = Document(filename=SHEET, original_filename=SHEET, workspace_id=mission.ws, file_type="text/csv",
                     file_size=3209, status="processed")
    mission.db.add(sheet)
    mission.db.flush()

    prompt = MissionDispatcher.build_task_prompt(mission.owed, goal=mission.run.goal)
    assert f"- {SHEET}: document_id {sheet.id}" in prompt
    assert "Read each one whole with platform_read_document" in prompt

    mission.owed.description = "Save the list as drafts/owed.md and as owed.md."    # files it writes, not reads
    assert "## Documents this work names" not in MissionDispatcher.build_task_prompt(mission.owed)
