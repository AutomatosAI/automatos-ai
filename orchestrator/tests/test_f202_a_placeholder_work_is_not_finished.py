"""F202 (night 6) — quality gates stop passing empty or wrong work.

- At 06:09 Auto filed "First Week's Activity Summary for Priya and Hana" to
  Reports, and the bell announced it. It read "Analysis … revealed [Insert
  insights from Task 1148 here]".
- At 03:56 a watch passed "New Cafe Onboarding" run exec-d52d3c5d3200 at 1.00:
  a welcome email to the wrong café. The run had no inputs, and the judge never
  saw what the run was for.

submit_report now refuses a template's placeholders. A watch never passes an
output that still holds one, judge or no judge. The judge sees the run's inputs.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

SUMMARY = ("## First Week's Activity Summary\n\nPriya and Hana finished the café onboarding. Analysis of the week's "
           "orders revealed [Insert insights from Task 1148 here].\n\nOne question is still open.")
CHECKLIST = ("**Monday Morning Dispatch Checklist: Subscription Orders**\n**For:** Tom\n"
             "**Date:** [Current Monday's Date]\n\n1. Verify Roasts\n2. Gather Materials")


@pytest.fixture
def filed(monkeypatch):
    import services.report_service as report_service

    made = []

    class _Reports:
        def __init__(self, db, workspace_id):
            pass

        async def create_report(self, **kwargs):
            made.append(kwargs["title"])
            return {"id": "r-1"}

    monkeypatch.setattr(report_service, "ReportService", _Reports)
    return made


def _submit(title, content):
    from modules.tools.discovery.handlers_reports import submit_report

    return asyncio.run(submit_report(None, uuid4(), {"title": title, "content": content, "report_type": "summary",
                                                     "_agent_id": 322, "_agent_name": "Auto"}))


def test_a_report_with_a_placeholder_is_refused(filed):
    result = _submit("First Week's Activity Summary for Priya and Hana", SUMMARY)

    assert result["success"] is False and filed == []
    assert result["error"].startswith("The report still has placeholders where its content belongs: "
                                      "[Insert insights from Task 1148 here].")


def test_a_printable_forms_blank_is_not_a_placeholder(filed):
    """Night 6's dispatch checklist leaves the date for Tom to write in."""
    _submit("Tom's Monday Dispatch Checklist", CHECKLIST)
    assert filed == ["Tom's Monday Dispatch Checklist"]


# ── the watch ──────────────────────────────────────────────────────────────

def _bundle(text, inputs=None):
    from modules.coordination.run_verdict import RunOutputBundle

    kwargs = {"inputs": inputs} if inputs is not None else {}
    return RunOutputBundle(text=text, kind="playbook_execution", terminal_state="completed",
                           mechanics_reliability=1.0, **kwargs)


def test_a_run_whose_output_still_has_a_placeholder_never_passes(monkeypatch):
    from modules.coordination import run_verdict
    from services import watch_decider

    monkeypatch.setattr(run_verdict.RunVerdictService, "collect_run_output",
                        staticmethod(lambda db, watch: _bundle(f"### Final output\n{SUMMARY}")))

    class _JudgeDown:
        async def score_run(self, db, watch):
            return None

        def apply_verdict(self, db, watch, verdict):
            pass

    closed = []

    async def _close(self, db, watch, *, passed, terminal_state, explanation):
        closed.append((passed, explanation))
        return "closed"

    monkeypatch.setattr(watch_decider.WatchDecider, "_close", _close)
    watch = NS(id="c57cb0bd", status="watching", policy="run_and_report", target_type="playbook_execution",
               target_id="exec-1", quality_threshold=0.8, actions_taken=0, action_budget=0, final_verdict=None)
    asyncio.run(watch_decider.WatchDecider(verdict_service=_JudgeDown()).decide_terminal(None, watch, "completed", None))

    ((passed, explanation),) = closed
    assert passed is False
    assert "The output still has placeholders where its content belongs ([Insert insights from Task 1148 here])" \
        in explanation


def test_the_judge_is_not_paid_to_score_a_placeholder():
    from modules.coordination.run_verdict import RunVerdictService

    def no_judge(*a, **k):
        raise AssertionError("a placeholder fails before any judge runs")

    verdict = asyncio.run(RunVerdictService().score_run(None, NS(id="w-1"), bundle=_bundle(SUMMARY),
                                                        llm_factory=no_judge))
    assert verdict.score == 0.0 and not verdict.passes(0.8)


class _Runs:
    def __init__(self, execution):
        self.execution = execution

    def query(self, *a):
        return self

    def filter(self, *a):
        return self

    def first(self):
        return self.execution


@pytest.mark.parametrize("inputs, shown", [({"cafe_name": "Gull & Anchor", "contact": "Maya"}, "Gull & Anchor"),
                                          ({}, "none given")], ids=["with-inputs", "night-6-none"])
def test_the_judge_grades_against_the_runs_inputs(inputs, shown):
    from modules.coordination.run_verdict import RunVerdictService, build_run_judge_prompt

    execution = NS(execution_id="exec-d52d3c5d3200", workspace_id=uuid4(), status="completed", error_message=None,
                   output_data={"final_output": "Welcome to Harbourline, Lamplight Café!"}, step_results=[],
                   input_data=inputs)
    bundle = RunVerdictService._collect_playbook(_Runs(execution), NS(target_id="exec-d52d3c5d3200"))
    prompt = build_run_judge_prompt(
        success_criteria="Playbook 'New Cafe Onboarding' completes and delivers its expected output.", bundle=bundle)

    assert "## The run's inputs" in prompt and shown in prompt
    assert "work about a different customer, place, product or date than they name is not complete" in prompt
