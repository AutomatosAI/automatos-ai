"""F123 — a run out of budget after finished work fails honestly.

Run 4's execution 120: step 1, a session step, completed after 51 minutes. The
check before step 2 then found the 40-minute total budget spent and failed the
run "at step 2". Board card #760 went to failed with an empty result: the owner
saw a failure and nothing of the finished work. A budget stop after a completed
step still fails the run, but its error_message says what happened, and the
card goes to review with every completed step named and the last one's output.
The dollar ceiling stops the same way. With no completed step, today's failure
stands. The card's cap names its cut, on the success path too.

These run the real step loop and the real board bridge, faking only the edges:
the session, the step's agent call, the clock, the log upload, notifications
(tests/helpers_playbook_run.py).
"""
from __future__ import annotations

import pytest

from api import recipe_executor as rex
from tests.helpers_playbook_run import LOG, TOKENS, done, run_playbook

OUTPUT = "I've redone the list and Callum's Monday sheet from your export: 14 boxes, 3 changes."
SESSION_STEP_S = 3082      # execution 120's step 1: 51 min 22 s
BUDGET_S = 2400            # the 40-minute total budget
CUT = "[Cut at 4,000 characters. The full output: {}]"   # what the owner reads at the cap


def _spent():
    usd = rex._tokens_to_usd(TOKENS, None)
    assert usd > 0, "the flat price must make step 1 cost something"
    return usd


def _stop(kind):
    """(step seconds, execution_config, the reason the run must give)."""
    finished = "Step 2 never started. Step 1 completed; its output is below."
    if kind == "time":
        return SESSION_STEP_S, {"total_timeout": BUDGET_S}, f"Out of time after step 1 of 2 (51 min; budget 40 min). {finished}"
    ceiling = _spent() * 1.5           # step 1 fits; step 1's cost again would not
    reason = f"Budget ceiling ${ceiling:.2f} reached after step 1 of 2 (${_spent():.4f} spent). {finished}"
    return 5, {"total_timeout": BUDGET_S, "cost_ceiling": ceiling}, reason


@pytest.mark.parametrize("kind", ["time", "ceiling"])
def test_a_budget_stop_after_a_finished_step_fails_honestly_with_the_work_on_the_card(monkeypatch, kind):
    step_seconds, exec_config, reason = _stop(kind)
    execution, card = run_playbook(monkeypatch, outcomes=[done(OUTPUT)], step_seconds=step_seconds, exec_config=exec_config)

    assert execution.status == "failed"
    assert execution.error_message.splitlines()[0] == reason
    assert execution.step_results[0]["status"] == "completed"
    assert card.status == "review" and card.review_feedback is None   # the reviewer's channel stays theirs
    assert card.result.splitlines()[0] == reason
    assert f"Step 1 (CLUB SECRETARY): completed in {'51 min' if kind == 'time' else '5 s'}, {TOKENS:,} tokens" in card.result
    assert LOG.format(1) in card.result and OUTPUT in card.result


def test_with_no_finished_step_the_failure_is_unchanged(monkeypatch):
    execution, card = run_playbook(monkeypatch, outcomes=[{"status": "error", "error": "the sheet would not open"}],
                           step_seconds=SESSION_STEP_S, exec_config={"total_timeout": BUDGET_S})

    message = f"Total execution timeout ({float(BUDGET_S)}s) exceeded after {SESSION_STEP_S}s at step 2"
    assert execution.status == "failed" and execution.error_message == message
    assert execution.step_results[0]["status"] == "failed"
    assert card.status == "failed" and card.error_message == message
    assert card.result is None and card.review_feedback is None


def test_a_long_output_on_the_stopped_card_says_where_it_was_cut(monkeypatch):
    long_output = "Row " * 2000
    execution, card = run_playbook(monkeypatch, outcomes=[done(long_output)], step_seconds=SESSION_STEP_S,
                           exec_config={"total_timeout": BUDGET_S})

    assert card.status == "review" and long_output not in card.result
    assert card.result.endswith(CUT.format(LOG.format(1)))


def test_the_success_path_names_its_cut_too(monkeypatch):
    long_output = "Box " * 2000
    execution, card = run_playbook(monkeypatch, outcomes=[done(OUTPUT), done(long_output)], step_seconds=5,
                           exec_config={"total_timeout": BUDGET_S})

    assert execution.status == "completed" and card.status == "done"
    assert card.result.startswith("Box Box")
    assert card.result.endswith(CUT.format(LOG.format(2)))
