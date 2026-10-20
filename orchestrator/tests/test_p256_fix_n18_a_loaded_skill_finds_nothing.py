"""#1211 follow-up (night 18): a loaded skill is no read of what the reply reports.

P256-FIX-N17-5 (F446) lets a third-person shape stand after a turn that only read: "It's now running."
says what the read found. It counted every done read, platform_load_skill too, so "Your post is now live
on LinkedIn." after a loaded skill alone drew no not-done line, though a skill's instructions observe
nothing of the workspace or of LinkedIn. Now a read that observes nothing (``claim_removed._OBSERVES_NOTHING``)
leaves the turn checked as one that read nothing; a read that looked still lets the finding stand.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claim_removed import without_unbacked_claims
from tests import test_prd256_fix_honesty_per_claim as per_claim

SKILL_LOADED, TASKS_LISTED, LINE = per_claim.SKILL_LOADED, per_claim.TASKS_LISTED, per_claim.NOTHING_DONE_LINE
LIVE = "Your post is now live on LinkedIn."
CARD_READ = per_claim._platform("platform_get_task", {"task_id": "#1836"})


@pytest.mark.parametrize("answer", [LIVE, "The new page is now live."])
def test_after_a_loaded_skill_alone_a_world_state_claim_draws_the_line(answer):
    assert per_claim._said(answer, SKILL_LOADED) == [LINE]


@pytest.mark.parametrize("looked", [TASKS_LISTED, CARD_READ])
def test_after_a_read_that_looked_the_finding_still_stands(looked):
    """N17-5 as it was: the read found it."""
    assert per_claim._said(LIVE, looked) == []


@pytest.mark.parametrize("looked", [TASKS_LISTED, CARD_READ])
def test_a_loaded_skill_beside_a_read_that_looked_changes_nothing(looked):
    assert per_claim._said(LIVE, SKILL_LOADED, looked) == []
    assert per_claim._said(LIVE, looked, SKILL_LOADED) == []


def test_a_search_before_the_loop_still_counts_as_a_read():
    """``read_ahead``: retrieval ran before the loop, so the turn looked at something."""
    skill = ("platform_load_skill", {"skill_name": "linkedin-posts"}, {"success": True})   # as the tracker records it
    assert without_unbacked_claims(LIVE, [skill], True, read_ahead=True) is None
    removed = without_unbacked_claims(LIVE, [skill], True, read_ahead=False)
    assert removed is not None and LINE in removed.text


def test_auto_s_own_report_is_caught_after_a_loaded_skill_as_after_any_read():
    """N17-5 unchanged for Auto's own report ("Done.", "Email sent."): caught after a read, so after a skill too."""
    for looked in (SKILL_LOADED, TASKS_LISTED):
        assert per_claim._said("Done.", looked) == [LINE]
