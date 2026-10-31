"""P256-FIX-SOC-5 (F437): the status machine's new moves, each kept to its own caller.

``render_failed`` now also moves a draft, so the platform's refusal of a plan's render is a
failure every reader shows; the table alone would let any caller fail a draft, so each function
checks its own status (pure, as ``tests/test_prd251_service.py``, whose pinned tables carry the new
moves; this file holds the new cases so that file does not grow past its size):

* ``fail_render`` still ends only a render that started: a draft raises and is unchanged;
* ``refuse_render`` fails only a draft: a rendering post raises and keeps its history;
* a render's entry says when it gives up, from the clock that stamps it; a caller that gives no
  limit (the crops' render) records none;
* the planned-slot sweep's ``slot_passed`` ends a failed post missed.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import modules.socials.service as service  # noqa: E402
from modules.socials.service import DRAFT, FAILED, MISSED, RENDERING, IllegalTransition  # noqa: E402
from tests.test_prd251_service import AUTHOR, NOW, _post_in  # noqa: E402

REFUSED = "The renderer cannot be reached right now."


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch):
    monkeypatch.setattr(service, "_utcnow", lambda: NOW)


def test_fail_render_ends_only_a_started_render_and_refuse_render_only_a_draft():
    draft = _post_in(DRAFT)
    with pytest.raises(IllegalTransition):
        service.fail_render(draft, AUTHOR, "The check failed.")
    assert draft.status == DRAFT and draft.review_log == []
    rendering = _post_in(RENDERING)
    log_before = list(rendering.review_log)
    with pytest.raises(IllegalTransition):
        service.refuse_render(rendering, AUTHOR, REFUSED)
    assert rendering.status == RENDERING and rendering.review_log == log_before
    refused = service.refuse_render(_post_in(DRAFT), AUTHOR, f"  {REFUSED}  ")
    assert refused.status == FAILED
    assert refused.review_log[-1] == {"at": NOW.isoformat(), "by": AUTHOR, "action": "render_failed", "comment": REFUSED}


def test_a_render_says_when_it_gives_up_from_the_same_clock():
    post = service.start_render(_post_in(DRAFT), AUTHOR, gives_up_after=1500)
    assert post.review_log[-1] == {"at": NOW.isoformat(), "by": AUTHOR, "action": "render", "comment": None,
                                   "gives_up_at": (NOW + timedelta(seconds=1500)).isoformat()}
    again = service.start_render(service.fail_render(post, AUTHOR, "The check failed."), AUTHOR)
    assert "gives_up_at" not in again.review_log[-1]


def test_the_slot_sweep_ends_a_failed_post_missed():
    post = service.refuse_render(_post_in(DRAFT), AUTHOR, REFUSED)
    service.set_planned_for(post, NOW - timedelta(days=1), "UTC")
    service.slot_passed(post, "scheduler", "Its planned slot passed with no approval.")
    assert post.status == MISSED and post.review_log[-1]["action"] == service.ACTION_SLOT_PASSED  # which posts: test_f437_b
