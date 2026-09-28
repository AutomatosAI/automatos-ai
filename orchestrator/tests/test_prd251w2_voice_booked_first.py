"""P251W1-RVW-5 — a voice's spend is booked before its first line is spoken.

Wave 1 booked a voice render in ``_speak_lines``'s ``finally``, after every line
was spoken: a process that died in between (a redeploy's SIGKILL, an OOM, a
crash) left what the toolkit charged with no booking. The script's priced
amount (P251W1-RVW-2) is now booked before its first line
(``modules/socials/media_ledger``) and settled in place when the script ends. On
the Wave 1 voice harness (``test_prd251w1_voice.py``) with the real usage
tracker and the real ``llm_usage`` (its ``ledger`` fixture). Pins:

* (c) Fish Audio speaks the first line, then the process dies: llm_usage holds
  the script's priced amount against the post;
* the priced amount is in llm_usage before the first line is spoken, and the
  ended script settles that same row to Fish Audio's balance difference: one
  amount, counted once by the caps and by the budget gate's spend-to-date;
* a booking that cannot be written speaks nothing.

The process's death is ``tests/helpers_socials_process_death.py``'s stand-in:
nothing after it reaches the database, as with a real SIGKILL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_voice as voice_harness  # noqa: E402
from tests.test_prd251w1_voice import (  # noqa: E402  (the voice harness's pieces)
    BALANCE, FISH_PER_BYTE, SPEAK, TOKEN, WS, FakeStore, Renderer, _fish_bytes, _fish_post, _post, _render,
)
from tests.helpers_socials_process_death import mortal, paid_media_now, render_until_it_dies  # noqa: E402

import modules.socials.media_caps as media_caps  # noqa: E402
import modules.socials.media_ledger as media_ledger  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.policy.budget import spend_to_date  # noqa: E402

# The voice harness's fixtures, bound by assignment so pytest finds them here and
# ruff does not read them as unused imports. ``ledger`` runs the real usage tracker.
env = voice_harness.env
ledger = voice_harness.ledger

# Fish Audio's balance reads 10.00, then 9.99 (_script_fish): the script cost $0.01.
BALANCE_DIFFERENCE = 0.01


def _priced():
    """The two-line script's price (P251W1-RVW-2): its UTF-8 bytes at Fish Audio's configured price."""
    return _fish_bytes() * FISH_PER_BYTE


def test_a_voice_is_booked_at_its_price_before_the_process_can_die_after_its_first_line(ledger, monkeypatch):
    """(c) Fish Audio speaks the first line, then the process dies: the script's
    priced amount is in llm_usage against the post."""
    post = _fish_post(ledger)
    process = mortal(ledger, monkeypatch)
    first_line = ledger.composio.answers[SPEAK]
    ledger.composio.answers[SPEAK] = [first_line, process.die]

    render_until_it_dies(ledger, post["id"], process, renderer=Renderer(FakeStore()), store=FakeStore(), token=TOKEN)

    assert ledger.composio.slugs()[:3] == [BALANCE, SPEAK, SPEAK]
    assert paid_media_now(ledger, post["id"]) == [
        {"provider": "fish_audio", "usd": pytest.approx(_priced()), "units": _fish_bytes(), "status": "pending"},
    ]
    assert _post(ledger, post["id"]).status == "rendering", "nothing ended the render: the boot reaper will"


def test_the_price_is_booked_before_the_first_line_and_the_script_settles_that_row(ledger):
    post = _fish_post(ledger)
    speak, before_first_line = ledger.composio.answers[SPEAK], []

    def speaking(params):
        if not before_first_line:
            before_first_line.append(paid_media_now(ledger, post["id"]))
        return speak(params)

    ledger.composio.answers[SPEAK] = speaking

    ok, _, _ = _render(ledger, post["id"], FakeStore())

    assert ok is True
    assert before_first_line == [
        [{"provider": "fish_audio", "usd": pytest.approx(_priced()), "units": _fish_bytes(), "status": "pending"}],
    ]
    assert paid_media_now(ledger, post["id"]) == [
        {"provider": "fish_audio", "usd": pytest.approx(BALANCE_DIFFERENCE), "units": _fish_bytes(), "status": "success"},
    ]
    # One amount, counted once, by the post's cap, the monthly cap and the budget gate.
    ledger.session.expire_all()
    spend = media_caps.media_spend(ledger.session, ledger.session.get(Workspace, WS), post["id"])
    assert (spend.post_usd, spend.month_usd) == (pytest.approx(BALANCE_DIFFERENCE), pytest.approx(BALANCE_DIFFERENCE))
    assert spend_to_date(ledger.session, WS, "month")["cost_usd"] == pytest.approx(BALANCE_DIFFERENCE)


def test_a_voice_whose_booking_cannot_be_written_speaks_nothing(ledger, monkeypatch):
    post = _fish_post(ledger)

    def cannot_book(*args, **kwargs):
        raise media_ledger.BookingFailed("fish_audio's spend could not be booked")

    monkeypatch.setattr(media_ledger, "commit", cannot_book)

    ok, renderer, _ = _render(ledger, post["id"], FakeStore())

    assert ok is False and renderer.bundles == []
    assert ledger.composio.slugs() == [BALANCE], "its balance read, then nothing spoken"
    entry = _post(ledger, post["id"]).review_log[-1]
    assert entry["report"]["code"] == "voice_failed"
    assert entry["comment"] == "What Fish Audio would spend could not be booked: nothing was spoken. Nothing was rendered."
    assert paid_media_now(ledger, post["id"]) == []
