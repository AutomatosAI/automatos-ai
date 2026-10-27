"""P256-FIX-SOC-5 (F436, objection 4): Auto reads why a plan's slot is empty.

Auto's only plan tool returned the plan's fixed keys, the whole content bank and the history,
and no slot; no tool listed the plans. ``platform_get_social_plan_slots`` answers one line a
slot in the plan's timezone and one line for each plan's current batch, at most 40 lines, so it
fits the chat's cut of a tool's answer. On the plan maker's harness, the handler called as the
executor calls it, pinned:

* with no plan named it reads every active or paused plan (an ended one is left out), and the
  Thursday slot past the batch's 7 days says which batch makes it and when;
* a 20-row, 14-day plan's answer, formatted for the model and cut to the chat's budget, is not
  cut, and keeps the asked-for day's lines; the rest is counted;
* ``days`` past 14 and a ``from`` that is not a day are refused; with Socials off the tool
  refuses as every Socials tool does.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import anyio

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from consumers.chatbot.tool_cut import TOOL_RESULT_TOKEN_BUDGET  # noqa: E402  (the chat service's cut)
from core.context_guard import truncate_to_token_budget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.tools.discovery import handlers_socials, social_plan_slots  # noqa: E402
from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402
from tests.test_f436_a_a_batch_makes_its_seven_days import plan_of, tick  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251cw2_batches import week  # noqa: E402,F401  (the fixture)

api = api_harness.api
ACTION = "platform_get_social_plan_slots"
NOW = datetime(2026, 10, 14, 13, 47, tzinfo=timezone.utc)  # Wednesday, just after the 13:44 batch
THURSDAY = "Thu 22 Oct 09:00 | Countdown | r1 | waiting: made by the batch of Wed 21 Oct 13:44"
WEEKLY_WED = {"rhythm": "weekly", "batch_day": "wed", "time": "13:44"}


def _ask(env, monkeypatch, **params):
    monkeypatch.setattr(social_plan_slots, "_now", lambda: NOW)
    return anyio.run(handlers_socials.get_social_plan_slots, env.session, WS_A, params)


def _what_chat_sees(result):
    """The answer as Auto reads it: formatted for the model, then cut to the chat's token budget."""
    return truncate_to_token_budget(ToolResultFormatter.format_for_llm(result, ACTION), TOOL_RESULT_TOKEN_BUDGET)


def test_with_no_plan_named_it_reads_every_active_or_paused_plan(week, monkeypatch):  # noqa: F811
    countdown = plan_of(week, ["mon", "tue", "wed", "thu"], batch_day="wed", time="13:44", starts="2026-10-19", ends="2026-10-25")
    tick(NOW)
    paused = plan_of(week, ["fri"], name="Paused plan", starts="2026-10-19", ends="2026-10-25")
    ended = plan_of(week, ["fri"], name="Ended plan", starts="2026-10-19", ends="2026-10-25")
    assert week.client.post(f"/api/socials/plans/{paused['id']}/pause").status_code == 200
    assert week.client.post(f"/api/socials/plans/{ended['id']}/end").status_code == 200

    answer = _ask(week, monkeypatch, **{"from": "2026-10-19", "days": 7})

    assert answer["success"] is True
    lines = answer["lines"]
    assert THURSDAY in lines
    assert "Wed 21 Oct 09:00 | Countdown | r1 | made (needs_approval)" in lines
    assert any(line.startswith("Fri 23 Oct 09:00 | Paused plan | r1 | ") for line in lines)
    assert not any("Ended plan" in line for line in lines)
    one = _ask(week, monkeypatch, plan_id=countdown["id"], **{"from": "2026-10-22", "days": 1})
    assert [line for line in one["lines"] if line.startswith("Thu 22 Oct")] == [THURSDAY]


def test_a_big_plans_answer_fits_the_chats_cut_and_keeps_the_day_asked(week, monkeypatch):  # noqa: F811
    rows = [{"channels": ["linkedin"], "format": "text", "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
             "time": f"{8 + n // 2:02d}:{(n % 2) * 30:02d}"} for n in range(20)]
    big = _create_plan(week, name="Big plan", timezone="UTC", starts_on="2026-10-19", ends_on="2026-11-30", cadence=rows,
                       make=WEEKLY_WED)

    answer = _ask(week, monkeypatch, plan_id=big["id"], **{"from": "2026-10-22", "days": 14})
    seen = _what_chat_sees(answer)

    assert answer["count"] == 20 * 14 + 1  # every slot and the current batch, counted
    assert len(answer["lines"]) == social_plan_slots.MAX_LINES + 1
    assert answer["lines"][-1] == social_plan_slots.MORE.format(count=20 * 14 + 1 - social_plan_slots.MAX_LINES)
    assert seen == ToolResultFormatter.format_for_llm(answer, ACTION)  # nothing cut
    assert "Thu 22 Oct 08:00 | Big plan | r1 | waiting: made by the batch of Wed 21 Oct 13:44" in seen


def test_too_many_days_a_bad_day_and_socials_off_are_refused(week, monkeypatch):  # noqa: F811
    plan_of(week, ["mon"])

    too_many = _ask(week, monkeypatch, days=15)
    assert (too_many["success"], too_many["error"]) == (False, social_plan_slots.DAYS_REFUSED)
    assert _ask(week, monkeypatch, **{"from": "next thursday"})["error"] == social_plan_slots.FROM_REFUSED
    assert _ask(week, monkeypatch, plan_id="not-a-plan")["success"] is False

    workspace = week.session.get(Workspace, WS_A)
    workspace.settings = {"socials": {"enabled": False}}
    week.session.commit()
    off = _ask(week, monkeypatch)
    assert off["success"] is False and off["socials_off"] is True
