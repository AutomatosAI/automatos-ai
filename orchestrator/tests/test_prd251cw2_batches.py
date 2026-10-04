"""PRD-251C Wave 2 — a plan's rhythm (US-C201) and its batches, made at once (US-C203).

Pinned:

* ``make.rhythm`` (daily, weekly or monthly), ``make.batch_day`` and ``make.batch_date`` are
  checked; a plan saved before PRD-251C has no rhythm and stays daily;
* a new plan makes its week on Sunday at 17:00 (O1, O3);
* a save changes what it sends: the rest of ``make``, and what the make tick recorded
  (notices sent, batches made), stay;
* a batch's window, moment and key (weekly on any day, monthly on its date); the slots due:
  the coming week once its day comes, never a skipped or taken one, a slot moved ahead of its
  batch or added after it; across the clock change; the next batch; the records kept;
* the tick, on the maker harness: a week made across ticks within the per-tick cap and
  announced once; a slot no channel posts skipped and recorded, the batch still announced;
  a row added after its batch made at the next tick; a daily plan unchanged;
* US-C207: a weekly plan researches the day before its batch day, follows it when it moves,
  and keeps a day the owner chose; the bank shows when research last ran.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import anyio  # noqa: E402

import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost  # noqa: E402
from modules.socials import batches, plan_notify, plans  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from services import socials_plan_research as research_mod  # noqa: E402
from tests.test_prd251bw2_maker import _topic as _maker_topic  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api


def test_rhythm_batch_day_and_date_are_checked():
    assert plans.validate_make({})["rhythm"] == plans.DAILY  # a plan saved before PRD-251C
    made = plans.validate_make({"rhythm": "weekly", "batch_day": "sun", "time": "17:00"})
    assert (made["rhythm"], made["batch_day"], made["batch_date"], made["time"]) == ("weekly", "sun", 25, "17:00")
    assert plans.validate_make({"rhythm": "monthly", "batch_date": 28})["batch_date"] == 28
    for bad in ({"rhythm": "hourly"}, {"batch_day": "sunday"}, {"batch_date": 29}, {"batch_date": 0}, {"batch_date": True}):
        with pytest.raises(plans.InvalidPlan):
            plans.validate_make(bad)


def test_a_new_plan_makes_its_week_on_sunday_at_17(bank):  # noqa: F811
    plan = _create_plan(bank)
    assert {key: plan["make"][key] for key in ("rhythm", "batch_day", "batch_date", "time")} == {
        "rhythm": "weekly", "batch_day": "sun", "batch_date": 25, "time": "17:00",
    }
    daily = _create_plan(bank, name="Every morning", make={"rhythm": "daily", "time": "07:00"})
    assert (daily["make"]["rhythm"], daily["make"]["time"]) == ("daily", "07:00")
    refused = bank.client.put(f"/api/socials/plans/{plan['id']}", json={"make": {"rhythm": "hourly"}})
    assert refused.status_code == 422, refused.text


def test_a_save_keeps_the_rest_of_make_and_what_the_tick_recorded(bank):  # noqa: F811
    plan = _create_plan(bank)
    row = bank.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    records = {"notified": {"social_plan_ready": "2026-10-14"}, "batches": {"2026-W42": {"announced": "2026-10-11T17:05:00+00:00"}}}
    row.make = {**row.make, **records}
    bank.session.commit()
    saved = bank.client.put(f"/api/socials/plans/{plan['id']}", json={"make": {"batch_day": "fri"}})
    assert saved.status_code == 200, saved.text
    make = saved.json()["make"]
    assert (make["rhythm"], make["batch_day"], make["time"]) == ("weekly", "fri", "17:00")
    assert {key: make[key] for key in records} == records


# ── US-C203: the batch, made at once ───────────────────────────────────────

UTC = timezone.utc
WEEKLY = {"rhythm": "weekly", "batch_day": "sun", "time": "17:00"}
DAILY_ROW = {"id": "r1", "channels": ["linkedin"], "format": "text", "days": list(plans.WEEKDAYS), "time": "09:00"}


def _plan(**fields):
    defaults = {"timezone": "UTC", "starts_on": date(2026, 10, 12), "ends_on": date(2026, 11, 8),
                "cadence": [DAILY_ROW], "slot_overrides": {}, "make": WEEKLY}
    return SimpleNamespace(**{**defaults, **fields})


def _at(*args):
    return datetime(*args, tzinfo=UTC)


def _days(slots):
    return [slot.local_date.day for slot in slots]


def test_a_batch_window_its_moment_and_its_key():
    week = batches.window_of(_plan(), date(2026, 10, 14))
    assert (week.start, week.end, week.moment, week.key) == (date(2026, 10, 12), date(2026, 10, 19), _at(2026, 10, 11, 17), "2026-W42")
    wednesday = batches.window_of(_plan(make={**WEEKLY, "batch_day": "wed"}), date(2026, 10, 14))
    assert (wednesday.start, wednesday.end, wednesday.moment) == (date(2026, 10, 8), date(2026, 10, 15), _at(2026, 10, 7, 17))
    month = batches.window_of(_plan(make={"rhythm": "monthly", "batch_date": 25, "time": "17:00"}), date(2026, 11, 10))
    assert (month.start, month.end, month.moment, month.key) == (date(2026, 11, 1), date(2026, 12, 1), _at(2026, 10, 25, 17), "2026-11")
    assert batches.window_of(_plan(make={"rhythm": "daily"}), date(2026, 10, 14)) is None


def test_a_weekly_plan_makes_the_coming_week_once_its_day_comes():
    plan = _plan()
    assert _days(batches.due_slots(plan, _at(2026, 10, 14, 7, 30), ())) == [14, 15, 16, 17, 18]  # this week, made last Sunday
    assert batches.due_slots(plan, _at(2026, 10, 18, 16, 59), ()) == []  # Sunday before 17:00: the next week waits
    assert _days(batches.due_slots(plan, _at(2026, 10, 18, 17, 0), ())) == [19, 20, 21, 22, 23, 24, 25]
    taken = {"r1|2026-10-19|09:00"}
    skipped = _plan(make={**WEEKLY, "batches": {"2026-W43": {"skipped": {"r1|2026-10-20|09:00": "no channel"}}}})
    assert _days(batches.due_slots(skipped, _at(2026, 10, 18, 17, 30), taken)) == [21, 22, 23, 24, 25]


def test_a_slot_moved_before_its_batch_or_added_after_it_is_made_now():
    moved = _plan(slot_overrides={"r1|2026-10-21|09:00": {"to": "2026-10-16T10:00:00+00:00"}})
    assert "r1|2026-10-21|09:00" in [slot.key for slot in batches.due_slots(moved, _at(2026, 10, 14, 7, 30), ())]
    evening = {**DAILY_ROW, "id": "r2", "days": ["fri"], "time": "18:00"}
    added = _plan(cadence=[DAILY_ROW, evening])
    assert "r2|2026-10-16|18:00" in [slot.key for slot in batches.due_slots(added, _at(2026, 10, 14, 7, 30), ())]


def test_the_batches_keep_their_local_times_across_the_clock_change():
    london = _plan(timezone="Europe/London")
    assert batches.window_of(london, date(2026, 10, 20)).moment == _at(2026, 10, 18, 16)  # 17:00 BST
    assert batches.window_of(london, date(2026, 10, 27)).moment == _at(2026, 10, 25, 17)  # 17:00 GMT
    after = batches.due_slots(london, _at(2026, 10, 25, 17, 30), ())
    assert [slot.at for slot in after][:2] == [_at(2026, 10, 26, 9), _at(2026, 10, 27, 9)]  # 09:00 GMT
    before = batches.due_slots(london, _at(2026, 10, 18, 16, 30), ())
    assert before[0].at == _at(2026, 10, 19, 8)  # 09:00 BST


def test_the_next_batch_and_the_records():
    plan = _plan()
    assert batches.next_window(plan, _at(2026, 10, 14, 7, 30)).moment == _at(2026, 10, 18, 17)
    assert batches.next_window(_plan(ends_on=date(2026, 10, 18)), _at(2026, 10, 14, 7, 30)) is None
    assert batches.next_window(_plan(starts_on=date(2026, 11, 2)), _at(2026, 10, 14, 7, 30)).start == date(2026, 11, 2)
    old = {"2026-W30": {"announced": "2026-07-19T17:00:00+00:00"}}
    make = batches.with_skip(_plan(make={**WEEKLY, "batches": old}), "2026-W43", "r1|2026-10-20|09:00", "no channel", date(2026, 10, 18))
    assert make["batches"] == {"2026-W43": {"skipped": {"r1|2026-10-20|09:00": "no channel"}}}  # W30 was dropped
    assert batches.skipped_keys(SimpleNamespace(make=make)) == {"r1|2026-10-20|09:00"}
    assert batches.label(plan, batches.window_of(plan, date(2026, 10, 20))) == "the week of 19 Oct"


# ── the tick, on the maker harness ─────────────────────────────────────────

SUNDAY_EVENING = _at(2026, 10, 18, 17, 30)  # the week of 19 Oct is due


@pytest.fixture
def week(maker, monkeypatch):  # noqa: F811
    """The maker harness, research's weekly pass switched off, and the review notices kept."""
    monkeypatch.setattr(research_mod, "launch_due", lambda now: 0)
    reviews = []
    monkeypatch.setattr(plan_notify, "notify_review", lambda ws, plan_id, event, title: reviews.append((event[1], title)))
    maker.reviews = reviews
    return maker


def _weekly_plan(api, rows=(DAILY_ROW,), **overrides):
    cadence = [{key: value for key, value in row.items() if key != "id"} for row in rows]
    body = {"name": "Countdown", "timezone": "UTC", "starts_on": "2026-10-12", "ends_on": "2026-11-08",
            "cadence": cadence, "make": WEEKLY, **overrides}
    plan = _create_plan(api, **body)
    for n in range(8):
        _maker_topic(api, plan, f"Topic {n}")
    return plan


def _tick(now=SUNDAY_EVENING):
    return anyio.run(maker_mod.run_tick, now)


def _posts(api, plan):
    api.session.expire_all()
    return api.session.query(SocialPost).filter(SocialPost.campaign_id == uuid.UUID(plan["id"])).order_by(SocialPost.slot_key).all()


def test_a_week_is_made_across_ticks_and_announced_once(week, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_PLAN_MAX_SLOTS_PER_TICK", 3)
    plan = _weekly_plan(week)
    for _ in range(2):
        _tick()
    assert len(_posts(week, plan)) == 6 and week.reviews == []
    _tick()
    made = _posts(week, plan)
    assert len(made) == 7 and {post.batch_key for post in made} == {"2026-W43"}
    assert {post.status for post in made} == {"needs_approval"}
    assert week.reviews == [("Your week is ready: ", "Countdown: 7 posts for the week of 19 Oct")]
    _tick()
    assert len(week.reviews) == 1 and not [event for event, _title in week.told.plan if event == "social_plan_ready"]


def test_a_slot_no_channel_posts_is_skipped_and_recorded_and_the_week_still_ends(week):
    tiktok = {**DAILY_ROW, "id": "r2", "channels": ["tiktok"], "days": ["tue"]}
    plan = _weekly_plan(week, rows=(DAILY_ROW, tiktok))
    _tick()
    assert len(_posts(week, plan)) == 7
    row = week.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    week.session.refresh(row)
    assert list(row.make["batches"]["2026-W43"]["skipped"]) == ["r2|2026-10-20|09:00"]
    assert [event for event, _title in week.told.plan if event == "social_plan_slot_skipped"] == ["social_plan_slot_skipped"]
    assert week.reviews == [("Your week is ready: ", "Countdown: 7 posts for the week of 19 Oct")]
    _tick(_at(2026, 10, 18, 17, 40))
    assert len(_posts(week, plan)) == 7  # the skipped slot is not tried again


def test_a_row_added_after_its_batch_is_made_at_the_next_tick(week):
    plan = _weekly_plan(week)
    _tick()
    evening = {"channels": ["linkedin"], "format": "text", "days": ["fri"], "time": "18:00"}
    resp = week.client.put(f"/api/socials/plans/{plan['id']}", json={"cadence": [*plan["cadence"], evening]})
    assert resp.status_code == 200, resp.text
    _tick(_at(2026, 10, 18, 17, 40))
    keys = [post.slot_key for post in _posts(week, plan)]
    assert len(keys) == 8 and any(key.endswith("|2026-10-23|18:00") for key in keys)
    assert len(week.reviews) == 1  # the week was announced once


def test_a_daily_plan_is_unchanged(week):
    plan = _weekly_plan(week, make={"rhythm": "daily", "time": "07:00"})
    _tick(_at(2026, 10, 14, 7, 30))
    made = _posts(week, plan)
    assert [post.slot_key for post in made] == ["r1|2026-10-14|09:00"] and made[0].batch_key is None
    assert week.reviews == [] and [event for event, _title in week.told.plan if event == "social_plan_ready"] == ["social_plan_ready"]


# ── US-C207: research the day before the batch ─────────────────────────────


def _put(api, plan, **fields):
    resp = api.client.put(f"/api/socials/plans/{plan['id']}", json=fields)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_a_weekly_plan_researches_the_day_before_its_batch_unless_the_owner_chose(bank):  # noqa: F811
    plan = _create_plan(bank)
    assert plan["research"]["day"] == "sat"
    assert _create_plan(bank, name="Midweek", make={"rhythm": "weekly", "batch_day": "wed"})["research"]["day"] == "tue"
    assert _create_plan(bank, name="Chosen", research={"day": "thu"})["research"]["day"] == "thu"
    assert _create_plan(bank, name="Daily", make={"rhythm": "daily"})["research"]["day"] == "mon"
    assert _put(bank, plan, make={"batch_day": "fri"})["research"]["day"] == "thu"  # it follows the batch day
    # The Plan page sends research's day as shown with every save: unchanged, it still follows.
    assert _put(bank, plan, make={"batch_day": "mon"}, research={"day": "thu"})["research"]["day"] == "sun"
    assert _put(bank, plan, research={"day": "wed"})["research"]["day"] == "wed"  # the owner's choice
    assert _put(bank, plan, make={"batch_day": "tue"})["research"]["day"] == "wed"  # kept


def test_the_bank_shows_when_research_last_ran(bank):  # noqa: F811
    plan = _create_plan(bank)
    assert bank.client.get(f"/api/socials/plans/{plan['id']}/topics").json()["research_last_run_at"] is None
    row = bank.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    row.research = {**row.research, "last_run_at": "2026-10-17T06:00:00+00:00", "last_run_id": "research-1"}
    bank.session.commit()
    assert bank.client.get(f"/api/socials/plans/{plan['id']}/topics").json()["research_last_run_at"] == "2026-10-17T06:00:00+00:00"


# ── US-C202: the next batch on the Plan page ───────────────────────────────


def test_the_plan_says_when_its_next_batch_is_made(bank):  # noqa: F811
    future = {"starts_on": "2030-01-07", "ends_on": "2030-02-03", "timezone": "UTC"}  # Monday 7 Jan 2030
    weekly = _create_plan(bank, name="Next year", **future)
    assert weekly["next_batch_at"] == "2030-01-06T17:00:00+00:00"  # the Sunday before, at 17:00
    assert bank.client.get(f"/api/socials/plans/{weekly['id']}").json()["next_batch_at"] == weekly["next_batch_at"]
    daily = _create_plan(bank, name="Every day", make={"rhythm": "daily"}, **future)
    assert daily["next_batch_at"] is None

