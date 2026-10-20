"""P256-FIX-SOC-5 (F436, socials day 14 Oct): a weekly batch makes its own 7 days, and says so.

The run read "the build fills n-1 slots" (4→3, 3→2 twice). There is no off-by-one: a weekly batch
makes the 7 local days from the next midnight after its batch day, so with a Wednesday batch the
window runs Thursday to Wednesday and Thursday 22 and Friday 23 Oct are made by the batch of
Wednesday 21. Every slot the run saw empty belonged to the next batch. On the plan maker's harness
(``tests/test_prd251cw2_batches.py``'s ``week`` fixture: SQLite, the composer and the notices
faked), pinned:

* a 4-slot week inside one window makes 4 posts, one batch key, announced once (passes on main:
  the batch was never short);
* a slot past the window waits for the batch that makes it, and its slot row says which and when;
  that batch makes it at its moment;
* a same-day change of the batch time never advertises the week already made and announced (it
  made nothing at the advertised time); the next batch is the following one;
* a batch day other than Sunday is warned about on the plan: which days the next batch makes.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import anyio  # noqa: E402

import api.socials_plans as socials_plans_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251bw2_maker import _topic as _maker_topic  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251cw2_batches import _posts, week  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
WEEK_READY = "Your week is ready: "
MON_TO_THU = ["mon", "tue", "wed", "thu"]


def _at(*args):
    return datetime(*args, tzinfo=UTC)


def plan_of(env, days, *, batch_day="sun", time="17:00", starts="2026-10-12", ends="2026-11-08", name="Countdown"):
    """A weekly text plan in UTC, one row at 09:00 on ``days``, with eight topics in its bank."""
    row = {"channels": ["linkedin"], "format": "text", "days": days, "time": "09:00"}
    plan = _create_plan(env, name=name, timezone="UTC", starts_on=starts, ends_on=ends, cadence=[row],
                        make={"rhythm": "weekly", "batch_day": batch_day, "time": time})
    for n in range(8):
        _maker_topic(env, plan, f"{name} topic {n}")
    return plan


def tick(now):
    return anyio.run(maker_mod.run_tick, now)


def at_clock(monkeypatch, now):
    """The plan routes read ``now`` (the next batch, the slots' phases) as this moment."""
    monkeypatch.setattr(socials_plans_api, "_now", lambda: now)


def test_a_four_slot_week_inside_one_window_makes_four_posts(week):  # noqa: F811
    plan = plan_of(week, MON_TO_THU)

    tick(_at(2026, 10, 18, 17, 30))

    made = _posts(week, plan)
    assert [post.slot_key[3:13] for post in made] == ["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]
    assert {post.batch_key for post in made} == {"2026-W43"}
    assert week.reviews == [(WEEK_READY, "Countdown: 4 posts for the week of 19 Oct")]


def test_a_slot_past_the_window_says_which_batch_makes_it_and_is_made_then(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, MON_TO_THU, batch_day="wed", time="13:44", starts="2026-10-19", ends="2026-10-25")

    tick(_at(2026, 10, 14, 13, 47))

    made = _posts(week, plan)
    assert len(made) == 3 and {post.batch_key for post in made} == {"2026-W42"}  # Mon 19 to Wed 21
    at_clock(monkeypatch, _at(2026, 10, 14, 13, 47))
    window = {"start": "2026-10-19T00:00:00Z", "end": "2026-10-26T00:00:00Z"}
    slots = week.client.get(f"/api/socials/plans/{plan['id']}/slots", params=window).json()["slots"]
    thursday = next(slot for slot in slots if slot["local_date"] == "2026-10-22")
    assert (thursday["state"], thursday["phase"]) == ("planned", "waiting")
    assert (thursday["batch"]["key"], thursday["batch"]["made_at"]) == ("2026-W43", "2026-10-21T13:44:00+00:00")
    assert {slot["phase"] for slot in slots if slot["local_date"] != "2026-10-22"} == {"made"}

    tick(_at(2026, 10, 21, 13, 47))

    fourth = [post for post in _posts(week, plan) if post.slot_key == "r1|2026-10-22|09:00"]
    assert len(fourth) == 1 and fourth[0].batch_key == "2026-W43"


def test_a_same_day_rearm_never_advertises_the_announced_week(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon", "wed", "fri"], batch_day="wed", time="11:00", starts="2026-10-19")
    tick(_at(2026, 10, 14, 11, 5))
    assert len(_posts(week, plan)) == 2 and len(week.reviews) == 1  # Mon 19 and Wed 21: the week of 15 Oct, announced

    at_clock(monkeypatch, _at(2026, 10, 14, 13, 19))
    saved = week.client.put(f"/api/socials/plans/{plan['id']}", json={"make": {"time": "13:21"}})
    assert saved.status_code == 200, saved.text
    assert saved.json()["next_batch_at"] == "2026-10-21T13:21:00+00:00"  # never 14 Oct 13:21: that week is made
    assert week.client.get(f"/api/socials/plans/{plan['id']}").json()["next_batch_at"] == "2026-10-21T13:21:00+00:00"

    told_before = list(week.told.plan)
    tick(_at(2026, 10, 14, 13, 21, 30))

    assert len(_posts(week, plan)) == 2 and len(week.reviews) == 1 and week.told.plan == told_before


def test_a_batch_day_that_splits_the_week_is_warned(week, monkeypatch):  # noqa: F811
    at_clock(monkeypatch, _at(2026, 10, 14, 13, 47))
    weekdays = ["mon", "tue", "wed", "thu", "fri"]
    midweek = plan_of(week, weekdays, batch_day="wed", time="13:44")
    sunday = plan_of(week, weekdays, name="Sunday batch")

    warnings = week.client.get(f"/api/socials/plans/{midweek['id']}").json()["warnings"]

    assert warnings == ["Your Wednesday 13:44 batch makes Thursday to Wednesday: Thu 22 and Fri 23 Oct are made in "
                        "the batch of Wed 21 Oct."]
    assert week.client.get(f"/api/socials/plans/{sunday['id']}").json()["warnings"] == []
