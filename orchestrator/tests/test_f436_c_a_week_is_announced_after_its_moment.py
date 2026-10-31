"""P256-FIX-SOC-5 (F436, objection 5b): a slot moved earlier never announces its future week.

A slot moved before its batch's moment is due at once and made: the tick then mapped the post to
its own week, found nothing of that week pending yet (the rest was not due) and sent "Your week is
ready" with a count of 1, days early. At the real moment the week was already announced, so the
real notice never went out. Now a week is announced only once its moment has come, and a week whose
only posts were made early is announced at its moment by the tick itself
(``batches.unannounced``). On the plan maker's harness, pinned:

* a W43 slot moved into today is made today, with no notice for W43; at the W43 moment the tick
  makes the rest and sends one "Your week is ready" that counts the early post;
* a week holding only an early-made post is announced at its moment.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from tests.test_f436_a_a_batch_makes_its_seven_days import plan_of, tick  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251cw2_batches import _posts, week  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
EVERY_DAY = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
TODAY = datetime(2026, 10, 14, 10, 0, tzinfo=UTC)  # Wednesday, in the week before the plan starts
W43_MOMENT = datetime(2026, 10, 18, 17, 30, tzinfo=UTC)  # Sunday after 17:00: the week of 19 Oct is made
MOVED_KEY = "r1|2026-10-21|09:00"


def _move_into_today(env, plan):
    resp = env.client.put(f"/api/socials/plans/{plan['id']}/slots/{MOVED_KEY}", json={"to": "2026-10-14T15:00:00Z"})
    assert resp.status_code == 200 and resp.json()["slot"]["moved"] is True, resp.text


def test_a_slot_moved_into_today_does_not_announce_its_future_week(week):  # noqa: F811
    plan = plan_of(week, EVERY_DAY, starts="2026-10-19", ends="2026-10-25")
    _move_into_today(week, plan)

    tick(TODAY)

    (early,) = _posts(week, plan)
    assert (early.slot_key, early.batch_key) == (MOVED_KEY, "2026-W43")
    assert week.reviews == []  # the week of 19 Oct is not made yet: nothing is announced

    tick(W43_MOMENT)

    made = _posts(week, plan)
    assert len(made) == 7 and {post.batch_key for post in made} == {"2026-W43"}
    assert week.reviews == [("Your week is ready: ", "Countdown: 7 posts for the week of 19 Oct")]  # the early one counted


def test_a_week_holding_only_an_early_made_post_is_announced_at_its_moment(week):  # noqa: F811
    plan = plan_of(week, ["wed"], starts="2026-10-19", ends="2026-10-25")
    _move_into_today(week, plan)

    tick(TODAY)
    assert len(_posts(week, plan)) == 1 and week.reviews == []

    tick(W43_MOMENT)  # nothing is left to make: only the tick's own pass announces it

    assert len(_posts(week, plan)) == 1
    assert week.reviews == [("Your week is ready: ", "Countdown: 1 post for the week of 19 Oct")]
    tick(datetime(2026, 10, 18, 17, 35, tzinfo=UTC))
    assert len(week.reviews) == 1  # once
