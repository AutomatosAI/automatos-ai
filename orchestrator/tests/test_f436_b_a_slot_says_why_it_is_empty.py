"""P256-FIX-SOC-5 (F436, socials day 14 Oct): a slot says why it is empty, and a stall never reads
as progress.

A plan in the middle of its batch looked the same as a failed one: a slot answered only made or
planned, the next batch time rolled on as soon as the batch's moment passed, and a stall recorded
nothing. Each slot now carries a phase derived from the same predicates the make tick uses
(``modules/socials/slot_phases.py``), and the plan its current batch. On the plan maker's harness,
pinned:

* at the batch's moment, before any tick, the batch reads building since its moment and its due
  slots building; after the tick it is ready;
* a due slot no unused topic suits reads stalled, naming the format, and still does after a tick
  (whose notice names the format too);
* Socials off, a paused plan and a day that has its ``max_per_day`` posts read stalled, saying why;
* a batch with no post made for three ticks reads late; one made just now reads building;
* a slot the batch skipped shows the reason it recorded;
* ``state`` stays made or planned: the calendar's contract.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import plan_store, service, slot_phases  # noqa: E402
from modules.socials.capabilities import SocialChannel  # noqa: E402
from modules.socials.settings import SOCIALS_OFF_FOR_WORKSPACE  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
import api.socials_targets as socials_targets  # noqa: E402
from tests.test_f436_a_a_batch_makes_its_seven_days import at_clock, plan_of, tick  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_maker import _kind, _text_plan  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251cw2_batches import week  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc
MOMENT = datetime(2026, 10, 18, 17, 0, tzinfo=UTC)  # the week of 19 Oct is made
WINDOW = {"start": "2026-10-19T00:00:00Z", "end": "2026-10-26T00:00:00Z"}
VIDEO_CHANNELS = [SocialChannel(toolkit="linkedin", label="LinkedIn", post_kinds=(_kind("text"), _kind("image"), _kind("video")),
                                verified=True, setup_note=None)]


def _slots(env, plan, window=WINDOW):
    resp = env.client.get(f"/api/socials/plans/{plan['id']}/slots", params=window)
    assert resp.status_code == 200, resp.text
    return resp.json()["slots"]


def _batch(env, plan):
    return env.client.get(f"/api/socials/plans/{plan['id']}").json()["current_batch"]


def _phases(slots):
    return {slot["local_date"]: (slot["phase"], slot["why"]) for slot in slots}


def test_mid_build_the_plan_says_building_since_its_moment(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon", "tue", "wed"])
    at_clock(monkeypatch, MOMENT + timedelta(minutes=1))

    batch = _batch(week, plan)
    assert {key: batch[key] for key in ("key", "since", "state", "made", "left", "why")} == {
        "key": "2026-W43", "since": "2026-10-18T17:00:00+00:00", "state": "building", "made": 0, "left": 3, "why": None,
    }
    assert set(_phases(_slots(week, plan)).values()) == {("building", None)}

    tick(MOMENT + timedelta(minutes=1))

    after = _batch(week, plan)
    assert (after["state"], after["made"], after["left"]) == ("ready", 3, 0)


def test_a_due_slot_no_topic_suits_reads_stalled_not_building(week, monkeypatch):  # noqa: F811
    monkeypatch.setattr(maker_mod, "social_channels", lambda db, ws: VIDEO_CHANNELS)
    monkeypatch.setattr(socials_targets, "social_channels", lambda db, ws: VIDEO_CHANNELS)
    monkeypatch.setattr(maker_mod, "fits_quota", lambda db, workspace, seconds: True)
    row = {"channels": ["linkedin"], "format": "video", "days": ["tue"], "time": "09:00"}
    plan = _create_plan(week, name="Countdown", timezone="UTC", cadence=[row], make={"rhythm": "weekly", "batch_day": "sun", "time": "17:00"})
    for n in range(3):
        resp = week.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": f"Still {n}", "formats": ["image"], "facts": []})
        assert resp.status_code == 201, resp.text
    at_clock(monkeypatch, MOMENT + timedelta(minutes=1))
    stalled = ("stalled", "no unused topic in the bank suits a video")

    assert _phases(_slots(week, plan))["2026-10-20"] == stalled
    assert (_batch(week, plan)["state"], _batch(week, plan)["why"]) == stalled

    tick(MOMENT + timedelta(minutes=1))

    assert _phases(_slots(week, plan))["2026-10-20"] == stalled
    assert ("social_plan_bank_empty", "Countdown: no unused topic in the bank suits a video; research again or add one") in week.told.plan


def test_socials_off_and_a_paused_plan_read_stalled_with_why(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon", "tue", "wed"])
    assert week.client.post(f"/api/socials/plans/{plan['id']}/pause").status_code == 200
    at_clock(monkeypatch, MOMENT + timedelta(minutes=1))

    assert set(_phases(_slots(week, plan)).values()) == {("stalled", slot_phases.PAUSED_WHY)}
    assert _batch(week, plan)["state"] == "stalled"

    workspace = week.session.get(Workspace, WS_A)
    workspace.settings = {"socials": {"enabled": False}}  # the routes answer 404 now: read the rows themselves
    week.session.commit()
    row = week.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    rows = plan_store.plan_slots(week.session, row, datetime(2026, 10, 19, tzinfo=UTC), datetime(2026, 10, 26, tzinfo=UTC),
                                 MOMENT + timedelta(minutes=1))
    assert {(slot["phase"], slot["why"]) for slot in rows} == {("stalled", SOCIALS_OFF_FOR_WORKSPACE)}


def test_the_daily_cap_reads_stalled(week):  # noqa: F811
    rows = [{"channels": ["linkedin"], "format": "text", "days": ["wed"], "time": "09:00"},
            {"channels": ["linkedin"], "format": "text", "days": ["wed"], "time": "12:00"}]
    plan = _text_plan(week, cadence=rows, make={"rhythm": "daily", "time": "07:00", "max_per_day": 1})
    for n in range(3):
        week.client.post(f"/api/socials/plans/{plan['id']}/topics", json={"title": f"Topic {n}", "facts": []})
    now = datetime(2026, 10, 14, 7, 30, tzinfo=UTC)
    tick(now)
    row = week.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    week.session.refresh(row)

    found = plan_store.plan_slots(week.session, row, datetime(2026, 10, 14, tzinfo=UTC), datetime(2026, 10, 15, tzinfo=UTC), now)

    assert [(slot["key"], slot["phase"], slot["why"]) for slot in found] == [
        ("r1|2026-10-14|09:00", "made", None),
        ("r2|2026-10-14|12:00", "stalled", slot_phases.DAILY_CAP_WHY.format(cap=1)),
    ]
    assert found[1]["batch"] == {"key": None, "label": None, "made_at": "2026-10-14T07:00:00+00:00"}  # a daily plan's make time


def _made_post(env, plan, key, created_at):
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="member-1", title="Made", content_hash="0" * 64,
                      campaign_id=uuid.UUID(plan["id"]), slot_key=key, batch_key="2026-W43", status="needs_approval",
                      format="text", created_at=created_at)
    post.content_hash = service.compute_content_hash(post)
    env.session.add(post)
    env.session.commit()


def test_a_window_with_no_progress_for_three_ticks_reads_late(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon", "tue", "wed"])
    later = MOMENT + timedelta(minutes=20)  # three 5-minute ticks and more, and nothing made
    at_clock(monkeypatch, later)
    late = ("late", "due since 17:00; nothing made for 3 ticks")

    assert set(_phases(_slots(week, plan)).values()) == {late}
    assert (_batch(week, plan)["state"], _batch(week, plan)["why"]) == late

    _made_post(week, plan, "r1|2026-10-19|09:00", later - timedelta(minutes=2))  # the tick made one just now

    phases = _phases(_slots(week, plan))
    assert phases["2026-10-19"] == ("made", None) and phases["2026-10-20"] == phases["2026-10-21"] == ("building", None)


def test_a_batch_skipped_slot_shows_its_recorded_why(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon"])
    tiktok = {"channels": ["tiktok"], "format": "text", "days": ["tue"], "time": "09:00"}
    resp = week.client.put(f"/api/socials/plans/{plan['id']}", json={"cadence": [*plan["cadence"], tiktok]})
    assert resp.status_code == 200, resp.text
    tick(MOMENT + timedelta(minutes=30))
    at_clock(monkeypatch, MOMENT + timedelta(minutes=31))

    phases = _phases(_slots(week, plan))

    assert phases["2026-10-20"] == ("skipped", "no connected channel posts a text for tiktok")
    assert phases["2026-10-19"] == ("made", None)


def test_state_stays_made_or_planned(week, monkeypatch):  # noqa: F811
    plan = plan_of(week, ["mon", "tue", "wed", "thu"], batch_day="wed", time="13:44")
    tick(datetime(2026, 10, 14, 13, 47, tzinfo=UTC))
    at_clock(monkeypatch, datetime(2026, 10, 14, 13, 50, tzinfo=UTC))

    slots = _slots(week, plan)

    assert {slot["state"] for slot in slots} == {"made", "planned"}
    assert all({"phase", "why", "batch"} <= set(slot) for slot in slots)
    assert {slot["phase"] for slot in slots if slot["state"] == "made"} == {"made"}
