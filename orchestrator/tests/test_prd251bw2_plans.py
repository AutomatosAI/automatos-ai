"""PRD-251B Wave 2, US-B202 (and US-B208's slot moves) — plans over HTTP, and their slots.

Pinned:

* ``expand_slots`` is pure: each cadence row on each of its days between the plan's dates,
  at its local time in the plan's zone — across both DST changes (Europe/London) and month
  ends — moved or skipped by ``slot_overrides``; ``slot_for_key`` finds one slot by key;
* when a slot's post is made (``make_at``): the plan's make time, a video days early, a day
  earlier when that leaves too little lead; ``due_slots`` and ``next_free_slot``;
* the cadence check: known formats and channels, a template of the right kind and a length
  it declares, days and times; the dates, the zone and the late policy;
* the routes: create (the fields required), list with the bank's counts, get, update (an
  ended plan is read-only), pause/resume/end (only the moves a status allows), the slots
  in a window (planned and made), a slot moved, put back or skipped (a made one is 409);
  another workspace's plan is a 404.
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

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialTopic  # noqa: E402
from modules.socials import plans  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _ctx  # noqa: E402

api = api_harness.api
UTC = timezone.utc
VIDEO_TEMPLATE = str(uuid.uuid4())
IMAGE_TEMPLATE = str(uuid.uuid4())
TEMPLATES = {VIDEO_TEMPLATE: ("social_video", [15, 30, 40]), IMAGE_TEMPLATE: ("social_image", [])}


def _plan(**fields):
    defaults = {
        "starts_on": date(2026, 3, 1), "ends_on": date(2026, 11, 30), "timezone": "Europe/London",
        "cadence": [{"id": "r1", "channels": ["linkedin"], "format": "image", "days": ["sun"], "time": "09:00"}],
        "slot_overrides": {}, "make": None,
    }
    return SimpleNamespace(**{**defaults, **fields})


def _at(*args):
    return datetime(*args, tzinfo=UTC)


# ── the slots ──────────────────────────────────────────────────────────────


def test_slots_keep_their_local_time_across_both_clock_changes():
    plan = _plan()
    spring = plans.expand_slots(plan, _at(2026, 3, 21), _at(2026, 4, 1))
    assert [s.at for s in spring] == [_at(2026, 3, 22, 9), _at(2026, 3, 29, 8)]  # 09:00 GMT, then 09:00 BST
    autumn = plans.expand_slots(plan, _at(2026, 10, 17), _at(2026, 10, 27))
    assert [s.at for s in autumn] == [_at(2026, 10, 18, 8), _at(2026, 10, 25, 9)]
    assert [s.key for s in autumn] == ["r1|2026-10-18|09:00", "r1|2026-10-25|09:00"]


def test_a_time_the_clocks_skip_lands_an_hour_on():
    plan = _plan(cadence=[{"id": "r1", "channels": ["linkedin"], "format": "image", "days": ["sun"], "time": "01:30"}])
    (slot,) = plans.expand_slots(plan, _at(2026, 3, 28), _at(2026, 3, 30))
    assert slot.at == _at(2026, 3, 29, 1, 30)  # 02:30 BST: the 01:30 that never happened, an hour on


def test_slots_run_over_month_ends_within_the_plans_dates():
    daily = [{"id": "d", "channels": ["x"], "format": "text", "days": list(plans.WEEKDAYS), "time": "12:00"}]
    plan = _plan(starts_on=date(2026, 1, 30), ends_on=date(2026, 2, 2), timezone="UTC", cadence=daily)
    slots = plans.expand_slots(plan, _at(2026, 1, 1), _at(2026, 3, 1))
    assert [s.local_date for s in slots] == [date(2026, 1, 30), date(2026, 1, 31), date(2026, 2, 1), date(2026, 2, 2)]


def test_overrides_move_and_skip_a_slot_and_the_key_finds_it():
    key, other = "r1|2026-10-18|09:00", "r1|2026-10-25|09:00"
    plan = _plan(slot_overrides={key: {"to": "2026-10-19T12:00:00+00:00"}, other: {"skip": True}})
    slots = plans.expand_slots(plan, _at(2026, 10, 17), _at(2026, 10, 27))
    assert [(s.key, s.at, s.moved) for s in slots] == [(key, _at(2026, 10, 19, 12), True)]
    assert plans.slot_for_key(plan, key).at == _at(2026, 10, 19, 12)
    assert plans.slot_for_key(plan, other) is None  # skipped
    assert plans.slot_for_key(plan, "r1|2026-10-19|09:00") is None  # a Monday: not the row's day
    assert plans.slot_for_key(plan, "r9|2026-10-18|09:00") is None and plans.slot_for_key(plan, "nonsense") is None


def test_make_time_videos_early_and_the_lead():
    video = [{"id": "v", "channels": ["tiktok"], "format": "video", "days": ["wed"], "time": "09:00"},
             {"id": "i", "channels": ["linkedin"], "format": "image", "days": ["wed"], "time": "09:00"},
             {"id": "e", "channels": ["linkedin"], "format": "image", "days": ["wed"], "time": "08:00"}]
    plan = _plan(timezone="UTC", cadence=video, make={"time": "07:00", "video_days_early": 1})
    slots = {s.row_id: s for s in plans.expand_slots(plan, _at(2026, 10, 14), _at(2026, 10, 15))}
    assert plans.make_at(plan, slots["v"]) == _at(2026, 10, 13, 7)   # the day before
    assert plans.make_at(plan, slots["i"]) == _at(2026, 10, 14, 7)   # its own day, two hours ahead
    assert plans.make_at(plan, slots["e"]) == _at(2026, 10, 13, 7)   # 07:00 leaves one hour: a day earlier
    day_before = _plan(timezone="UTC", cadence=video, make={"time": "07:00", "image_days_early": 1})
    assert plans.make_at(day_before, slots["i"]) == _at(2026, 10, 13, 7)  # images the day before too
    # At 07:30 on the Wednesday: the video and the 08:00 image were due yesterday; the 09:00 image is made.
    due = plans.due_slots(plan, _at(2026, 10, 14, 7, 30), made={"i|2026-10-14|09:00"})
    assert [s.row_id for s in due] == ["e", "v"]


def test_the_next_free_slot_of_the_same_row():
    rows = [{"id": "r1", "channels": ["linkedin"], "format": "image", "days": ["mon", "wed", "fri"], "time": "09:00"}]
    plan = _plan(timezone="UTC", cadence=rows)
    taken = {"r1|2026-10-16|09:00"}
    slot = plans.next_free_slot(plan, "r1", _at(2026, 10, 14, 10), taken)
    assert slot.key == "r1|2026-10-19|09:00"  # Friday is taken: the Monday after
    assert plans.next_free_slot(_plan(timezone="UTC", cadence=rows, ends_on=date(2026, 10, 16)), "r1", _at(2026, 10, 14, 10), taken) is None


# ── the checks ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("row, message", [
    ({"channels": ["linkedin"], "format": "poster", "days": ["mon"], "time": "09:00"}, "format"),
    ({"channels": [], "format": "image", "days": ["mon"], "time": "09:00"}, "channels"),
    ({"channels": ["Linked In!"], "format": "image", "days": ["mon"], "time": "09:00"}, "channels"),
    ({"channels": ["linkedin"], "format": "image", "days": ["someday"], "time": "09:00"}, "days"),
    ({"channels": ["linkedin"], "format": "image", "days": ["mon"], "time": "9am"}, "time"),
    ({"channels": ["tiktok"], "format": "video", "template_id": IMAGE_TEMPLATE, "days": ["mon"], "time": "09:00"}, "template_id"),
    ({"channels": ["tiktok"], "format": "video", "template_id": VIDEO_TEMPLATE, "length_seconds": 20, "days": ["mon"], "time": "09:00"}, "length_seconds"),
    ({"channels": ["x"], "format": "text", "template_id": IMAGE_TEMPLATE, "days": ["mon"], "time": "09:00"}, "template_id"),
])
def test_a_cadence_row_the_plan_cannot_keep_is_refused(row, message):
    with pytest.raises(plans.InvalidPlan) as caught:
        plans.validate_cadence([row], TEMPLATES)
    assert message in str(caught.value)


def test_a_good_cadence_is_cleaned():
    rows = plans.validate_cadence([
        {"channels": ["Instagram", "tiktok"], "format": "video", "template_id": VIDEO_TEMPLATE, "length_seconds": 30,
         "days": ["fri", "mon"], "time": "17:30"},
        {"channels": ["linkedin"], "format": "image", "days": ["tue"], "time": "08:00"},
    ], TEMPLATES)
    assert rows[0] == {"id": "r1", "channels": ["instagram", "tiktok"], "format": "video", "length_seconds": 30,
                       "template_id": VIDEO_TEMPLATE, "days": ["mon", "fri"], "time": "17:30"}
    assert rows[1]["id"] == "r2" and rows[1]["length_seconds"] is None
    with pytest.raises(plans.InvalidPlan):
        plans.validate_cadence([{**rows[0], "id": "same"}, {**rows[1], "id": "same"}], TEMPLATES)


def test_dates_zone_and_late_policy_are_checked():
    with pytest.raises(plans.InvalidPlan):
        plans.validate_dates(date(2026, 2, 1), date(2026, 1, 1))
    with pytest.raises(plans.InvalidPlan):
        plans.validate_dates(date(2026, 1, 1), date(2027, 6, 1))
    with pytest.raises(plans.InvalidPlan):
        plans.validate_timezone("Mars/Olympus")
    with pytest.raises(plans.InvalidPlan):
        plans.validate_late_policy("whenever")
    assert plans.validate_make({"visual_mix": {"templates": 70, "ai_images": 30}})["visual_mix"] == {"templates": 70, "ai_images": 30}
    with pytest.raises(plans.InvalidPlan):
        plans.validate_make({"visual_mix": {"templates": 70}})


# ── the routes ─────────────────────────────────────────────────────────────


@pytest.fixture
def bank(api):
    """The api harness with the content bank's table."""
    SocialPost.metadata.create_all(api.session.get_bind(), tables=[SocialTopic.__table__])
    return api


def _body(**overrides):
    body = {
        "name": "Countdown to Lisbon", "goal": "Fill the stand", "starts_on": "2026-10-12", "ends_on": "2026-11-08",
        "timezone": "Europe/London",
        "cadence": [{"channels": ["linkedin"], "format": "image", "days": ["mon", "wed", "fri"], "time": "09:00"}],
    }
    body.update(overrides)
    return body


def _create(api, **overrides):
    resp = api.client.post("/api/socials/plans", json=_body(**overrides))
    assert resp.status_code == 201, resp.text
    return resp.json()


def test_a_plan_is_created_listed_and_read(bank):
    plan = _create(bank)
    assert (plan["kind"], plan["status"], plan["late_policy"], plan["approval_mode"]) == ("plan", "active", "skip", "per_post")
    assert plan["make"]["time"] == "07:00" and plan["research"] == {"enabled": True, "day": "mon", "time": "06:00"}
    assert plan["cadence"][0]["id"] == "r1" and plan["bank"] == {"topics": 0, "unused": 0}
    listed = bank.client.get("/api/socials/plans").json()
    assert [p["id"] for p in listed["plans"]] == [plan["id"]]
    assert bank.client.get("/api/socials/campaigns").status_code == 200  # a plan is a campaign
    assert bank.client.get(f"/api/socials/plans/{plan['id']}").json()["name"] == "Countdown to Lisbon"


@pytest.mark.parametrize("missing", ["name", "starts_on", "ends_on", "timezone", "cadence"])
def test_a_plan_needs_its_core_fields(bank, missing):
    body = _body()
    body.pop(missing)
    assert bank.client.post("/api/socials/plans", json=body).status_code == 422


def test_update_pause_resume_end(bank):
    plan = _create(bank)
    updated = bank.client.put(f"/api/socials/plans/{plan['id']}", json={"late_policy": "next_slot", "goal": "Book demos"})
    assert updated.status_code == 200 and updated.json()["late_policy"] == "next_slot"
    assert bank.client.post(f"/api/socials/plans/{plan['id']}/resume").status_code == 422  # active already
    assert bank.client.post(f"/api/socials/plans/{plan['id']}/pause").json()["status"] == "paused"
    assert bank.client.post(f"/api/socials/plans/{plan['id']}/resume").json()["status"] == "active"
    assert bank.client.post(f"/api/socials/plans/{plan['id']}/end").json()["status"] == "ended"
    assert bank.client.put(f"/api/socials/plans/{plan['id']}", json={"goal": "x"}).status_code == 422
    assert bank.client.post(f"/api/socials/plans/{plan['id']}/resume").status_code == 422


def test_the_slots_of_a_window_and_a_slot_moved_put_back_and_skipped(bank):
    plan = _create(bank)
    window = {"start": "2026-10-12T00:00:00Z", "end": "2026-10-19T00:00:00Z"}
    slots = bank.client.get(f"/api/socials/plans/{plan['id']}/slots", params=window).json()["slots"]
    assert [s["key"] for s in slots] == ["r1|2026-10-12|09:00", "r1|2026-10-14|09:00", "r1|2026-10-16|09:00"]
    assert {s["state"] for s in slots} == {"planned"} and slots[0]["at"] == "2026-10-12T08:00:00+00:00"
    key = slots[1]["key"]
    moved = bank.client.put(f"/api/socials/plans/{plan['id']}/slots/{key}", json={"to": "2026-10-15T11:00:00Z"})
    assert moved.status_code == 200 and moved.json()["slot"]["moved"] is True
    back = bank.client.put(f"/api/socials/plans/{plan['id']}/slots/{key}", json={"to": None})
    assert back.json()["slot"]["at"] == "2026-10-14T08:00:00+00:00"
    skipped = bank.client.put(f"/api/socials/plans/{plan['id']}/slots/{key}", json={"skip": True})
    assert skipped.json()["skipped"] is True
    after = bank.client.get(f"/api/socials/plans/{plan['id']}/slots", params=window).json()["slots"]
    assert key not in [s["key"] for s in after]
    too_far = bank.client.put(f"/api/socials/plans/{plan['id']}/slots/{slots[0]['key']}", json={"to": "2026-12-01T09:00:00Z"})
    assert too_far.status_code == 422


def test_a_made_slot_moves_with_its_post_never_here(bank):
    plan = _create(bank)
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title="Made", content_hash="0" * 64,
                      campaign_id=uuid.UUID(plan["id"]), slot_key="r1|2026-10-12|09:00", status="needs_approval")
    bank.session.add(post)
    bank.session.commit()
    window = {"start": "2026-10-12T00:00:00Z", "end": "2026-10-13T00:00:00Z"}
    (slot,) = bank.client.get(f"/api/socials/plans/{plan['id']}/slots", params=window).json()["slots"]
    assert slot["state"] == "made" and slot["post"]["id"] == str(post.id)
    assert bank.client.put(f"/api/socials/plans/{plan['id']}/slots/{slot['key']}", json={"skip": True}).status_code == 409


def test_a_slots_window_is_bounded(bank):
    plan = _create(bank)
    resp = bank.client.get(f"/api/socials/plans/{plan['id']}/slots", params={"start": "2026-10-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"})
    assert resp.status_code == 422


def test_another_workspaces_plan_is_not_found(bank):
    plan = _create(bank)
    bank.ctx = _ctx(WS_B)
    assert bank.client.get(f"/api/socials/plans/{plan['id']}").status_code == 404
    assert bank.client.put(f"/api/socials/plans/{plan['id']}", json={"goal": "x"}).status_code == 404
    assert bank.client.get("/api/socials/plans").json()["plans"] == []
