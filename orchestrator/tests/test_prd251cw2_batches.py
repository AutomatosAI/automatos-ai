"""PRD-251C Wave 2 — a plan's rhythm (US-C201).

Pinned:

* ``make.rhythm`` (daily, weekly or monthly), ``make.batch_day`` and ``make.batch_date`` are
  checked; a plan saved before PRD-251C has no rhythm and stays daily;
* a new plan makes its week on Sunday at 17:00 (O1, O3);
* a save changes what it sends: the rest of ``make``, and what the make tick recorded
  (notices sent, batches made), stay.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialCampaign  # noqa: E402
from modules.socials import plans  # noqa: E402
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
