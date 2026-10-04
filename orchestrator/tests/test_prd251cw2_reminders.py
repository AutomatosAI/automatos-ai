"""PRD-251C Wave 2, US-C206 — the evening-before reminder.

On the S0.3b API harness (SQLite), the tick's reminder pass run directly. Pinned:

* from the plan's ``make.remind_at`` (20:00 unless set), one reminder per plan per day counts
  tomorrow's posts still waiting for a person, linked to the week's review;
* none before that time, none twice the same evening, none when tomorrow's posts are approved,
  and none for another day's posts.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.socials import plan_notify  # noqa: E402
from services import socials_plan_reminders as reminders  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc


def _at(*args):
    return datetime(*args, tzinfo=UTC)


@pytest.fixture
def evening(bank, monkeypatch):  # noqa: F811
    told = []
    monkeypatch.setattr(plan_notify, "notify_review", lambda ws, plan_id, event, title: told.append((event[0], title)))
    bank.told, bank.factory = told, sessionmaker(bind=bank.session.get_bind())
    return bank


def _post(api, plan, planned_for, status="needs_approval"):
    api.session.add(SocialPost(
        id=uuid.uuid4(), workspace_id=WS_A, created_by="the plan", title="Planned", content_hash="0" * 64,
        campaign_id=uuid.UUID(plan["id"]), planned_for=planned_for, status=status, format="text",
    ))
    api.session.commit()


def test_one_reminder_the_evening_before_counts_tomorrows_waiting_posts(evening):
    plan = _create_plan(evening, timezone="UTC")
    _post(evening, plan, _at(2026, 10, 19, 9))
    _post(evening, plan, _at(2026, 10, 19, 12), status="changes_requested")
    _post(evening, plan, _at(2026, 10, 19, 15), status="scheduled")  # approved already
    _post(evening, plan, _at(2026, 10, 20, 9))  # another day's
    assert reminders.remind_due(evening.factory, _at(2026, 10, 18, 19, 59)) == 0  # before 20:00
    assert reminders.remind_due(evening.factory, _at(2026, 10, 18, 20, 30)) == 1
    assert evening.told == [("social_plan_reminder", "Countdown to Lisbon: 2 posts for tomorrow still wait for approval")]
    assert reminders.remind_due(evening.factory, _at(2026, 10, 18, 21, 30)) == 0  # once that evening


def test_no_reminder_when_tomorrows_posts_are_approved_and_the_time_is_the_plans(evening):
    plan = _create_plan(evening, timezone="UTC", make={"rhythm": "weekly", "remind_at": "18:00"})
    _post(evening, plan, _at(2026, 10, 19, 9), status="scheduled")
    assert reminders.remind_due(evening.factory, _at(2026, 10, 18, 20, 30)) == 0
    _post(evening, plan, _at(2026, 10, 20, 9))
    assert reminders.remind_due(evening.factory, _at(2026, 10, 19, 17, 59)) == 0
    assert reminders.remind_due(evening.factory, _at(2026, 10, 19, 18, 0)) == 1  # the plan's own time
    assert evening.told == [("social_plan_reminder", "Countdown to Lisbon: 1 post for tomorrow still waits for approval")]
