"""PRD-251C Wave 4, US-C403 — the weekly note.

Pinned:

* **Its contents from fixtures:** the week's posts with their numbers, the best and the worst,
  what the best share (format, time of day, the best one's topic), the plan's health and
  Auto's proposals, and the link to the plan's Posted view.
* **When:** with the weekly batch for a weekly plan (once its moment has come), on Mondays at
  the make time for a daily one.
* **One per plan per week** (the S0.3b harness and the plan tick's sender): the week's posts
  and numbers are read, the note is sent once, and a second pass that week sends nothing.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget  # noqa: E402
from modules.socials import plan_health, plan_notify, weekly_note  # noqa: E402
from services import socials_weekly_notes  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
UTC = timezone.utc


def _post(title, fmt, clock, engagement, topic=None):
    line = "no numbers yet" if engagement is None else f"{engagement} likes"
    return weekly_note.WeekPost(title=title, format=fmt, clock=clock, topic=topic, line=line, engagement=engagement)


def test_the_note_says_the_week_the_best_and_worst_what_the_best_share_and_what_to_do():
    posts = [
        _post("Stand map", "image", "09:30", 40, topic="Where we are"), _post("Speaker clip", "video", "18:00", 4),
        _post("Countdown", "image", "08:00", 30), _post("Late news", "image", "12:30", None),
    ]
    title, message = weekly_note.compose("Countdown to Lisbon", posts, ["A cap is close"], ["Post the image row at 08:00"],
                                         "https://app.test/deliverables?tab=socials&view=posted&plan=p1")
    assert title == "Countdown to Lisbon: 4 posts went out this week"
    lines = message.splitlines()
    assert lines[:4] == ["• Stand map: 40 likes", "• Speaker clip: 4 likes", "• Countdown: 30 likes", "• Late news: no numbers yet"]
    assert "Best: “Stand map”, 40 engagements." in lines and "Worst: “Speaker clip”, 4 engagements." in lines
    assert "What the best share: image posts, in the mornings; the best was on “Where we are”." in lines
    assert "Plan health: A cap is close." in lines
    assert "Auto proposes: Post the image row at 08:00. Open the plan to apply one." in lines
    assert lines[-1] == "Everything that went out: https://app.test/deliverables?tab=socials&view=posted&plan=p1"


def test_a_week_with_nothing_read_says_so():
    _, message = weekly_note.compose("Plan", [_post("Only one", "image", "09:00", None)], [], [], "u")
    assert "No numbers read yet: posts are read a day after they go out." in message.splitlines()


def _plan(rhythm, time="17:00"):
    return SimpleNamespace(starts_on=date(2026, 10, 1), ends_on=date(2026, 12, 31), timezone="UTC", cadence=[], slot_overrides={},
                           make={"rhythm": rhythm, "batch_day": "sun", "time": time})


def test_a_weekly_plans_note_comes_with_its_batch_and_a_daily_plans_on_monday():
    weekly = _plan("weekly")
    assert weekly_note.note_key(weekly, datetime(2026, 10, 18, 16, 59, tzinfo=UTC)) == "2026-W42"  # this week's batch, made last Sunday
    assert weekly_note.note_key(weekly, datetime(2026, 10, 18, 17, 0, tzinfo=UTC)) == "2026-W43"  # next week's batch is made: its note
    daily = _plan("daily", time="07:00")
    assert weekly_note.note_key(daily, datetime(2026, 10, 19, 6, 59, tzinfo=UTC)) is None  # Monday, before the make time
    assert weekly_note.note_key(daily, datetime(2026, 10, 19, 7, 0, tzinfo=UTC)) == "2026-W43"
    assert weekly_note.note_key(daily, datetime(2026, 10, 20, 9, 0, tzinfo=UTC)) is None  # Tuesday


@pytest.fixture
def notes(bank, monkeypatch):  # noqa: F811
    SocialPost.metadata.create_all(bank.session.get_bind(), tables=[SocialPostStat.__table__])
    sent = []
    monkeypatch.setattr(plan_notify, "notify_weekly_note", lambda ws, plan_id, title, message: sent.append((str(plan_id), title, message)))
    monkeypatch.setattr(plan_health, "health_for", lambda db, plan, workspace, now: [])
    bank.sent, bank.factory = sent, sessionmaker(bind=bank.session.get_bind())
    return bank


def test_one_note_per_plan_per_week_with_the_weeks_posts_and_numbers(notes):
    plan = _create_plan(notes, timezone="UTC", make={"rhythm": "daily", "time": "07:00"})
    now = datetime.now(UTC)
    monday = (now - timedelta(days=now.weekday())).replace(hour=8, minute=0, second=0, microsecond=0)
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title="Stand map", content_hash="0" * 64, status="published",
                      format="image", campaign_id=uuid.UUID(plan["id"]))
    target = SocialPostTarget(id=uuid.uuid4(), post_id=post.id, toolkit="twitter", post_kind="image", action_plan={},
                              idempotency_key=f"sp:{post.id}:twitter", status="published", remote_id="1850",
                              published_at=monday - timedelta(days=2))
    stat = SocialPostStat(workspace_id=WS_A, post_id=post.id, target_id=target.id, reading=1, read_at=monday, numbers={"likes": 12, "views": 300},
                          source_action="TWITTER_POST_LOOKUP_BY_POST_ID")
    notes.session.add_all([post, target, stat])
    notes.session.commit()
    assert socials_weekly_notes.send_due(notes.factory, monday) == 1
    ((plan_id, title, message),) = notes.sent
    assert (plan_id, title) == (plan["id"], "Countdown to Lisbon: 1 post went out this week")
    assert "• Stand map: 300 views · 12 likes" in message and f"view=posted&plan={plan['id']}" in message
    assert socials_weekly_notes.send_due(notes.factory, monday + timedelta(hours=3)) == 0  # once that week
    assert len(notes.sent) == 1
