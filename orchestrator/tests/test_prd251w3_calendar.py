"""PRD-251 Wave 3, US-307 (S3.1b, D10) — scheduled social posts in the Command Center calendar.

The sixth source of the schedule feed (``services/activity_social_items.py``): each
scheduled post of the caller's workspace with a slot in the window is one item,
``social-<post_id>``, with its title, its slot (UTC ISO), its id and its own timezone.
Another workspace's post, a post that is not scheduled and a slot outside the window
never appear; with Socials off for the workspace there are no social items. On the
CI Postgres the item reaches ``GET /api/activity/schedule``'s service beside the five
other sources.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialPostTarget  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.socials import settings as socials_settings  # noqa: E402
from services.activity_service import ActivityService  # noqa: E402
from services.activity_social_items import social_post_items  # noqa: E402

WS, OTHER = uuid.uuid4(), uuid.uuid4()
NOW = datetime.now(timezone.utc)
HORIZON = NOW + timedelta(days=7)


@pytest.fixture
def db(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    copies = sa.MetaData()
    api_harness._sqlite_copy(Workspace.__table__, copies)
    copies.create_all(engine)
    SocialPost.metadata.create_all(engine, tables=[SocialPost.__table__, SocialPostTarget.__table__])
    session = sessionmaker(bind=engine)()
    for ws in (WS, OTHER):
        session.add(Workspace(id=ws, name=f"w-{ws.hex[:4]}", plan="basic", plan_limits={}, settings={"socials": {"enabled": True}},
                              onboarding={}, created_at=NOW, updated_at=NOW))
    session.commit()
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    yield session
    session.close()


def _post(db, workspace_id, status, slot, title="Harvest Club opens", tz="Europe/Lisbon"):
    post = service.create_draft(db, workspace_id=workspace_id, created_by="a", title=title, copy={"base": "x"})
    post.status, post.scheduled_for, post.timezone = status, slot, tz
    db.commit()
    return post.id


def test_a_scheduled_post_is_a_social_item_with_its_slot_and_timezone(db):
    slot = NOW + timedelta(days=2, hours=3)
    post_id = _post(db, WS, service.SCHEDULED, slot)

    (item,) = social_post_items(db, WS, NOW, HORIZON)

    assert item == {
        "id": f"social-{post_id}",
        "post_id": str(post_id),
        "type": "social",
        "name": "Harvest Club opens",
        "next_run_at": slot.isoformat(),
        "timezone": "Europe/Lisbon",
        "frequency": "One-off",
        "agent_name": None,
        "agent_id": None,
        "recurrence": {"cron_expression": None, "interval_minutes": None, "timezone": "Europe/Lisbon", "active_hours": None},
    }
    assert ActivityService(db, WS)._social_post_items(NOW, HORIZON) == [item]


def test_another_workspace_s_post_an_unscheduled_post_and_a_slot_outside_the_window_never_appear(db):
    slot = NOW + timedelta(days=1)
    mine = _post(db, WS, service.SCHEDULED, slot)
    _post(db, OTHER, service.SCHEDULED, slot, title="Not yours")
    for status in (service.APPROVED, service.MISSED, service.PUBLISHED, service.PUBLISHING):
        _post(db, WS, status, slot, title=status)
    _post(db, WS, service.SCHEDULED, NOW + timedelta(days=9), title="Too far")
    _post(db, WS, service.SCHEDULED, NOW - timedelta(hours=1), title="Already due")

    assert [item["post_id"] for item in social_post_items(db, WS, NOW, HORIZON)] == [str(mine)]


def test_with_socials_off_there_are_no_social_items(db, monkeypatch):
    _post(db, WS, service.SCHEDULED, NOW + timedelta(days=1))
    db.get(Workspace, WS).settings = {"socials": {"enabled": False}}
    db.commit()
    assert social_post_items(db, WS, NOW, HORIZON) == []

    db.get(Workspace, WS).settings = {"socials": {"enabled": True}}
    db.commit()
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "false")  # the master switch
    assert social_post_items(db, WS, NOW, HORIZON) == []


def test_a_post_with_no_timezone_shows_in_utc(db):
    post_id = _post(db, WS, service.SCHEDULED, NOW + timedelta(days=1), tz=None)
    (item,) = social_post_items(db, WS, NOW, HORIZON)
    assert item["post_id"] == str(post_id) and item["timezone"] == "UTC"


@pytest.mark.integration
def test_the_schedule_feed_carries_the_social_item_on_postgres(monkeypatch):
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM social_posts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI"):
            raise
        pytest.skip(f"the Postgres check needs the test database: {exc}")
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    session = sessionmaker(bind=engine)()
    workspace_id = uuid.uuid4()
    try:
        session.add(Workspace(id=workspace_id, name="w3-calendar", plan="basic", plan_limits={}, settings={"socials": {"enabled": True}}))
        session.commit()
        post_id = _post(session, workspace_id, service.SCHEDULED, NOW + timedelta(days=1, hours=2))
        feed = ActivityService(session, workspace_id).get_schedule(range_days=7)
        social = [item for item in feed["scheduled"] if item["type"] == "social"]
        assert [item["id"] for item in social] == [f"social-{post_id}"]
    finally:
        session.rollback()
        session.execute(sa.text("DELETE FROM social_posts WHERE workspace_id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
        session.execute(sa.text("DELETE FROM workspaces WHERE id = CAST(:ws AS uuid)"), {"ws": str(workspace_id)})
        session.commit()
        session.close()
        engine.dispose()
