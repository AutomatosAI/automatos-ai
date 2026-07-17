"""PRD-205 S5 -- the watcher speaks: verdict/action/escalation reach chat.

The shared seam (``dispatch_watch_notification``) is the single hook: every
verdict/action/escalation producer (decider close, decider action, ticker
sweep, escalate_watch_now) flows through it, so chat delivery lands once.

Covers: verdict posts to the origin chat; no-origin falls back to the
creator's Auto thread (clerk passthrough); the bell still fires and its
behavior/kwargs are unchanged; ordering + independence (a raising bell does
not silence the chat, a raising chat does not cost the bell or the return
value); non-conversational events (approval_pending) stay bell-only;
notify_watch_verdict end-to-end through the seam.

Pure unit tests -- mock db + SimpleNamespace watches (the seam is
getattr-defensive by contract). BOTH seams are patched: the dispatcher
(the ``notifications`` table is migration-only -- PRD-204 lesson) and
``deliver_background_message`` (never a real session from a unit test).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def capture_notifications(monkeypatch):
    """Capture NotificationDispatcher.dispatch at the seam (never SELECT
    notifications -- the table is migration-only)."""
    from core.services.notification_dispatcher import NotificationDispatcher

    sent = []

    async def _capture(self, event_type, title, message=None, **kwargs):
        sent.append({"event_type": event_type, "title": title,
                     "message": message, **kwargs})
        return {"dispatched_to": ["in_app"]}

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _capture)
    return sent


@pytest.fixture
def capture_chat(monkeypatch):
    """Capture deliver_background_message at the messenger seam."""
    import services.chat_messenger as cm

    delivered = []

    def _capture(**kwargs):
        delivered.append(kwargs)
        return None

    monkeypatch.setattr(cm, "deliver_background_message", _capture)
    return delivered


def _watch(**overrides):
    base = {
        "id": uuid.uuid4(),
        "workspace_id": uuid.uuid4(),
        "created_by": "user_creator",
        "origin_chat_id": None,
        "title": "Watch: quarterly report",
        "quality_threshold": 0.8,
        "final_score": 0.84,
        "final_verdict": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _mock_db_without_user():
    db = MagicMock()
    q = MagicMock()
    q.filter.return_value = q
    q.first.return_value = None
    db.query.return_value = q
    return db


def _dispatch(db, watch, event_type, message="the composed prose", title="t"):
    from services.watch_notifications import dispatch_watch_notification

    return asyncio.run(
        dispatch_watch_notification(
            db, watch, event_type=event_type, title=title, message=message
        )
    )


# ---------------------------------------------------------------------------
# Delivery targeting
# ---------------------------------------------------------------------------


def test_verdict_posts_to_origin_chat(capture_notifications, capture_chat):
    origin = uuid.uuid4()
    watch = _watch(origin_chat_id=origin)

    ok = _dispatch(
        _mock_db_without_user(), watch, "watch_verdict",
        message="Run scored 8.4/10 against a bar of 8.0/10. Tightened SCRIBE.",
    )

    assert ok is True
    assert len(capture_notifications) == 1  # bell unchanged
    assert len(capture_chat) == 1
    sent = capture_chat[0]
    assert sent["chat_id"] == str(origin)
    assert sent["workspace_id"] == watch.workspace_id
    assert sent["text"].startswith("Run scored 8.4/10")  # prose, not re-composed
    assert sent["source"]["origin"] == "watcher"
    assert sent["source"]["label"]  # the badge label rides along
    assert sent["link_type"] == "watch"
    assert sent["link_id"] == str(watch.id)
    assert sent["clerk_user_id"] == "user_creator"


def test_no_origin_falls_back_to_creator_auto_thread(capture_notifications, capture_chat):
    watch = _watch(origin_chat_id=None)
    _dispatch(_mock_db_without_user(), watch, "watch_verdict")

    assert len(capture_chat) == 1
    assert capture_chat[0]["chat_id"] is None  # messenger resolves the Auto thread
    assert capture_chat[0]["clerk_user_id"] == "user_creator"


@pytest.mark.parametrize("event_type", ["watch_action", "watch_escalation"])
def test_actions_and_escalations_also_speak(event_type, capture_notifications, capture_chat):
    watch = _watch()
    ok = _dispatch(_mock_db_without_user(), watch, event_type, message="because reasons")
    assert ok is True
    assert len(capture_notifications) == 1
    assert len(capture_chat) == 1
    assert capture_chat[0]["text"] == "because reasons"


def test_non_conversational_events_stay_bell_only(capture_notifications, capture_chat):
    watch = _watch()
    ok = _dispatch(_mock_db_without_user(), watch, "approval_pending")
    assert ok is True
    assert len(capture_notifications) == 1
    assert capture_chat == []


def test_message_none_falls_back_to_title(capture_notifications, capture_chat):
    watch = _watch()
    _dispatch(
        _mock_db_without_user(), watch, "watch_escalation",
        message=None, title="Watch escalated: quarterly report",
    )
    assert capture_chat[0]["text"] == "Watch escalated: quarterly report"


# ---------------------------------------------------------------------------
# Ordering + independence (each surface fail-soft on its own)
# ---------------------------------------------------------------------------


def test_bell_failure_does_not_silence_the_chat(monkeypatch, capture_chat):
    from core.services.notification_dispatcher import NotificationDispatcher

    async def _boom(self, *a, **k):
        raise RuntimeError("bell wiring down")

    monkeypatch.setattr(NotificationDispatcher, "dispatch", _boom)

    ok = _dispatch(_mock_db_without_user(), _watch(), "watch_verdict")
    assert ok is False  # the seam still reports the bell failure
    assert len(capture_chat) == 1  # ...but the verdict reached the chat


def test_chat_failure_does_not_cost_the_bell(monkeypatch, capture_notifications):
    import services.chat_messenger as cm

    def _boom(**kwargs):
        raise RuntimeError("messenger exploded")

    monkeypatch.setattr(cm, "deliver_background_message", _boom)

    ok = _dispatch(_mock_db_without_user(), _watch(), "watch_verdict")
    assert ok is True  # bell contract unchanged
    assert len(capture_notifications) == 1


def test_workspaceless_watch_fires_nothing(capture_notifications, capture_chat):
    watch = _watch(workspace_id=None)
    ok = _dispatch(_mock_db_without_user(), watch, "watch_verdict")
    assert ok is False
    assert capture_notifications == []
    assert capture_chat == []


# ---------------------------------------------------------------------------
# End-to-end through notify_watch_verdict (the S6-of-PRD-204 producer edge)
# ---------------------------------------------------------------------------


def test_notify_watch_verdict_delivers_composed_prose(capture_notifications, capture_chat):
    from services.watch_notifications import notify_watch_verdict

    origin = uuid.uuid4()
    watch = _watch(origin_chat_id=origin)
    ok = asyncio.run(
        notify_watch_verdict(
            _mock_db_without_user(),
            watch,
            score=0.84,
            explanation="The report was too operational; tightened and reran.",
            passed=True,
        )
    )
    assert ok is True
    assert len(capture_notifications) == 1
    assert len(capture_chat) == 1
    text = capture_chat[0]["text"]
    # The chat gets the SAME prose the bell got (score x10 display + reason).
    assert capture_notifications[0]["message"] == text
    assert "8.4/10" in text
    assert "tightened and reran" in text.lower()
    assert capture_chat[0]["chat_id"] == str(origin)
