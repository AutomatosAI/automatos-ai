"""PRD-264 EUAI-03 — the rule and the state behind the channel bots' AI notice.

Article 50(1): a person messaging a workspace's bot is told, on first contact,
that the reply comes from an AI and for whom. One function in
``modules/policy/ai_disclosure.py`` decides when (``channel_notice_due``): the
first message to the person on the connection, the first after the idle gap,
and any reply to "are you human?". ``channels/ai_notice.py`` keeps the
last-notice time in Redis with a TTL equal to the gap; when Redis can't be
read the notice is added. The principal is named only by a public name
(FR-15, ``public_principal_name``).

Nothing here is gated by ``AUTH_EDITION``; the ``edition`` fixture runs the
stateful cases in both editions to prove it.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config as app_config  # noqa: E402

HOUR = 3600
T0 = 1_800_000_000.0
SHOP = "Harbour Coffee"
WS = str(uuid4())


@pytest.fixture(params=("local", "saas"))
def edition(request, monkeypatch):
    monkeypatch.setattr(app_config, "AUTH_EDITION", request.param)
    return request.param


class FakeRedis:
    """The two calls the notice state makes: ``get`` and ``set(..., ex=)``."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def get(self, key: str) -> Optional[str]:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self.values = {**self.values, key: value}
        self.ttls = {**self.ttls, key: ex}
        return True


class BrokenRedis:
    def get(self, key: str) -> Optional[str]:
        raise ConnectionError("redis down")

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        raise ConnectionError("redis down")


class FakeDb:
    """Answers the two principal lookups: the business profile, then the workspace."""

    def __init__(self, company: Any = None, workspace_name: Any = None) -> None:
        self.company = company
        self.workspace_name = workspace_name
        self.rolled_back = False

    def execute(self, statement: Any, params: Any = None) -> Any:
        sql = str(statement)
        value = self.company if "business_profiles" in sql else self.workspace_name
        return _Result(None if value is None else (value,))

    def begin_nested(self) -> "_Savepoint":
        return _Savepoint(self)


class _Savepoint:
    """``Session.begin_nested()``: rolls back on an exception and lets it through."""

    def __init__(self, db: FakeDb) -> None:
        self.db = db

    def __enter__(self) -> "_Savepoint":
        return self

    def __exit__(self, exc_type: Any, *_rest: Any) -> bool:
        if exc_type is not None:
            self.db.rolled_back = True
        return False


class _Result:
    def __init__(self, row: Any) -> None:
        self._row = row

    def fetchone(self) -> Any:
        return self._row


class Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def notice_state(monkeypatch):
    """The real notice module on a fake Redis and a fake clock."""
    import channels.ai_notice as ai_notice

    redis = FakeRedis()
    clock = Clock(T0)
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: redis)
    monkeypatch.setattr(ai_notice, "_clock", clock)
    monkeypatch.setattr(app_config, "AI_DISCLOSURE_IDLE_HOURS", 24)
    return ai_notice, redis, clock


def _decide(ai_notice, db, *, person="person-1", inbound=None, ws=None, conn="conn-1"):
    return asyncio.run(ai_notice.notice_for(
        db, workspace_id=ws or WS, connection_id=conn, person_id=person, inbound_text=inbound,
    ))


def _send(ai_notice, db, **kwargs):
    """Decide, then record as a successful send does; returns the notice text."""
    notice = _decide(ai_notice, db, **kwargs)
    asyncio.run(ai_notice.record_notice_sent(notice))
    return notice.text


# ── the one rule (ai_disclosure.channel_notice_due) ────────────────────────

def test_first_contact_is_due():
    from modules.policy.ai_disclosure import channel_notice_due

    assert channel_notice_due(last_notice_at=None, now=T0, idle_seconds=24 * HOUR) is True


def test_within_the_gap_is_not_due_and_after_it_is():
    from modules.policy.ai_disclosure import channel_notice_due

    gap = 24 * HOUR
    assert channel_notice_due(last_notice_at=T0, now=T0 + gap - 1, idle_seconds=gap) is False
    assert channel_notice_due(last_notice_at=T0, now=T0 + gap, idle_seconds=gap) is True


@pytest.mark.parametrize("question", [
    "are you human?", "Are you a bot?", "Are you an AI", "am I talking to a real person?",
    "is this a bot", "Você é humano?", "estou a falar com um robô?", "¿Eres un bot?",
    "Êtes-vous un robot ?", "Bist du ein Mensch?", "Sind Sie eine KI?", "sei un bot?",
])
def test_asking_if_it_is_human_is_always_due(question):
    from modules.policy.ai_disclosure import asks_if_human, channel_notice_due

    assert asks_if_human(question) is True
    assert channel_notice_due(last_notice_at=T0, now=T0 + 1, idle_seconds=24 * HOUR,
                              inbound_text=question) is True


@pytest.mark.parametrize("text", [
    "What are your opening hours?", "Is this available in blue?", "I'm a human resources manager",
    "", None, 42,
])
def test_an_ordinary_message_is_not_a_human_question(text):
    from modules.policy.ai_disclosure import asks_if_human

    assert asks_if_human(text) is False


# ── the principal's public name (FR-15) ────────────────────────────────────

@pytest.mark.parametrize("company, workspace, expected", [
    (SHOP, "Ops Team", SHOP),                 # the business profile's name first
    ("  Harbour Coffee ", None, SHOP),        # trimmed
    (None, "Ops Team", "Ops Team"),           # then a workspace name the owner chose
    ("", "Ops Team", "Ops Team"),
    (None, "Ada Lovelace's Workspace", None),  # the auto-created form carries a person's name
    (None, "ada@example.com's Workspace", None),
    (None, "ada@example.com", None),           # an email is never a public name
    ("sales@harbour.example", None, None),
    (None, None, None),
    (7, ["x"], None),
])
def test_public_principal_name(company, workspace, expected):
    from modules.policy.ai_disclosure import public_principal_name

    assert public_principal_name(company, workspace) == expected


def test_the_notice_names_the_principal_or_says_ai_assistant():
    from modules.policy.ai_disclosure import channel_notice

    assert channel_notice(SHOP) == "AI assistant for Harbour Coffee."
    assert channel_notice(None) == "AI assistant."


def test_the_principal_is_looked_up_profile_first_then_workspace():
    from channels.ai_notice import principal_name

    assert principal_name(FakeDb(company=SHOP, workspace_name="Ops Team"), WS) == SHOP
    assert principal_name(FakeDb(company=None, workspace_name="Ops Team"), WS) == "Ops Team"
    assert principal_name(FakeDb(company=None, workspace_name="Ada's Workspace"), WS) is None
    assert principal_name(FakeDb(company=SHOP), "not-a-uuid") is None


def test_a_failed_principal_lookup_names_no_one_and_rolls_back():
    from channels.ai_notice import principal_name

    class Exploding(FakeDb):
        def execute(self, statement, params=None):
            raise RuntimeError("db gone")

    db = Exploding()
    assert principal_name(db, WS) is None
    assert db.rolled_back is True


# ── the state: first contact, the gap, the human question, Redis down ──────

def test_first_message_carries_the_notice_once(edition, notice_state):
    ai_notice, _, clock = notice_state
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db) == "AI assistant for Harbour Coffee."
    clock.now = T0 + 60
    assert _send(ai_notice, db) is None


def test_after_the_idle_gap_the_notice_comes_back(edition, notice_state):
    ai_notice, _, clock = notice_state
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db) is not None
    clock.now = T0 + 24 * HOUR - 1
    assert _send(ai_notice, db) is None
    clock.now = T0 + 24 * HOUR
    assert _send(ai_notice, db) == "AI assistant for Harbour Coffee."


def test_the_gap_follows_the_setting(notice_state, monkeypatch):
    ai_notice, redis, clock = notice_state
    monkeypatch.setattr(app_config, "AI_DISCLOSURE_IDLE_HOURS", 2)
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db) is not None
    assert set(redis.ttls.values()) == {2 * HOUR}
    clock.now = T0 + 2 * HOUR
    assert _send(ai_notice, db) is not None


def test_are_you_human_brings_the_notice_back_within_the_gap(edition, notice_state):
    ai_notice, _, clock = notice_state
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db) is not None
    clock.now = T0 + 60
    assert _send(ai_notice, db, inbound="are you a real person?") == "AI assistant for Harbour Coffee."


def test_the_state_is_per_person_and_per_connection(notice_state):
    ai_notice, _, _ = notice_state
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db, person="ana") is not None
    assert _send(ai_notice, db, person="ben") is not None
    assert _send(ai_notice, db, person="ana", conn="conn-2") is not None
    assert _send(ai_notice, db, person="ana") is None


def test_the_key_carries_no_raw_person_id(notice_state):
    ai_notice, redis, _ = notice_state

    _send(ai_notice, FakeDb(), person="+351912345678")
    assert redis.values
    assert all("+351912345678" not in key for key in redis.values)


def test_no_record_until_the_message_went_out(notice_state):
    """A failed send never uses up the notice: only record_notice_sent writes."""
    ai_notice, redis, _ = notice_state
    db = FakeDb(company=SHOP)

    assert _decide(ai_notice, db).text is not None
    assert redis.values == {}
    assert _decide(ai_notice, db).text is not None


def test_redis_not_configured_adds_the_notice_every_time(edition, notice_state, monkeypatch):
    ai_notice, _, _ = notice_state
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: None)
    db = FakeDb(company=SHOP)

    assert _send(ai_notice, db) == "AI assistant for Harbour Coffee."
    assert _send(ai_notice, db) == "AI assistant for Harbour Coffee."


def test_redis_failing_adds_the_notice(edition, notice_state, monkeypatch):
    ai_notice, _, _ = notice_state
    monkeypatch.setattr(ai_notice, "_get_redis", lambda: BrokenRedis())
    db = FakeDb()

    assert _send(ai_notice, db) == "AI assistant."
    assert _send(ai_notice, db) == "AI assistant."


def test_an_unreadable_record_adds_the_notice(notice_state):
    ai_notice, redis, _ = notice_state
    key = ai_notice.notice_key(WS, "conn-1", "person-1")
    redis.values = {key: "not-a-time"}

    assert _send(ai_notice, FakeDb()) == "AI assistant."


def test_the_notice_is_the_first_paragraph():
    from channels.ai_notice import ChannelNotice

    assert ChannelNotice(key="k", text="AI assistant.").apply("Hi!") == "AI assistant.\n\nHi!"
    assert ChannelNotice(key="k", text=None).apply("Hi!") == "Hi!"


def test_idle_hours_is_a_setting_named_in_the_config_surface():
    import json

    assert isinstance(app_config.AI_DISCLOSURE_IDLE_HOURS, int)
    surface = json.loads((ORCH_ROOT / "reports" / "config-surface.json").read_text(encoding="utf-8"))
    assert "AI_DISCLOSURE_IDLE_HOURS" in surface["settings"]
