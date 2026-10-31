"""PRD-268 US-C05 (§6.2.2, FR-5) — failing streaks: one card per streak, the interval doubling, the reset.

Proven here:

* **the error class** a failed check-in records (``services.proactive.failures``): a provider's
  refusal is ``http_<status>`` (from the SDK error's ``status_code``, the one it was raised from,
  or its message), out of credit ``out_of_credit``, a timeout ``timeout``, else the class name;
* **the card's copy** (``services.proactive.streak_text``): built from structured fields, from
  tables in English and Portuguese (Brazil), within the title and detail limits;
* **the hold** (``failing_streaks.holds``): from the third failure the next check-in waits for
  the doubled interval, never less than it but for the scheduler's few minutes of grace, so a
  tick a moment early is still due; the base interval is the heartbeat's own, as the scheduler
  reads it;
* **against Postgres** (the suite's rolled-back transaction, joined by SAVEPOINT; every tick
  through the bridge's job in live mode, with the clock frozen at the tick; the model scripted
  to fail as VECTOR's provider did; the bell recorded at the dispatcher, the CI database having
  no notifications table): the 26 Oct VECTOR case replayed from
  ``tests/fixtures/prd268_vector_26oct.json``. Its 12 failing ticks make one suggestion and
  zero board tickets, running at 09, 10, 11, 13 and 17:00 as the interval doubles from the
  third failure; the card says so; the first success expires it and the hour is back. Also:
  a failure of another class is its own streak, and a streak it interrupted counts afresh; an
  open card stays open while the agent fails; a hidden card is never made again; another
  agent's card is left alone; an agent that is gone, or Auto's workspace check-in, makes no
  card; bookkeeping that fails never fails the check-in; shadow keeps the streak beside the tick.

Each fire's run row is stamped with its tick's time after the fire: the suite's one transaction
would give every row the transaction's own start, where production commits each row as it runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.proactive.dedupe import agent_error_key, agent_error_prefix
from services.proactive import failing_streaks, heartbeat_bell, heartbeat_bridge, streak_text
from services.proactive.failing_streaks import base_minutes, holds
from services.proactive.failures import failure_class
from services.proactive.heartbeat_bridge import agent_job, orchestrator_job

VECTOR = json.loads((Path(__file__).parent / "fixtures" / "prd268_vector_26oct.json").read_text(encoding="utf-8"))
MODEL = "claude-haiku-5-5"
HB = {"enabled": True, "mode": "check_in", "report_to": "auto", "interval_minutes": VECTOR["interval_minutes"],
      "prompt": VECTOR["prompt"]}


def _at(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


class ProviderRefused(Exception):
    """A provider SDK's refusal: its HTTP status, and its message as the SDK words it."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class APITimeoutError(Exception):
    """Named as the SDKs name a timeout."""


def _vector_refusal() -> ProviderRefused:
    return ProviderRefused(VECTOR["provider_error"], VECTOR["provider_status"])


# ── The error class ─────────────────────────────────────────────────────────

def _raised_from(outer: Exception, inner: Exception) -> Exception:
    """``outer`` as ``raise outer from inner`` leaves it."""
    outer.__cause__ = inner
    return outer


@pytest.mark.parametrize("exc, named", [
    (ProviderRefused("Error code: 401 - {'error': {'code': 401}}", 401), "http_401"),
    (ProviderRefused("refused", 503), "http_503"),
    (RuntimeError("Error code: 500 - {'error': {'message': 'upstream'}}"), "http_500"),
    (ValueError("Model 'x' is unavailable or has been removed."), "ValueError"),
    (TimeoutError(), "timeout"),
    (APITimeoutError("Request timed out."), "timeout"),
    (RuntimeError("Error code: 402 - {'error': {'message': 'This request requires more credits'}}"),
     "out_of_credit"),
    (RuntimeError("provider refused the key (HTTP 401)"), "RuntimeError"),  # words, not the SDK's form
    (RuntimeError("boom"), "RuntimeError"),
])
def test_a_failed_check_in_is_named_by_what_failed(exc, named):
    assert failure_class(exc) == named


def test_a_refusal_wrapped_by_the_manager_keeps_its_status():
    wrapped = _raised_from(ValueError("Model 'x' is unavailable"), ProviderRefused("gone", 404))
    timed_out = _raised_from(RuntimeError("the call failed"), APITimeoutError("Request timed out."))

    assert failure_class(wrapped) == "http_404" and failure_class(timed_out) == "timeout"


def test_the_providers_status_is_read_from_its_message_as_before():
    from core.llm.credit import PROVIDER_REFUSED_TEXT, plain_failure, provider_status

    assert provider_status(VECTOR["provider_error"]) == "401"
    assert provider_status("the provider said no (HTTP 401)") is None and provider_status(None) is None
    assert plain_failure(VECTOR["provider_error"]) == PROVIDER_REFUSED_TEXT.format(status="401")


# ── The copy ────────────────────────────────────────────────────────────────

SINCE = _at("2026-10-26T09:00:00+00:00")


def test_the_streak_text_has_both_tables_and_names_each_failure():
    assert set(streak_text.STRINGS["en"]) == set(streak_text.STRINGS["pt-BR"])
    title = streak_text.streak_title

    assert title("VECTOR", "http_401") == "VECTOR's check-ins fail: the AI provider refuses its key (HTTP 401)"
    assert title("VECTOR", "http_500") == "VECTOR's check-ins fail: the AI provider refuses the request (HTTP 500)"
    assert title("VECTOR", "out_of_credit") == "VECTOR's check-ins fail: the AI credit ran out"
    assert title("VECTOR", "timeout") == "VECTOR's check-ins fail: they time out"
    assert title("VECTOR", "cost_ceiling") == "VECTOR's check-ins fail: they pass their cost ceiling"
    assert title("VECTOR", "RuntimeError") == "VECTOR's check-ins fail (RuntimeError)"
    assert title("VECTOR", "http_401", "pt-BR") == "Os check-ins de VECTOR falham: o provedor de IA recusa a chave (HTTP 401)"


def test_the_detail_says_since_when_how_often_and_from_the_third_failure_how_slowly():
    detail = streak_text.streak_detail

    assert detail("http_401", 1, SINCE, 60, 60) == (
        "Its check-in at 2026-10-26 09:00 UTC failed (http_401). No ticket was filed. "
        "Start gives Auto a ticket to find out why and fix it.")
    assert "It now checks in" not in detail("http_401", 2, SINCE, 60, 60)
    assert detail("http_401", 3, SINCE, 60, 120) == (
        "Since 2026-10-26 09:00 UTC, its last 3 check-ins failed (http_401). No ticket was filed for them. "
        "It now checks in every 2 hours instead of every hour, until a check-in works. "
        "Start gives Auto a ticket to find out why and fix it.")
    assert "a cada 2 horas, em vez de a cada hora" in detail("http_401", 3, SINCE, 60, 120, "pt-BR")
    assert streak_text.works_again_detail("VECTOR", "http_401", 5) == (
        "VECTOR's check-ins work again, after 5 failures in a row (http_401).")


@pytest.mark.parametrize("minutes, words", [(60, "hour"), (120, "2 hours"), (1440, "24 hours"), (90, "90 minutes"),
                                            (30, "30 minutes")])
def test_an_interval_is_said_in_words(minutes, words):
    assert streak_text.duration(minutes) == words


def test_the_copy_stays_within_the_cards_limits_whatever_the_name_or_class():
    from core.models.proactive import DETAIL_LENGTH, TITLE_LENGTH

    long_name, long_class = "V" * 300, "E" * 64
    assert len(streak_text.streak_title(long_name, long_class)) <= TITLE_LENGTH
    assert streak_text.streak_title("VECTOR", "Bad\u202eError") == "VECTOR's check-ins fail (BadError)"
    assert streak_text.streak_title("VEC\nTOR\u202e\u200b", "timeout") == "VEC TOR's check-ins fail: they time out"
    assert len(streak_text.streak_detail(long_class, 10 ** 6, SINCE, 1, 1440)) <= DETAIL_LENGTH
    assert len(streak_text.works_again_detail(long_name, long_class, 10 ** 6)) <= DETAIL_LENGTH


# ── The hold ────────────────────────────────────────────────────────────────

def _failures(last: str, count: int, error_class: str = "http_401") -> list:
    """An agent's newest finished runs, newest first: ``count`` failures, the newest at ``last``."""
    newest = _at(last)
    return [NS(outcome="error", error_class=error_class, window_until=newest - timedelta(hours=index),
               created_at=newest - timedelta(hours=index)) for index in range(count)]


def test_from_the_third_failure_the_next_check_in_waits_for_the_doubled_interval():
    three = _failures("2026-10-26T11:00:05+00:00", 3)  # the third failure started five seconds late

    assert not holds(three, 60, _at("2026-10-26T13:00:00+00:00"))  # its 120 minutes, a moment early
    assert holds(three, 60, _at("2026-10-26T12:00:00+00:00"))
    assert not holds(three[:2], 60, _at("2026-10-26T11:01:00+00:00"))  # before the third: never held
    assert holds(three, 60, _at("2026-10-26T12:54:00+00:00"))  # past the grace: early is early


def test_a_success_newest_or_another_class_holds_nothing():
    worked = [NS(outcome="silent", error_class=None, window_until=_at("2026-10-26T12:00:00+00:00"),
                 created_at=_at("2026-10-26T12:00:00+00:00")), *_failures("2026-10-26T11:00:00+00:00", 5)]
    mixed = [*_failures("2026-10-26T12:00:00+00:00", 1, "timeout"), *_failures("2026-10-26T11:00:00+00:00", 5)]

    assert not holds(worked, 60, _at("2026-10-26T12:30:00+00:00"))
    assert not holds(mixed, 60, _at("2026-10-26T12:30:00+00:00")) and not holds([], 60, SINCE)


def test_a_check_in_never_runs_before_its_interval_whatever_the_base():
    seven_hundred = _failures("2026-10-26T00:00:00+00:00", 4)  # next_interval(700, 4) is the 24 hours
    daily = _failures("2026-10-26T09:00:00+00:00", 5)          # a daily heartbeat stays daily

    assert holds(seven_hundred, 700, _at("2026-10-26T00:00:00+00:00") + timedelta(minutes=1400))
    assert not holds(seven_hundred, 700, _at("2026-10-26T00:00:00+00:00") + timedelta(minutes=2100))
    assert holds(daily, 1440, _at("2026-10-27T08:00:00+00:00"))
    assert not holds(daily, 1440, _at("2026-10-27T08:59:58+00:00"))


@pytest.mark.parametrize("hb_config, minutes", [
    ({}, 60), (None, 60), ({"interval_minutes": 240}, 240), ({"interval_minutes": "30"}, 30),
    ({"interval_minutes": 0}, 60), ({"interval_minutes": -5}, 60), ({"interval_minutes": "often"}, 60),
    ({"interval_minutes": True}, 60),
])
def test_the_base_interval_is_the_heartbeats_own_as_the_scheduler_reads_it(hb_config, minutes):
    assert base_minutes(hb_config) == minutes


def test_an_agents_streak_keys_share_its_prefix_and_no_other_agents():
    assert agent_error_key(578, "http_401") == "agent:578:error:http_401"
    assert agent_error_key(578, "Timeout").startswith(agent_error_prefix(578))
    assert not agent_error_key(57, "http_401").startswith(agent_error_prefix(5))


# ── Against Postgres ────────────────────────────────────────────────────────

@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the runner, the streak and the bell commit, and none may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _agent_row(db, ws: UUID, name: str, configuration: dict, system: bool = False) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, is_system_agent, configuration) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', :s, CAST(:c AS json)) RETURNING id"),
        {"n": name, "w": str(ws), "s": system, "c": json.dumps(configuration)}).scalar())


@pytest.fixture
def shop(db_session):
    """A workspace with check-ins on, its owner, Auto, and VECTOR: a ``check_in`` agent on an
    hourly heartbeat that reports to Auto."""
    name = f"prd268st_{uuid4().hex[:10]}"
    owner = int(db_session.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": name, "e": f"{name}@t.test", "c": name}).scalar())
    ws = uuid4()
    db_session.execute(text(
        "INSERT INTO workspaces (id, name, slug, plan, owner_id, settings) VALUES (CAST(:id AS uuid), "
        "'prd268 streaks', :slug, 'pro', :o, CAST(:settings AS jsonb))"),
        {"id": str(ws), "slug": f"prd268-streaks-{ws.hex[:10]}", "o": owner,
         "settings": json.dumps({"proactive": {"enabled": True}})})
    auto = _agent_row(db_session, ws, "Auto", {}, system=True)
    agent = _agent_row(db_session, ws, VECTOR["agent"], {"heartbeat": HB})
    db_session.commit()
    return NS(db=db_session, ws=ws, owner=owner, auto=auto, agent=agent)


class FakeLLM:
    """The heartbeat model: it fails with ``failure`` while one is set, else answers that
    nothing is new. ``calls`` holds the clock at each call."""

    def __init__(self, rig):
        self.config, self.rig, self.calls = NS(model=MODEL), rig, []
        self.failure = _vector_refusal()

    async def generate_response(self, messages, tools=None):
        self.calls.append(self.rig.clock)
        if self.failure is not None:
            raise self.failure
        return NS(content="Nothing new.", tool_calls=[], usage={"input_tokens": 1000, "output_tokens": 20})

    def call_cost(self, input_tokens, output_tokens, reported_cost=None):
        return 0.001, "estimate"


async def no_delivery(db, run_id):
    """The runner's delivery, left out: these tests ring and count W1's heartbeat bell only."""
    from services.proactive.delivery import Delivery

    return Delivery()


@pytest.fixture
def live(shop, monkeypatch):
    """Platform mode live; every session the test's; the clock frozen at the tick; the model
    and the context build scripted; the tick, the report filer and the dispatcher recorded."""
    from core.llm import manager
    from core.services import notification_dispatcher
    from modules.context.result import ContextResult
    from modules.context.service import ContextService
    from modules.tools.discovery import platform_executor
    from services import heartbeat_report
    from services.heartbeat_service import HeartbeatService
    from services.proactive import runner

    rig = NS(mode="live", clock=_at(VECTOR["ticks"][0]), bells=[], ticks=[], reports=[], service=HeartbeatService())
    rig.llm = FakeLLM(rig)

    async def build_context(self, **kwargs):
        return ContextResult(system_prompt=f"SYSTEM\n{kwargs['task_description']}", tools=[])

    async def tick(agent_id, workspace_id, hb_config):
        rig.ticks.append(rig.clock)
        return {"status": "success"}

    async def report(*args, **kwargs):
        rig.reports.append(args)

    async def within_active_hours(hb_config, workspace_id=""):
        return True

    class Dispatcher:
        def __init__(self, db, workspace_id):
            pass

        async def dispatch(self, **kwargs):
            rig.bells.append({"at": rig.clock, **kwargs})
            return {"dispatched_to": ["in_app"]}

    for module in (runner, heartbeat_bridge, heartbeat_bell):
        monkeypatch.setattr(module, "_session", lambda: nullcontext(shop.db))
    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: rig.mode if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(ContextService, "build_context", build_context)
    monkeypatch.setattr(runner, "heartbeat_llm", lambda workspace_id, agent_id: rig.llm)
    monkeypatch.setattr(runner, "_now", lambda: rig.clock)
    # W2's delivery (the thread post and suggestion_ready) is proven in test_prd268w2_suggestion_delivery.py.
    monkeypatch.setattr(runner, "deliver_run", no_delivery)
    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "_full_autonomy", lambda self: False)
    monkeypatch.setattr(notification_dispatcher, "NotificationDispatcher", Dispatcher)
    monkeypatch.setattr(heartbeat_report, "auto_create_heartbeat_report", report)
    monkeypatch.setattr(rig.service, "_agent_tick", tick)
    monkeypatch.setattr(rig.service, "_is_within_active_hours", within_active_hours)
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        yield rig


def _stamp(shop, run_id: str, at: datetime) -> None:
    shop.db.execute(text("UPDATE proactive_runs SET created_at = :at WHERE id = CAST(:id AS uuid)"),
                    {"at": at, "id": run_id})
    shop.db.commit()


def _fire(shop, live, at: datetime, agent_id=None) -> dict:
    """One fire of the agent's heartbeat at ``at``, its run row stamped with that time."""
    live.clock = at
    result = asyncio.run(agent_job(live.service)(agent_id or shop.agent, str(shop.ws), dict(HB)))
    _stamp(shop, result["run_id"], at)
    return result


def _replay(shop, live, ticks) -> list:
    return [_fire(shop, live, _at(tick)) for tick in ticks]


def _cards(shop):
    from core.models.proactive import ProactiveSuggestion

    shop.db.expire_all()
    return shop.db.query(ProactiveSuggestion).filter(ProactiveSuggestion.workspace_id == shop.ws).order_by(
        ProactiveSuggestion.created_at, ProactiveSuggestion.subject_key).all()


def _count(shop, table: str) -> int:
    return int(shop.db.execute(text(f"SELECT count(*) FROM {table} WHERE workspace_id = CAST(:w AS uuid)"),  # noqa: S608 — fixed names
                               {"w": str(shop.ws)}).scalar())


def _hhmm(moments) -> list:
    return [moment.strftime("%H:%M") for moment in moments]


SCHEDULE_26_OCT = ["error", "error", "error", "skipped_cap", "error", "skipped_cap", "skipped_cap", "skipped_cap",
                   "error", "skipped_cap", "skipped_cap", "skipped_cap"]


def test_the_26_oct_vector_case_twelve_failing_ticks_make_one_suggestion_and_no_ticket(shop, live):
    first = _fire(shop, live, _at(VECTOR["ticks"][0]))
    [card] = _cards(shop)
    assert card.detail.startswith("Its check-in at 2026-10-26 09:00 UTC failed (http_401). No ticket was filed.")
    results = [first, *_replay(shop, live, VECTOR["ticks"][1:3])]
    [card] = _cards(shop)
    assert "It now checks in every 2 hours instead of every hour, until a check-in works." in card.detail
    results += _replay(shop, live, VECTOR["ticks"][3:])

    assert [result["outcome"] for result in results] == SCHEDULE_26_OCT
    assert {result["error_class"] for result in results} == {"http_401", "failing_streak"}
    assert _hhmm(live.llm.calls) == ["09:00", "10:00", "11:00", "13:00", "17:00"]  # one call per run that ran
    [card] = _cards(shop)
    assert (card.subject_key, card.source_type, card.status) == (f"agent:{shop.agent}:error:http_401", "streak", "open")
    assert (card.user_id, card.source_agent_id, card.occurrences) == (shop.owner, shop.agent, 5)
    assert (card.created_at, card.last_seen_at) == (_at(VECTOR["ticks"][0]), _at(VECTOR["ticks"][8]))
    assert str(card.run_id) == first["run_id"] and card.severity == "today"
    assert card.title == "VECTOR's check-ins fail: the AI provider refuses its key (HTTP 401)"
    assert card.detail.startswith("Since 2026-10-26 09:00 UTC, its last 5 check-ins failed (http_401).")
    assert "It now checks in every 8 hours instead of every hour" in card.detail
    failed = [result["run_id"] for result in results if result["outcome"] == "error"]
    assert card.evidence == [{"type": "run", "id": run_id} for run_id in reversed(failed)]
    assert card.proposed_action == {"kind": "board_task", "payload": {"agent_id": shop.auto}}
    # Where the tick filed a blocked card per failure (12), the check-ins filed none.
    assert VECTOR["board_cards_filed_by_the_tick"] == len(VECTOR["ticks"])
    assert _count(shop, "board_tasks") == 0 and _count(shop, "heartbeat_results") == 0
    assert live.ticks == [] and live.reports == []
    assert [(bell["at"], bell["status"]) for bell in live.bells] == [(_at(VECTOR["ticks"][0]), "error")]


def test_a_held_back_tick_is_a_skipped_run_with_no_model_call(shop, live):
    from core.models.proactive import ProactiveRun

    held = _replay(shop, live, VECTOR["ticks"][:4])[-1]

    shop.db.expire_all()
    run = shop.db.query(ProactiveRun).filter(ProactiveRun.id == UUID(held["run_id"])).one()
    assert (run.outcome, run.error_class, run.model, run.tokens, run.agent_id) == (
        "skipped_cap", "failing_streak", None, 0, shop.agent)
    assert len(live.llm.calls) == 3 and _cards(shop)[0].occurrences == 3


def test_the_first_success_expires_the_suggestion_and_brings_the_hour_back(shop, live):
    _replay(shop, live, VECTOR["ticks"])
    live.llm.failure = None

    held = _fire(shop, live, _at("2026-10-27T00:00:00+00:00"))    # 7 hours after 17:00: 8 are due
    worked = _fire(shop, live, _at("2026-10-27T01:00:00+00:00"))
    after = _fire(shop, live, _at("2026-10-27T02:00:00+00:00"))   # the heartbeat's own hour again

    assert [held["outcome"], worked["outcome"], after["outcome"]] == ["skipped_cap", "silent", "silent"]
    [card] = _cards(shop)
    assert (card.status, card.occurrences) == ("expired", 5)
    assert card.detail == "VECTOR's check-ins work again, after 5 failures in a row (http_401)."
    assert card.expires_at == _at("2026-10-27T01:00:00+00:00") and card.decided_at is None
    assert _hhmm(live.llm.calls[-2:]) == ["01:00", "02:00"]

    live.llm.failure = _vector_refusal()
    again = _fire(shop, live, _at("2026-10-27T03:00:00+00:00"))
    assert again["outcome"] == "error" and again["rang"] is True
    assert [(c.status, c.occurrences) for c in _cards(shop)] == [("expired", 5), ("open", 1)]


def test_a_failure_of_another_class_is_its_own_streak_and_a_success_ends_both(shop, live):
    _replay(shop, live, VECTOR["ticks"][:2])
    live.llm.failure = APITimeoutError("Request timed out.")
    _fire(shop, live, _at(VECTOR["ticks"][2]))

    cards = {card.subject_key: card for card in _cards(shop)}
    key, timeout = f"agent:{shop.agent}:error:http_401", f"agent:{shop.agent}:error:timeout"
    assert set(cards) == {key, timeout} and len(live.bells) == 2
    assert (cards[key].occurrences, cards[timeout].occurrences) == (2, 1)
    assert cards[timeout].title == "VECTOR's check-ins fail: they time out"

    live.llm.failure = None
    _fire(shop, live, _at(VECTOR["ticks"][3]))
    assert {card.subject_key: card.status for card in _cards(shop)} == {key: "expired", timeout: "expired"}


def test_a_streak_another_class_interrupted_counts_afresh_on_its_open_card(shop, live):
    _replay(shop, live, VECTOR["ticks"][:2])
    live.llm.failure = ProviderRefused("Error code: 500 - {'error': {'message': 'upstream'}}", 500)
    _fire(shop, live, _at(VECTOR["ticks"][2]))
    live.llm.failure = _vector_refusal()
    _fire(shop, live, _at(VECTOR["ticks"][3]))       # 401 again: a streak of one, the 500 between

    cards = {card.subject_key: card for card in _cards(shop)}
    card = cards[f"agent:{shop.agent}:error:http_401"]
    assert card.occurrences == 3 and card.status == "open"  # every 401 failure the open card has seen
    assert card.detail.startswith("Its check-in at 2026-10-26 12:00 UTC failed (http_401).")
    assert cards[f"agent:{shop.agent}:error:http_500"].occurrences == 1


def test_an_open_streak_card_stays_open_while_the_agent_fails(shop, live):
    _replay(shop, live, VECTOR["ticks"][:2])

    [card] = _cards(shop)
    assert card.expires_at == _at(VECTOR["ticks"][1]) + timedelta(days=7)


def test_a_hidden_streak_card_is_neither_counted_expired_nor_made_again(shop, live):
    _fire(shop, live, _at(VECTOR["ticks"][0]))
    shop.db.execute(text(
        "UPDATE proactive_suggestions SET status = 'hidden', hide_reason = 'not_now', decided_at = :at "
        "WHERE workspace_id = CAST(:w AS uuid)"), {"at": _at("2026-10-26T09:30:00+00:00"), "w": str(shop.ws)})
    shop.db.commit()

    _fire(shop, live, _at(VECTOR["ticks"][1]))       # the same streak: its card is hidden
    live.llm.failure = None
    _fire(shop, live, _at(VECTOR["ticks"][2]))       # works: a hidden card stays the person's decision
    live.llm.failure = _vector_refusal()
    _fire(shop, live, _at(VECTOR["ticks"][3]))       # a new streak, hidden within 14 days

    [card] = _cards(shop)
    assert (card.status, card.occurrences, card.hide_reason) == ("hidden", 1, "not_now")


def test_this_agents_success_leaves_another_agents_streak_card_open(shop, live):
    other = _agent_row(shop.db, shop.ws, "LEDGER", {"heartbeat": HB})
    shop.db.commit()
    _fire(shop, live, _at(VECTOR["ticks"][0]), agent_id=other)
    live.llm.failure = None

    _fire(shop, live, _at(VECTOR["ticks"][1]))

    [card] = _cards(shop)
    assert (card.subject_key, card.status) == (f"agent:{other}:error:http_401", "open")
    assert card.title.startswith("LEDGER's check-ins fail")


def test_a_check_in_whose_agent_is_gone_makes_no_card(shop, live):
    from services.proactive import runner

    result = asyncio.run(runner.run_agent_check_in(shop.agent + 10_000, shop.ws, dict(HB)))

    assert (result.outcome, result.error_class) == ("error", runner.AGENT_NOT_FOUND)
    assert _cards(shop) == []


def test_streak_bookkeeping_that_fails_never_fails_the_check_in(shop, live, monkeypatch, caplog):
    def broken(db, failure):
        raise RuntimeError("proactive_suggestions is locked")

    monkeypatch.setattr(failing_streaks, "_open_card", broken)
    with caplog.at_level(logging.ERROR, logger=failing_streaks.__name__):
        result = _fire(shop, live, _at(VECTOR["ticks"][0]))

    assert (result["outcome"], result["error_class"], result["rang"]) == ("error", "http_401", True)
    assert _cards(shop) == [] and "could not be recorded" in caplog.text


def test_in_shadow_the_tick_runs_and_the_check_in_keeps_its_streak_beside_it(shop, live):
    live.mode = "shadow"
    results = []
    for tick in VECTOR["ticks"][:4]:
        live.clock = _at(tick)
        result = asyncio.run(agent_job(live.service)(shop.agent, str(shop.ws), dict(HB)))
        _stamp(shop, result["check_in"]["run_id"], live.clock)
        results.append(result["check_in"])

    assert _hhmm(live.ticks) == ["09:00", "10:00", "11:00", "12:00"]  # today's tick, unchanged
    assert [result["outcome"] for result in results] == SCHEDULE_26_OCT[:4]
    assert [card.occurrences for card in _cards(shop)] == [3] and live.bells == []


def test_auto_workspace_check_in_failing_makes_no_streak_card(shop, live, monkeypatch):
    from services.proactive import runner
    from services.proactive.signals import Signal

    signal = Signal(kind="ticket_failed", subject_key="ticket:0846", ref=("ticket", "0846"),
                    at=_at(VECTOR["ticks"][0]) - timedelta(minutes=5), summary="Ticket #0846 failed")
    monkeypatch.setattr(runner, "read_signals", lambda db, workspace_id, since, until: (signal,))

    result = asyncio.run(orchestrator_job(live.service)(str(shop.ws), {"enabled": True, "proactive_level": "notify"}))

    assert (result["outcome"], result["error_class"]) == ("error", "http_401")
    assert _cards(shop) == []
