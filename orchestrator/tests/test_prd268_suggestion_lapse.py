"""PRD-268 P268W1-RVW-3 (§6.2.4, §6.9) — open suggestions lapse at their ``expires_at``.

``services.proactive.expiry.expire_lapsed`` moves a workspace's open suggestions whose 7 days
have passed to ``expired``, in one UPDATE, before anything reads them. Against Postgres,
inside a rolled-back transaction (joined by SAVEPOINT, so the commits stay inside it).
Proven here:

* **at a check-in's start**: an agent whose three open suggestions lapsed (8 days old) is no
  longer at ``max_open_per_agent``, and the subject one of them held may be suggested again
  (dedupe passes it over); a 6-day-old suggestion stays open, and another workspace's
  lapsed one is left alone. Auto's workspace check-in lapses them too, and its prompt no
  longer lists a lapsed title as open. The check-ins run through ``runner`` with the model
  scripted (the runner harness of ``tests/test_prd268_runner.py``) and the suggestion
  recorded by ``platform_suggest_next_step`` itself;
* **Start and Hide** of a lapsed suggestion are 409 ``expired``, a Mission's too: no ticket,
  no audit row;
* **the list**: ``?status=open`` leaves a lapsed suggestion out, ``?status=expired`` lists it;
  a 6-day-old one is still open, and another workspace's lapsed row is untouched;
* **the UPDATE** touches only open rows of its workspace past their time: not a started or
  hidden one, not one with no ``expires_at``, not a streak card made 8 days ago whose
  ``expires_at`` its failures pushed on.
"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

import tests.test_prd268_runner as harness
import tests.test_prd268_suggestions_api as api
from modules.proactive import title_key
from services.proactive import runner

SUGGESTIONS = "/api/proactive/suggestions"
LIFETIME = timedelta(days=7)
LAPSED_DAYS, LIVE_DAYS = 8, 6
AGAIN = "Ticket #0912 failed: Refresh the sitemap"
LAPSED_TITLES = (AGAIN, "VECTOR's report is late", "The sitemap is stale")
LIVE_TITLE = "The October invoice is unpaid"


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the runner, the suggestion tool and the routes commit, and none may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def saas(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "SPEND_CEILING_USD", 0)


def _seed(db, ws, user, title, *, age_days, agent=None, status="open", lifetime=LIFETIME, action=None):
    """One suggestion made ``age_days`` ago, expiring ``lifetime`` after it was made (None: never)."""
    from core.models.proactive import ProactiveSuggestion

    made = datetime.now(timezone.utc) - timedelta(days=age_days)
    row = ProactiveSuggestion(
        workspace_id=ws, user_id=user, source_type="heartbeat" if agent else "check_in", source_agent_id=agent,
        title=title, detail="", evidence=[{"type": "ticket", "id": "0912"}],
        proposed_action=action or {"kind": "chat", "payload": {}}, subject_key=title_key(title), status=status,
        last_seen_at=made, created_at=made, expires_at=None if lifetime is None else made + lifetime)
    db.add(row)
    db.commit()
    return row.id


def _status(db, sid) -> str:
    from core.models.proactive import ProactiveSuggestion

    db.expire_all()
    return db.get(ProactiveSuggestion, sid).status


# ── At a check-in's start ───────────────────────────────────────────────────

@pytest.fixture
def shop(db_session):
    """A workspace with check-ins switched on, its owner and an agent; another workspace."""
    owner = harness._user(db_session, "owner")
    ws = harness._workspace(db_session, owner)
    other_owner = harness._user(db_session, "other")
    shop = NS(db=db_session, ws=ws, owner=owner, agent=harness._agent(db_session, ws), other_owner=other_owner,
              other_ws=harness._workspace(db_session, other_owner))
    db_session.commit()
    return shop


@pytest.fixture
def scripted(shop, monkeypatch):
    """The check-ins on the test's session, proactive mode shadow: the context build recorded,
    one signal for Auto's, and the model scripted (a read of the board, then the suggestion
    of ``AGAIN``); the executor real."""
    from core.llm import manager
    from modules.context.result import ContextResult
    from modules.context.service import ContextService
    from modules.tools.discovery import platform_executor
    from modules.tools.discovery.action_registry import get_action_registry

    llm = harness.FakeLLM([harness._reply(harness._call(harness.LIST_TASKS, {})),
                           harness._reply(harness._suggest("0912", AGAIN)), harness._reply(content="Done.")])
    rig = NS(llm=llm, tasks=[])

    async def build_context(self, **kwargs):
        rig.tasks.append(kwargs["task_description"])
        tools = [get_action_registry().get(name).to_openai_schema() for name in (harness.LIST_TASKS, harness.SUGGEST)]
        return ContextResult(system_prompt=kwargs["task_description"], tools=tools)

    async def list_tasks(db, workspace_id, params):
        return {"success": True, "tasks": [dict(harness.READ_TICKET)], "total": 1}

    monkeypatch.setattr(manager, "read_system_setting",
                        lambda category, key: "shadow" if (category, key) == ("proactive", "mode") else None)
    monkeypatch.setattr(ContextService, "build_context", build_context)
    monkeypatch.setattr(runner, "_session", lambda: nullcontext(shop.db))
    monkeypatch.setattr(runner, "heartbeat_llm", lambda workspace_id, agent_id: llm)
    monkeypatch.setattr(runner, "read_signals",
                        lambda db, workspace_id, since, until: (harness._signal(datetime.now(timezone.utc)),))
    monkeypatch.setattr(platform_executor, "list_board_tasks", list_tasks)
    monkeypatch.setattr(platform_executor.PlatformActionExecutor, "_full_autonomy", lambda self: False)
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        yield rig


def _open_again(shop):
    """The open suggestion of ``AGAIN`` in the shop's workspace, the only one there may be."""
    from core.models.proactive import ProactiveSuggestion

    shop.db.expire_all()
    [again] = shop.db.query(ProactiveSuggestion).filter(
        ProactiveSuggestion.workspace_id == shop.ws, ProactiveSuggestion.status == "open",
        ProactiveSuggestion.subject_key == title_key(AGAIN)).all()
    return again


def test_lapsed_suggestions_expire_at_the_next_check_in_and_the_agent_suggests_again(shop, scripted):
    lapsed = [_seed(shop.db, shop.ws, shop.owner, title, age_days=LAPSED_DAYS, agent=shop.agent)
              for title in LAPSED_TITLES]  # max_open_per_agent is 3: before the lapse, the agent is at it
    live = _seed(shop.db, shop.ws, shop.owner, LIVE_TITLE, age_days=LIVE_DAYS)
    elsewhere = _seed(shop.db, shop.other_ws, shop.other_owner, AGAIN, age_days=LAPSED_DAYS, agent=shop.agent)

    result = asyncio.run(runner.run_agent_check_in(shop.agent, shop.ws, {}))

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    assert [_status(shop.db, sid) for sid in lapsed] == ["expired"] * 3
    assert (_status(shop.db, live), _status(shop.db, elsewhere)) == ("open", "open")
    again = _open_again(shop)
    assert (again.run_id, again.source_agent_id, again.title) == (result.run_id, shop.agent, AGAIN)


def test_autos_check_in_lapses_them_first_and_its_prompt_lists_them_no_longer(shop, scripted):
    lapsed = _seed(shop.db, shop.ws, shop.owner, AGAIN, age_days=LAPSED_DAYS)
    live = _seed(shop.db, shop.ws, shop.owner, LIVE_TITLE, age_days=LIVE_DAYS)

    result = asyncio.run(runner.run_workspace_check_in(shop.ws))

    assert (result.outcome, result.suggestions) == ("suggested", 1)
    assert (_status(shop.db, lapsed), _status(shop.db, live)) == ("expired", "open")
    [task] = scripted.tasks
    assert LIVE_TITLE in task and AGAIN not in task
    again = _open_again(shop)
    assert (again.run_id, again.source_agent_id, again.source_type) == (result.run_id, None, "check_in")


# ── Start, Hide and the list ────────────────────────────────────────────────

@pytest.fixture
def team(db_session, saas) -> NS:
    """The suggestions API's workspace (owner, admin, editor, viewer, an agent) and another workspace."""
    return api._team(db_session)


def _lapsed_ticket(team, title: str = AGAIN):
    return _seed(team.db, team.ws, team.owner.pk, title, age_days=LAPSED_DAYS,
                 action={"kind": "board_task", "payload": {"agent_id": team.agent}})


def test_start_of_a_lapsed_suggestion_is_409_expired_and_files_nothing(team):
    sid = _lapsed_ticket(team)
    mission = _seed(team.db, team.ws, team.owner.pk, "Plan the November launch", age_days=LAPSED_DAYS,
                    action={"kind": "mission", "payload": {}})

    response = api._start(team, team.owner, sid)
    lapsed_mission = api._start(team, team.owner, mission)

    assert response.status_code == 409 and response.json()["code"] == "expired"
    assert lapsed_mission.status_code == 409 and lapsed_mission.json()["code"] == "expired"
    assert (_status(team.db, sid), _status(team.db, mission)) == ("expired", "expired")
    assert api._tickets(team.db, sid) == [] and api._audit(team.db, sid) == []


def test_hide_of_a_lapsed_suggestion_is_409_expired(team):
    sid = _lapsed_ticket(team)

    response = api._hide(team, team.owner, sid, json={"reason": "not_now"})

    assert response.status_code == 409 and response.json()["code"] == "expired"
    assert api._row(team, sid).hide_reason is None and api._audit(team.db, sid) == []


def test_the_list_shows_a_lapsed_suggestion_as_expired_never_open(team):
    sid = _lapsed_ticket(team)
    live = _seed(team.db, team.ws, team.owner.pk, LIVE_TITLE, age_days=LIVE_DAYS)
    elsewhere = _seed(team.db, team.other_ws, team.other_owner.pk, AGAIN, age_days=LAPSED_DAYS)

    listed = api._as(team, team.owner).get(SUGGESTIONS, params={"status": "open"})
    expired = api._as(team, team.owner).get(SUGGESTIONS, params={"status": "expired"})

    assert listed.status_code == 200 and expired.status_code == 200
    assert [row["id"] for row in listed.json()["suggestions"]] == [str(live)]
    assert [row["id"] for row in expired.json()["suggestions"]] == [str(sid)]
    assert _status(team.db, elsewhere) == "open"


# ── The UPDATE ──────────────────────────────────────────────────────────────

def test_expire_lapsed_touches_only_open_rows_of_its_workspace_past_their_time(shop):
    from services.proactive.expiry import expire_lapsed

    db, ws, owner = shop.db, shop.ws, shop.owner
    lapsed = _seed(db, ws, owner, AGAIN, age_days=LAPSED_DAYS)
    live = _seed(db, ws, owner, LIVE_TITLE, age_days=LIVE_DAYS)
    pushed_on = _seed(db, ws, owner, "VECTOR's check-ins fail", age_days=LAPSED_DAYS, agent=shop.agent,
                      lifetime=timedelta(days=LAPSED_DAYS) + LIFETIME)  # a streak card seen again today
    timeless = _seed(db, ws, owner, "No expiry", age_days=30, lifetime=None)
    decided = [_seed(db, ws, owner, title, age_days=LAPSED_DAYS, status=status)
               for title, status in (("Started long ago", "started"), ("Hidden long ago", "hidden"))]
    elsewhere = _seed(db, shop.other_ws, shop.other_owner, AGAIN, age_days=LAPSED_DAYS)

    assert expire_lapsed(db, ws, datetime.now(timezone.utc)) == 1
    assert expire_lapsed(db, ws, datetime.now(timezone.utc)) == 0

    assert _status(db, lapsed) == "expired"
    assert [_status(db, sid) for sid in (live, pushed_on, timeless, elsewhere)] == ["open"] * 4
    assert [_status(db, sid) for sid in decided] == ["started", "hidden"]
