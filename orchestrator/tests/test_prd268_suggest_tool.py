"""PRD-268 US-C03b (§6.2.3-6.2.5, FR-6) — ``platform_suggest_next_step``: a check-in records one suggestion.

The check-in's one write. Proven here, against Postgres (inside a rolled-back transaction):

* it runs only in a check-in: the run's facts reach it server-side, from ``caller_context``
  (``check_in_lane.carries_the_check_in``); a ``_check_in`` the model sends is stripped;
* it refuses evidence the run did not read (provenance), a fourth suggestion in one run,
  the other §6.9 caps it counts (ten new per person a day, three open per agent), an open
  duplicate and one hidden in the last 14 days ("Already suggested: <title>"), arguments
  that are not a suggestion, a ticket for another workspace's agent, and a workspace with
  no owner;
* otherwise it inserts one open row addressed to the workspace owner (K11) with its run
  and source; a concurrent check-in holding the same open key loses cleanly;
* it is registered as a write in the ``proactive`` category, handled, offered in a
  check-in, and on the hierarchy gate's allow-list.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery.check_in_lane import CHECK_IN_PARAM, check_in_params

ORCH = Path(__file__).resolve().parents[1]
SUGGEST = "platform_suggest_next_step"
TICKET_READ = {"type": "ticket", "id": "0846"}
REPORT_READ = {"type": "report", "id": "17"}
READS = (TICKET_READ, REPORT_READ)
TITLE = "Ticket #0846 has been blocked for two days"


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here):
    the tool commits, and a lost race rolls its savepoint back; neither may take the seed."""
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _user(db, tag: str) -> int:
    name = f"prd268st_{tag}_{uuid4().hex[:10]}"
    return int(db.execute(text(
        "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
        {"u": name, "e": f"{name}@t.test", "c": name}).scalar())


def _workspace(db, owner) -> UUID:
    ws = uuid4()
    db.execute(text("INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), 'prd268 cafe', :o)"),
               {"id": str(ws), "o": owner})
    return ws


def _agent(db, ws: UUID) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('VECTOR', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"w": str(ws)}).scalar())


@pytest.fixture
def shop(db_session):
    """A workspace with its owner and an agent, and another workspace with its own agent."""
    owner = _user(db_session, "owner")
    ws = _workspace(db_session, owner)
    other_ws = _workspace(db_session, _user(db_session, "other"))
    shop = NS(db=db_session, ws=ws, owner=owner, other_ws=other_ws,
              agent=_agent(db_session, ws), stranger=_agent(db_session, other_ws))
    db_session.commit()
    return shop


def _row(shop, key: str, *, title: str = "Earlier suggestion", status: str = "open", decided_at=None,
         created_at=None, agent=None, user=None, ws=None) -> None:
    """Seed one suggestion row, committed."""
    created = created_at or datetime.now(timezone.utc)
    shop.db.execute(text(
        "INSERT INTO proactive_suggestions (id, workspace_id, user_id, source_type, source_agent_id, title, "
        "proposed_action, subject_key, status, decided_at, created_at, last_seen_at) VALUES (CAST(:id AS uuid), "
        "CAST(:ws AS uuid), :user, 'check_in', :agent, :title, CAST(:action AS jsonb), :key, :status, :decided, "
        ":created, :created)"),
        {"id": str(uuid4()), "ws": str(ws or shop.ws), "user": user or shop.owner, "agent": agent, "title": title,
         "action": json.dumps({"kind": "chat", "payload": {}}), "key": key, "status": status,
         "decided": decided_at, "created": created})
    shop.db.commit()


def _rows(shop):
    from core.models.proactive import ProactiveSuggestion

    shop.db.expire_all()
    return shop.db.query(ProactiveSuggestion).filter(ProactiveSuggestion.workspace_id == shop.ws).all()


def _check_in(run=None, reads=READS, agent=None) -> dict:
    return {"run_id": str(run or uuid4()), "reads": list(reads), "agent_id": agent}


def _call(title: str = TITLE, **over) -> dict:
    return {"title": title, "detail": "VECTOR waits on the owner's answer.", "evidence": [TICKET_READ],
            "proposed_action": {"kind": "chat"}, **over}


def _suggest(shop, params: dict, check_in=None) -> dict:
    from modules.tools.discovery.handlers_proactive import suggest_next_step

    sent = {**params, CHECK_IN_PARAM: check_in} if check_in is not None else dict(params)
    return asyncio.run(suggest_next_step(shop.db, shop.ws, sent))


def _refused(reply: dict, code: str) -> bool:
    return reply.get("success") is False and reply.get("error_code") == code


# ── Only in a check-in ──────────────────────────────────────────────────────

@pytest.mark.parametrize("check_in", [None, {"reads": list(READS)}, {"run_id": "not-a-run", "reads": []},
                                      {"run_id": str(uuid4()), "agent_id": "VECTOR"}])
def test_outside_a_check_in_the_tool_refuses_and_records_nothing(shop, check_in):
    from modules.tools.discovery.handlers_proactive import NOT_IN_A_CHECK_IN

    reply = _suggest(shop, _call(), check_in)
    assert _refused(reply, "not_in_a_check_in") and reply["error"] == NOT_IN_A_CHECK_IN
    assert _rows(shop) == []


def test_check_in_params_strips_a_forged_run_and_hands_the_suggestion_tool_the_servers():
    forged = {"title": "x", CHECK_IN_PARAM: {"run_id": "forged", "reads": [{"type": "ticket", "id": "1"}]}}
    lane = {"lane": "check_in", "run_id": "r-1", "check_in_reads": [TICKET_READ], "check_in_agent_id": 7}

    assert check_in_params(SUGGEST, forged, None) == {"title": "x"}
    assert check_in_params(SUGGEST, forged, {"lane": "heartbeat", "run_id": "r-1"}) == {"title": "x"}
    assert check_in_params("platform_list_agents", forged, lane) == {"title": "x"}
    assert check_in_params(SUGGEST, json.dumps(forged), lane) == {
        "title": "x", CHECK_IN_PARAM: {"run_id": "r-1", "reads": [TICKET_READ], "agent_id": 7}}
    assert forged[CHECK_IN_PARAM]["run_id"] == "forged"  # the caller's dict is untouched


# ── One insert ──────────────────────────────────────────────────────────────

def test_a_check_in_records_one_open_suggestion_for_the_workspace_owner(shop):
    run = uuid4()
    reply = _suggest(shop, _call(), _check_in(run))

    assert reply["success"] is True
    [row] = _rows(shop)
    assert reply["data"]["suggestion_id"] == str(row.id)
    assert (row.user_id, row.source_type, row.source_agent_id, row.run_id) == (shop.owner, "check_in", None, run)
    assert (row.title, row.status, row.occurrences, row.severity) == (TITLE, "open", 1, "can_wait")
    assert row.subject_key == "title:ticket #0846 has been blocked for two days"
    assert row.evidence == [TICKET_READ] and row.proposed_action == {"kind": "chat", "payload": {}}
    assert row.expires_at - row.created_at == timedelta(days=7)


def test_an_agents_check_in_records_its_agent_and_proposes_a_ticket_for_it(shop):
    action = {"kind": "board_task", "payload": {"agent_id": str(shop.agent)}}
    reply = _suggest(shop, _call(subject_key="ticket:846", severity="today", proposed_action=action),
                     _check_in(agent=shop.agent))

    assert reply["success"] is True
    [row] = _rows(shop)
    assert (row.source_type, row.source_agent_id, row.severity) == ("heartbeat", shop.agent, "today")
    assert row.subject_key == "ticket:0846"
    assert row.proposed_action == {"kind": "board_task", "payload": {"agent_id": shop.agent}}


# ── Provenance ──────────────────────────────────────────────────────────────

def test_evidence_the_run_did_not_read_is_refused(shop):
    reply = _suggest(shop, _call(), _check_in(reads=[REPORT_READ]))
    assert _refused(reply, "no_provenance") and _rows(shop) == []


def test_a_ticket_read_is_matched_by_its_board_number(shop):
    reply = _suggest(shop, _call(evidence=[{"type": "Ticket", "id": "#0846"}]),
                     _check_in(reads=[{"type": "ticket", "id": 846}]))
    assert reply["success"] is True
    assert _rows(shop)[0].evidence == [TICKET_READ]


# ── The caps ────────────────────────────────────────────────────────────────

def test_a_fourth_suggestion_in_one_run_is_refused(shop):
    run = uuid4()
    for n in range(3):
        assert _suggest(shop, _call(f"Ticket #084{n} is stuck"), _check_in(run))["success"] is True

    reply = _suggest(shop, _call("Ticket #0849 is stuck"), _check_in(run))
    assert _refused(reply, "suggestion_cap") and "max_suggestions_per_run" in reply["error"]
    assert len(_rows(shop)) == 3
    assert _suggest(shop, _call("Ticket #0849 is stuck"), _check_in())["success"] is True  # a new run may


def test_a_fourth_open_suggestion_from_one_agent_is_refused(shop):
    for n in range(3):
        _row(shop, f"title:agent finding {n}", agent=shop.agent)

    reply = _suggest(shop, _call(), _check_in(agent=shop.agent))
    assert _refused(reply, "suggestion_cap") and "max_open_per_agent" in reply["error"]
    assert _suggest(shop, _call(), _check_in())["success"] is True  # Auto's own check-in has no agent cap


def test_an_eleventh_new_suggestion_for_a_person_in_a_day_is_refused(shop):
    for n in range(10):
        _row(shop, f"title:earlier {n}", status="started")

    reply = _suggest(shop, _call(), _check_in())
    assert _refused(reply, "suggestion_cap") and "max_new_per_person_per_day" in reply["error"]
    assert len(_rows(shop)) == 10


# ── Dedupe (FR-6) ───────────────────────────────────────────────────────────

def test_an_open_duplicate_is_refused_by_key_and_by_title(shop):
    _row(shop, "ticket:0846", title="VECTOR's ticket #0846 is blocked")

    by_key = _suggest(shop, _call("Look at #0846", subject_key="ticket:0846"), _check_in())
    by_title = _suggest(shop, _call("  vector's TICKET #0846 is blocked!! "), _check_in())

    for reply in (by_key, by_title):
        assert _refused(reply, "already_suggested")
        assert reply["error"] == "Already suggested: VECTOR's ticket #0846 is blocked"
    assert len(_rows(shop)) == 1


def test_a_suggestion_hidden_13_days_ago_blocks_it_and_one_hidden_15_days_ago_does_not(shop):
    now = datetime.now(timezone.utc)
    _row(shop, "ticket:0846", title="Hidden lately", status="hidden", decided_at=now - timedelta(days=13),
         created_at=now - timedelta(days=20))
    _row(shop, "ticket:0901", title="Hidden long ago", status="hidden", decided_at=now - timedelta(days=15),
         created_at=now - timedelta(days=20))

    recent = _suggest(shop, _call("About #0846", subject_key="ticket:0846"), _check_in())
    assert _refused(recent, "already_suggested") and recent["error"] == "Already suggested: Hidden lately"
    ticket_0901 = {"type": "ticket", "id": "0901"}
    old = _suggest(shop, _call("About #0901", subject_key="ticket:0901", evidence=[ticket_0901]),
                   _check_in(reads=[ticket_0901]))
    assert old["success"] is True


def test_a_started_or_expired_suggestion_or_someone_elses_never_blocks_a_new_one(shop):
    _row(shop, "ticket:0846", status="started")
    _row(shop, "ticket:0846", status="expired")
    _row(shop, "ticket:0846", ws=shop.other_ws, user=shop.owner)

    reply = _suggest(shop, _call(subject_key="ticket:0846"), _check_in())
    assert reply["success"] is True


def test_a_concurrent_check_in_holding_the_key_loses_cleanly(shop, monkeypatch):
    """The other check-in committed between the dedupe read and this insert: the open-row
    unique index refuses it, its savepoint rolls back, and the session goes on."""
    from modules.tools.discovery import handlers_proactive

    _row(shop, "ticket:0846", title="Filed a moment ago")
    monkeypatch.setattr(handlers_proactive, "_duplicate", lambda *args: None)

    reply = _suggest(shop, _call(subject_key="ticket:0846"), _check_in())
    assert _refused(reply, "already_suggested") and reply["error"] == f"Already suggested: {TITLE}"
    assert [row.title for row in _rows(shop)] == ["Filed a moment ago"]


# ── Arguments ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("over", [
    {"title": "x" * 121},
    {"title": " !? "},
    {"detail": "x" * 601},
    {"evidence": []},
    {"evidence": [TICKET_READ] * 3 + [REPORT_READ] * 3},
    {"evidence": [{"type": "ticket"}]},
    {"proposed_action": {"kind": "mission", "payload": {}}},
    {"proposed_action": {"kind": "chat", "payload": {"agent_id": 3}}},
    {"proposed_action": {"kind": "board_task", "payload": {"brief": "do it"}}},
    {"proposed_action": {"kind": "board_task", "payload": {"agent_id": -2}}},
    {"proposed_action": {"kind": "board_task", "payload": {"agent_id": 2**31}}},
    {"severity": "urgent"},
    {"subject_key": "agent:578:error:401"},  # a failing streak's key is the check-in's own, never its model's
    {"subject_key": "mission:7"},
    {"subject_key": "ticket:0901"},  # a ticket key without that ticket's read in the evidence
])
def test_arguments_that_are_not_a_suggestion_are_refused(shop, over):
    reply = _suggest(shop, _call(**over), _check_in())
    assert _refused(reply, "not_a_suggestion") and reply["error"]
    assert _rows(shop) == []


@pytest.mark.parametrize("given, stored", [
    ("ticket:#846", "ticket:0846"),
    ("Title:  Look at THIS!", "title:look at this"),
])
def test_a_subject_key_is_stored_in_its_canonical_form(shop, given, stored):
    assert _suggest(shop, _call(subject_key=given), _check_in())["success"] is True
    assert _rows(shop)[0].subject_key == stored


def test_title_and_detail_are_stored_as_plain_text(shop):
    """Controls and format characters (zero-width, bidi overrides) are dropped and spaces
    collapsed: the title is one line; the detail keeps its line breaks."""
    title = "Ticket\u202e #0846\n is\u200b   blocked\x07"
    detail = "First line\u200b\n\tsecond   line\x00"
    assert _suggest(shop, _call(title, detail=detail), _check_in())["success"] is True
    [row] = _rows(shop)
    assert (row.title, row.detail) == ("Ticket #0846 is blocked", "First line\nsecond line")


def test_a_zero_width_character_does_not_dodge_dedupe(shop):
    _row(shop, "title:ticket #0846 is blocked", title="Ticket #0846 is blocked")
    reply = _suggest(shop, _call("Ticket #08\u200b46 is blocked"), _check_in())
    assert _refused(reply, "already_suggested") and len(_rows(shop)) == 1


def test_a_ticket_for_another_workspaces_agent_is_refused(shop):
    action = {"kind": "board_task", "payload": {"agent_id": shop.stranger}}
    reply = _suggest(shop, _call(proposed_action=action), _check_in())
    assert _refused(reply, "no_such_agent") and _rows(shop) == []


def test_a_workspace_with_no_owner_gets_no_suggestion(shop, monkeypatch):
    from core.workspaces import owner

    monkeypatch.setattr(owner, "config", NS(IS_LOCAL_EDITION=False))  # no local operator to fall back to
    shop.ws = _workspace(shop.db, None)
    shop.db.commit()

    reply = _suggest(shop, _call(), _check_in())
    assert _refused(reply, "no_addressee") and _rows(shop) == []


# ── Through the executor ────────────────────────────────────────────────────

def _execute(shop, params: dict, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(shop.db, shop.ws)
    executor._full_autonomy = lambda: False
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(SUGGEST, params, caller_context))


def test_through_the_executor_a_check_ins_call_records_its_suggestion_with_the_servers_run(shop):
    run = uuid4()
    lane = {"lane": "check_in", "run_id": str(run), "check_in_reads": [TICKET_READ],
            "check_in_agent_id": shop.agent}
    forged = {CHECK_IN_PARAM: {"run_id": str(uuid4()), "reads": [REPORT_READ], "agent_id": shop.stranger}}

    reply = _execute(shop, {**_call(), **forged}, lane)

    assert reply["success"] is True
    [row] = _rows(shop)
    assert (row.run_id, row.source_agent_id) == (run, shop.agent)


@pytest.mark.parametrize("caller_context", [None, {"lane": "heartbeat"}])
def test_through_the_executor_a_forged_check_in_outside_the_lane_is_refused(shop, caller_context):
    forged = {CHECK_IN_PARAM: _check_in()}
    reply = _execute(shop, {**_call(), **forged}, caller_context)
    assert _refused(reply, "not_in_a_check_in") and _rows(shop) == []


# ── Registered, handled, offered, gated ─────────────────────────────────────

def test_the_tool_is_a_write_in_the_proactive_category_handled_and_offered_in_a_check_in():
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.check_in_lane import offered_in_a_check_in
    from modules.tools.discovery.handlers_proactive import suggest_next_step
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    action = get_action_registry().get(SUGGEST)
    assert (action.permission_level, action.category, action.requires_confirmation) == ("write", "proactive", False)
    assert not (action.admin_only or action.super_admin_only)
    assert action.parameters["required"] == ["title", "evidence", "proposed_action"]
    assert set(action.parameters["properties"]) == {
        "title", "detail", "evidence", "proposed_action", "severity", "subject_key"}
    assert action.parameters["properties"]["proposed_action"]["properties"]["kind"]["enum"] == ["board_task", "chat"]
    assert PLATFORM_HANDLERS[SUGGEST] is suggest_next_step
    assert offered_in_a_check_in(action) is True
    assert SUGGEST in _hierarchy_gate().ALLOW_LIST


def _hierarchy_gate():
    """scripts/check_hierarchy_gate.py, loaded by path as test_f147 loads it."""
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
