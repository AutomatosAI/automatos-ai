"""PRD-256 P256-FIX-RVW-23 (D1, D7, FX-010, FX-012): a timer Auto sets for an agent waits
for the owner's click, and runs on the one active agent its card named.

The third fix-wave review: platform_schedule_task was not owner-only though
platform_schedule_playbook is, and what it filed read as the owner's own work: each fire's
ticket was created_by 'user' with review_mode 'auto', so its agent's GMAIL_SEND_EMAIL ran
with no card and the ticket closed itself ('Email the supplier every Monday to confirm the
order' is #2318 on a timer). And the handler took the first agent whose name matched,
switched-off ones included. Now, from a person's chat, it raises the card naming the agent
(name and #id), when it runs in words, its delivery and the brief's first line; the click
runs exactly that call. A scheduled ticket whose brief sends or orders is filed for the
owner's review unless the call named a review_mode. The agent is the one active agent the
name or the agent_id names: a clash lists the candidates; an agent's own run is unchanged.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery import owner_only

SCHEDULE = "platform_schedule_task"
BRIEF = "Email the supplier every Monday to confirm the order\nCopy the owner in."
MONDAYS = {"task_type": "recurring", "schedule": "0 9 * * 1", "deliver_as": "board_task", "description": BRIEF}


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


def _agent(db, ws, name, status="active"):
    from core.models import Agent

    agent = Agent(name=name, agent_type="worker", description="", status=status, configuration={},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


@pytest.fixture
def club(db_session, seed_workspace):
    """Auto (the caller), two active OPS, an active CLUB DESK beside a switched-off one."""
    db = db_session
    ws = UUID(seed_workspace())
    tag = uuid4().hex[:6]
    owner = _user(db, "gerard")
    auto = _agent(db, ws, f"AUTO-{tag}")
    ops = [_agent(db, ws, f"OPS-{tag}"), _agent(db, ws, f"OPS-{tag}")]
    off = _agent(db, ws, f"CLUB DESK-{tag}", status="inactive")
    desk = _agent(db, ws, f"CLUB DESK-{tag}")
    return NS(db=db, ws=ws, tag=tag, owner=owner, auto=auto, ops=ops, off=off, desk=desk)


@pytest.fixture
def rows(monkeypatch):
    """What ScheduledTaskService.create_task was asked to file, one entry per row."""
    from services.scheduled_task_service import ScheduledTaskService

    filed = []

    async def _create(self, **kw):
        filed.append(kw)
        return {"success": True, "task_id": len(filed), "deliver_as": kw.get("deliver_as")}

    monkeypatch.setattr(ScheduledTaskService, "create_task", _create)
    return filed


def _owners_chat(club):
    return {"driving_user_id": str(club.owner), "user_id": "user_owner", "conversation_id": str(uuid4())}


class _Executor:
    """PlatformActionExecutor._run_cleared's shape: the gates have cleared, the handler runs."""

    def __init__(self, club):
        self.club = club

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(self.club.db, self.club.ws, params)


def _run(club, params, caller_context):
    from modules.tools.discovery.handlers_scheduling import schedule_task

    return asyncio.run(_Executor(club)._run_cleared(SCHEDULE, params, caller_context, None, schedule_task))


def _handler(club, params):
    from modules.tools.discovery.handlers_scheduling import schedule_task

    return asyncio.run(schedule_task(club.db, club.ws, params))


def _grants(club):
    from core.models.approval_grants import ApprovalGrant

    return club.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == club.ws).count()


# ── The list, its verb, its card text and its schema ─────────────────────────────────

def test_schedule_task_is_owner_only_with_a_verb_card_text_and_an_agent_id():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.card_question import READERS

    assert SCHEDULE in owner_only.OWNER_ONLY_ACTIONS and owner_only.is_owner_only(SCHEDULE, {})
    assert owner_only.VERBS[SCHEDULE] == "set an agent's timer" and SCHEDULE in READERS
    assert "agent_id" in get_action_registry().get(SCHEDULE).parameters["properties"]


# ── From the owner's chat: the card, then the click files exactly that row ───────────

def test_the_chats_schedule_task_asks_runs_nothing_and_names_agent_schedule_and_brief(club, rows):
    params = {**MONDAYS, "target_agent_name": club.desk.name, "_agent_id": club.auto.id}
    ask = _run(club, params, _owners_chat(club))

    assert ask["requires_confirmation"] is True and ask["owner_only"] is True and ask["grant_id"]
    assert rows == []                                                       # nothing was scheduled
    asked = ask["question_md"]
    assert asked.startswith(f"Set an agent's timer '{club.desk.name}' (agent #{club.desk.id})")
    assert f"- agent: '{club.desk.name}' (agent #{club.desk.id})" in asked   # the active one, never #off
    assert f"(agent #{club.off.id})" not in asked
    assert "- runs: Mondays at 09:00 (UTC)" in asked
    assert "- delivered as: a ticket on the board, filed when it fires" in asked
    assert "- brief: Email the supplier every Monday to confirm the order (+1 more line)" in asked
    assert "Copy the owner in" not in asked                                  # the first line, the rest counted
    assert "- review: human (waits in Review for you before it closes)" in asked   # as the click files it


def test_the_click_files_exactly_the_row_asked(club, rows):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    chat = _owners_chat(club)
    ask = _run(club, {**MONDAYS, "target_agent_name": club.desk.name, "_agent_id": club.auto.id}, chat)
    clicked = {key: value for key, value in club.db.get(ApprovalGrant, ask["grant_id"]).details["params"].items()
               if not str(key).startswith("_")}
    assert clicked["agent_id"] == club.desk.id and "target_agent_name" not in clicked   # bound before the card

    grant_grant(club.db.get(ApprovalGrant, ask["grant_id"]), granted_by=f"user:{club.owner}")
    club.db.flush()
    done = _run(club, {**clicked, "_agent_id": club.auto.id, "_user_id": "user_owner"}, chat)

    assert done["success"] is True and done["approved_via_grant_id"] == ask["grant_id"]
    assert len(rows) == 1
    row = rows[0]
    assert row["target_agent_id"] == club.desk.id and row["created_by_agent_id"] == club.auto.id
    assert (row["task_type"], row["schedule"], row["description"], row["deliver_as"]) == (
        "recurring", "0 9 * * 1", BRIEF, "board_task")
    assert row["payload"]["review_mode"] == "human"                           # its brief sends: reviewed

    again = _run(club, {**clicked, "_agent_id": club.auto.id, "_user_id": "user_owner"}, chat)
    assert again["requires_confirmation"] is True and len(rows) == 1          # one click, one row


def test_a_chat_delivery_card_names_the_caller_and_a_one_shot_in_words(club):
    from modules.tools.discovery.card_question_timers import schedule_task_lines

    lines = schedule_task_lines(club.db, club.ws, SCHEDULE, {
        "task_type": "one_shot", "schedule": "2030-01-07T09:00:00Z", "description": "Check the float",
        "_agent_id": club.auto.id})
    assert f"- agent: '{club.auto.name}' (agent #{club.auto.id})" in lines
    assert "- runs: once, on Mon 07 Jan 2030 at 09:00 (UTC)" in lines
    assert "- delivered as: a chat with the agent, opened when it fires" in lines
    assert "- brief: Check the float" in lines
    inbox = schedule_task_lines(club.db, club.ws, SCHEDULE, {**MONDAYS, "_agent_id": club.auto.id})
    assert "- agent: none named; the ticket goes to the board's Inbox" in inbox
    named = schedule_task_lines(club.db, club.ws, SCHEDULE, {**MONDAYS, "review_mode": "auto"})
    assert "- review: auto (closes by itself when done)" in named              # the call named it: shown


# ── A scheduled brief that sends is reviewed by a person (FX-010, D7) ───────────────

def test_a_scheduled_brief_that_sends_is_filed_for_review(club, rows):
    from modules.tools.discovery.brief_sends import REVIEW_HELD

    out = _handler(club, {**MONDAYS, "description": "Email the supplier to confirm the order",
                          "_agent_id": club.auto.id, "_user_id": "user_owner"})
    assert rows[-1]["payload"]["review_mode"] == "human"
    assert out["success"] is True and out[REVIEW_HELD] is True and out["review_mode"] == "human"


@pytest.mark.parametrize("said, user, review", [
    ({"review_mode": "auto"}, "user_owner", "auto"),                       # the call named one
    ({"description": "Draft a reply to Declan"}, "user_owner", "auto"),    # a draft only
    ({}, None, "auto"),                                                    # an agent's own run (FX-010)
], ids=["named", "drafts", "agent-run"])
def test_a_named_review_mode_a_draft_or_an_agents_own_run_keeps_its_review(club, rows, said, user, review):
    params = {**MONDAYS, **said, "_agent_id": club.auto.id}
    if user:
        params["_user_id"] = user
    _handler(club, params)
    assert rows[-1]["payload"]["review_mode"] == review


# ── The agent: an id wins; a clash lists the candidates; never a switched-off namesake ─

def test_two_active_ops_list_the_candidates_then_the_id_call_schedules(club, rows):
    first, second = club.ops
    refused = _handler(club, {**MONDAYS, "target_agent_name": first.name, "_agent_id": club.auto.id})
    assert refused["success"] is False and rows == []
    assert f"{first.id} · {first.name}" in refused["error"] and f"{second.id} · {second.name}" in refused["error"]
    assert "call again with agent_id" in refused["error"]

    done = _handler(club, {**MONDAYS, "agent_id": second.id, "target_agent_name": first.name,
                           "_agent_id": club.auto.id})
    assert done["success"] is True and rows[-1]["target_agent_id"] == second.id


def test_a_clash_from_the_owners_chat_is_refused_before_any_card(club, rows):
    reply = _run(club, {**MONDAYS, "target_agent_name": club.ops[0].name, "_agent_id": club.auto.id},
                 _owners_chat(club))
    assert reply["success"] is False and "requires_confirmation" not in reply
    assert "call again with agent_id" in reply["error"] and _grants(club) == 0 and rows == []


def test_an_inactive_namesake_beside_an_active_one_is_never_picked(club, rows):
    _handler(club, {**MONDAYS, "target_agent_name": club.desk.name.lower(), "_agent_id": club.auto.id})
    assert rows[-1]["target_agent_id"] == club.desk.id

    off = _handler(club, {**MONDAYS, "agent_id": club.off.id, "_agent_id": club.auto.id})
    assert off["success"] is False and "switched off" in off["error"] and len(rows) == 1


def test_a_name_no_active_agent_carries_is_not_found(club, rows):
    out = _handler(club, {**MONDAYS, "target_agent_name": f"NOBODY-{club.tag}", "_agent_id": club.auto.id})
    assert out["success"] is False and "not found" in out["error"] and rows == []


# ── An agent's run, a ticket or a playbook step is unchanged ─────────────────────────

@pytest.mark.parametrize("lane", [None, {"board_task_id": 2318}, {"playbook_execution_id": "exec-1"}],
                         ids=["agent-run", "board-ticket", "playbook-step"])
def test_an_agent_runs_schedule_task_does_not_ask(club, rows, lane):
    out = _run(club, {**MONDAYS, "target_agent_name": club.desk.name, "_agent_id": club.auto.id}, lane)
    assert out["success"] is True and "requires_confirmation" not in out
    assert rows[-1]["target_agent_id"] == club.desk.id and _grants(club) == 0
