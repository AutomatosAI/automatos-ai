"""F249 (night 7): a Reject's lesson reaches the agent's next card.

"Just the email" went to the Support Agent three times (#0155, #0158, #0171), and
"leave off Perfect!" three times (#0141, #0152, #0161): each Reject's note stayed on
the ticket it was written on. Every run of an agent's ticket now carries the owner's
recent corrections to that agent on its other tickets, newest first and each once,
on both claim paths (the API dispatcher's and the CLI host's).
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

from services.ticket_redo import STANDING_HEADING   # F318 (night 9b): headed as the owner's rules now

JUST_THE_EMAIL = "Just the email please - no subject line options, no notes to me."
NO_PERFECT = "Leave off 'Perfect!' at the start."


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the F249 tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


class _Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture
def shop(engine, new_session, monkeypatch):
    import api.board_tasks as bt

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f249')"), {"id": ws})
    agents = {name: s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"n": name, "w": ws}).scalar()
        for name in ("Support Agent", "Content Creator")}
    s.commit()
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: None)
    yield NS(ws=ws, agents=agents, new=new_session, ctx=NS(workspace_id=UUID(ws), user=NS(id=7, email="owner@cafe.test")))
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})  # noqa: S608
    s.commit()


def _ticket(shop, agent, title, status="assigned"):
    s = shop.new()
    tid = s.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, raw_prompt, status, assigned_agent_id, review_mode) "
        "VALUES (CAST(:w AS uuid), :t, :t, :s, :a, 'human') RETURNING id"),
        {"w": shop.ws, "t": title, "s": status, "a": shop.agents[agent]}).scalar()
    s.commit()
    return tid


def _sent_back(shop, ticket_id, note):
    import api.board_tasks as bt

    s = shop.new()
    s.execute(text("UPDATE board_tasks SET status = 'review', result = 'Subject options: …' WHERE id = :i"),
              {"i": ticket_id})
    s.commit()
    asyncio.run(bt.reject_task(ticket_id, _Req({"feedback": note}), ctx=shop.ctx, db=shop.new()))


def _finish(shop, *ticket_ids):
    s = shop.new()
    s.execute(text("UPDATE board_tasks SET status = 'done' WHERE id = ANY(:ids)"), {"ids": list(ticket_ids)})
    s.commit()


def _session_prompt(shop, ticket_id):
    from core.models.core import BoardTask
    from services.cli_host_service import _ticket_prompt

    return _ticket_prompt(shop.new().get(BoardTask, ticket_id))


def _dispatch_prompt(shop, ticket_id):
    from config import config
    from services.board_dispatcher import _claim_and_sweep

    claimed = _claim_and_sweep(shop.new, config, "w-f249")["claimed"]
    return next(c["prompt"] for c in claimed if c["task_id"] == ticket_id)


def test_the_agents_next_card_carries_the_owners_corrections_newest_first_each_once(shop):
    first = _ticket(shop, "Support Agent", "Reply to Theo Frost", status="review")
    second = _ticket(shop, "Support Agent", "Reply to Ellie Shaw", status="review")
    _sent_back(shop, first, NO_PERFECT)
    _sent_back(shop, second, JUST_THE_EMAIL)
    _sent_back(shop, first, "just the email please -  no subject line options, no notes to me.")  # the same, again
    _finish(shop, first, second)                         # only the new card is the dispatcher's to claim
    nxt = _ticket(shop, "Support Agent", "Reply to the Lantern Hill café")

    for prompt in (_session_prompt(shop, nxt), _dispatch_prompt(shop, nxt)):
        assert prompt.startswith("Reply to the Lantern Hill café")
        block = prompt.split(f"{STANDING_HEADING}\n", 1)[1]
        assert block.splitlines()[1:] == ["- just the email please -  no subject line options, no notes to me.",
                                          f"- {NO_PERFECT}"]


def test_another_agent_never_gets_them(shop):
    _sent_back(shop, _ticket(shop, "Support Agent", "Reply to Theo Frost", status="review"), JUST_THE_EMAIL)
    theirs = _ticket(shop, "Content Creator", "Instagram caption for the Guji")

    assert STANDING_HEADING not in _session_prompt(shop, theirs)


def test_a_redo_keeps_its_own_corrections_in_the_redo_and_adds_the_rest(shop):
    other = _ticket(shop, "Support Agent", "Reply to Theo Frost", status="review")
    mine = _ticket(shop, "Support Agent", "Reply to Ellie Shaw", status="review")
    _sent_back(shop, other, NO_PERFECT)
    _sent_back(shop, mine, JUST_THE_EMAIL)

    prompt = _session_prompt(shop, mine)

    assert f"1. {JUST_THE_EMAIL}" in prompt                       # its own, in the redo part
    standing = prompt.split(f"{STANDING_HEADING}\n", 1)[1]
    assert f"- {NO_PERFECT}" in standing and JUST_THE_EMAIL not in standing


def test_a_ticket_with_no_session_or_agent_has_none():
    from services.ticket_redo import redo_block

    assert redo_block(NS(id=1, assigned_agent_id=7, review_feedback=None, planning_data={})) is None
