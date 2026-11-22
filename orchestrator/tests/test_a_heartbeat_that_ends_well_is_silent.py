"""A heartbeat's ticket, and Auto's review of it, end done without a task_complete notice.

On an install where an agent's heartbeat runs as a board ticket, one routine tick that
found nothing to do told the owner four times: the heartbeat ticket's ``task_complete``
and its report's ``report_submitted``, then the same pair for the ticket Auto was given
to review it (``services.heartbeats.auto_review``). The report's notice is gone for every
ticket (test_a_tickets_ending_rings_the_bell_once); this is the ``task_complete``.

Both tickets are filed with ``source_type == "heartbeat"`` (``cli_ticket_lane`` for the
heartbeat's session, ``auto_review`` for Auto's), whatever their titles. For them
``_dispatch_task_complete`` sends nothing and still runs its watch and knowledge hooks.
A failure still raises ``task_failed``; any other ticket still raises ``task_complete``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"


def _ticket(**over):
    base = dict(id=330, title="Heartbeat: News & Research", status="done", assigned_agent_id=None,
                result="Nothing is assigned to me.", description=None, error_message=None,
                source_type="heartbeat", source_id="agent:17:2026-11-17T09:00")
    base.update(over)
    return SimpleNamespace(**base)


# A heartbeat's session ticket; Auto's review of a heartbeat (its source id is the tick's).
HEARTBEAT_TICKET = _ticket()
AUTO_REVIEW_TICKET = _ticket(id=331, title="Review News & Research heartbeat — Nothing is assigned to me.",
                             source_id="171")


@pytest.fixture
def ending(monkeypatch):
    import core.services.notification_dispatcher as nd
    import services.report_knowledge as knowledge
    import services.watch_hooks as hooks

    seen = SimpleNamespace(notices=[], watched=[], filed=[])

    class _Dispatcher:
        def __init__(self, db, ws):
            pass

        async def dispatch(self, **kw):
            seen.notices.append(kw["event_type"])
            return {}

    async def _file_done(db, workspace_id, task):
        seen.filed.append(task.id)

    monkeypatch.setattr(nd, "NotificationDispatcher", _Dispatcher)
    monkeypatch.setattr(hooks, "watch_ingest_terminal", lambda db, **kw: seen.watched.append(kw["terminal_state"]))
    monkeypatch.setattr(knowledge, "file_done_ticket", _file_done)
    return seen


@pytest.mark.parametrize("task", [HEARTBEAT_TICKET, AUTO_REVIEW_TICKET], ids=["heartbeat", "auto_review"])
def test_a_heartbeat_ticket_that_ends_done_sends_no_notice_but_runs_its_hooks(ending, task):
    from api.board_tasks import _dispatch_task_complete

    asyncio.run(_dispatch_task_complete(object(), WS, task))
    assert ending.notices == []
    assert ending.watched == ["completed"] and ending.filed == [task.id]


def test_any_other_ticket_still_says_it_is_complete(ending):
    from api.board_tasks import _dispatch_task_complete

    asyncio.run(_dispatch_task_complete(object(), WS, _ticket(source_type="user", source_id=None)))
    assert ending.notices == ["task_complete"]


def test_a_heartbeat_ticket_that_fails_still_says_so(ending):
    from api.board_tasks import _dispatch_task_failed

    asyncio.run(_dispatch_task_failed(object(), WS, _ticket(status="failed", error_message="The feed was down.")))
    assert ending.notices == ["task_failed"]


def test_both_kinds_of_heartbeat_ticket_are_filed_with_the_field_the_rule_reads():
    from services.heartbeats.auto_review import HEARTBEAT_SOURCE as REVIEW_SOURCE
    from services.heartbeats.session_ticks import HEARTBEAT_SOURCE, is_heartbeat_ticket

    assert REVIEW_SOURCE == HEARTBEAT_SOURCE
    assert is_heartbeat_ticket(HEARTBEAT_TICKET) and is_heartbeat_ticket(AUTO_REVIEW_TICKET)
    assert not is_heartbeat_ticket(_ticket(source_type="scheduled_task", title="Heartbeat check"))
