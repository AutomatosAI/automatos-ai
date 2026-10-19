"""PRD-263 US-020 — the /approve lookup sees every [HARNESS] card until PR-3A deletes it.

``handle_harness_command`` listed the workspace's HARNESS cards with one
``platform_list_tasks`` call, which returns 20 cards by default
(modules/tools/discovery/board_task_list.py), so a card older than the newest 20 was
"not found". It now follows ``next_offset`` until the last page. PR-3A (US-047) deletes
this reader, and this test with it.

The setup (the apscheduler stub, the faked database and board task) comes from
test_harness_commands.py, the suite of the same handler.
"""
from __future__ import annotations

import asyncio

from tests.test_harness_commands import (
    _ADMIN,
    _ADMIN_MEMBER,
    _WS_ID,
    _FakeBoardTask,
    _FakeDB,
    _harness_task,
    _patch_executor,
    config,
    hc,
)

PAGE = 20  # platform_list_tasks' default page (board_task_list.DEFAULT_LIST_TASKS_LIMIT)
CARDS = 25


class _PagedExecutor:
    """platform_list_tasks as the handler returns it: one page, ``next_offset`` until the last."""

    def __init__(self, cards):
        self._cards = cards
        self.offsets = []

    async def execute(self, action, params, caller_context=None):
        assert action == "platform_list_tasks" and params["tags"] == ["harness"]
        offset = params.get("offset") or 0
        self.offsets.append(offset)
        page = self._cards[offset:offset + PAGE]
        after = offset + len(page)
        return {"success": True, "tasks": page, "total": len(self._cards), "offset": offset, "limit": PAGE,
                "returned": len(page), "next_offset": after if after < len(self._cards) else None}


def test_the_25th_harness_card_is_found(monkeypatch):
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    cards = [_harness_task(task_id=n, rx_id=f"rx-{n}") for n in range(1, CARDS + 1)]
    executor = _PagedExecutor(cards)
    _patch_executor(monkeypatch, executor)
    oldest = _FakeBoardTask(CARDS)
    db = _FakeDB(member=_ADMIN_MEMBER, board_task=oldest)

    result = asyncio.run(hc.handle_harness_command(db, _WS_ID, "/reject", f"rx-{CARDS}", _ADMIN))

    assert result["success"] is True and result["rejected"] is True
    assert oldest.status == "rejected"
    assert executor.offsets == [0, PAGE]  # two pages, then next_offset is None


def test_a_page_that_does_not_move_forward_ends_the_walk(monkeypatch):
    class _Stuck(_PagedExecutor):
        async def execute(self, action, params, caller_context=None):
            self.offsets.append(params.get("offset") or 0)
            return {"success": True, "tasks": [], "next_offset": 0}

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    executor = _Stuck([])
    _patch_executor(monkeypatch, executor)

    result = asyncio.run(hc.handle_harness_command(_FakeDB(member=_ADMIN_MEMBER), _WS_ID, "/reject", "rx-1", _ADMIN))

    assert result["success"] is False and executor.offsets == [0]
