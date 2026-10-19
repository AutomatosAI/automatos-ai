"""#1073 — the debounced Knowledge Graph rebuild is scheduled on the running loop.

``schedule_incremental_update`` called the deprecated ``asyncio.get_event_loop()``
to set its debounce timer. It now uses ``asyncio.get_running_loop()``, like the
async methods in ``graph_service.py``, and the timer fires on that loop.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from modules.knowledge import graph_service
from modules.knowledge.graph_service import GraphifyService

_FIRE_TIMEOUT_SECONDS = 1


def test_schedule_uses_running_loop():
    """Regression test for issue #1073: the rebuild timer fires on the active event loop."""

    async def check_scheduling():
        service = GraphifyService()
        fired = asyncio.Event()
        rebuilt: list[str] = []

        async def fake_rebuild(workspace_id: str) -> None:
            rebuilt.append(workspace_id)
            fired.set()

        with patch(
            "modules.knowledge.graph_service.asyncio.get_event_loop",
            side_effect=AssertionError("Legacy event loop API was called"),
        ), patch.object(graph_service, "_DEBOUNCE_SECONDS", 0), patch.object(
            service, "_debounced_rebuild", new=fake_rebuild
        ):
            service.schedule_incremental_update(
                "test-workspace",
                [{"type": "document", "id": 1}],
            )
            await asyncio.wait_for(fired.wait(), timeout=_FIRE_TIMEOUT_SECONDS)

        assert rebuilt == ["test-workspace"]

    asyncio.run(check_scheduling())
