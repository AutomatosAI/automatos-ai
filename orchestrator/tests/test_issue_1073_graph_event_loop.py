
import asyncio
from unittest.mock import patch

from modules.knowledge.graph_service import GraphifyService


def test_schedule_uses_running_loop():
    """Regression test for issue #1073: ensure graph updates use the active asyncio event loop."""

    async def check_scheduling():
        service = GraphifyService()

        # Schedule an update using the currently running event loop.
        with patch(
            "modules.knowledge.graph_service.asyncio.get_event_loop",
            side_effect=AssertionError("Legacy event loop API was called"),
        ):
            service.schedule_incremental_update(
                "test-workspace",
                [{"type": "document", "id": 1}],
            )

        handle = service._debounce_handles["test-workspace"]

        # Verify the scheduled handle belongs to the running event loop.
        try:
            assert not handle.cancelled()
            assert handle._loop is asyncio.get_running_loop()
        finally:
            handle.cancel()

    asyncio.run(check_scheduling())
