"""F105 (4 Nov): an agent's plugin context is fetched without stopping the event loop.

The prompt's sections render together (``asyncio.gather``). The plugins section built
its tier 2 (each plugin's SKILL.md from the S3/Redis cache) through a sync wrapper that,
on a running loop, ran the fetch on another thread and blocked on ``.result(timeout=30)``:
every turn of an agent with plugins could hold the loop, and every other request, for
up to 30 s. The section now awaits ``build_tier2_content`` itself, and the wrapper is gone.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

FETCH_S = 0.2
TICK_S = 0.01
MIN_TICKS = 5  # the loop ran other work while the fetch waited


class _Plugins:
    """The plugin context service, with a slow tier-2 fetch."""

    def __init__(self, db) -> None:
        self.db = db

    def get_assigned_plugins(self, agent_id):
        return [(None, SimpleNamespace(materialized_skill_ids=None))]

    def build_tier1_summary(self, rows) -> str:
        return "tier one"

    async def build_tier2_content(self, rows, task_context=None, max_plugins=2) -> str:
        await asyncio.sleep(FETCH_S)
        return "tier two"


def test_tier_two_is_awaited_so_the_loop_keeps_serving(monkeypatch):
    from core.services import plugin_context_service
    from modules.context.sections.plugins import PluginsSection

    monkeypatch.setattr(plugin_context_service, "PluginContextService", _Plugins)
    ctx = SimpleNamespace(agent=SimpleNamespace(id=7), db_session=object(), task_description="launch copy")

    async def build_while_ticking():
        ticks = 0

        async def tick():
            nonlocal ticks
            while True:
                await asyncio.sleep(TICK_S)
                ticks += 1

        ticker = asyncio.create_task(tick())
        content = await PluginsSection()._build(ctx)
        ticker.cancel()
        return content, ticks

    content, ticks = asyncio.run(build_while_ticking())
    assert content == "tier one\n\ntier two"
    assert ticks >= MIN_TICKS


def test_the_blocking_wrapper_is_gone():
    from core.services.plugin_context_service import PluginContextService

    assert not hasattr(PluginContextService, "build_tier2_content_sync")
