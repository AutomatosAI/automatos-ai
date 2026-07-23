"""Tool-surface bug fixes (PR-A of TOOL-SURFACE-DEEP-REVIEW-2026-07-23).

Three plumbing faults, no design change:

1. The full chat path computed its tool surface (with ``is_super_admin`` and
   PRD-221 ``page_actions`` threaded in) at service.py::_get_tools — then
   ``smart_chat.prepare(available_tools=…)`` documented it as ignored and
   ToolsSection rebuilt the surface WITHOUT them. The page-prior union and su
   widening never reached the LLM, and get_tools_for_agent_async ran twice
   per turn. Fix: the prebuilt surface rides ``prebuilt_tools`` through
   build_context into ToolsSection, which uses it instead of rebuilding.

2. Heartbeat / task-execution build_context calls passed ``task_description``
   but no ``query`` — so the dispatcher enum never narrowed on those lanes
   (full 137 actions on every heartbeat, forever).

3. ActionSemanticIndex launched a live query-embed per caller with no
   in-flight dedup — concurrent same-query turns each paid for (and raced)
   the same embedding.

Pure unit tests — no DB, no Redis, no network.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

_ORCH = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1a. ToolsSection honors a prebuilt surface (FILTERED + FULL strategies)
# ---------------------------------------------------------------------------


class _Boom:
    """get_tools_for_agent_async stand-in that must NOT be called."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls += 1
        raise AssertionError(
            "get_tools_for_agent_async was called although a prebuilt surface "
            "was supplied — the double-build is back"
        )


class _Rebuilt:
    """get_tools_for_agent_async stand-in returning a sentinel surface."""

    def __init__(self) -> None:
        self.calls = 0
        self.kwargs: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> List[Dict[str, Any]]:
        self.calls += 1
        self.kwargs.append(kwargs)
        return [_tool("rebuilt_tool")]


def _tool(name: str) -> Dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": name, "parameters": {}}}


@pytest.mark.asyncio
async def test_filtered_strategy_ships_the_prebuilt_surface() -> None:
    """A non-empty prebuilt surface is authoritative: no rebuild happens and
    the prebuilt tools (which carry the page-prior + su threading from the
    chat entrypoint) are what ships."""
    from modules.context.sections.tools import ToolsSection, ToolLoadingStrategy

    prebuilt = [_tool("platform_execute"), _tool("page_prior_tool")]
    boom = _Boom()
    with patch("modules.tools.tool_router.get_tools_for_agent_async", new=boom):
        tools, tool_choice = await ToolsSection().load_tools(
            agent_id=1,
            workspace_id="ws-1",
            strategy=ToolLoadingStrategy.FILTERED,
            db_session=None,
            prebuilt_tools=prebuilt,
            # No query/hints/context → SmartToolRouter step is skipped; the
            # prebuilt short-circuit is what's under test here.
        )
    assert boom.calls == 0
    assert tools == prebuilt
    assert tool_choice == "auto"


@pytest.mark.asyncio
async def test_full_strategy_ships_the_prebuilt_surface() -> None:
    from modules.context.sections.tools import ToolsSection, ToolLoadingStrategy

    prebuilt = [_tool("platform_execute")]
    boom = _Boom()
    with patch("modules.tools.tool_router.get_tools_for_agent_async", new=boom):
        tools, _ = await ToolsSection().load_tools(
            agent_id=1,
            workspace_id="ws-1",
            strategy=ToolLoadingStrategy.FULL,
            db_session=None,
            prebuilt_tools=prebuilt,
        )
    assert boom.calls == 0
    assert tools == prebuilt


@pytest.mark.asyncio
async def test_empty_prebuilt_falls_back_to_rebuild() -> None:
    """An EMPTY prebuilt list means the entrypoint's build failed (tool_router
    returns [] on error) — the section rebuilds rather than shipping nothing."""
    from modules.context.sections.tools import ToolsSection, ToolLoadingStrategy

    rebuilt = _Rebuilt()
    with patch("modules.tools.tool_router.get_tools_for_agent_async", new=rebuilt):
        tools, _ = await ToolsSection().load_tools(
            agent_id=1,
            workspace_id="ws-1",
            strategy=ToolLoadingStrategy.FILTERED,
            db_session=None,
            prebuilt_tools=[],
        )
    assert rebuilt.calls == 1
    assert tools == [_tool("rebuilt_tool")]


@pytest.mark.asyncio
async def test_no_prebuilt_keeps_todays_rebuild_path() -> None:
    from modules.context.sections.tools import ToolsSection, ToolLoadingStrategy

    rebuilt = _Rebuilt()
    with patch("modules.tools.tool_router.get_tools_for_agent_async", new=rebuilt):
        tools, _ = await ToolsSection().load_tools(
            agent_id=7,
            workspace_id="ws-2",
            strategy=ToolLoadingStrategy.FILTERED,
            db_session=None,
        )
    assert rebuilt.calls == 1
    assert tools == [_tool("rebuilt_tool")]
    assert rebuilt.kwargs[0]["agent_id"] == 7


# ---------------------------------------------------------------------------
# 1b. ContextService threads ctx.kwargs["prebuilt_tools"] into ToolsSection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_service_threads_prebuilt_tools() -> None:
    from modules.context.service import ContextService
    from modules.context.sections import SectionContext
    from modules.context.sections.tools import ToolsSection

    captured: Dict[str, Any] = {}

    async def _capture(self: Any, **kwargs: Any):  # noqa: ANN401
        captured.update(kwargs)
        return kwargs.get("prebuilt_tools") or [], "auto"

    ctx = SectionContext(
        agent=SimpleNamespace(id=1),
        workspace_id="ws-1",
        db_session=None,
        messages=[],
        kwargs={"prebuilt_tools": [_tool("threaded")], "query": "hello"},
    )
    config = SimpleNamespace(tool_loading="filtered")
    with patch.object(ToolsSection, "load_tools", _capture):
        tools, _ = await ContextService._load_tools(config, ctx)

    assert captured.get("prebuilt_tools") == [_tool("threaded")]
    assert tools == [_tool("threaded")]


# ---------------------------------------------------------------------------
# 1c. prepare_request forwards available_tools as the prebuilt surface
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prepare_request_forwards_available_tools_as_prebuilt() -> None:
    """The orchestrator's ``available_tools`` parameter stops being decorative:
    it must reach build_context as ``prebuilt_tools``."""
    from consumers.chatbot.smart_orchestrator import SmartChatOrchestrator, ConversationState
    import modules.context as context_pkg

    orch = SmartChatOrchestrator.__new__(SmartChatOrchestrator)
    orch.workspace_id = "ws-1"
    orch.agent_id = 1
    orch.agent_name = "Auto"
    orch.widget_mode = False
    orch._db_session = None
    orch.state = ConversationState()
    orch.classifier = SimpleNamespace(
        classify=lambda q, m: SimpleNamespace(
            primary_intent=SimpleNamespace(value="chat"),
            requires_tools=False,
            requires_memory=False,
            confidence=1.0,
        )
    )
    orch._load_agent = lambda: SimpleNamespace(id=1)
    orch._build_compat_memory_result = lambda context: None

    captured: Dict[str, Any] = {}

    class _FakeContextService:
        def __init__(self, _db: Any) -> None:  # noqa: ANN401
            pass

        async def build_context(self, **kwargs: Any):  # noqa: ANN401
            captured.update(kwargs)
            return SimpleNamespace(
                system_prompt="",
                messages=[],
                tools=kwargs.get("prebuilt_tools") or [],
                tool_choice="auto",
                memory_context=None,
                user_name=None,
                to_assembly_trace=lambda: [],
            )

    sentinel = [_tool("prebuilt_marker")]
    with patch.object(context_pkg, "ContextService", _FakeContextService):
        result = await orch.prepare_request(
            messages=[{"role": "user", "content": "hi"}],
            available_tools=sentinel,
        )

    assert captured.get("prebuilt_tools") == sentinel
    assert result.tools == sentinel


# ---------------------------------------------------------------------------
# 2. Heartbeat + task-execution lanes thread a query for enum narrowing
# ---------------------------------------------------------------------------
# Source-probes (authz_sweep_probe precedent): these call sites sit deep in
# DB-bound methods; what matters — and what regressed silently before — is
# that the build_context call passes a query so _load_dispatcher_only /
# _load_full can narrow the enum.


def _call_site(text: str, anchor: str, window: int = 900) -> str:
    idx = text.index(anchor)
    return text[idx : idx + window]


def test_heartbeat_build_context_threads_query() -> None:
    src = (_ORCH / "services" / "heartbeat_service.py").read_text()
    site = _call_site(src, "ContextMode.HEARTBEAT_ORCHESTRATOR")
    assert re.search(r"query\s*=", site), (
        "heartbeat build_context passes no query — the dispatcher enum ships "
        "all 137 actions on every heartbeat run"
    )


def test_task_execution_build_context_threads_query() -> None:
    src = (_ORCH / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    site = _call_site(src, "context_result = await ContextService")
    assert re.search(r"query\s*=", site), (
        "agent_factory build_context passes no query — the dispatcher enum "
        "ships all 137 actions on every task execution"
    )


# ---------------------------------------------------------------------------
# 3. Concurrent same-query embeds dedup to ONE live call
# ---------------------------------------------------------------------------


class _SlowEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_embedding(self, text: str) -> List[float]:
        self.calls += 1
        await asyncio.sleep(0.05)
        return [1.0, 0.0, 0.0]


class _MissCache:
    def __init__(self) -> None:
        self.set_calls: List[Dict[str, Any]] = []

    def get_embeddings_batch(self, texts: List[str], model: str = "d") -> Dict[str, Optional[List[float]]]:
        return {t: None for t in texts}

    def set_embeddings_batch(self, embeddings: Dict[str, List[float]], model: str = "d") -> None:
        self.set_calls.append(dict(embeddings))


def _bare_index() -> Any:  # noqa: ANN401
    from modules.tools.discovery.action_semantic_index import ActionSemanticIndex

    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = _SlowEmbedder()
    idx._cache = _MissCache()
    idx._inflight = {}
    return idx


@pytest.mark.asyncio
async def test_concurrent_same_query_embeds_once() -> None:
    idx = _bare_index()
    v1, v2 = await asyncio.gather(
        idx._embed_query_bounded("hello there", "mk", timeout_s=5.0),
        idx._embed_query_bounded("hello there", "mk", timeout_s=5.0),
    )
    assert idx._embedding_manager.calls == 1, "two live embeds for one query"
    assert v1[0] == v2[0] == [1.0, 0.0, 0.0]
    assert not v1[2] and not v2[2]  # neither timed out


@pytest.mark.asyncio
async def test_inflight_entry_cleared_after_completion() -> None:
    idx = _bare_index()
    await idx._embed_query_bounded("hello there", "mk", timeout_s=5.0)
    await asyncio.sleep(0)  # let done-callbacks run
    assert idx._inflight == {}, "in-flight table leaked a completed task"


@pytest.mark.asyncio
async def test_distinct_queries_do_not_dedup() -> None:
    idx = _bare_index()
    await asyncio.gather(
        idx._embed_query_bounded("alpha", "mk", timeout_s=5.0),
        idx._embed_query_bounded("beta", "mk", timeout_s=5.0),
    )
    assert idx._embedding_manager.calls == 2
