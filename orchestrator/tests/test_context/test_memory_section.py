"""Unit tests for MemorySection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.context.sections.base import SectionContext
from modules.context.sections.memory import MemorySection


@pytest.fixture
def memory():
    return MemorySection()


def _ctx(messages=None, skip_memory=False, **kwargs):
    """Build a minimal SectionContext for memory tests."""
    kw = dict(kwargs)
    if skip_memory:
        kw["skip_memory"] = True
    return SectionContext(
        agent=SimpleNamespace(id=42, name="MemAgent"),
        workspace_id="ws_mem_test",
        messages=messages or [{"role": "user", "content": "What do you remember?"}],
        kwargs=kw,
    )


class TestMemorySectionRender:
    """Tests for successful memory rendering."""

    @pytest.mark.asyncio
    async def test_skip_memory_returns_empty(self, memory):
        ctx = _ctx(skip_memory=True)
        result = await memory.render(ctx)
        assert result == ""

    @pytest.mark.asyncio
    @patch("modules.context.sections.memory.MemorySection._try_context_router")
    @patch("modules.context.sections.memory.MemorySection._build_from_smart_memory")
    async def test_falls_back_to_smart_memory_when_context_router_returns_none(
        self, mock_smart, mock_router, memory
    ):
        mock_router.return_value = None
        mock_smart.return_value = "## What You Know About This User\n\n- Likes coffee"

        ctx = _ctx()
        result = await memory.render(ctx)

        assert "Likes coffee" in result
        mock_smart.assert_called_once()

    @pytest.mark.asyncio
    @patch("modules.context.sections.memory.MemorySection._try_context_router")
    async def test_uses_context_router_when_available(self, mock_router, memory):
        bundle = SimpleNamespace(
            long_term_memories=[{"memory": "User prefers dark mode"}],
            session_summary=None,
            daily_logs=None,
            temporal_results=None,
            knowledge_awareness=None,
        )
        mock_router.return_value = bundle

        ctx = _ctx()
        result = await memory.render(ctx)

        assert "dark mode" in result

    @pytest.mark.asyncio
    async def test_stashes_memory_context_in_kwargs(self, memory):
        """Rendered memory text should be stored in ctx.kwargs['_memory_context']."""
        bundle = SimpleNamespace(
            long_term_memories=[{"memory": "remembers this"}],
            session_summary=None,
            daily_logs=None,
            temporal_results=None,
            knowledge_awareness=None,
        )
        with patch.object(memory, "_try_context_router", return_value=bundle):
            ctx = _ctx()
            await memory.render(ctx)

        assert "_memory_context" in ctx.kwargs
        assert "remembers this" in ctx.kwargs["_memory_context"]


class TestMemorySectionFailureResilience:
    """Memory retrieval failures must NEVER crash the prompt build."""

    @pytest.mark.asyncio
    async def test_render_catches_all_exceptions(self, memory):
        """If _build() raises, render() returns empty string."""
        with patch.object(
            memory, "_build", side_effect=RuntimeError("total failure")
        ):
            ctx = _ctx()
            result = await memory.render(ctx)

        assert result == ""

    @pytest.mark.asyncio
    @patch("modules.context.sections.memory.MemorySection._try_context_router")
    @patch("modules.context.sections.memory.MemorySection._build_from_smart_memory")
    async def test_smart_memory_failure_returns_empty(
        self, mock_smart, mock_router, memory
    ):
        mock_router.return_value = None
        mock_smart.side_effect = Exception("smart memory exploded")

        ctx = _ctx()
        # _build will raise, render catches it
        result = await memory.render(ctx)
        assert result == ""

    @pytest.mark.asyncio
    async def test_no_messages_no_task_returns_empty(self, memory):
        """With no user messages and no task_description, query is empty -> no memories."""
        with (
            patch.object(memory, "_try_context_router", return_value=None),
            patch.object(memory, "_build_from_smart_memory", return_value=""),
        ):
            ctx = _ctx(messages=[])
            result = await memory.render(ctx)
            assert result == ""


class TestMemorySectionAttributes:
    """Tests for section metadata."""

    def test_name(self, memory):
        assert memory.name == "memory"

    def test_priority(self, memory):
        assert memory.priority == 6

    def test_max_tokens(self, memory):
        assert memory.max_tokens == 1500


class TestMemorySectionExtractQuery:
    """Tests for the _extract_query helper."""

    def test_extracts_latest_user_message(self):
        ctx = _ctx(messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second question"},
        ])
        result = MemorySection._extract_query(ctx)
        assert result == "second question"

    def test_falls_back_to_task_description(self):
        ctx = SectionContext(
            agent=SimpleNamespace(id=1, name="A"),
            workspace_id="ws_1",
            messages=[],
            task_description="Write a report",
            kwargs={},
        )
        result = MemorySection._extract_query(ctx)
        assert result == "Write a report"

    def test_returns_empty_when_nothing_available(self):
        ctx = SectionContext(
            agent=SimpleNamespace(id=1, name="A"),
            workspace_id="ws_1",
            messages=[],
            task_description=None,
            kwargs={},
        )
        result = MemorySection._extract_query(ctx)
        assert result == ""
