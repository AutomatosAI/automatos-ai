"""Integration tests for ContextService.build_context().

Verifies the full assembly pipeline: section composition, parallel
rendering, budget allocation, tool loading, message formatting, and
ContextResult immutability.

External dependencies (DB, memory, tools, ActionRegistry) are mocked
with realistic return values.
"""

import asyncio
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure orchestrator package is importable
_orchestrator_root = str(Path(__file__).resolve().parent.parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.context.modes import ContextMode
from modules.context.result import ContextResult
from modules.context.service import ContextService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent():
    """Realistic Agent-like object for integration tests."""
    skill = SimpleNamespace(
        name="sentinel",
        prompt_template="You are SENTINEL, a security monitoring agent.",
        is_active=True,
    )
    return SimpleNamespace(
        id=42,
        name="Test Agent",
        agent_type="assistant",
        description="A helpful test agent for the platform",
        use_custom_persona=False,
        custom_persona_prompt=None,
        persona=None,
        skills=[skill],
    )


@pytest.fixture
def mock_db():
    """Mock SQLAlchemy session with chainable query API."""
    db = MagicMock()
    q = MagicMock()
    q.join.return_value = q
    q.filter.return_value = q
    q.order_by.return_value = q
    q.all.return_value = []
    q.first.return_value = None
    db.query.return_value = q
    # For custom section's select-based queries
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = []
    db.execute.return_value = result_mock
    return db


@pytest.fixture
def messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "Search the web for AI news"},
        {"role": "assistant", "content": "I'll search for the latest AI news."},
        {"role": "user", "content": "Thanks, also check competitor updates"},
    ]


@pytest.fixture
def mock_platform_actions():
    """Patch PlatformActionsSection._build to return realistic action summary.

    The actual import (get_action_registry) is lazy inside _build(), so we
    mock _build() directly to avoid import-path issues in tests.
    """
    from modules.context.sections.platform_actions import PlatformActionsSection

    summary = (
        "## Available Platform Actions\n\n"
        "**Communication:**\n"
        "- `platform_send_message` — Send a message to a user\n"
        "- `platform_send_email` — Send an email\n\n"
        "**Data:**\n"
        "- `platform_search_knowledge` — Search knowledge base\n"
        "- `platform_store_memory` — Store a memory"
    )
    with patch.object(PlatformActionsSection, "_build", return_value=summary):
        yield summary


@pytest.fixture
def mock_memory():
    """Patch MemorySection._build to return realistic memory content.

    The actual imports (get_unified_memory_service, get_smart_memory_manager)
    are lazy inside _build(), so we mock _build() to avoid import issues
    and stash the memory context as the real section does.
    """
    from modules.context.sections.memory import MemorySection

    memory_content = (
        "## What You Know About This User\n\n"
        "- User prefers concise responses\n"
        "- User works on AI platform\n\n"
        "## Recent Activity\n\n"
        "Agent completed 3 tasks today."
    )

    async def _fake_render(self, ctx):
        if ctx.kwargs.get("skip_memory"):
            return ""
        ctx.kwargs["_memory_context"] = memory_content
        ctx.kwargs["_user_name"] = "Gar"
        return memory_content

    with patch.object(MemorySection, "render", _fake_render):
        yield memory_content


_MOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "platform_execute",
            "description": "Execute a platform action",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_MOCK_DISPATCHER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "platform_execute",
        "description": "Execute a platform action",
        "parameters": {"type": "object", "properties": {}},
    },
}


@pytest.fixture
def mock_tools_full():
    """Patch ToolsSection.load_tools for FULL/FILTERED strategies."""
    from modules.context.sections.tools import ToolsSection

    mock_load = AsyncMock(return_value=(list(_MOCK_TOOLS), "auto"))
    with patch.object(ToolsSection, "load_tools", mock_load):
        yield _MOCK_TOOLS


@pytest.fixture
def mock_tools_dispatcher():
    """Patch ToolsSection.load_tools for DISPATCHER_ONLY strategy."""
    from modules.context.sections.tools import ToolsSection

    mock_load = AsyncMock(return_value=([_MOCK_DISPATCHER_SCHEMA], "auto"))
    with patch.object(ToolsSection, "load_tools", mock_load):
        yield _MOCK_DISPATCHER_SCHEMA


# ---------------------------------------------------------------------------
# CHATBOT mode
# ---------------------------------------------------------------------------


class TestBuildContextChatbot:
    """build_context(CHATBOT) — identity, memory, platform actions, tools, messages."""

    @pytest.mark.asyncio
    async def test_chatbot_contains_identity(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """System prompt contains agent identity."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert "Test Agent" in result.system_prompt
        assert "identity" in result.sections_included

    @pytest.mark.asyncio
    async def test_chatbot_contains_memory(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """System prompt contains memory section."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert "What You Know About This User" in result.system_prompt
        assert "memory" in result.sections_included

    @pytest.mark.asyncio
    async def test_chatbot_contains_platform_actions(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """System prompt contains platform actions catalog."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert "Available Platform Actions" in result.system_prompt
        assert "platform_actions" in result.sections_included

    @pytest.mark.asyncio
    async def test_chatbot_has_tools(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """Tools list is non-empty for chatbot mode."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert len(result.tools) > 0

    @pytest.mark.asyncio
    async def test_chatbot_formats_messages(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """Messages are formatted (system messages stripped)."""
        msgs_with_system = [
            {"role": "system", "content": "Old system prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=msgs_with_system,
        )
        # System messages should be stripped
        roles = [m["role"] for m in result.messages]
        assert "system" not in roles
        assert "user" in roles

    @pytest.mark.asyncio
    async def test_chatbot_memory_context_in_result(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """ContextResult.memory_context is populated for SSE events."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert result.memory_context is not None
        assert "User prefers" in result.memory_context

    @pytest.mark.asyncio
    async def test_chatbot_mode_string(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """Result mode field is 'chatbot'."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert result.mode == "chatbot"

    @pytest.mark.asyncio
    async def test_chatbot_skills_in_prompt(
        self, agent, mock_db, messages, mock_platform_actions, mock_memory, mock_tools_full    ):
        """PRD-202 S2: the skill's L1 metadata (name) appears in the system prompt.

        The fixture skill ('sentinel') is not in the core always-on set, so its
        full body is trigger-loaded (load_skill), not pre-injected — the section
        contributes L1 metadata (name + description) instead.
        """
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id="ws_test",
            messages=messages,
        )
        assert "sentinel" in result.system_prompt  # L1 metadata (name)
        assert "skills" in result.sections_included


# ---------------------------------------------------------------------------
# TASK_EXECUTION mode
# ---------------------------------------------------------------------------


class TestBuildContextTaskExecution:
    """build_context(TASK_EXECUTION) — identity, task_context, tools."""

    @pytest.mark.asyncio
    async def test_task_execution_has_identity(
        self, agent, mock_db, mock_platform_actions, mock_memory, mock_tools_full    ):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Search the web and write a report on AI trends",
        )
        assert "Test Agent" in result.system_prompt
        assert "identity" in result.sections_included

    @pytest.mark.asyncio
    async def test_task_execution_has_task_description(
        self, agent, mock_db, mock_platform_actions, mock_memory, mock_tools_full    ):
        """task_description appears in system_prompt."""
        task_desc = "Search the web and write a report on AI trends"
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description=task_desc,
        )
        assert task_desc in result.system_prompt
        assert "task_context" in result.sections_included

    @pytest.mark.asyncio
    async def test_task_execution_has_tools(
        self, agent, mock_db, mock_platform_actions, mock_memory, mock_tools_full    ):
        """TASK_EXECUTION uses FULL tool loading strategy."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Do stuff",
        )
        assert len(result.tools) >= 2
        assert result.tool_choice == "auto"

    @pytest.mark.asyncio
    async def test_task_execution_mode_string(
        self, agent, mock_db, mock_platform_actions, mock_memory, mock_tools_full    ):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Do stuff",
        )
        assert result.mode == "task_execution"

    @pytest.mark.asyncio
    async def test_task_execution_with_metadata(
        self, agent, mock_db, mock_platform_actions, mock_memory, mock_tools_full    ):
        """Task metadata (status, priority, board) appears when provided."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Write blog post",
            task_status="in_progress",
            task_priority="high",
            board_name="Content Board",
        )
        assert "Status: in_progress" in result.system_prompt
        assert "Priority: high" in result.system_prompt
        assert "Board: Content Board" in result.system_prompt


# ---------------------------------------------------------------------------
# HEARTBEAT mode
# ---------------------------------------------------------------------------


class TestBuildContextHeartbeat:
    """build_context(HEARTBEAT) — small budget, dispatcher-only tools."""

    @pytest.mark.asyncio
    async def test_heartbeat_token_estimate_under_8000(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """Heartbeat mode budget enforces small token usage."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        # Token estimate should be well under 8000 (heartbeat max_tokens)
        assert result.token_estimate < 8000

    @pytest.mark.asyncio
    async def test_heartbeat_dispatcher_only_tools(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """Heartbeat mode loads only the platform_execute dispatcher."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        assert len(result.tools) == 1
        assert result.tools[0]["function"]["name"] == "platform_execute"

    @pytest.mark.asyncio
    async def test_heartbeat_no_messages(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """Heartbeat mode has no conversation messages."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.messages == []

    @pytest.mark.asyncio
    async def test_heartbeat_has_identity(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        assert "Test Agent" in result.system_prompt

    @pytest.mark.asyncio
    async def test_heartbeat_has_datetime(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """Heartbeat includes datetime context."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        assert "Current UTC time:" in result.system_prompt


# ---------------------------------------------------------------------------
# RECIPE mode
# ---------------------------------------------------------------------------


class TestBuildContextRecipe:
    """build_context(RECIPE) — recipe_context with step info."""

    @pytest.mark.asyncio
    async def test_recipe_contains_step_info(
        self, agent, mock_db, mock_platform_actions, mock_tools_full
    ):
        """Recipe context appears with step details."""
        recipe_step = {
            "name": "Content Pipeline",
            "step_number": 2,
            "total_steps": 5,
            "step_name": "Research Phase",
            "instructions": "Search for recent AI developments and compile findings.",
            "previous_output": "Step 1 completed: identified 3 target topics.",
        }
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.RECIPE,
            agent=agent,
            workspace_id="ws_test",
            recipe_step=recipe_step,
        )
        assert "Content Pipeline" in result.system_prompt
        assert "2/5" in result.system_prompt
        assert "Research Phase" in result.system_prompt
        assert "Search for recent AI developments" in result.system_prompt
        assert "playbook_context" in result.sections_included

    @pytest.mark.asyncio
    async def test_recipe_has_tools(
        self, agent, mock_db, mock_platform_actions, mock_tools_full
    ):
        """RECIPE uses FULL tool loading strategy."""
        recipe_step = {
            "name": "Test Recipe",
            "step_number": 1,
            "total_steps": 1,
            "instructions": "Do the thing.",
        }
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.RECIPE,
            agent=agent,
            workspace_id="ws_test",
            recipe_step=recipe_step,
        )
        assert len(result.tools) >= 2

    @pytest.mark.asyncio
    async def test_recipe_previous_output(
        self, agent, mock_db, mock_platform_actions, mock_tools_full
    ):
        """Previous step output appears in recipe context."""
        recipe_step = {
            "name": "Multi-Step Recipe",
            "step_number": 3,
            "total_steps": 4,
            "instructions": "Summarize findings.",
            "previous_output": "Found 5 relevant articles on topic X.",
        }
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.RECIPE,
            agent=agent,
            workspace_id="ws_test",
            recipe_step=recipe_step,
        )
        assert "Previous Step Results" in result.system_prompt
        assert "Found 5 relevant articles" in result.system_prompt


# ---------------------------------------------------------------------------
# ROUTER / ORCHESTRATOR_STAGE / NL2SQL modes (minimal, no tools)
# ---------------------------------------------------------------------------


class TestBuildContextMinimalModes:
    """Modes with minimal sections and no tools."""

    @pytest.mark.asyncio
    async def test_router_no_tools(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.tools == []
        assert result.tool_choice == "none"
        assert result.mode == "router"

    @pytest.mark.asyncio
    async def test_orchestrator_stage_no_tools(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ORCHESTRATOR_STAGE,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.tools == []
        assert result.tool_choice == "none"

    @pytest.mark.asyncio
    async def test_nl2sql_no_tools(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.NL2SQL,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.tools == []
        assert result.tool_choice == "none"

    @pytest.mark.asyncio
    async def test_router_has_identity_and_datetime(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert "Test Agent" in result.system_prompt
        assert "Current UTC time:" in result.system_prompt


# ---------------------------------------------------------------------------
# Failure resilience
# ---------------------------------------------------------------------------


class TestSectionFailureResilience:
    """Section render() failures don't crash build_context()."""

    @pytest.mark.asyncio
    async def test_single_section_failure_doesnt_crash(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """If one section raises, others still render."""
        with patch(
            "modules.context.sections.skills.SkillsSection.render",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Skill loading exploded"),
        ):
            svc = ContextService(mock_db)
            result = await svc.build_context(
                mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
                agent=agent,
                workspace_id="ws_test",
            )
            # Identity and datetime should still be present
            assert "Test Agent" in result.system_prompt
            assert "Current UTC time:" in result.system_prompt
            # Skills should NOT be in sections_included
            assert "skills" not in result.sections_included

    @pytest.mark.asyncio
    async def test_tool_loading_failure_returns_empty_tools(
        self, agent, mock_db, mock_platform_actions
    ):
        """Tool loading failure (inside ToolsSection) returns empty tools, not a crash.

        We mock the internal _load_dispatcher_only to raise, but load_tools
        itself has a try/except that catches and returns empty tools.
        """
        from modules.context.sections.tools import ToolsSection

        with patch.object(
            ToolsSection,
            "_load_dispatcher_only",
            side_effect=RuntimeError("ActionRegistry exploded"),
        ):
            svc = ContextService(mock_db)
            result = await svc.build_context(
                mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
                agent=agent,
                workspace_id="ws_test",
            )
            # ToolsSection.load_tools catches the error → empty tools
            assert isinstance(result.tools, list)
            # System prompt still renders
            assert "Test Agent" in result.system_prompt

    @pytest.mark.asyncio
    async def test_memory_failure_doesnt_crash(
        self, agent, mock_db, messages, mock_platform_actions, mock_tools_full
    ):
        """Memory retrieval failure degrades gracefully."""
        from modules.context.sections.memory import MemorySection

        async def _exploding_render(self, ctx):
            raise RuntimeError("Memory service unavailable")

        with patch.object(MemorySection, "render", _exploding_render):
            svc = ContextService(mock_db)
            result = await svc.build_context(
                mode=ContextMode.CHATBOT,
                agent=agent,
                workspace_id="ws_test",
                messages=messages,
            )
            # Build succeeds — identity still present
            assert "Test Agent" in result.system_prompt
            # Memory is absent but build didn't crash
            assert result.memory_context is None


# ---------------------------------------------------------------------------
# ContextResult immutability
# ---------------------------------------------------------------------------


class TestContextResultImmutability:
    """ContextResult is a frozen dataclass."""

    @pytest.mark.asyncio
    async def test_result_is_frozen(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        with pytest.raises(FrozenInstanceError):
            result.system_prompt = "hacked"  # type: ignore[misc]

    @pytest.mark.asyncio
    async def test_result_is_context_result_type(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert isinstance(result, ContextResult)


# ---------------------------------------------------------------------------
# Metadata fields
# ---------------------------------------------------------------------------


class TestMetadataFields:
    """preparation_time_ms, token_estimate, token_budget, sections_included."""

    @pytest.mark.asyncio
    async def test_preparation_time_populated(self, agent, mock_db):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.preparation_time_ms > 0

    @pytest.mark.asyncio
    async def test_token_estimate_positive(
        self, agent, mock_db, mock_platform_actions, mock_tools_full, mock_memory    ):
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Write a report",
        )
        assert result.token_estimate > 0

    @pytest.mark.asyncio
    async def test_token_budget_matches_mode(self, agent, mock_db):
        """ROUTER mode should have the default budget."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        # ROUTER: total=128000, reserved_response=4096, reserved_messages=0
        # available = 128000 - 4096 - 0 = 123904
        assert result.token_budget == 123904

    @pytest.mark.asyncio
    async def test_heartbeat_budget_uses_max_tokens_override(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """HEARTBEAT has max_tokens=8000 override."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        # HEARTBEAT: max_tokens=8000, reserved_response=2048, reserved_messages=0
        # available = 8000 - 2048 - 0 = 5952
        assert result.token_budget == 5952

    @pytest.mark.asyncio
    async def test_sections_included_lists_rendered_sections(
        self, agent, mock_db, mock_platform_actions, mock_tools_dispatcher
    ):
        """sections_included lists all sections that rendered non-empty content."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.HEARTBEAT_ORCHESTRATOR,
            agent=agent,
            workspace_id="ws_test",
        )
        # HEARTBEAT sections: identity, skills, platform_actions, task_context, datetime_context
        # identity always renders, skills renders (agent has skills fixture),
        # platform_actions renders (mocked), task_context is empty (no task_description),
        # datetime_context renders
        assert "identity" in result.sections_included
        assert "platform_actions" in result.sections_included
        assert "datetime_context" in result.sections_included

    @pytest.mark.asyncio
    async def test_sections_trimmed_empty_when_under_budget(self, agent, mock_db):
        """No sections trimmed when total is well under budget."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert result.sections_trimmed == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: None agent, empty messages, no task_description."""

    @pytest.mark.asyncio
    async def test_none_agent(self, mock_db):
        """build_context with None agent still works."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=None,
            workspace_id="ws_test",
        )
        assert "Agent" in result.system_prompt  # fallback name
        assert isinstance(result, ContextResult)

    @pytest.mark.asyncio
    async def test_no_messages(
        self, agent, mock_db, mock_platform_actions, mock_tools_full, mock_memory    ):
        """build_context with no messages returns empty messages list."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="Do stuff",
        )
        assert result.messages == []

    @pytest.mark.asyncio
    async def test_empty_task_description(
        self, agent, mock_db, mock_platform_actions, mock_tools_full, mock_memory    ):
        """Empty task_description means task_context section is empty."""
        svc = ContextService(mock_db)
        result = await svc.build_context(
            mode=ContextMode.TASK_EXECUTION,
            agent=agent,
            workspace_id="ws_test",
            task_description="",
        )
        assert "task_context" not in result.sections_included

    @pytest.mark.asyncio
    async def test_no_db_session(self, agent):
        """build_context works without a DB session (some sections return empty)."""
        svc = ContextService(db_session=None)
        result = await svc.build_context(
            mode=ContextMode.ROUTER,
            agent=agent,
            workspace_id="ws_test",
        )
        assert isinstance(result, ContextResult)
        assert "Test Agent" in result.system_prompt
