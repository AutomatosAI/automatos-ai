"""PRD-137 Fix #3: IdentitySection is the single owner of identity content."""
import asyncio
import importlib.util
import pathlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Stub the personality module so IdentitySection can import it
# ---------------------------------------------------------------------------

_personality_stub = types.ModuleType("consumers.chatbot.personality")


class _FakePersonality:
    @staticmethod
    def get_base_system_prompt(**kw):
        return "BASE_PROMPT"

    @staticmethod
    def get_platform_skill():
        return "PLATFORM_SKILL"

    @staticmethod
    def get_tool_guidance_prompt(**kw):
        return "TOOL_GUIDANCE"

    @staticmethod
    def get_action_response_style():
        return "ACTION_STYLE"

    @staticmethod
    def get_anti_patterns():
        return "ANTI_PATTERNS"

    @staticmethod
    def get_self_learning_instruction():
        return "SELF_LEARNING"


_personality_stub.AutomatosPersonality = _FakePersonality
_personality_stub.load_orchestrator_settings = lambda ws_id: {}

sys.modules.setdefault("consumers", types.ModuleType("consumers"))
sys.modules["consumers"].__path__ = []
sys.modules.setdefault("consumers.chatbot", types.ModuleType("consumers.chatbot"))
sys.modules["consumers.chatbot"].__path__ = []
sys.modules["consumers.chatbot.personality"] = _personality_stub

# Stub context estimator
_estimator_stub = types.ModuleType("modules.context.estimator")


class _FakeEstimator:
    def estimate(self, text):
        return len(text) // 4


_estimator_stub.TokenEstimator = _FakeEstimator
sys.modules.setdefault("modules", types.ModuleType("modules"))
sys.modules["modules"].__path__ = []
sys.modules.setdefault("modules.context", types.ModuleType("modules.context"))
sys.modules["modules.context"].__path__ = []
sys.modules["modules.context.estimator"] = _estimator_stub
sys.modules.setdefault("modules.context.sections", types.ModuleType("modules.context.sections"))
sys.modules["modules.context.sections"].__path__ = []

# Now load base and identity
_base_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "modules.context.sections.base",
        _ROOT / "modules" / "context" / "sections" / "base.py",
    )
)
sys.modules["modules.context.sections.base"] = _base_mod
_base_mod.__spec__.loader.exec_module(_base_mod)

_identity_mod = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location(
        "modules.context.sections.identity",
        _ROOT / "modules" / "context" / "sections" / "identity.py",
    )
)
_identity_mod.__spec__.loader.exec_module(_identity_mod)

IdentitySection = _identity_mod.IdentitySection
SectionContext = _base_mod.SectionContext


def _make_agent(name="TestAgent", description=None, persona_prompt=None, use_custom=False):
    agent = MagicMock()
    agent.name = name
    agent.agent_type = "assistant"
    agent.description = description
    agent.use_custom_persona = use_custom
    agent.custom_persona_prompt = persona_prompt
    agent.persona = None
    return agent


def _make_ctx(agent, personality=False, **kwargs):
    return SectionContext(
        agent=agent,
        workspace_id="ws_test",
        workspace_name="Test Workspace",
        kwargs={"personality": personality, **kwargs},
    )


# ── Non-chatbot path ───────────────────────────────────────────────


def test_basic_identity_includes_name_and_role():
    agent = _make_agent(name="Scout")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Scout" in result
    assert "assistant" in result
    assert "Test Workspace" in result


def test_basic_identity_includes_description():
    agent = _make_agent(description="I find leads for the sales team.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "I find leads for the sales team." in result


def test_basic_identity_includes_persona():
    agent = _make_agent(use_custom=True, persona_prompt="Speak like a pirate.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Speak like a pirate." in result
    assert "Persona & Communication Style" in result


def test_basic_identity_no_duplicate_description():
    agent = _make_agent(description="Welcome! I'm your helper.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert result.count("Welcome! I'm your helper.") == 1


# ── Chatbot path (personality=True) ─────────────────────────────────


def test_chatbot_identity_includes_description():
    agent = _make_agent(description="Hello! Welcome to our store.")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Hello! Welcome to our store." in result
    assert result.count("Hello! Welcome to our store.") == 1


def test_chatbot_identity_includes_persona():
    agent = _make_agent(use_custom=True, persona_prompt="Be concise and formal.")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Be concise and formal." in result
    assert "Persona & Communication Style" in result
    assert result.count("Be concise and formal.") == 1


def test_chatbot_identity_includes_personality_parts():
    agent = _make_agent()
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "BASE_PROMPT" in result
    assert "PLATFORM_SKILL" in result
    assert "TOOL_GUIDANCE" in result
    assert "SELF_LEARNING" in result


def test_chatbot_identity_no_description_when_empty():
    agent = _make_agent(description="")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Agent Description" not in result


def test_chatbot_identity_no_persona_when_not_set():
    agent = _make_agent()
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.get_event_loop().run_until_complete(section.render(ctx))
    assert "Persona & Communication Style" not in result


# ── Render failure fallback ─────────────────────────────────────────


def test_render_failure_returns_minimal_identity():
    agent = _make_agent(name="BrokenBot")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()

    with patch.object(
        _FakePersonality, "get_base_system_prompt",
        side_effect=RuntimeError("boom"),
    ):
        result = asyncio.get_event_loop().run_until_complete(section.render(ctx))

    assert "BrokenBot" in result
    assert "Automatos platform" in result
