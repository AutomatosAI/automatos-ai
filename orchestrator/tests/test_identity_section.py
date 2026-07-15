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

# PRD-201 S2: base.py counts + truncates via core.context_guard now (the char/4
# TokenEstimator was deleted). Stub core.context_guard (cheap, no tiktoken) so
# the section still loads in isolation with the char/4 behaviour the assertions
# were written against.
_cg_stub = types.ModuleType("core.context_guard")
_cg_stub.count_tokens = lambda text: len(text or "") // 4


def _cg_truncate(text, max_tokens, *, suffix=""):
    if not text or max_tokens <= 0:
        return text
    limit = max_tokens * 4
    return text if len(text) <= limit else text[:limit] + suffix


_cg_stub.truncate_to_token_budget = _cg_truncate


def _load_sections_isolated():
    """Load base.py + identity.py under a fake module graph, then restore.

    The fakes only need to exist while the target files execute — the loaded
    classes capture their dependencies at exec time. Restoring sys.modules
    afterwards stops the fake ``consumers`` / ``modules`` packages (with
    emptied ``__path__``) from leaking into the collection of sibling test
    modules. (PRD-142 W2-S2b.)
    """
    _keys = (
        "consumers",
        "consumers.chatbot",
        "consumers.chatbot.personality",
        "modules",
        "modules.context",
        "core.context_guard",
        "modules.context.sections",
        "modules.context.sections.base",
    )
    _saved = {k: sys.modules.get(k) for k in _keys}
    try:
        # Assign fresh fake packages — never mutate a real cached package's
        # __path__ in place (setdefault + __path__=[] would corrupt the real
        # ``consumers`` / ``modules`` packages if already imported).
        for _name in (
            "consumers",
            "consumers.chatbot",
            "modules",
            "modules.context",
            "modules.context.sections",
        ):
            _pkg = types.ModuleType(_name)
            _pkg.__path__ = []
            sys.modules[_name] = _pkg
        sys.modules["consumers.chatbot.personality"] = _personality_stub
        sys.modules["core.context_guard"] = _cg_stub

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

        return _identity_mod.IdentitySection, _base_mod.SectionContext
    finally:
        for _k, _v in _saved.items():
            if _v is None:
                sys.modules.pop(_k, None)
            else:
                sys.modules[_k] = _v


IdentitySection, SectionContext = _load_sections_isolated()


@pytest.fixture(autouse=True)
def _runtime_personality_stub():
    """identity.py imports ``consumers.chatbot.personality`` LAZILY inside its
    chatbot render path (render() and _get_personality_block), so the stub must
    be live during this file's test phase — not just at import time. Installing
    it here (after collection) keeps the fake out of sibling modules' collection,
    and restoring per-test keeps it from bleeding into their test phase. The
    leaf key alone resolves ``from consumers.chatbot.personality import X``
    without needing pathless parent packages. (PRD-142 W2-S2b.)"""
    _key = "consumers.chatbot.personality"
    _saved = sys.modules.get(_key)
    sys.modules[_key] = _personality_stub
    try:
        yield
    finally:
        if _saved is None:
            sys.modules.pop(_key, None)
        else:
            sys.modules[_key] = _saved


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
    result = asyncio.run(section.render(ctx))
    assert "Scout" in result
    assert "assistant" in result
    assert "Test Workspace" in result


def test_basic_identity_includes_description():
    agent = _make_agent(description="I find leads for the sales team.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "I find leads for the sales team." in result


def test_basic_identity_includes_persona():
    agent = _make_agent(use_custom=True, persona_prompt="Speak like a pirate.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "Speak like a pirate." in result
    assert "Persona & Communication Style" in result


def test_basic_identity_no_duplicate_description():
    agent = _make_agent(description="Welcome! I'm your helper.")
    ctx = _make_ctx(agent)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert result.count("Welcome! I'm your helper.") == 1


# ── Chatbot path (personality=True) ─────────────────────────────────


def test_chatbot_identity_includes_description():
    agent = _make_agent(description="Hello! Welcome to our store.")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "Hello! Welcome to our store." in result
    assert result.count("Hello! Welcome to our store.") == 1


def test_chatbot_identity_includes_persona():
    agent = _make_agent(use_custom=True, persona_prompt="Be concise and formal.")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "Be concise and formal." in result
    assert "Persona & Communication Style" in result
    assert result.count("Be concise and formal.") == 1


def test_chatbot_identity_includes_personality_parts():
    agent = _make_agent()
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "BASE_PROMPT" in result
    assert "PLATFORM_SKILL" in result
    assert "TOOL_GUIDANCE" in result
    assert "SELF_LEARNING" in result


def test_chatbot_identity_no_description_when_empty():
    agent = _make_agent(description="")
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
    assert "Agent Description" not in result


def test_chatbot_identity_no_persona_when_not_set():
    agent = _make_agent()
    ctx = _make_ctx(agent, personality=True)
    section = IdentitySection()
    result = asyncio.run(section.render(ctx))
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
        result = asyncio.run(section.render(ctx))

    assert "BrokenBot" in result
    assert "Automatos platform" in result
