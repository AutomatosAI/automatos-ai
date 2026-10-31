"""PRD-268 US-C02c (§6.2.3, K9) — a check-in's context is lean and shows only what it may run.

``ContextMode.CHECK_IN`` builds a 12,000-token context (identity, field digest, date,
task) and loads tools with the ``read_only`` strategy: the platform actions labelled
read, plus ``platform_suggest_next_step``, each first-class, from the registry alone. No
Composio, plugin, workspace or sub-agent tool, and no ``platform_execute`` dispatcher,
ever reaches the surface, so the model is never shown a tool the lane would refuse
(``check_in_lane``). No other mode's config or tool loading changes.

Pure: no database. The hidden categories are stubbed; one test reads the real registry.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from modules.context.budget import DEFAULT_BUDGETS
from modules.context.modes import MODE_CONFIGS, ContextMode, ModeConfig
from modules.context.sections.base import SectionContext
from modules.context.service import ContextService
from modules.tools.discovery.check_in_lane import SUGGEST_NEXT_STEP, may_clear_in_a_check_in

WS = "5b0c7a1e-2f43-4a8e-9c61-0d7e2b9f4a10"
CHECK_IN = MODE_CONFIGS[ContextMode.CHECK_IN]

READ = "platform_mode_test_read"
WRITE = "platform_mode_test_write"
DESTROY = "platform_mode_test_destroy"
ADMIN_READ = "platform_mode_test_admin_read"
SUPER_READ = "platform_mode_test_super_read"
OFFLINE_READ = "platform_mode_test_offline_read"
BROKEN_CHECK_READ = "platform_mode_test_broken_check_read"
HIDDEN_READ = "platform_mode_test_hidden_read"
HIDDEN_CATEGORY = "socials"
# Named like the tools a check-in must never be shown, each labelled read so only the
# surface's own rule keeps it out.
NEVER_SHOWN = ("platform_execute", "workspace_read_file", "GMAIL_SEND_EMAIL", "composio_execute", "plugin_mode_test")

# Every other mode's tool loading and token cap, as on the base: CHECK_IN changes none.
OTHER_MODES = {
    ContextMode.CHATBOT: ("filtered", None),
    ContextMode.TASK_EXECUTION: ("full", None),
    ContextMode.HEARTBEAT_ORCHESTRATOR: ("dispatcher_only", 8000),
    ContextMode.HEARTBEAT_AGENT: ("full", 128000),
    ContextMode.RECIPE: ("full", None),
    ContextMode.ROUTER: ("none", None),
    ContextMode.ORCHESTRATOR_STAGE: ("none", None),
    ContextMode.NL2SQL: ("none", None),
    ContextMode.COORDINATOR: ("full", 131072),
    ContextMode.PLANNING: ("none", None),
}


def _action(name, level, category="test", **flags):
    from modules.tools.discovery.action_registry import ActionDefinition

    return ActionDefinition(name=name, description=f"{name} (mode test)", category=category,
                            parameters={"type": "object", "properties": {}}, permission_level=level, **flags)


def _raises():
    raise RuntimeError("availability check failed")


class _Registry:
    def __init__(self, actions):
        self._actions = list(actions)

    def get_all(self):
        return list(self._actions)


def _stub_registry():
    return _Registry([
        _action(READ, "read"),
        _action(WRITE, "write"),
        _action(DESTROY, "destructive", requires_confirmation=True),
        _action(SUGGEST_NEXT_STEP, "write"),
        _action(ADMIN_READ, "read", admin_only=True),
        _action(SUPER_READ, "read", super_admin_only=True),
        _action(OFFLINE_READ, "read", available=lambda: False),
        _action(BROKEN_CHECK_READ, "read", available=_raises),
        _action(HIDDEN_READ, "read", category=HIDDEN_CATEGORY),
        *(_action(name, "read") for name in NEVER_SHOWN),
    ])


@pytest.fixture
def agent_surface(monkeypatch):
    """The agent's own surface (Composio, a plugin, the dispatcher): a check-in must never
    build or use it. Records every build."""
    import modules.tools.tool_router as tool_router

    built = []
    agent_tools = [{"type": "function", "function": {"name": name}} for name in NEVER_SHOWN]

    async def _get_tools_for_agent_async(**kwargs):
        built.append(kwargs)
        return agent_tools

    monkeypatch.setattr(tool_router, "get_tools_for_agent_async", _get_tools_for_agent_async)
    return NS(built=built, tools=agent_tools)


@pytest.fixture
def nothing_hidden(monkeypatch):
    import modules.tools.discovery.hidden_categories as hidden_categories

    monkeypatch.setattr(hidden_categories, "hidden_categories_for_workspace", lambda workspace_id, db=None: ())


def _load_check_in_tools(agent_surface, workspace_id=WS):
    ctx = SectionContext(agent=NS(id=7), workspace_id=workspace_id, context_mode=ContextMode.CHECK_IN.value,
                         kwargs={"prebuilt_tools": agent_surface.tools, "query": "What changed?"})
    return asyncio.run(ContextService._load_tools(CHECK_IN, ctx))


def _names(tools):
    return [tool["function"]["name"] for tool in tools]


# ── the mode ────────────────────────────────────────────────────────────────

def test_the_check_in_mode_is_lean_and_loads_read_only():
    assert ContextMode.CHECK_IN.value == "check_in"
    assert CHECK_IN == ModeConfig(
        sections=["identity", "field_memory", "datetime_context", "task_context"],
        tool_loading="read_only",
        personality=False,
        max_tokens=12000,
    )


def test_the_budget_is_12000_with_800_for_the_answer():
    budget = DEFAULT_BUDGETS[ContextMode.CHECK_IN]
    assert (budget.total, budget.reserved_for_response, budget.reserved_for_messages) == (12_000, 800, 0)
    # The mode's own cap wins over any model's window.
    for model in ("claude-haiku-5-5", "a-model-with-a-million-tokens", None):
        assert ContextService._get_budget(ContextMode.CHECK_IN, CHECK_IN, model, db_session=None).total == 12_000


def test_no_other_mode_changes():
    assert {mode: (MODE_CONFIGS[mode].tool_loading, MODE_CONFIGS[mode].max_tokens) for mode in OTHER_MODES} == OTHER_MODES
    assert [mode for mode, config in MODE_CONFIGS.items() if config.tool_loading == "read_only"] == [ContextMode.CHECK_IN]


# ── the read_only strategy ──────────────────────────────────────────────────

def test_check_in_loads_only_reads_and_the_suggestion(monkeypatch, agent_surface, nothing_hidden):
    import modules.tools.discovery.action_registry as action_registry

    monkeypatch.setattr(action_registry, "get_action_registry", _stub_registry)
    tools, tool_choice = _load_check_in_tools(agent_surface)

    # Nothing is hidden here, so the Socials read is a read like any other; the two tests
    # below prove it stays out while its category is hidden.
    assert _names(tools) == [READ, SUGGEST_NEXT_STEP, HIDDEN_READ]
    assert tool_choice == "auto"
    assert all(tool["type"] == "function" for tool in tools)


def test_the_agents_composio_and_plugin_tools_are_never_built_or_used(monkeypatch, agent_surface, nothing_hidden):
    import modules.tools.discovery.action_registry as action_registry

    monkeypatch.setattr(action_registry, "get_action_registry", _stub_registry)
    tools, _ = _load_check_in_tools(agent_surface)

    assert agent_surface.built == []
    assert not set(_names(tools)) & set(NEVER_SHOWN)


def test_a_category_the_workspace_is_not_shown_stays_out(monkeypatch, agent_surface):
    import modules.tools.discovery.action_registry as action_registry
    import modules.tools.discovery.hidden_categories as hidden_categories

    asked = []

    def _hidden(workspace_id, db=None):
        asked.append(workspace_id)
        return (HIDDEN_CATEGORY,)

    monkeypatch.setattr(action_registry, "get_action_registry", _stub_registry)
    monkeypatch.setattr(hidden_categories, "hidden_categories_for_workspace", _hidden)
    tools, _ = _load_check_in_tools(agent_surface)

    assert asked == [WS]
    assert HIDDEN_READ not in _names(tools)
    assert READ in _names(tools)


def test_with_no_workspace_every_gated_category_stays_out(monkeypatch, agent_surface):
    """The real hidden-categories rule: no workspace hides Socials and every integration."""
    import modules.tools.discovery.action_registry as action_registry

    monkeypatch.setattr(action_registry, "get_action_registry", _stub_registry)
    tools, _ = _load_check_in_tools(agent_surface, workspace_id=None)

    assert HIDDEN_READ not in _names(tools)
    assert _names(tools) == [READ, SUGGEST_NEXT_STEP]


def test_a_registry_with_nothing_to_offer_loads_no_tools(monkeypatch, agent_surface, nothing_hidden):
    import modules.tools.discovery.action_registry as action_registry

    monkeypatch.setattr(action_registry, "get_action_registry", lambda: _Registry([_action(WRITE, "write")]))
    assert _load_check_in_tools(agent_surface) == ([], "none")


def test_a_registry_that_cannot_be_read_loads_no_tools(monkeypatch, agent_surface, nothing_hidden):
    import modules.tools.discovery.action_registry as action_registry

    def _broken():
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(action_registry, "get_action_registry", _broken)
    assert _load_check_in_tools(agent_surface) == ([], "none")
    assert agent_surface.built == []


def test_every_tool_check_in_loads_is_a_read_action_or_the_suggestion(agent_surface, nothing_hidden):
    """The real registry: every name loaded is a registered read action or
    platform_suggest_next_step, none is Composio, a plugin, a workspace tool or the
    dispatcher, and the lane clears every one of them."""
    from modules.tools.discovery.action_registry import get_action_registry

    registry = get_action_registry()
    tools, tool_choice = _load_check_in_tools(agent_surface)
    names = _names(tools)

    assert tool_choice == "auto" and "platform_list_tasks" in names
    assert len(names) == len(set(names))
    for name in names:
        action = registry.get(name)
        assert action is not None, f"{name} is not a registered platform action"
        assert action.permission_level == "read" or name == SUGGEST_NEXT_STEP, f"{name} is {action.permission_level}"
        assert name.startswith("platform_") and name != "platform_execute", name
        assert may_clear_in_a_check_in(name), f"{name} is shown but the lane refuses it"
    reads = {action.name for action in registry.get_by_permission("read") if action.name.startswith("platform_")}
    assert set(names) - {SUGGEST_NEXT_STEP} <= reads
    assert "platform_ask_orchestrator" not in names
    assert agent_surface.built == []
