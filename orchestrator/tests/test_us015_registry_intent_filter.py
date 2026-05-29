"""
PRD-141 US-015: intent->tool filtering reads ActionRegistry categories.
=======================================================================

The chatbot router's category fallback no longer carries hardcoded
``TOOL_CATEGORIES`` / ``INTENT_TO_TOOLS`` dicts. ``_filter_tools_by_intent``
maps the classified intent to ActionRegistry *category names* via
``_INTENT_TO_REGISTRY_CATEGORIES`` and pulls the matching action names from the
registry at call time — so an action registered under an already-mapped
category is auto-discoverable with no edit to the router. The kept set is
unioned with the classifier's suggested tools plus the always-on CORE_TOOLS /
ALWAYS_INCLUDE sets.

Leaf-load pattern as in US-013/US-014: ``consumers.chatbot.__init__`` pulls the
DB-backed chat service, so we load ``intent_classifier`` + ``smart_tool_router``
under a synthetic package and inject a fake ``action_registry`` module into
``sys.modules`` (the registry import inside ``_filter_tools_by_intent`` is lazy).
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_chatbot_dir = _orchestrator_root / "consumers" / "chatbot"
_PKG = "_us015_chatbot"


def _load_modules():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_chatbot_dir)]
        sys.modules[_PKG] = pkg

    def _leaf(mod_name):
        full = f"{_PKG}.{mod_name}"
        if full in sys.modules:
            return sys.modules[full]
        spec = importlib.util.spec_from_file_location(full, _chatbot_dir / f"{mod_name}.py")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _PKG
        sys.modules[full] = module
        spec.loader.exec_module(module)
        return module

    intent_mod = _leaf("intent_classifier")
    router_mod = _leaf("smart_tool_router")
    return router_mod, intent_mod


_router_mod, _intent_mod = _load_modules()
SmartToolRouter = _router_mod.SmartToolRouter
_INTENT_TO_REGISTRY_CATEGORIES = _router_mod._INTENT_TO_REGISTRY_CATEGORIES
Intent = _intent_mod.Intent
IntentResult = _intent_mod.IntentResult


def _intent_result(primary=Intent.DATA_QUERY, suggested=None):
    return IntentResult(
        primary_intent=primary,
        confidence=0.9,
        requires_tools=True,
        requires_memory=False,
        suggested_tools=suggested or [],
        reasoning="stub",
        is_simple=False,
    )


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} desc"}}


@pytest.fixture
def registry_env(monkeypatch):
    """Inject a fake ActionRegistry whose get_by_category is test-controlled."""
    state = {"by_category": {}}

    class _FakeRegistry:
        def get_by_category(self, category):
            return state["by_category"].get(category, [])

    fake_mod = types.ModuleType("modules.tools.discovery.action_registry")
    fake_mod.get_action_registry = lambda: _FakeRegistry()
    monkeypatch.setitem(sys.modules, "modules.tools.discovery.action_registry", fake_mod)
    return state


# ---------------------------------------------------------------------------
# The hardcoded dicts are gone
# ---------------------------------------------------------------------------

def test_legacy_dicts_deleted():
    assert not hasattr(SmartToolRouter, "TOOL_CATEGORIES")
    assert not hasattr(SmartToolRouter, "INTENT_TO_TOOLS")
    assert not hasattr(SmartToolRouter, "_filter_tools_by_categories")
    assert not hasattr(SmartToolRouter, "_tool_matches_query")


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def test_intent_to_registry_categories_mapping():
    """Every tool-requiring intent maps to >=1 ActionRegistry category."""
    tool_intents = [
        Intent.DATA_QUERY,
        Intent.SEARCH,
        Intent.EXTERNAL_ACTION,
        Intent.CREATION,
        Intent.MEMORY_RECALL,
        Intent.MULTI_STEP,
    ]
    for intent in tool_intents:
        cats = _INTENT_TO_REGISTRY_CATEGORIES.get(intent)
        assert cats, f"{intent} must map to >=1 registry category"
    assert all(len(cats) >= 1 for cats in _INTENT_TO_REGISTRY_CATEGORIES.values())


# ---------------------------------------------------------------------------
# Registry-backed filtering
# ---------------------------------------------------------------------------

def test_category_filter_uses_registry(registry_env):
    """Tool names come from ActionRegistry.get_by_category for the mapped categories."""
    registry_env["by_category"]["analytics"] = [
        SimpleNamespace(name="metrics_read"),
        SimpleNamespace(name="kpi_dump"),
    ]
    r = SmartToolRouter()
    intent = _intent_result(primary=Intent.DATA_QUERY, suggested=[])
    tools = [_tool("metrics_read"), _tool("kpi_dump"), _tool("totally_unrelated_tool")]

    filtered = r._filter_tools_by_intent(tools, intent)

    names = {t["function"]["name"] for t in filtered}
    assert "metrics_read" in names
    assert "kpi_dump" in names
    assert "totally_unrelated_tool" not in names


def test_new_action_auto_discoverable(registry_env):
    """A new action under an already-mapped category appears with NO router code change."""
    registry_env["by_category"]["analytics"] = [SimpleNamespace(name="brand_new_kpi_action")]
    r = SmartToolRouter()
    intent = _intent_result(primary=Intent.DATA_QUERY, suggested=[])
    tools = [_tool("brand_new_kpi_action")]

    filtered = r._filter_tools_by_intent(tools, intent)

    assert {t["function"]["name"] for t in filtered} >= {"brand_new_kpi_action"}


def test_filter_always_keeps_core_and_always_include(registry_env):
    """CORE_TOOLS, ALWAYS_INCLUDE, and suggested tools survive filtering."""
    registry_env["by_category"]["analytics"] = [SimpleNamespace(name="metrics_read")]
    r = SmartToolRouter()
    intent = _intent_result(primary=Intent.DATA_QUERY, suggested=["search_codebase"])
    core = sorted(SmartToolRouter.CORE_TOOLS)
    always = sorted(SmartToolRouter.ALWAYS_INCLUDE)
    tools = [_tool("metrics_read")] + [_tool(n) for n in core] + [_tool(n) for n in always]

    filtered = r._filter_tools_by_intent(tools, intent)

    names = {t["function"]["name"] for t in filtered}
    for n in core + always:
        assert n in names
    assert "metrics_read" in names  # from registry


def test_no_categories_no_suggested_returns_all(registry_env):
    """An intent with no category mapping and no suggestions can't narrow → keep all."""
    r = SmartToolRouter()
    # GREETING is not a key in _INTENT_TO_REGISTRY_CATEGORIES and has no suggestions
    intent = _intent_result(primary=Intent.GREETING, suggested=[])
    tools = [_tool("a"), _tool("b")]

    filtered = r._filter_tools_by_intent(tools, intent)

    assert {t["function"]["name"] for t in filtered} == {"a", "b"}


# ---------------------------------------------------------------------------
# Widget callback signal tool survives filtering (production regression)
# ---------------------------------------------------------------------------

def test_widget_callback_tool_survives_multi_step_filter(registry_env):
    """The widget callback signal tool must NOT be filtered out of a widget turn.

    Production regression: "Can someone call me back" classified as MULTI_STEP,
    the registry's mapped categories did not surface widget_open_callback_form,
    and it was dropped from the LLM's toolset. The LLM then improvised
    composio_execute(action="widget_open_callback_form"), which fails with
    "'WIDGET' is not assigned to agent N". Pinning the tool in ALWAYS_INCLUDE
    guarantees it survives whenever it is present in available_tools (it is only
    present when the Site has callback.enabled — gated upstream).
    """
    # Registry returns categories that do NOT include the widget tool, mirroring
    # production: nothing in the mapped categories surfaces it.
    registry_env["by_category"]["agents"] = [SimpleNamespace(name="platform_list_agents")]
    r = SmartToolRouter()
    intent = _intent_result(primary=Intent.MULTI_STEP, suggested=[])
    tools = [_tool("widget_open_callback_form"), _tool("some_unrelated_tool")]

    filtered = r._filter_tools_by_intent(tools, intent)

    names = {t["function"]["name"] for t in filtered}
    assert "widget_open_callback_form" in names
    assert "some_unrelated_tool" not in names
