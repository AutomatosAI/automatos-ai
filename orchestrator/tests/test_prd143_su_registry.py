"""PRD-143 S1 — super_admin_only tier on ActionRegistry.

Fail-closed: every registry listing/selection path EXCLUDES
super_admin_only actions by default; they are included ONLY when
include_super_admin=True is passed explicitly. The existing
admin_only/exclude_admin mechanism is unchanged — super_admin_only is a
second, stricter tier layered on top.

Pure unit tests: action_registry.py is loaded directly (no package
imports, no DB/config), mirroring test_action_registry_filtered.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_THIS = Path(__file__).resolve()
_AR_PATH = _THIS.parents[1] / "modules" / "tools" / "discovery" / "action_registry.py"
_spec = importlib.util.spec_from_file_location("action_registry_prd143_s1", _AR_PATH)
action_registry_mod = importlib.util.module_from_spec(_spec)
sys.modules["action_registry_prd143_s1"] = action_registry_mod
_spec.loader.exec_module(action_registry_mod)

ActionDefinition = action_registry_mod.ActionDefinition
ActionRegistry = action_registry_mod.ActionRegistry


def _make_action(
    name: str,
    category: str = "agents",
    description: str = "Test action",
    admin_only: bool = False,
    super_admin_only: bool = False,
    promoted: bool = False,
) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description,
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        admin_only=admin_only,
        super_admin_only=super_admin_only,
        promoted=promoted,
    )


def _enum_of(schema: dict) -> list[str]:
    return schema["function"]["parameters"]["properties"]["action"]["enum"]


@pytest.fixture
def registry() -> ActionRegistry:
    reg = ActionRegistry()
    # Bypass the lazy initializer so tests don't pull live platform_actions.
    reg._initialized = True
    reg.register(_make_action("platform_list_agents", category="agents"))
    reg.register(_make_action(
        "platform_admin_action", category="admin", admin_only=True,
    ))
    reg.register(_make_action(
        "platform_admin_promoted", category="admin", admin_only=True, promoted=True,
    ))
    reg.register(_make_action(
        "platform_promoted_action", category="promoted", promoted=True,
    ))
    reg.register(_make_action(
        "platform_query_loki_logs", category="monitoring", super_admin_only=True,
    ))
    reg.register(_make_action(
        "platform_su_promoted", category="monitoring", super_admin_only=True, promoted=True,
    ))
    return reg


def test_su_action_excluded_from_first_class_schemas_by_default(registry):
    """A promoted su action never gets a first-class schema unless explicitly included."""
    names = [s["function"]["name"] for s in registry.to_first_class_schemas()]
    assert "platform_su_promoted" not in names
    assert "platform_promoted_action" in names

    # exclude_admin in either position must not re-admit su actions.
    names_admin_excl = [
        s["function"]["name"] for s in registry.to_first_class_schemas(exclude_admin=True)
    ]
    assert "platform_su_promoted" not in names_admin_excl


def test_su_action_excluded_from_dispatcher_schema_by_default(registry):
    """The dispatcher enum never carries an su action for a default caller."""
    enum = _enum_of(registry.to_dispatcher_schema())
    assert "platform_query_loki_logs" not in enum
    assert "platform_list_agents" in enum

    # Even when the ranker explicitly allow-lists the su action.
    enum_allowed = _enum_of(registry.to_dispatcher_schema(
        allowed_names=["platform_list_agents", "platform_query_loki_logs"],
    ))
    assert "platform_query_loki_logs" not in enum_allowed
    assert "platform_list_agents" in enum_allowed

    # The empty-intersection fallback path must also stay su-clean.
    enum_fallback = _enum_of(registry.to_dispatcher_schema(
        allowed_names=["platform_query_loki_logs"],
    ))
    assert "platform_query_loki_logs" not in enum_fallback

    # to_openai_tools (full schema listing) is fail-closed too.
    tool_names = [t["function"]["name"] for t in registry.to_openai_tools()]
    assert "platform_query_loki_logs" not in tool_names
    assert "platform_su_promoted" not in tool_names
    assert "platform_list_agents" in tool_names


def test_su_action_excluded_from_prompt_summary_by_default(registry):
    """Prompt summaries (full and filtered) never mention su actions by default."""
    summary = registry.build_prompt_summary()
    assert "platform_query_loki_logs" not in summary
    assert "platform_su_promoted" not in summary
    assert "platform_list_agents" in summary

    # Filtered variant: su excluded even when requested by name.
    filtered = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_query_loki_logs"],
    )
    assert "platform_query_loki_logs" not in filtered
    assert "platform_list_agents" in filtered


def test_su_action_included_only_with_explicit_include_flag(registry):
    """include_super_admin=True is the ONLY way su actions surface."""
    names = [
        s["function"]["name"]
        for s in registry.to_first_class_schemas(include_super_admin=True)
    ]
    assert "platform_su_promoted" in names

    enum = _enum_of(registry.to_dispatcher_schema(include_super_admin=True))
    assert "platform_query_loki_logs" in enum

    enum_allowed = _enum_of(registry.to_dispatcher_schema(
        include_super_admin=True,
        allowed_names=["platform_list_agents", "platform_query_loki_logs"],
    ))
    assert "platform_query_loki_logs" in enum_allowed

    summary = registry.build_prompt_summary(include_super_admin=True)
    assert "platform_query_loki_logs" in summary

    filtered = registry.build_filtered_prompt_summary(
        ["platform_query_loki_logs"], include_super_admin=True,
    )
    assert "platform_query_loki_logs" in filtered

    tool_names = [
        t["function"]["name"] for t in registry.to_openai_tools(include_super_admin=True)
    ]
    assert "platform_query_loki_logs" in tool_names

    # The flag default is False on the definition itself.
    assert ActionDefinition(
        name="x", description="d", category="c", parameters={},
    ).super_admin_only is False


def test_admin_only_filter_behaviour_unchanged(registry):
    """The pre-existing admin_only/exclude_admin mechanism is intact."""
    # Prompt summary: admin included by default, dropped with exclude_admin=True.
    assert "platform_admin_action" in registry.build_prompt_summary()
    assert "platform_admin_action" not in registry.build_prompt_summary(exclude_admin=True)

    # Dispatcher: same opt-in exclusion semantics.
    assert "platform_admin_action" in _enum_of(registry.to_dispatcher_schema())
    assert "platform_admin_action" not in _enum_of(
        registry.to_dispatcher_schema(exclude_admin=True)
    )

    # First-class schemas: promoted admin action included by default,
    # dropped with exclude_admin=True.
    default_names = [s["function"]["name"] for s in registry.to_first_class_schemas()]
    assert "platform_admin_promoted" in default_names
    excl_names = [
        s["function"]["name"] for s in registry.to_first_class_schemas(exclude_admin=True)
    ]
    assert "platform_admin_promoted" not in excl_names

    # Filtered prompt summary: exclude_admin still honoured.
    filtered = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_admin_action"], exclude_admin=True,
    )
    assert "platform_admin_action" not in filtered
    assert "platform_list_agents" in filtered
