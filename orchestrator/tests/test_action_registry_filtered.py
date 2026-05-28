"""Tests for ActionRegistry.build_filtered_prompt_summary (PRD-138 US-002).

These tests build a fresh ActionRegistry with mock ActionDefinitions so they
stay pure unit tests with no DB / config dependencies.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load action_registry directly without triggering modules.tools.* package
# imports (which would pull in the live platform_actions registrar).
_THIS = Path(__file__).resolve()
_AR_PATH = _THIS.parents[1] / "modules" / "tools" / "discovery" / "action_registry.py"
_spec = importlib.util.spec_from_file_location("action_registry_under_test", _AR_PATH)
action_registry_mod = importlib.util.module_from_spec(_spec)
sys.modules["action_registry_under_test"] = action_registry_mod
_spec.loader.exec_module(action_registry_mod)

ActionDefinition = action_registry_mod.ActionDefinition
ActionRegistry = action_registry_mod.ActionRegistry


def _make_action(
    name: str,
    category: str = "agents",
    description: str = "Test action",
    properties: dict | None = None,
    required: list[str] | None = None,
    admin_only: bool = False,
    promoted: bool = False,
) -> ActionDefinition:
    """Build a simple ActionDefinition for tests."""
    return ActionDefinition(
        name=name,
        description=description,
        category=category,
        parameters={
            "type": "object",
            "properties": properties or {},
            "required": required or [],
        },
        admin_only=admin_only,
        promoted=promoted,
    )


@pytest.fixture
def registry() -> ActionRegistry:
    """A pre-populated ActionRegistry with five known actions across categories."""
    reg = ActionRegistry()
    # Bypass the lazy initializer so tests don't pull live platform_actions.
    reg._initialized = True
    reg.register(_make_action(
        "platform_list_agents",
        category="agents",
        description="List all agents",
        properties={"limit": {"type": "integer"}},
    ))
    reg.register(_make_action(
        "platform_create_agent",
        category="agents",
        description="Create a new agent",
        properties={"name": {"type": "string"}, "role": {"type": "string"}},
        required=["name"],
    ))
    reg.register(_make_action(
        "platform_list_missions",
        category="missions",
        description="List active missions",
    ))
    reg.register(_make_action(
        "platform_admin_only_action",
        category="admin",
        description="Admin-only thing",
        admin_only=True,
    ))
    reg.register(_make_action(
        "platform_promoted_action",
        category="promoted",
        description="A promoted first-class action",
        promoted=True,
    ))
    return reg


# ---- Acceptance criteria: 3 known names ----

def test_filter_returns_only_requested_actions(registry):
    """AC #7: Pass 3 known names, assert markdown contains exactly those 3."""
    summary = registry.build_filtered_prompt_summary([
        "platform_list_agents",
        "platform_create_agent",
        "platform_list_missions",
    ])
    # All three actions appear
    assert "platform_list_agents" in summary
    assert "platform_create_agent" in summary
    assert "platform_list_missions" in summary
    # Their categories appear as headers (Title Case via .title())
    assert "### Agents" in summary
    assert "### Missions" in summary
    # Other actions absent
    assert "platform_admin_only_action" not in summary
    assert "platform_promoted_action" not in summary


# ---- Acceptance criteria: empty list ----

def test_empty_list_emits_no_action_lines(registry):
    """AC #8: Empty list returns empty summary — no actions, no headers."""
    summary = registry.build_filtered_prompt_summary([])
    assert "- `platform_" not in summary
    assert "### " not in summary


def test_empty_list_does_not_fall_back_to_all(registry):
    """AC #5: Empty list does NOT fall back to listing all actions."""
    summary = registry.build_filtered_prompt_summary([])
    # None of the registered action names should appear
    assert "platform_list_agents" not in summary
    assert "platform_create_agent" not in summary
    assert "platform_list_missions" not in summary


# ---- Acceptance criteria: mix of valid and unknown ----

def test_unknown_names_silently_skipped(registry):
    """AC #3 + AC #9: Mix of valid + unknown names, only valid ones appear."""
    summary = registry.build_filtered_prompt_summary([
        "platform_list_agents",
        "platform_does_not_exist",
        "platform_create_agent",
        "another_unknown_action",
    ])
    # Valid ones present
    assert "platform_list_agents" in summary
    assert "platform_create_agent" in summary
    # Unknown ones absent (and no error raised)
    assert "platform_does_not_exist" not in summary
    assert "another_unknown_action" not in summary


# ---- AC #4: exclude_admin / exclude_promoted ----

def test_exclude_admin_filters_admin_only(registry):
    """exclude_admin=True drops admin_only actions even if requested."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_admin_only_action"],
        exclude_admin=True,
    )
    assert "platform_list_agents" in summary
    assert "platform_admin_only_action" not in summary


def test_admin_included_by_default(registry):
    """exclude_admin defaults to False — admin actions appear when requested."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_admin_only_action"],
    )
    assert "platform_list_agents" in summary
    assert "platform_admin_only_action" in summary


def test_exclude_promoted_filters_promoted(registry):
    """exclude_promoted=True drops promoted actions even if requested."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_promoted_action"],
        exclude_promoted=True,
    )
    assert "platform_list_agents" in summary
    assert "platform_promoted_action" not in summary


def test_promoted_included_by_default(registry):
    """exclude_promoted defaults to False — promoted actions appear in Direct Tools section."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_promoted_action"],
    )
    assert "platform_list_agents" in summary
    assert "platform_promoted_action" in summary
    assert "Direct Tools" in summary


# ---- AC #2: format identical to build_prompt_summary ----

def test_format_matches_build_prompt_summary(registry):
    """AC #2 + #6: Filtered output must match the unfiltered output for the
    same subset, so callers can swap one method for the other.

    Strategy: include ALL non-admin non-promoted actions in the filter,
    expect identical bytes to the unfiltered (default flags) summary.
    """
    full = registry.build_prompt_summary()  # exclude_admin=False, exclude_promoted=False
    all_names = [a.name for a in registry.get_all()]
    filtered = registry.build_filtered_prompt_summary(all_names)
    assert filtered == full


def test_filtered_preserves_param_hints(registry):
    """Param-hint logic (required marker, backticks) must match build_prompt_summary."""
    summary = registry.build_filtered_prompt_summary(["platform_create_agent"])
    # required=["name"] should produce "(required)" marker
    assert "`name` (required)" in summary
    # role is not required — no marker
    assert "`role`" in summary
    assert "`role` (required)" not in summary


# ---- AC #6: build_prompt_summary unchanged ----

def test_build_prompt_summary_still_works(registry):
    """AC #6: existing build_prompt_summary() behaviour preserved."""
    summary = registry.build_prompt_summary()
    assert "## Available Platform Actions" in summary
    assert "platform_list_agents" in summary
    assert "platform_create_agent" in summary
    assert "platform_list_missions" in summary
    # promoted included by default (exclude_promoted=False) in Direct Tools section
    assert "platform_promoted_action" in summary
    assert "Direct Tools" in summary
    # admin included by default (exclude_admin=False)
    assert "platform_admin_only_action" in summary


def test_build_prompt_summary_exclude_admin(registry):
    """exclude_admin=True drops admin_only actions in unfiltered summary too."""
    summary = registry.build_prompt_summary(exclude_admin=True)
    assert "platform_list_agents" in summary
    assert "platform_admin_only_action" not in summary


# ---- PRD-138 US-008: to_dispatcher_schema(allowed_names=...) ----


def _enum_of(schema: dict) -> list[str]:
    """Pull the action enum out of a dispatcher schema."""
    return schema["function"]["parameters"]["properties"]["action"]["enum"]


def test_dispatcher_allowed_names_narrows_enum(registry):
    """AC #1: allowed_names with two known non-admin/non-promoted names yields
    an enum equal to exactly those two names (sorted)."""
    schema = registry.to_dispatcher_schema(
        allowed_names=["platform_list_agents", "platform_create_agent"],
    )
    assert _enum_of(schema) == ["platform_create_agent", "platform_list_agents"]


def test_dispatcher_allowed_names_none_matches_legacy(registry):
    """AC #2: allowed_names=None must produce byte-identical schema to the
    pre-US-008 call (no allowed_names argument)."""
    legacy = registry.to_dispatcher_schema()
    new = registry.to_dispatcher_schema(allowed_names=None)
    assert legacy == new


def test_dispatcher_empty_allowed_names_falls_back_to_full(registry, caplog):
    """AC #3: allowed_names=[] returns the full eligible enum and logs a WARNING
    (empty allow-list is treated as 'ranker returned nothing', not 'block everything')."""
    import logging
    with caplog.at_level(logging.WARNING):
        schema = registry.to_dispatcher_schema(allowed_names=[])
    full = registry.to_dispatcher_schema(allowed_names=None)
    assert _enum_of(schema) == _enum_of(full)
    assert any(
        "allowed_names=[]" in record.message and "empty allow-list" in record.message
        for record in caplog.records
    ), f"Expected empty-allow-list warning, got: {[r.message for r in caplog.records]}"


def test_dispatcher_admin_excluded_even_when_in_allowed_names(registry):
    """AC #4: An admin action listed in allowed_names is still excluded when
    exclude_admin=True. Permission filters run before the allow-list."""
    schema = registry.to_dispatcher_schema(
        exclude_admin=True,
        allowed_names=["platform_list_agents", "platform_admin_only_action"],
    )
    enum = _enum_of(schema)
    assert "platform_list_agents" in enum
    assert "platform_admin_only_action" not in enum


def test_dispatcher_admin_passes_when_exclude_admin_false(registry):
    """Companion to AC #4: when exclude_admin=False (default), an admin action
    in allowed_names is allowed through."""
    schema = registry.to_dispatcher_schema(
        allowed_names=["platform_list_agents", "platform_admin_only_action"],
    )
    enum = _enum_of(schema)
    assert "platform_list_agents" in enum
    assert "platform_admin_only_action" in enum


def test_dispatcher_unknown_names_silently_dropped(registry):
    """AC #5: Names in allowed_names that aren't registered are silently skipped,
    not an error. Valid names from the same call still appear."""
    schema = registry.to_dispatcher_schema(
        allowed_names=[
            "platform_list_agents",
            "platform_does_not_exist",
            "another_phantom_action",
        ],
    )
    enum = _enum_of(schema)
    assert enum == ["platform_list_agents"]


def test_dispatcher_only_unknown_names_falls_back_to_full(registry, caplog):
    """When every name in allowed_names is unknown, the intersection is empty,
    so the dispatcher falls back to the full enum and logs a WARNING. The LLM
    must never be handed a schema with zero callable actions."""
    import logging
    with caplog.at_level(logging.WARNING):
        schema = registry.to_dispatcher_schema(
            allowed_names=["nope_one", "nope_two"],
        )
    full = registry.to_dispatcher_schema(allowed_names=None)
    assert _enum_of(schema) == _enum_of(full)
    assert any(
        "intersection is empty" in record.message
        for record in caplog.records
    ), f"Expected intersection-empty warning, got: {[r.message for r in caplog.records]}"


def test_dispatcher_promoted_excluded_even_when_in_allowed_names(registry):
    """exclude_promoted=True (default) drops promoted actions even if listed
    in allowed_names. Same precedence rule as exclude_admin."""
    schema = registry.to_dispatcher_schema(
        allowed_names=["platform_list_agents", "platform_promoted_action"],
    )
    enum = _enum_of(schema)
    assert "platform_list_agents" in enum
    assert "platform_promoted_action" not in enum


def test_dispatcher_allowed_names_preserves_sorted_order(registry):
    """The enum order must be deterministic (sorted) so prompt cache keys are
    stable across calls with the same allow-list in different orders."""
    schema_a = registry.to_dispatcher_schema(
        allowed_names=["platform_list_missions", "platform_list_agents", "platform_create_agent"],
    )
    schema_b = registry.to_dispatcher_schema(
        allowed_names=["platform_create_agent", "platform_list_missions", "platform_list_agents"],
    )
    assert _enum_of(schema_a) == _enum_of(schema_b)
    assert _enum_of(schema_a) == sorted(_enum_of(schema_a))
