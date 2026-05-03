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
    """AC #8: Empty list returns header-only summary, no action lines."""
    summary = registry.build_filtered_prompt_summary([])
    # Header is present
    assert "## Available Platform Actions" in summary
    # No action bullet lines (no "- `platform_..." entries)
    assert "- `platform_" not in summary
    # No category headers
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
    """exclude_promoted=True (default) drops promoted actions even if requested."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_promoted_action"],
    )
    assert "platform_list_agents" in summary
    assert "platform_promoted_action" not in summary


def test_include_promoted_when_disabled(registry):
    """exclude_promoted=False keeps promoted actions in the summary."""
    summary = registry.build_filtered_prompt_summary(
        ["platform_list_agents", "platform_promoted_action"],
        exclude_promoted=False,
    )
    assert "platform_list_agents" in summary
    assert "platform_promoted_action" in summary


# ---- AC #2: format identical to build_prompt_summary ----

def test_format_matches_build_prompt_summary(registry):
    """AC #2 + #6: Filtered output must match the unfiltered output for the
    same subset, so callers can swap one method for the other.

    Strategy: include ALL non-admin non-promoted actions in the filter,
    expect identical bytes to the unfiltered (default flags) summary.
    """
    full = registry.build_prompt_summary()  # exclude_admin=False, exclude_promoted=True
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
    # promoted excluded by default
    assert "platform_promoted_action" not in summary
    # admin included by default (exclude_admin=False)
    assert "platform_admin_only_action" in summary


def test_build_prompt_summary_exclude_admin(registry):
    """exclude_admin=True drops admin_only actions in unfiltered summary too."""
    summary = registry.build_prompt_summary(exclude_admin=True)
    assert "platform_list_agents" in summary
    assert "platform_admin_only_action" not in summary
