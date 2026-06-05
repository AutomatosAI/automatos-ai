"""Unit tests for ContextMode, ModeConfig, and MODE_CONFIGS validation."""

import pytest

from modules.context.modes import ContextMode, ModeConfig, MODE_CONFIGS
from modules.context.sections import SECTION_REGISTRY


class TestContextMode:
    """Tests for the ContextMode enum."""

    EXPECTED_MODES = {
        "chatbot", "task_execution", "heartbeat_orchestrator",
        "heartbeat_agent", "recipe",
        "router", "orchestrator_stage", "nl2sql",
        "coordinator",
    }

    def test_all_expected_modes_exist(self):
        actual = {m.value for m in ContextMode}
        assert actual == self.EXPECTED_MODES

    def test_mode_is_string_enum(self):
        """ContextMode values are strings (str, Enum)."""
        for mode in ContextMode:
            assert isinstance(mode.value, str)
            assert isinstance(mode, str)


class TestModeConfig:
    """Tests for the ModeConfig frozen dataclass."""

    def test_frozen(self):
        config = ModeConfig(sections=["identity"], tool_loading="none")
        with pytest.raises(AttributeError):
            config.personality = True  # type: ignore[misc]

    def test_defaults(self):
        config = ModeConfig()
        assert config.sections == []
        assert config.tool_loading == "none"
        assert config.personality is False
        assert config.max_tokens is None


class TestModeConfigs:
    """Tests that MODE_CONFIGS is complete and consistent with SECTION_REGISTRY."""

    def test_every_mode_has_config(self):
        """Every ContextMode must have an entry in MODE_CONFIGS."""
        for mode in ContextMode:
            assert mode in MODE_CONFIGS, f"Missing MODE_CONFIG for {mode}"

    def test_all_section_names_in_registry(self):
        """Every section name used in any ModeConfig must exist in SECTION_REGISTRY."""
        for mode, config in MODE_CONFIGS.items():
            for section_name in config.sections:
                assert section_name in SECTION_REGISTRY, (
                    f"Section '{section_name}' in {mode} config "
                    f"not found in SECTION_REGISTRY"
                )

    def test_no_duplicate_sections_in_config(self):
        """No mode should list the same section twice."""
        for mode, config in MODE_CONFIGS.items():
            assert len(config.sections) == len(set(config.sections)), (
                f"{mode} has duplicate sections: {config.sections}"
            )

    def test_chatbot_has_personality(self):
        assert MODE_CONFIGS[ContextMode.CHATBOT].personality is True

    def test_non_chatbot_modes_no_personality(self):
        for mode in ContextMode:
            if mode == ContextMode.CHATBOT:
                continue
            assert MODE_CONFIGS[mode].personality is False, (
                f"{mode} should not have personality=True"
            )

    def test_heartbeat_orchestrator_has_max_tokens(self):
        assert MODE_CONFIGS[ContextMode.HEARTBEAT_ORCHESTRATOR].max_tokens == 8000

    def test_heartbeat_agent_has_max_tokens(self):
        assert MODE_CONFIGS[ContextMode.HEARTBEAT_AGENT].max_tokens == 128000

    def test_tool_loading_values_valid(self):
        valid_strategies = {"full", "filtered", "dispatcher_only", "none"}
        for mode, config in MODE_CONFIGS.items():
            assert config.tool_loading in valid_strategies, (
                f"{mode} has invalid tool_loading: {config.tool_loading}"
            )

    def test_chatbot_uses_filtered_tools(self):
        assert MODE_CONFIGS[ContextMode.CHATBOT].tool_loading == "filtered"

    def test_task_execution_uses_full_tools(self):
        assert MODE_CONFIGS[ContextMode.TASK_EXECUTION].tool_loading == "full"

    def test_heartbeat_orchestrator_uses_dispatcher_only(self):
        assert MODE_CONFIGS[ContextMode.HEARTBEAT_ORCHESTRATOR].tool_loading == "dispatcher_only"

    def test_heartbeat_agent_uses_full_tools(self):
        assert MODE_CONFIGS[ContextMode.HEARTBEAT_AGENT].tool_loading == "full"

    def test_router_uses_no_tools(self):
        assert MODE_CONFIGS[ContextMode.ROUTER].tool_loading == "none"


class TestSectionRegistry:
    """Tests for SECTION_REGISTRY completeness."""

    EXPECTED_SECTIONS = {
        "identity", "skills", "composio", "plugins",
        "platform_actions", "memory", "tools",
        "task_context", "playbook_context", "datetime_context",
        "business_graph", "conversation", "custom",
        "onboarding", "mission_context", "agent_roster",
    }

    def test_all_expected_sections_registered(self):
        assert set(SECTION_REGISTRY.keys()) == self.EXPECTED_SECTIONS

    def test_registry_values_are_classes(self):
        from modules.context.sections.base import BaseSection

        for name, cls in SECTION_REGISTRY.items():
            assert isinstance(cls, type), f"{name} is not a class"
            assert issubclass(cls, BaseSection), (
                f"{name} -> {cls} is not a BaseSection subclass"
            )

    def test_section_names_match_registry_keys(self):
        """Each section class's .name attribute matches its registry key."""
        for key, cls in SECTION_REGISTRY.items():
            instance = cls()
            assert instance.name == key, (
                f"Registry key '{key}' != section.name '{instance.name}'"
            )
