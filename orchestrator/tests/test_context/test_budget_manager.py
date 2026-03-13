"""Unit tests for TokenBudgetManager and related dataclasses."""

import pytest

from modules.context.budget import (
    DEFAULT_BUDGETS,
    RenderedSection,
    TokenBudget,
    TokenBudgetManager,
)
from modules.context.modes import ContextMode


class TestTokenBudget:
    """Tests for the TokenBudget dataclass."""

    def test_available_for_sections_computed(self):
        budget = TokenBudget(total=128_000, reserved_for_response=4_096, reserved_for_messages=60_000)
        assert budget.available_for_sections == 128_000 - 4_096 - 60_000

    def test_available_for_sections_zero_messages(self):
        budget = TokenBudget(total=128_000, reserved_for_response=2_048, reserved_for_messages=0)
        assert budget.available_for_sections == 125_952

    def test_frozen(self):
        budget = TokenBudget(total=100, reserved_for_response=10, reserved_for_messages=20)
        with pytest.raises(AttributeError):
            budget.total = 200  # type: ignore[misc]


class TestRenderedSection:
    """Tests for the RenderedSection dataclass."""

    def test_frozen(self):
        section = RenderedSection(name="test", priority=5, content="hello", token_estimate=1)
        with pytest.raises(AttributeError):
            section.name = "other"  # type: ignore[misc]

    def test_max_tokens_optional(self):
        section = RenderedSection(name="test", priority=5, content="x", token_estimate=1)
        assert section.max_tokens is None


class TestTokenBudgetManager:
    """Tests for the budget allocation algorithm."""

    def setup_method(self):
        self.manager = TokenBudgetManager()

    def _make_section(self, name, priority, tokens, max_tokens=None):
        content = "x" * (tokens * 4)
        return RenderedSection(
            name=name,
            priority=priority,
            content=content,
            token_estimate=tokens,
            max_tokens=max_tokens,
        )

    def _budget(self, available):
        """Create a budget where available_for_sections equals `available`."""
        return TokenBudget(total=available + 100, reserved_for_response=50, reserved_for_messages=50)

    # -- Within budget (no trimming) --

    def test_all_within_budget(self):
        sections = [
            self._make_section("identity", 1, 100),
            self._make_section("skills", 4, 200),
            self._make_section("memory", 6, 150),
        ]
        budget = self._budget(1000)  # 450 tokens needed, 1000 available

        included, trimmed = self.manager.allocate(sections, budget)

        assert len(included) == 3
        assert trimmed == []

    def test_empty_sections(self):
        included, trimmed = self.manager.allocate([], self._budget(1000))
        assert included == []
        assert trimmed == []

    # -- Max tokens capping --

    def test_section_capped_to_max_tokens(self):
        """Section exceeding its own max_tokens gets truncated."""
        section = self._make_section("memory", 6, 500, max_tokens=100)
        budget = self._budget(10_000)

        included, trimmed = self.manager.allocate([section], budget)

        assert len(included) == 1
        assert included[0].token_estimate <= 100

    # -- Priority-based dropping --

    def test_drops_lowest_priority_first(self):
        """When over budget, highest priority number (lowest importance) dropped first."""
        sections = [
            self._make_section("identity", 1, 200),
            self._make_section("skills", 4, 200),
            self._make_section("memory", 6, 200),
            self._make_section("custom", 9, 200),
        ]
        # Only 500 available, need 800 -> must drop 300+ tokens
        budget = self._budget(500)

        included, trimmed = self.manager.allocate(sections, budget)

        # custom (9) should be dropped first, then memory (6)
        included_names = {s.name for s in included}
        assert "identity" in included_names  # priority 1, never dropped
        assert "custom" not in included_names  # priority 9, dropped first
        assert "custom" in trimmed

    def test_never_drops_priority_1(self):
        """Priority 1 sections are never dropped, even if over budget."""
        sections = [
            self._make_section("identity", 1, 5000),
        ]
        budget = self._budget(100)  # Way under the 5000 needed

        included, trimmed = self.manager.allocate(sections, budget)

        assert len(included) == 1
        assert included[0].name == "identity"
        assert trimmed == []

    def test_never_drops_priority_2(self):
        """Priority 2 sections are never dropped, even if over budget."""
        sections = [
            self._make_section("identity", 1, 3000),
            self._make_section("task_context", 2, 3000),
            self._make_section("memory", 6, 200),
        ]
        budget = self._budget(1000)  # Way under the 6200 needed

        included, trimmed = self.manager.allocate(sections, budget)

        included_names = {s.name for s in included}
        assert "identity" in included_names
        assert "task_context" in included_names
        assert "memory" not in included_names

    def test_preserves_original_order(self):
        """After dropping, included sections maintain their original order."""
        sections = [
            self._make_section("identity", 1, 100),
            self._make_section("custom", 9, 100),
            self._make_section("skills", 4, 100),
            self._make_section("memory", 6, 100),
        ]
        budget = self._budget(250)  # Need to drop 150 tokens

        included, _ = self.manager.allocate(sections, budget)

        names = [s.name for s in included]
        # Verify remaining sections are in original order
        for i in range(len(names) - 1):
            orig_idx_a = next(j for j, s in enumerate(sections) if s.name == names[i])
            orig_idx_b = next(j for j, s in enumerate(sections) if s.name == names[i + 1])
            assert orig_idx_a < orig_idx_b

    def test_drops_multiple_to_fit(self):
        """May need to drop several sections to fit budget."""
        sections = [
            self._make_section("identity", 1, 100),
            self._make_section("skills", 4, 300),
            self._make_section("platform_actions", 5, 300),
            self._make_section("memory", 6, 300),
            self._make_section("datetime", 8, 50),
            self._make_section("custom", 9, 200),
        ]
        budget = self._budget(500)  # Need 1250, have 500

        included, trimmed = self.manager.allocate(sections, budget)

        # At minimum, identity (priority 1) must survive
        assert any(s.name == "identity" for s in included)
        # Several sections should be trimmed
        assert len(trimmed) >= 2


class TestDefaultBudgets:
    """Tests for DEFAULT_BUDGETS coverage."""

    def test_all_modes_have_budgets(self):
        for mode in ContextMode:
            assert mode in DEFAULT_BUDGETS, f"Missing DEFAULT_BUDGET for {mode}"

    def test_heartbeat_orchestrator_budget_compact(self):
        """Heartbeat orchestrator has 0 message reserve (no conversation history)."""
        budget = DEFAULT_BUDGETS[ContextMode.HEARTBEAT_ORCHESTRATOR]
        assert budget.reserved_for_messages == 0
        assert budget.reserved_for_response == 2_048

    def test_heartbeat_agent_budget(self):
        """Heartbeat agent has 0 message reserve, 4K response reserve."""
        budget = DEFAULT_BUDGETS[ContextMode.HEARTBEAT_AGENT]
        assert budget.reserved_for_messages == 0
        assert budget.reserved_for_response == 4_096

    def test_chatbot_budget_generous_messages(self):
        """Chatbot reserves 60K for message history."""
        budget = DEFAULT_BUDGETS[ContextMode.CHATBOT]
        assert budget.reserved_for_messages == 60_000

    def test_all_budgets_have_positive_sections_budget(self):
        for mode, budget in DEFAULT_BUDGETS.items():
            assert budget.available_for_sections > 0, (
                f"{mode} has non-positive section budget"
            )
