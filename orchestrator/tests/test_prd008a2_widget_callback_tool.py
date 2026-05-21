"""PRD-008-A.2 — LLM-driven callback form trigger.

Tests the proper agent-tool architecture that replaced the deprecated
server-side keyword/regex matcher:

* the tool handler returns a frontend-data signal the SSE bridge can act on
* the result formatter preserves that signal end-to-end
* the registry no longer references the old keyword constants
"""

from __future__ import annotations

import pytest

from modules.tools.widget_callback import (
    WIDGET_OPEN_CALLBACK_FORM_NAME,
    WIDGET_SIGNAL_KEY,
    WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
    handle_widget_open_callback_form,
)
from modules.tools.formatting.result_formatter import ToolResultFormatter


class TestKeywordMatcherIsGone:
    """The keyword matcher was a regression — make sure nobody re-imports it."""

    def test_old_symbols_no_longer_exported(self) -> None:
        import api.widgets.chat as widget_chat

        assert not hasattr(widget_chat, "_DEFAULT_CALLBACK_INTENT_PHRASES")
        assert not hasattr(widget_chat, "_matches_callback_intent")


class TestWidgetCallbackHandler:
    """The handler is a pure signal — no side effects, always succeeds."""

    @pytest.mark.asyncio
    async def test_handler_returns_signal_in_frontend_data(self) -> None:
        result = await handle_widget_open_callback_form(
            WIDGET_OPEN_CALLBACK_FORM_NAME,
            {"product_context": "EN 12101-9 Control Panel"},
        )
        assert result["success"] is True
        assert result["frontend_data"][WIDGET_SIGNAL_KEY] == WIDGET_SIGNAL_OPEN_CALLBACK_FORM
        assert result["frontend_data"]["product_context"] == "EN 12101-9 Control Panel"
        # The LLM context must instruct against suggesting email.
        assert "email" in result["llm_context"].lower()

    @pytest.mark.asyncio
    async def test_handler_normalises_missing_product_context(self) -> None:
        result = await handle_widget_open_callback_form(
            WIDGET_OPEN_CALLBACK_FORM_NAME,
            {},
        )
        assert result["success"] is True
        assert result["frontend_data"]["product_context"] is None

    @pytest.mark.asyncio
    async def test_handler_normalises_blank_product_context(self) -> None:
        result = await handle_widget_open_callback_form(
            WIDGET_OPEN_CALLBACK_FORM_NAME,
            {"product_context": "   "},
        )
        assert result["frontend_data"]["product_context"] is None

    @pytest.mark.asyncio
    async def test_handler_ignores_non_string_product_context(self) -> None:
        result = await handle_widget_open_callback_form(
            WIDGET_OPEN_CALLBACK_FORM_NAME,
            {"product_context": 123},
        )
        assert result["frontend_data"]["product_context"] is None


class TestResultFormatterPassthrough:
    """The formatter must preserve the widget signal so the SSE bridge sees it."""

    def test_format_for_frontend_passes_signal_through(self) -> None:
        handler_result = {
            "success": True,
            "frontend_data": {
                WIDGET_SIGNAL_KEY: WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
                "product_context": "Smoke Damper",
            },
        }
        frontend = ToolResultFormatter.format_for_frontend(
            handler_result, WIDGET_OPEN_CALLBACK_FORM_NAME,
        )
        assert frontend[WIDGET_SIGNAL_KEY] == WIDGET_SIGNAL_OPEN_CALLBACK_FORM
        assert frontend["product_context"] == "Smoke Damper"

    def test_format_for_llm_uses_handler_message(self) -> None:
        handler_result = {
            "success": True,
            "llm_context": "Callback form is now open in the shopper's chat panel. Confirm briefly.",
        }
        out = ToolResultFormatter.format_for_llm(
            handler_result, WIDGET_OPEN_CALLBACK_FORM_NAME,
        )
        assert "Callback form is now open" in out

    def test_format_for_llm_handles_failure(self) -> None:
        handler_result = {"success": False, "error": "boom"}
        out = ToolResultFormatter.format_for_llm(
            handler_result, WIDGET_OPEN_CALLBACK_FORM_NAME,
        )
        assert "failed" in out.lower()
        assert "boom" in out


class TestRegistryRegistration:
    """The tool must be registered and gated on Site callback.enabled."""

    def test_tool_is_in_registry(self) -> None:
        from modules.tools.registry.tool_registry import (
            get_tool_registry,
            reset_tool_registry,
        )

        reset_tool_registry()
        registry = get_tool_registry()
        spec = registry.get_tool(WIDGET_OPEN_CALLBACK_FORM_NAME)
        assert spec is not None
        assert spec.is_active is True
        # The skill / LLM only needs product_context.
        param_names = {p.name for p in spec.parameters}
        assert param_names == {"product_context"}

    def test_validate_access_requires_workspace_context(self) -> None:
        from modules.tools.registry.tool_registry import (
            get_tool_registry,
            reset_tool_registry,
        )

        reset_tool_registry()
        registry = get_tool_registry()
        allowed, reason = registry.validate_tool_access(
            agent_id=1,
            tool_name=WIDGET_OPEN_CALLBACK_FORM_NAME,
            db=None,
            workspace_id=None,
        )
        assert allowed is False
        assert reason is not None
