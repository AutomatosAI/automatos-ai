"""
Widget Callback Tool
====================

Tool exposed to the Shopify Support agent (and any future widget-bound
agent) that asks the SDK to open the inline phone-capture form when the
shopper requests a callback / phone / human contact.

The tool is intentionally a *signal* tool — it does not perform any
external action. Its sole responsibility is to emit a marker in the
tool-data SSE chunk that ``api.widgets.chat`` translates into the
``event: open-callback-form`` SSE event the widget SDK already listens
for (see ``automatos-widget-sdk/packages/core/src/client.ts``).

Replaces the deprecated server-side keyword/regex matcher in
``api.widgets.chat`` (PRD-008-A.2 v1). The LLM owns intent recognition
end-to-end through its skill; this tool is the affordance the LLM uses
to open the form.

Signal contract
---------------
The handler returns a result whose ``frontend_data`` carries::

    {
        "_widget_signal": "open-callback-form",
        "product_context": "<optional product title>"
    }

``api/widgets/chat.py`` reads this from the ``tool-data`` chunk and
emits the SSE event with the correct payload.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Public constants — imported by registry, executor, and the SSE bridge
# in api.widgets.chat. Keep them in one place so they cannot drift.

WIDGET_OPEN_CALLBACK_FORM_NAME: str = "widget_open_callback_form"

WIDGET_SIGNAL_KEY: str = "_widget_signal"
WIDGET_SIGNAL_OPEN_CALLBACK_FORM: str = "open-callback-form"


async def handle_widget_open_callback_form(
    tool_name: str,
    parameters: Dict[str, Any],
    agent_id: int = 0,
    workspace_id: Any = None,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the open-callback-form signal tool.

    Returns a tool result shaped for the standard formatter so the LLM
    receives a clear confirmation and the streaming pipeline emits a
    ``tool-data`` chunk the SSE bridge can act on.
    """
    raw_product_context = parameters.get("product_context")
    product_context: Optional[str]
    if isinstance(raw_product_context, str) and raw_product_context.strip():
        product_context = raw_product_context.strip()
    else:
        product_context = None

    frontend_data: Dict[str, Any] = {
        WIDGET_SIGNAL_KEY: WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
        "product_context": product_context,
    }

    return {
        "success": True,
        "tool": tool_name,
        "frontend_data": frontend_data,
        "llm_context": (
            "Callback form is now open in the shopper's chat panel. Confirm "
            "briefly that you've opened a quick form and ask them to fill in "
            "their name and number. Do NOT offer email or alternative contact "
            "methods — the form is the contact method."
        ),
        "result": "open-callback-form signal emitted",
    }
