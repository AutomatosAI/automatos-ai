"""Generic vertical plugin — pass-through with opaque-context prefix.

PRD-141 US-002. Registered as ``PLUGIN_REGISTRY["generic"]`` and used by
any workspace whose ``settings.vertical`` is unset or equal to
``"generic"``.

Behaviour:

* ``trigger_reason`` set (any proactive trigger): the generic plugin
  returns the message unchanged. Building proactive opener directives
  is vertical-specific — generic has no opinion on the shape of a
  ``[PROACTIVE_OPENER]`` block, so it stays out of the way.
* ``trigger_reason`` is ``None`` AND ``page_context`` is non-empty:
  prepend ``"(Context: <json>)\\n\\n"`` to the message, where
  ``<json>`` is ``json.dumps(page_context, sort_keys=True)``. The
  ``sort_keys`` is deliberate — it keeps the prefix stable across
  dict-iteration orders, which matters for snapshot tests downstream.
* otherwise (no context): pass through unchanged.

The plugin treats ``page_context`` as an **opaque dict** — it never
reads vertical-specific keys (``productHandle``, ``cartItems`` and the
like). That is the whole point of having a generic plugin: a new
vertical adds its own plugin under ``integrations/<vertical>/`` and
generic stays neutral.
"""

from __future__ import annotations

import json
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from integrations import WidgetPluginResult


async def handle_widget_message(
    *,
    message: str,
    page_context: Optional[dict],
    trigger_reason: Optional[str],
    workspace_id: UUID,
    db: Session,
) -> WidgetPluginResult:
    if trigger_reason is not None:
        return WidgetPluginResult(message=message)

    if not page_context:
        return WidgetPluginResult(message=message)

    prefix = f"(Context: {json.dumps(page_context, sort_keys=True)})\n\n"
    return WidgetPluginResult(
        message=f"{prefix}{message}",
        context_note="generic: prepended opaque page_context",
    )
