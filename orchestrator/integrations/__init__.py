"""Vertical-plugin registry for widget chat dispatch.

PRD-141 — Widget Vertical-Agnostic Refactor.

The generic widget chat endpoint (`orchestrator/api/widgets/chat.py`) must
remain free of vertical-specific code (no Shopify keys, no booking fields,
etc.). Instead, it dispatches per-workspace via this registry:

    vertical = (workspace.settings or {}).get("vertical") or "generic"
    plugin = PLUGIN_REGISTRY[vertical]
    result = await plugin.handle_widget_message(...)
    body.message = result.message

Plugins live under `orchestrator/integrations/<vertical>/widget_proactive.py`
and self-register into `PLUGIN_REGISTRY` at import time. The registry is a
plain module-level dict — no startup hook needed; importing the plugin
module is enough (see `orchestrator/integrations/<vertical>/__init__.py`).

Contract: a plugin is any object exposing `handle_widget_message(...)`
matching the `WidgetPlugin` protocol below. The protocol is intentionally
narrow — vertical-specific helpers (graph walks, message builders) stay
private to the plugin module.

Phase 0 (this story, US-001) sets up the empty registry. US-002 registers
`"generic"`, US-003 registers `"shopify"`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy.orm import Session


@dataclass
class WidgetPluginResult:
    """Outcome of a plugin's `handle_widget_message` call.

    `message` is the possibly-rewritten user message that will be fed to
    the streaming agent. For pass-through cases (no trigger, empty
    context) it equals the input message verbatim.

    `context_note` is an optional short string the dispatcher MAY log or
    surface (e.g. "applied shopify proactive_opener rewrite"). It is not
    sent to the agent.

    `system_preamble` is an optional grounding block the dispatcher injects
    into the LLM message history for the CURRENT turn only — prepended to
    the latest user message in-memory. Unlike `message`, it is never
    persisted to the transcript nor used for conversation titling, so it
    grounds the agent fresh each turn without polluting stored history or
    accumulating stale context. `None` (the default) means no grounding.

    `telemetry` is a free-form dict for the dispatcher to attach to its
    structured log line (e.g. counts of related products resolved,
    fixture identifiers used). Plugins should keep keys short and
    JSON-serialisable.
    """

    message: str
    context_note: Optional[str] = None
    system_preamble: Optional[str] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class WidgetPlugin(Protocol):
    """Protocol every vertical plugin implements.

    Implementations live at `orchestrator/integrations/<vertical>/widget_proactive.py`
    and register themselves into `PLUGIN_REGISTRY` at import time.
    """

    async def handle_widget_message(
        self,
        *,
        message: str,
        page_context: Optional[dict],
        trigger_reason: Optional[str],
        workspace_id: UUID,
        db: Session,
    ) -> WidgetPluginResult: ...


PLUGIN_REGISTRY: Dict[str, WidgetPlugin] = {}


__all__ = [
    "PLUGIN_REGISTRY",
    "WidgetPlugin",
    "WidgetPluginResult",
]


# Sub-packages self-register into PLUGIN_REGISTRY at import time. Imports go
# here, after the registry is defined, so each vertical can do
# ``from integrations import PLUGIN_REGISTRY`` without circularity.
from . import generic  # noqa: E402,F401  (registers "generic")
from . import shopify  # noqa: E402,F401  (registers "shopify")
