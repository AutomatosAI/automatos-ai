"""Shopify widget plugin package.

Importing this package self-registers the ``"shopify"`` plugin into
``integrations.PLUGIN_REGISTRY``. The plugin module itself
(:mod:`integrations.shopify.widget_proactive`) is what the dispatcher
calls — its module-level ``handle_widget_message`` satisfies the
:class:`integrations.WidgetPlugin` protocol structurally.

US-003 ships this as a SHIM that delegates to existing chat.py
helpers. US-006/007/008 move those helpers into this folder and US-010
deletes the shim layer. See ``widget_proactive.py`` for the current
behaviour and migration story. PRD-141 US-003.
"""

from __future__ import annotations

from integrations import PLUGIN_REGISTRY

from . import widget_proactive

PLUGIN_REGISTRY["shopify"] = widget_proactive

__all__ = ["widget_proactive"]
