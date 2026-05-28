"""Generic widget plugin package.

Importing this package self-registers the ``"generic"`` plugin into
``integrations.PLUGIN_REGISTRY``. The plugin module itself
(:mod:`integrations.generic.widget_proactive`) is what the dispatcher
calls — its module-level ``handle_widget_message`` satisfies the
:class:`integrations.WidgetPlugin` protocol structurally.

See ``widget_proactive.py`` for behaviour. PRD-141 US-002.
"""

from __future__ import annotations

from integrations import PLUGIN_REGISTRY

from . import widget_proactive

PLUGIN_REGISTRY["generic"] = widget_proactive

__all__ = ["widget_proactive"]
