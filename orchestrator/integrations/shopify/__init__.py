"""Shopify widget plugin package.

Importing this package self-registers the ``"shopify"`` plugin into
``integrations.PLUGIN_REGISTRY``. The plugin module itself
(:mod:`integrations.shopify.widget_proactive`) is what the dispatcher
calls — its module-level ``handle_widget_message`` satisfies the
:class:`integrations.WidgetPlugin` protocol structurally.

After PRD-141 US-010 the plugin owns the full proactive opener and
cart-idle path end-to-end: resolvers, builders, and message dispatch.
``orchestrator/api/widgets/chat.py`` calls this module only through
``PLUGIN_REGISTRY["shopify"].handle_widget_message`` — no direct
imports of the underlying helpers from outside this package.
"""

from __future__ import annotations

from integrations import PLUGIN_REGISTRY

from . import widget_proactive
from . import provision  # PRD-183 S5: registers the Shopify VerticalProvisioner

PLUGIN_REGISTRY["shopify"] = widget_proactive

# Register the provisioner + graph-source mappers into the generic
# provisioning plane (PROVISIONER_REGISTRY / GRAPH_SOURCE_MAPPERS).
provision.register()

__all__ = ["widget_proactive", "provision"]
