"""BudStacks vertical package.

Importing this package self-registers the ``"budstacks"`` VerticalProvisioner
into ``integrations.provisioning.PROVISIONER_REGISTRY`` — same import-time
registration pattern as :mod:`integrations.shopify`. BudStacks has no widget
plugin (the generic widget chat path serves it), so unlike the Shopify package
nothing is added to ``PLUGIN_REGISTRY``.
"""

from __future__ import annotations

from . import provision

provision.register()

__all__ = ["provision"]
