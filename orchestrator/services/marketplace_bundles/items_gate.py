"""PRD-277 US-704 (D-12): the one check before a request for an item brought from Automatos Cloud.

An item's bundle (``items_feed``) and a plugin's package (``plugin_import``) are requested only where this
install brings items from Automatos Cloud: ``calls_cloud_items`` (the local edition, or an enterprise install
whose operator switched ``ENTERPRISE_CALLS_CLOUD_ITEMS`` on) and ``calls_cloud_sync`` (items arrive only in a
Sync, and one setting never turns on another). Each function that makes such a request asks it first, so a
caller other than the Sync's passes cannot go round it; the passes ask it too (``sync_items.items_enabled``).

It decides requests only. What the install already holds is read whatever it says: the "Items brought from
Cloud" routes answer in every edition but the hosted one, so an item brought earlier is still reviewed.

A leaf: it imports only ``core.editions``, so ``items_feed`` (which ``marketplace_cloud.feed`` imports) can
ask it without reaching ``marketplace_cloud.pull``.
"""

from __future__ import annotations

from core import editions

ITEMS_NOT_SWITCHED_ON = "Items aren't brought from Automatos Cloud on this install: its operator hasn't switched that on."


def brings_items() -> bool:
    """True where this install may request items from Automatos Cloud (``calls_cloud_items`` and ``calls_cloud_sync``)."""
    capabilities = editions.current()
    return capabilities.calls_cloud_items and capabilities.calls_cloud_sync


__all__ = ["ITEMS_NOT_SWITCHED_ON", "brings_items"]
