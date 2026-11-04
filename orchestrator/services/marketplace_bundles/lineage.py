"""PRD-270 W2 (K10) — an exported item's identity: its lineage, and the externalId hashed from it.

Automatos Cloud has no version flow for community items: a resubmission is a new marketplace row whose
``cloned_from_id`` is the submitter's own row, the same one each time. So an item is keyed by its
lineage, ``from:<cloned_from_id>`` when the row has one, else ``row:<its natural key>`` (an agent's
public_id, a Playbook's template_id, a package's or a plugin's slug), and its externalId is the first
32 hex characters of sha256('automatos-cloud:<type>:<lineage>'): stable across resubmissions, opaque,
and never an id a route takes. A submitter who deletes the row they submitted from starts a new
lineage (the column is ON DELETE SET NULL), so their next resubmission is a new item.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional

from services.marketplace_bundles.schema import EXTERNAL_ID_LENGTH
from services.themes.digest import sha256_hex

EXTERNAL_ID_NAMESPACE = "automatos-cloud"
FROM_LINEAGE = "from:"
ROW_LINEAGE = "row:"
# A row's natural key, by bundle type: the column a lineage names when the row was cloned from none.
NATURAL_KEYS: Mapping[str, str] = MappingProxyType({
    "agent": "public_id",
    "playbook": "template_id",
    "package": "slug",
    "plugin": "slug",
})


def external_id(item_type: str, lineage: str) -> str:
    """The externalId of the ``item_type`` item of ``lineage``: 32 lower-case hex characters."""
    seed = f"{EXTERNAL_ID_NAMESPACE}:{item_type}:{lineage}".encode("utf-8")
    return sha256_hex(seed)[:EXTERNAL_ID_LENGTH]


def lineage_of(item_type: str, row: Any) -> Optional[str]:
    """The lineage of ``row``, a row (or a column query's result) of ``item_type``; None when the row
    has neither a ``cloned_from_id`` nor its natural key, so nothing stable names it."""
    cloned_from = getattr(row, "cloned_from_id", None)
    if cloned_from is not None:
        return f"{FROM_LINEAGE}{cloned_from}"
    key = getattr(row, NATURAL_KEYS[item_type], None)
    return f"{ROW_LINEAGE}{key}" if key is not None and str(key) else None


__all__ = ["external_id", "lineage_of"]
