"""Cross-sell persistence integrity (PRD-189 S2).

The F1 wipe stayed invisible for weeks because the sync status blocks reported
that a sync *ran* while the graph no longer held what it computed — the pilot's
``orders_sync`` block said ``fbt_edges_added: 16`` with 0
``frequently_bought_with`` edges actually present, and nothing read the drift.

This module is the read-only guardrail: a pure comparison between the FBT
edges the last orders sync REPORTED and the FBT edges PRESENT in the workspace
Knowledge Graph. It is asserted in the sync path after every catalog/orders
import (``api/shopify.py``) and surfaced as the Command Center Commerce tile
(``api/analytics_real.py``, the PRD-185 S12 own-workspace strip).

Pure functions only — no DB, no IO, no imports beyond typing. Callers load the
graph and the status blocks themselves.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

FBT_RELATION = "frequently_bought_with"


def count_fbt_edges(graph: Any) -> int:
    """Number of ``frequently_bought_with`` edges in a NetworkX graph.

    Duck-typed on ``graph.edges(data=True)`` so it needs no networkx import;
    works for Graph and MultiGraph alike.
    """
    return sum(
        1
        for _u, _v, attrs in graph.edges(data=True)
        if (attrs.get("relation") or "").lower() == FBT_RELATION
    )


def fbt_integrity(reported: Optional[int], present: int) -> Dict[str, Any]:
    """Compare reported-vs-present FBT edges — the query that catches the wipe.

    Args:
        reported: ``fbt_edges_added`` from the last ``orders_sync`` status
            block, or ``None`` when no orders sync has ever run.
        present: ``frequently_bought_with`` edges currently in the graph.

    Returns a new dict:
        ``reported`` / ``present`` — the two sides of the comparison.
        ``drift`` — ``present - reported`` (``None`` when nothing reported).
        ``ok`` — ``True`` iff reported == present; ``None`` when nothing
        reported (honest unknown, not a fabricated green).
    """
    if reported is None:
        return {"reported": None, "present": present, "drift": None, "ok": None}
    return {
        "reported": int(reported),
        "present": present,
        "drift": present - int(reported),
        "ok": present == int(reported),
    }
