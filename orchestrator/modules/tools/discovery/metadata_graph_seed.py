"""Metadata-based graph cold-start for the tool-routing graph (PRD-143).

Telemetry seeding (``edge_builder`` / ``seed_tool_routing_graph``) only helps
tools that already appear in ``tool_execution_logs``. A brand-new platform tool
has ZERO telemetry, so the routing graph gives it no signal until it has been
used enough times — exactly when discoverability matters least. This closes
that gap: it seeds GLOBAL ``meta_sibling`` edges straight from the registry's
own metadata so a new tool is reachable via the graph the moment a sibling is
relevant, before any usage accrues.

Design (consistent with the existing multi-edge-type graph):
  * Two operator tools that share a ``category`` are likely co-relevant, so we
    add a directed ``meta_sibling`` edge between every same-category pair.
  * Edges are GLOBAL (``workspace_id=None``, ``agent_id=None``) — metadata is
    app-wide, and ``GraphRouter._query_edges`` already reads global edges.
  * ``confidence`` sits at the graph's min-confidence floor (nudged up slightly
    by shared-tag overlap) so meta edges PASS the router's filter but rank
    BELOW real ``used_after`` edges (higher Wilson confidence) as telemetry
    accrues — i.e. metadata is the floor, real usage always wins.
  * They coexist with ``used_after`` rows (the unique key includes
    ``edge_type``) and SURVIVE the nightly ``edge_builder`` recompute, which
    only rewrites ``used_after`` and intent-cluster rows.
  * ``super_admin_only`` tools are NEVER seeded — the obs tier stays off Auto's
    graph, same invariant the rest of PRD-143 enforces.

Idempotent: re-running upserts the same rows (ON CONFLICT UPDATE), so it
converges and is safe to apply like a migration.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Tuple

from .action_registry import ActionRegistry

META_EDGE_TYPE = "meta_sibling"

# The graph's default min-confidence floor (GraphRouter._min_confidence). Meta
# edges sit AT the floor so they pass the filter but never outrank real signal.
_META_BASE_CONFIDENCE = 0.6
# Shared tags nudge confidence within a tiny band so the most-related sibling in
# a category surfaces first — capped well below a strong real edge (Wilson ~0.7+).
_META_TAG_STEP = 0.01
_META_MAX_TAG_BONUS = 4


def _operator_actions(registry: ActionRegistry):
    """Operator-tier actions only — super_admin_only (the obs tier) is excluded
    so the metadata graph never makes an obs tool reachable."""
    return [
        a for a in registry.get_all()
        if not getattr(a, "super_admin_only", False)
    ]


def compute_meta_sibling_pairs(
    registry: ActionRegistry,
) -> List[Tuple[str, str, float, float]]:
    """Directed same-category operator pairs with a metadata weight/confidence.

    Returns ``(from_action, to_action, weight, confidence)`` for every ordered
    pair of distinct operator tools that share a non-empty ``category``.
    ``confidence`` rises slightly with shared-tag overlap (capped), ``weight``
    records the overlap for observability.
    """
    actions = _operator_actions(registry)
    by_category: dict = {}
    for a in actions:
        cat = (a.category or "").strip()
        if not cat:
            continue
        by_category.setdefault(cat, []).append(a)

    pairs: List[Tuple[str, str, float, float]] = []
    for siblings in by_category.values():
        if len(siblings) < 2:
            continue
        for src in siblings:
            src_tags = set(src.tags or [])
            for dst in siblings:
                if src.name == dst.name:
                    continue
                shared = len(src_tags & set(dst.tags or []))
                bonus = min(shared, _META_MAX_TAG_BONUS) * _META_TAG_STEP
                confidence = _META_BASE_CONFIDENCE + bonus
                weight = float(shared + 1)
                pairs.append((src.name, dst.name, weight, confidence))
    return pairs


def seed_meta_sibling_edges(
    db,
    registry: ActionRegistry,
    *,
    dry_run: bool = False,
) -> int:
    """Upsert global ``meta_sibling`` edges from registry metadata.

    Returns the number of edges that would be (or were) written. With
    ``dry_run`` nothing is written. Reuses ``edge_builder._upsert_edge_row`` so
    the ON CONFLICT upsert and unique-key semantics match real edges exactly.
    """
    from core.services.edge_builder import _upsert_edge_row

    pairs = compute_meta_sibling_pairs(registry)
    if dry_run:
        return len(pairs)

    now = datetime.utcnow()
    for from_action, to_action, weight, confidence in pairs:
        _upsert_edge_row(
            db,
            from_action,
            to_action,
            META_EDGE_TYPE,
            None,  # workspace_id — global
            None,  # agent_id — global
            weight,
            confidence,
            0,     # sample_count — metadata, not observed usage
            now,
        )
    db.flush()
    return len(pairs)
