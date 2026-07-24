"""PRD-187 S4 — L2→L3 promotion eligibility, one definition (P2-06, memory J4).

The old gate — ``importance > 0.7 AND access_count > 3`` — was mathematically
unreachable: the population averages 0.40–0.60 importance and ``access_count``
could only climb through a recall path that never fired (bootstrap deadlock:
promotion needs access → access needs recall → recall couldn't match). Zero
promotions in the table's lifetime was the forced output.

S3 made recall real (access now climbs as a *signal*); this policy removes the
dead AND-gate and promotes on distilled IMPORTANCE with type-aware thresholds:

- noise types (heartbeat digests, playbook summaries — the injection-filter
  exclusion set, one taxonomy) NEVER promote, at any importance: dropping the
  access gate must not open the floodgate to chatter;
- high-signal types (``user_fact`` / ``preference`` / ``procedure``) promote
  from a lower importance bar — these are exactly what durable memory is for;
- everything else keeps the standard importance threshold.

Field→durable promotion is a DIFFERENT policy and keeps its access gate
(``FIELD_PROMOTION_MIN_ACCESS_COUNT``, ``jobs/promote_field_memory.py``) —
there, access_count is genuine usage of a live recall path.

``promotion_eligible`` (pure) is the source of truth; ``eligibility_conditions``
is its SQLAlchemy mirror — they live side by side so drift is visible.
"""
from __future__ import annotations

from typing import Any, FrozenSet, List, Optional

from modules.memory.injection_filter import EXCLUDED_INJECTION_CONTENT_TYPES


def high_signal_types(raw: Optional[str] = None) -> FrozenSet[str]:
    """The comma-separated config list as a set (config-sourced by default)."""
    if raw is None:
        from config import config

        raw = config.MEMORY_PROMOTION_HIGH_SIGNAL_TYPES
    return frozenset(t.strip() for t in (raw or "").split(",") if t.strip())


def promotion_eligible(
    content_type: Optional[str],
    importance: Optional[float],
    *,
    min_importance: float,
    high_signal_min_importance: float,
    high_signal: FrozenSet[str],
    excluded: FrozenSet[str] = EXCLUDED_INJECTION_CONTENT_TYPES,
) -> bool:
    """PURE eligibility predicate — the one definition of 'promotable'."""
    ctype = content_type or ""
    score = importance or 0.0
    if ctype in excluded:
        return False
    if ctype in high_signal:
        return score >= high_signal_min_importance
    return score > min_importance


def eligibility_conditions(model: Any, workspace_id: str) -> List[Any]:
    """SQLAlchemy mirror of ``promotion_eligible`` over ``memory_short_term``,
    plus the standing row-state guards (not promoted, not archived).

    ``model`` is the ``MemoryShortTerm`` class — passed in so this module stays
    import-light for pure tests.
    """
    from sqlalchemy import and_, or_

    from config import config

    signal_types = high_signal_types()
    return [
        model.workspace_id == workspace_id,
        model.promoted_to_l3.is_(False),
        model.archived_at.is_(None),
        model.content_type.notin_(list(EXCLUDED_INJECTION_CONTENT_TYPES)),
        or_(
            and_(
                model.content_type.in_(list(signal_types)),
                model.importance >= config.MEMORY_PROMOTION_HIGH_SIGNAL_MIN_IMPORTANCE,
            ),
            model.importance > config.MEMORY_PROMOTION_MIN_IMPORTANCE,
        ),
    ]
