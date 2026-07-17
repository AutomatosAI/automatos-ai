"""PRD-206 S7 — recall that ranks like an auntie remembers.

Composite recall score over the merged candidate set, applied AFTER the
relevance floor and type exclusions (``injection_filter`` — load-bearing,
untouched):

    semantic × recency-decay × importance × pin-boost × same-page × same-project

- ``semantic`` is the store's similarity score; unscored rows get a neutral
  0.5 (they were kept by the floor rule "cannot judge", so they must not
  outrank genuinely-scored rows by default).
- ``recency-decay`` halves per ``half_life_days`` of age; rows without a
  parseable ``created_at`` get 1.0 (cannot judge — don't punish legacy rows).
- ``importance`` (S1 contract, [0,1]) maps to a [0.5, 1.5] factor; absent →
  neutral 1.0.
- ``pinned`` (S1 contract / S5 panel toggle) multiplies by the pin boost.
- Page/project boosts stay inert until callers pass ``query_page`` (S6) /
  ``query_project`` (S4) — the scoring seam is ready, the signals arrive
  with their stories.

Pure — no I/O, no config reads; callers resolve knobs (``MEMORY_RANK_*``)
and pass them in. Returns a NEW sorted list; input never mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_HALF_LIFE_DAYS = 30.0
DEFAULT_PIN_BOOST = 2.0
DEFAULT_PAGE_BOOST = 1.15
DEFAULT_PROJECT_BOOST = 1.15
# Neutral semantic stand-in for unscored rows (kept by the floor's
# "cannot judge" rule) — must sit below a confident match, above junk.
UNSCORED_SEMANTIC = 0.5


def _parse_created_at(value: Any) -> Optional[datetime]:
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def composite_score(
    mem: Dict[str, Any],
    *,
    now: datetime,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    pin_boost: float = DEFAULT_PIN_BOOST,
    page_boost: float = DEFAULT_PAGE_BOOST,
    project_boost: float = DEFAULT_PROJECT_BOOST,
    query_page: Optional[str] = None,
    query_project: Optional[str] = None,
) -> float:
    meta = mem.get("metadata")
    meta = meta if isinstance(meta, dict) else {}

    semantic = mem.get("score")
    semantic = float(semantic) if isinstance(semantic, (int, float)) else UNSCORED_SEMANTIC

    recency = 1.0
    created = _parse_created_at(mem.get("created_at"))
    if created is not None and half_life_days > 0:
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        recency = 0.5 ** (age_days / half_life_days)

    try:
        importance = float(meta.get("importance", 0.5))
    except (TypeError, ValueError):
        importance = 0.5
    importance_factor = 0.5 + max(0.0, min(1.0, importance))

    pin = pin_boost if meta.get("pinned") else 1.0
    page = page_boost if query_page and meta.get("page") == query_page else 1.0
    project = (
        project_boost
        if query_project and str(meta.get("project_id") or "") == str(query_project)
        else 1.0
    )

    return semantic * recency * importance_factor * pin * page * project


def rank_memories(
    memories: Iterable[Dict[str, Any]],
    *,
    now: Optional[datetime] = None,
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
    pin_boost: float = DEFAULT_PIN_BOOST,
    page_boost: float = DEFAULT_PAGE_BOOST,
    project_boost: float = DEFAULT_PROJECT_BOOST,
    query_page: Optional[str] = None,
    query_project: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """New list, best-first by composite score. Stable: equal scores keep
    their incoming order (semantic order from the store)."""
    now = now or datetime.now(timezone.utc)
    items = [m for m in memories if isinstance(m, dict)]
    return sorted(
        items,
        key=lambda m: composite_score(
            m,
            now=now,
            half_life_days=half_life_days,
            pin_boost=pin_boost,
            page_boost=page_boost,
            project_boost=project_boost,
            query_page=query_page,
            query_project=query_project,
        ),
        reverse=True,
    )
