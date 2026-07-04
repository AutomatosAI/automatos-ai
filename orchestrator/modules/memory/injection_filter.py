"""PRD-185 S11 — assembly-side memory-injection guard.

Sub-floor and noise-typed memories must never reach the prompt. The relevance
floor is already applied at the L3 search boundary
(``mem0_client.filter_by_relevance_floor``, PRD-159 S3); this is the guard over
the **merged** candidate set at the one chokepoint that feeds
``_format_memories_for_llm`` — so it re-asserts the floor over every source
(L3 global + agent tiers, and any future L2) AND adds the content-type exclusion
the search layer lacks (heartbeat digests, playbook execution summaries).

Pure — no I/O — so it unit-tests with plain dicts.
"""
from typing import Any, Dict, Iterable, List, Optional

# ``content_type`` / ``metadata.type`` values that are operational noise, never
# user context: heartbeat digests and playbook/recipe execution summaries. These
# are excluded from prompt injection. ``recipe_summary`` is the legacy name for
# ``playbook_summary`` (pre-March rename) — kept so old rows are filtered too.
EXCLUDED_INJECTION_CONTENT_TYPES = frozenset({
    "heartbeat_log",
    "playbook_summary",
    "recipe_summary",
})


def _content_type_of(mem: Dict[str, Any]) -> Optional[str]:
    """Best-effort content-type signal across the shapes a memory row takes.

    Mem0 search rows expose ``{id, memory, score, metadata, created_at}`` — the
    type, when present, rides in ``metadata`` (``content_type`` or ``type``);
    L2-shaped rows carry a top-level ``content_type`` (or ``category``). Checking
    every path means the filter bites wherever the tag lands instead of silently
    no-op-ing on a shape mismatch — the exact failure class this wave exists for.
    """
    if not isinstance(mem, dict):
        return None
    meta = mem.get("metadata")
    meta = meta if isinstance(meta, dict) else {}
    return (
        mem.get("content_type")
        or meta.get("content_type")
        or meta.get("type")
        or mem.get("category")
    )


def filter_injectable_memories(
    memories: Iterable[Dict[str, Any]],
    *,
    floor: float,
    excluded_types: Iterable[str] = EXCLUDED_INJECTION_CONTENT_TYPES,
) -> List[Dict[str, Any]]:
    """Drop sub-floor and noise-typed memories before prompt injection.

    - Scored-but-below-``floor`` rows are dropped; unscored rows are kept (cannot
      judge — same rule as ``filter_by_relevance_floor``). ``floor <= 0`` disables
      the score check.
    - Rows whose content-type signal is in ``excluded_types`` are dropped.
    """
    excluded = frozenset(excluded_types)
    out: List[Dict[str, Any]] = []
    for mem in memories:
        if not isinstance(mem, dict):
            continue
        score = mem.get("score")
        if floor and floor > 0 and score is not None and (score or 0) < floor:
            continue
        if _content_type_of(mem) in excluded:
            continue
        out.append(mem)
    return out
