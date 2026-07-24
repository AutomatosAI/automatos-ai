"""Pure field-memory scoring — PRD-166 S2.

No Qdrant, no IO, no clock: every function takes its inputs explicitly so the
ranking is unit-testable (the golden-ranking suite) without a vector store.

Three-factor resonance — ``similarity × stability × recency``:

* **similarity** — cosine², supplied by the vector store at query time.
* **stability** — how *established* a pattern is: base strength lifted by
  reinforcement (access_count), independent of time. A pattern reused across
  tasks/missions converges to a higher, time-independent floor.
* **recency** — exponential time-decay with an **adaptive half-life**: each
  access lengthens the effective half-life, so durable, frequently-recalled
  knowledge fades slowly while one-off noise decays fast.

``VectorFieldSharedContext`` delegates all scoring here so the query path, the
viz/list path, the archival filter, and the compaction predicate share one
honest definition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringParams:
    """Scoring knobs, sourced from ``config`` (D11: budgets/curves are config,
    never hardcoded). Built once per query via ``VectorFieldSharedContext``."""

    decay_rate: float            # base λ per hour (FIELD_DECAY_RATE)
    reinforce_bonus: float       # stability lift per access (FIELD_REINFORCE_BONUS)
    reinforce_cap: float         # max stability multiple (FIELD_REINFORCE_CAP)
    archival_threshold: float    # decayed strength below this → archived (filtered from results)
    half_life_access_scale: float  # adaptive half-life: each access slows decay (FIELD_HALF_LIFE_ACCESS_SCALE)


def stability_factor(
    strength: float,
    access_count: int,
    *,
    reinforce_bonus: float,
    reinforce_cap: float,
) -> float:
    """Time-independent establishment of a pattern: base strength × reinforcement
    boost, capped. Reused patterns are intrinsically stronger regardless of age."""
    boost = min(1.0 + max(0, access_count) * reinforce_bonus, reinforce_cap)
    return max(0.0, strength) * boost


def recency_factor(
    age_hours: float,
    access_count: int,
    *,
    decay_rate: float,
    half_life_access_scale: float,
) -> float:
    """Adaptive-half-life exponential decay in (0, 1]. The effective decay rate
    shrinks as ``access_count`` grows, so a frequently-accessed pattern has a
    *longer* half-life — durable knowledge persists, noise fades."""
    effective_decay = decay_rate / (1.0 + half_life_access_scale * max(0, access_count))
    return math.exp(-effective_decay * max(0.0, age_hours))


def decayed_strength(
    strength: float,
    age_hours: float,
    access_count: int,
    params: ScoringParams,
) -> float:
    """Scalar ``stability × recency`` — the time-aware strength used by the
    archival filter, the viz list, and the compaction predicate. (Same role as
    the old ``_compute_decayed_strength``, now with an adaptive half-life.)"""
    return stability_factor(
        strength, access_count,
        reinforce_bonus=params.reinforce_bonus,
        reinforce_cap=params.reinforce_cap,
    ) * recency_factor(
        age_hours, access_count,
        decay_rate=params.decay_rate,
        half_life_access_scale=params.half_life_access_scale,
    )


def resonance(
    cosine: float,
    strength: float,
    age_hours: float,
    access_count: int,
    params: ScoringParams,
) -> float:
    """Full three-factor query score: ``similarity² × stability × recency``."""
    similarity = max(0.0, cosine) ** 2
    return similarity * decayed_strength(strength, age_hours, access_count, params)


def is_prunable(
    strength: float,
    age_hours: float,
    access_count: int,
    params: ScoringParams,
    prune_threshold: float,
) -> bool:
    """Compaction predicate (PRD-166 S1): a pattern is *prunable* (deleted to
    bound Qdrant) when its decayed strength falls below the hard prune
    threshold. This is stricter than ``archival_threshold`` — archived patterns
    stay queryable; pruned ones are removed."""
    return decayed_strength(strength, age_hours, access_count, params) < prune_threshold


def is_tainted(provenance, untrusted_sources) -> bool:
    """PRD-178 S4 taint gate (top-risk #4 — promotion is the memory-poisoning
    surface): a field pattern is *tainted* when its provenance carries untrusted
    external content and must NEVER be promoted to durable memory.

    Tainted when the provenance names a source in ``untrusted_sources``
    (inbound email/web/webhook/…), or carries an explicit ``untrusted``/
    ``tainted`` truthy flag. Pure — provenance dict + source set in, bool out.
    Absent/empty provenance is treated as clean (internal agent/user origin)."""
    if not provenance:
        return False
    if provenance.get("untrusted") or provenance.get("tainted"):
        return True
    source = provenance.get("source") or provenance.get("source_type")
    if source and str(source).strip().lower() in untrusted_sources:
        return True
    return False


def is_promotable(
    decayed_strength: float,
    access_count: int,
    provenance,
    *,
    min_strength: float,
    min_access_count: int,
    untrusted_sources,
) -> bool:
    """PRD-178 S4: a field pattern promotes to durable memory only when it is
    strong AND reused AND clean. The taint gate is checked FIRST and is
    absolute — a tainted trajectory never promotes regardless of strength."""
    if is_tainted(provenance, untrusted_sources):
        return False
    return (
        decayed_strength >= min_strength
        and max(0, access_count) >= min_access_count
    )


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for budgeting a result block."""
    return (len(text or "") + 3) // 4


def budget_results(
    results: list,
    token_budget: int,
    *,
    value_key: str = "value",
) -> tuple:
    """PRD-166 S2/D11: trim a ranked result list to a token budget — NO silent
    cap. Keeps results in rank order until the budget is spent (always keeps at
    least the top one so a single large pattern still surfaces), and reports
    whether anything was dropped. Returns ``(kept, truncated)``."""
    kept: list = []
    used = 0
    for r in results:
        cost = estimate_tokens(str(r.get(value_key, "")))
        if kept and used + cost > token_budget:
            break
        kept.append(r)
        used += cost
    return kept, len(kept) < len(results)


def format_digest(
    patterns: list,
    *,
    key_key: str = "key",
    value_key: str = "value",
    truncated: bool = False,
) -> str:
    """PRD-166 S3: render ranked field patterns as a compact prompt block pinned
    into a task's dispatch prompt — the agent sees accumulated knowledge without
    having to call the query tool first. Pure (string in, string out)."""
    lines = [
        "## Field memory (accumulated knowledge for this work)",
        "Relevant findings from earlier tasks/missions, ranked by resonance:",
    ]
    for p in patterns:
        key = str(p.get(key_key, "")).strip()
        value = str(p.get(value_key, "")).strip()
        lines.append(f"- **{key}**: {value}" if key else f"- {value}")
    if truncated:
        lines.append("- _(more lower-ranked patterns omitted for budget)_")
    return "\n".join(lines)


def merge_dispatch_rows(
    upstream_rows: list,
    field_rows: list,
    *,
    key_key: str = "key",
) -> list:
    """PRD-164 S4 (Q22): merge a task's immediate upstream-dependency outputs
    ahead of semantic field hits for the dispatch digest. Deterministic — no
    new ranking algorithm: dependency order first, then field rank order.

    Field rows whose key matches an upstream row are dropped: a completed
    task's output is injected into the field under the task title, so the
    field would otherwise echo a staler copy of a row we already carry.
    Pure; never mutates its inputs.
    """
    def _norm(row: dict) -> str:
        return str(row.get(key_key, "")).strip().lower()

    seen = set()
    merged: list = []
    for row in list(upstream_rows or []) + list(field_rows or []):
        norm = _norm(row)
        if norm and norm in seen:
            continue
        if norm:
            seen.add(norm)
        merged.append(row)
    return merged
