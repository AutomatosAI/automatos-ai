"""Contradiction-based consolidation (PRD-159 S4).

The primary memory lifecycle is **consolidation**, not time-decay: near-duplicates
merge into one canonical memory (with provenance), and contradictions are resolved
by recency + confidence (the newer/more-confident fact supersedes; the loser is
archived with a reason) instead of being aged out by the old 15h decay.

These are pure functions over plain memory dicts of the shape returned by the
Mem0 layer — ``{"id", "memory"|"content", "importance", "created_at",
"metadata": {...}}`` — so they unit-test without a DB. The job seam
(``plan_consolidation``) turns a workspace's memories into an actionable plan
(merges / supersessions / promotions) that the caller applies via the memory
service.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

# Similarity at/above this is a near-duplicate (merge); a contradiction is high
# subject-overlap with a conflicting value, handled separately.
DEFAULT_DUP_THRESHOLD = 0.82
DEFAULT_SUBJECT_OVERLAP = 0.5


def _text(m: Dict[str, Any]) -> str:
    return str(m.get("memory") or m.get("content") or "").strip()


def _importance(m: Dict[str, Any]) -> float:
    md = m.get("metadata") or {}
    try:
        return float(m.get("importance", md.get("importance", 0.5)))
    except (TypeError, ValueError):
        return 0.5


def _created(m: Dict[str, Any]) -> str:
    return str(m.get("created_at") or (m.get("metadata") or {}).get("timestamp") or "")


def similarity(a: str, b: str) -> float:
    """Normalised text similarity in [0,1] (stdlib, no embeddings)."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _tokens(s: str) -> set:
    return {t for t in "".join(c if c.isalnum() else " " for c in s.lower()).split() if len(t) > 2}


def subject_overlap(a: str, b: str) -> float:
    """Jaccard token overlap — proxy for 'about the same subject'."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class MergeGroup:
    canonical: Dict[str, Any]
    merged_from: List[str] = field(default_factory=list)   # ids folded in


@dataclass
class Supersession:
    winner: Dict[str, Any]
    loser: Dict[str, Any]
    reason: str


@dataclass
class ConsolidationPlan:
    merges: List[MergeGroup] = field(default_factory=list)
    supersessions: List[Supersession] = field(default_factory=list)
    promotions: List[Dict[str, Any]] = field(default_factory=list)


def group_near_duplicates(
    memories: List[Dict[str, Any]], threshold: float = DEFAULT_DUP_THRESHOLD
) -> List[List[Dict[str, Any]]]:
    """Greedy single-link grouping of near-duplicate memories."""
    groups: List[List[Dict[str, Any]]] = []
    for m in memories:
        placed = False
        for g in groups:
            if similarity(_text(m), _text(g[0])) >= threshold:
                g.append(m)
                placed = True
                break
        if not placed:
            groups.append([m])
    return groups


def merge_group(group: List[Dict[str, Any]]) -> MergeGroup:
    """Pick the canonical (highest importance, then newest) and record provenance."""
    canonical = max(group, key=lambda m: (_importance(m), _created(m)))
    merged_from = [str(m.get("id")) for m in group if m is not canonical and m.get("id")]
    return MergeGroup(canonical=canonical, merged_from=merged_from)


def is_contradiction(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """High subject overlap but NOT a near-duplicate → conflicting statements.

    Same topic (overlapping subject tokens) with materially different wording is
    treated as a contradiction to resolve by recency+confidence, rather than two
    coexisting truths.
    """
    ta, tb = _text(a), _text(b)
    if subject_overlap(ta, tb) < DEFAULT_SUBJECT_OVERLAP:
        return False
    return similarity(ta, tb) < DEFAULT_DUP_THRESHOLD


def resolve_contradiction(a: Dict[str, Any], b: Dict[str, Any]) -> Supersession:
    """Newer wins; ties broken by confidence (importance)."""
    ca, cb = _created(a), _created(b)
    if ca != cb:
        winner, loser = (a, b) if ca > cb else (b, a)
        reason = "superseded by a more recent fact"
    else:
        winner, loser = (a, b) if _importance(a) >= _importance(b) else (b, a)
        reason = "superseded by a higher-confidence fact"
    return Supersession(winner=winner, loser=loser, reason=reason)


def select_promotions(
    memories: List[Dict[str, Any]], *, min_access: int = 2, min_importance: float = 0.6
) -> List[Dict[str, Any]]:
    """L2 memories that have proven stable (accessed enough / important) → L3."""
    out = []
    for m in memories:
        md = m.get("metadata") or {}
        level = str(md.get("tier") or md.get("level") or "").lower()
        if level and level not in ("l2", "short", "short_term", "session"):
            continue
        access = int(m.get("access_count", md.get("access_count", 0)) or 0)
        if access >= min_access or _importance(m) >= min_importance:
            out.append(m)
    return out


def plan_consolidation(
    memories: List[Dict[str, Any]],
    *,
    dup_threshold: float = DEFAULT_DUP_THRESHOLD,
) -> ConsolidationPlan:
    """Turn a memory set into an actionable consolidation plan.

    1. group + merge near-duplicates (provenance preserved),
    2. resolve contradictions among the surviving canonicals (recency+confidence),
    3. select stable L2 memories for L3 promotion.
    """
    plan = ConsolidationPlan()

    groups = group_near_duplicates(memories, dup_threshold)
    survivors: List[Dict[str, Any]] = []
    for g in groups:
        mg = merge_group(g)
        if mg.merged_from:
            plan.merges.append(mg)
        survivors.append(mg.canonical)

    # Contradictions among survivors — compare each pair once; archive losers.
    archived_ids = set()
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            if id(a) in map(id, [m.loser for m in plan.supersessions]):
                continue
            if is_contradiction(a, b):
                s = resolve_contradiction(a, b)
                loser_id = str(s.loser.get("id"))
                if loser_id in archived_ids:
                    continue
                archived_ids.add(loser_id)
                plan.supersessions.append(s)

    remaining = [m for m in survivors if str(m.get("id")) not in archived_ids]
    plan.promotions = select_promotions(remaining)
    return plan
