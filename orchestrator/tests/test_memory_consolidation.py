"""PRD-159 S4 — contradiction-based consolidation (pure logic).

Consolidation, not time-decay, is the primary lifecycle: near-duplicates merge
into one canonical with provenance; contradictions resolve by recency+confidence
(loser archived with a reason); stable L2 memories are promoted to L3.
"""
import os
import sys
from pathlib import Path

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

from modules.memory.operations.contradiction import (  # noqa: E402
    plan_consolidation,
    group_near_duplicates,
    merge_group,
    is_contradiction,
    resolve_contradiction,
    select_promotions,
)


def _m(id_, text, *, importance=0.5, created="", tier="l3", access=0):
    return {
        "id": id_, "memory": text, "importance": importance,
        "created_at": created,
        "metadata": {"tier": tier, "importance": importance, "access_count": access},
        "access_count": access,
    }


def test_five_near_dups_merge_to_one_canonical_with_provenance():
    dups = [
        _m("1", "InBuildUK is a UK smoke ventilation contractor.", importance=0.5),
        _m("2", "InBuildUK is a UK smoke-ventilation contractor.", importance=0.9),
        _m("3", "InBuildUK is a UK smoke ventilation contractor", importance=0.4),
        _m("4", "InBuildUK is a U.K. smoke ventilation contractor.", importance=0.6),
        _m("5", "InBuildUK is a UK smoke ventilation contractor!", importance=0.3),
    ]
    groups = group_near_duplicates(dups)
    assert len(groups) == 1
    mg = merge_group(groups[0])
    assert mg.canonical["id"] == "2"                     # highest importance wins
    assert set(mg.merged_from) == {"1", "3", "4", "5"}   # provenance preserved


def test_contradiction_newer_supersedes_older_with_reason():
    older = _m("a", "The deploy target is the staging cluster.", created="2026-06-01")
    newer = _m("b", "The deploy target is the production cluster now.", created="2026-06-10")
    assert is_contradiction(older, newer)
    s = resolve_contradiction(older, newer)
    assert s.winner["id"] == "b"
    assert s.loser["id"] == "a"
    assert "recent" in s.reason


def test_contradiction_tie_breaks_on_confidence():
    a = _m("a", "Primary on-call is Alice for releases.", importance=0.4, created="2026-06-01")
    b = _m("b", "Primary on-call is Bob for releases now.", importance=0.9, created="2026-06-01")
    s = resolve_contradiction(a, b)
    assert s.winner["id"] == "b"        # same timestamp → higher importance wins
    assert "confidence" in s.reason


def test_promotion_selects_accessed_l2():
    mems = [
        _m("1", "frequently used fact", tier="l2", access=5, importance=0.5),
        _m("2", "rarely used l2 fact", tier="l2", access=0, importance=0.2),
        _m("3", "important l2 fact", tier="l2", access=0, importance=0.8),
        _m("4", "an l3 fact", tier="l3", access=9, importance=0.9),
    ]
    promoted = {m["id"] for m in select_promotions(mems)}
    assert "1" in promoted        # accessed enough
    assert "3" in promoted        # important enough
    assert "2" not in promoted    # neither
    assert "4" not in promoted    # already L3


def test_plan_consolidation_end_to_end():
    mems = [
        _m("1", "InBuildUK is a UK smoke ventilation contractor.", importance=0.5),
        _m("2", "InBuildUK is a UK smoke-ventilation contractor.", importance=0.9),
        _m("3", "The deploy target is staging.", created="2026-06-01"),
        _m("4", "The deploy target is production now.", created="2026-06-10"),
        _m("5", "Posts should cite EN 12101-2.", tier="l2", access=4),
    ]
    plan = plan_consolidation(mems)
    # near-dup 1+2 merged
    assert any(set(mg.merged_from) == {"1"} or "1" in mg.merged_from for mg in plan.merges)
    # contradiction 3 vs 4 resolved → newer (4) wins
    assert any(s.winner["id"] == "4" and s.loser["id"] == "3" for s in plan.supersessions)
    # stable L2 fact promoted
    assert any(m["id"] == "5" for m in plan.promotions)
