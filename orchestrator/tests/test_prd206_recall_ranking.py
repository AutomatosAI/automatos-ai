"""PRD-206 S7 — composite recall ranking above the floor + exclusions.

semantic × recency × importance × pin (page/project boosts inert until S6/S4
pass their signals). The floor and type exclusions stay load-bearing and
untouched; ranking only reorders. The Q7 private-scope guard rides the same
chokepoint, and the retrieval cache is keyed per viewer.

Pure + one mocked retrieve_memories integration test.
"""
import os
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from modules.memory.injection_filter import filter_injectable_memories  # noqa: E402
from modules.memory.recall_ranking import (  # noqa: E402
    composite_score,
    rank_memories,
)

NOW = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _mem(text, score=None, days_ago=0.0, importance=0.5, pinned=False, **meta):
    metadata = {"importance": importance, **meta}
    if pinned:
        metadata["pinned"] = True
    return {"memory": text, "score": score, "created_at": _iso(days_ago), "metadata": metadata}


def test_recall_ranking_composite_deterministic_order():
    fixture = [
        _mem("old strong match", score=0.9, days_ago=90, importance=0.5),
        _mem("fresh good match", score=0.8, days_ago=1, importance=0.5),
        _mem("fresh weak match", score=0.45, days_ago=1, importance=0.5),
        _mem("pinned modest match", score=0.5, days_ago=10, importance=0.5, pinned=True),
    ]
    ranked = rank_memories(fixture, now=NOW)
    texts = [m["memory"] for m in ranked]
    # pinned: 0.5 × ~0.79 × 1.0 × 2 ≈ 0.79 — beats everything;
    # fresh 0.8 ≈ 0.78 beats 90-day-old 0.9 (× 0.125 decay) and weak 0.45.
    assert texts == [
        "pinned modest match",
        "fresh good match",
        "fresh weak match",
        "old strong match",
    ]


def test_importance_breaks_ties():
    a = _mem("load-bearing", score=0.6, days_ago=2, importance=0.9)
    b = _mem("minor", score=0.6, days_ago=2, importance=0.2)
    assert [m["memory"] for m in rank_memories([b, a], now=NOW)] == ["load-bearing", "minor"]


def test_unscored_rows_stay_neutral():
    strong = _mem("scored strong", score=0.9, days_ago=0)
    unscored = _mem("unscored legacy", score=None, days_ago=0)
    ranked = rank_memories([unscored, strong], now=NOW)
    assert ranked[0]["memory"] == "scored strong"
    # But unscored is not zeroed — it still outranks a sub-floor-ish match.
    weak = _mem("weak", score=0.35, days_ago=0)
    assert rank_memories([weak, unscored], now=NOW)[0]["memory"] == "unscored legacy"


def test_missing_created_at_not_punished():
    legacy = {"memory": "no timestamp", "score": 0.7, "metadata": {}}
    fresh = _mem("timestamped", score=0.7, days_ago=0)
    scores = {
        m["memory"]: composite_score(m, now=NOW)
        for m in (legacy, fresh)
    }
    assert scores["no timestamp"] == pytest.approx(scores["timestamped"], rel=1e-6)


def test_page_and_project_boosts_inert_until_signals_arrive():
    mem = _mem("page-stamped", score=0.6, days_ago=0, page="/activity", project_id="p1")
    base = composite_score(mem, now=NOW)
    boosted_page = composite_score(mem, now=NOW, query_page="/activity")
    boosted_proj = composite_score(mem, now=NOW, query_project="p1")
    other_page = composite_score(mem, now=NOW, query_page="/settings")
    assert boosted_page > base and boosted_proj > base
    assert other_page == pytest.approx(base)


def test_rank_is_stable_and_does_not_mutate_input():
    a = _mem("first", score=0.6, days_ago=1)
    b = _mem("second", score=0.6, days_ago=1)
    original = [a, b]
    ranked = rank_memories(original, now=NOW)
    assert [m["memory"] for m in ranked] == ["first", "second"]  # stable
    assert original == [a, b] and ranked is not original         # new list


def test_floor_and_exclusions_still_apply_before_ranking():
    """Composition: the guard filters, ranking only reorders what survives."""
    mems = [
        _mem("keeper", score=0.8, days_ago=0),
        _mem("sub-floor", score=0.1, days_ago=0),
        {"memory": "noise", "score": 0.9, "metadata": {"type": "heartbeat_log"}},
        _mem("private theirs", score=0.9, days_ago=0, scope="private", owner="user:9"),
        _mem("private mine", score=0.7, days_ago=0, scope="private", owner="user:7"),
    ]
    guarded = filter_injectable_memories(mems, floor=0.3, viewer_subject_id="user:7")
    assert {m["memory"] for m in guarded} == {"keeper", "private mine"}
    ranked = rank_memories(guarded, now=NOW)
    assert {m["memory"] for m in ranked} == {"keeper", "private mine"}


# ---------------------------------------------------------------------------
# retrieve_memories integration — viewer guard + per-viewer cache
# ---------------------------------------------------------------------------

class _FakeRecallService:
    def __init__(self, rows):
        self.rows = rows
        self.searches = 0

    async def search_long_term(self, workspace_id, query, agent_id=None, limit=8):
        self.searches += 1
        return [dict(r) for r in self.rows] if agent_id is None else []


@pytest.mark.asyncio
async def test_retrieve_memories_applies_scope_and_pin_ranking():
    from consumers.chatbot.smart_memory import SmartMemoryManager

    rows = [
        _mem("workspace fact", score=0.8, days_ago=1),
        _mem("pinned fact", score=0.6, days_ago=1, pinned=True),
        _mem("their private pref", score=0.9, days_ago=0, scope="private", owner="user:9"),
    ]
    mgr = SmartMemoryManager()
    mgr._unified_service = _FakeRecallService(rows)

    result = await mgr.retrieve_memories(
        "ws-1", None, "what do you know", viewer_subject_id="user:7",
    )
    texts = [m["memory"] for m in result.memories]
    assert "their private pref" not in texts
    assert texts[0] == "pinned fact"                 # pin boost outranks 0.8

    # The other user gets their private row — and does NOT hit the first
    # viewer's cache entry (per-viewer cache key).
    result9 = await mgr.retrieve_memories(
        "ws-1", None, "what do you know", viewer_subject_id="user:9",
    )
    assert "their private pref" in [m["memory"] for m in result9.memories]
