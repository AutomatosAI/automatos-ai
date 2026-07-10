"""PRD-159 S3 — recall relevance floor.

Low-relevance junk below the server-side similarity floor is never injected.
Scored-but-below-floor results are dropped; unscored results are kept (cannot
judge). Tested on the pure ``filter_by_relevance_floor`` helper (no network).
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

from modules.memory.durable_store import filter_by_relevance_floor  # noqa: E402


def _r(score):
    return {"memory": f"m{score}", "score": score}


def test_drops_below_floor():
    results = [_r(0.9), _r(0.4), _r(0.2), _r(0.05)]
    kept = filter_by_relevance_floor(results, 0.3)
    assert [r["score"] for r in kept] == [0.9, 0.4]


def test_keeps_unscored_results():
    results = [_r(0.9), {"memory": "no-score", "score": None}, _r(0.1)]
    kept = filter_by_relevance_floor(results, 0.3)
    # unscored kept (cannot judge), 0.1 dropped
    assert {"memory": "no-score", "score": None} in kept
    assert _r(0.1) not in kept


def test_floor_zero_disables_filter():
    results = [_r(0.9), _r(0.01)]
    assert filter_by_relevance_floor(results, 0.0) == results
    assert filter_by_relevance_floor(results, None) == results


def test_boundary_is_inclusive():
    results = [_r(0.3), _r(0.2999)]
    kept = filter_by_relevance_floor(results, 0.3)
    assert [r["score"] for r in kept] == [0.3]


def test_config_floor_default_is_03():
    from config import config
    assert abs(float(config.MEMORY_RELEVANCE_FLOOR) - 0.3) < 1e-9
