"""PRD-261 — the theme services' in-memory caches of checked, content-addressed files.

``services/themes/bounded_cache.py`` is the one bounded cache they share (``looks.py``'s checked
look schemas by file sha256): at most ``size`` entries, the least recently used dropped first,
safe across FastAPI's threadpool workers.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.themes import looks as looks_service  # noqa: E402
from services.themes.bounded_cache import BoundedCache  # noqa: E402

THREADS = 8


# --- the bounded cache ---------------------------------------------------------------------------


def test_a_full_cache_drops_the_least_recently_used_entry():
    cache = BoundedCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # a is now the most recently used
    cache.put("c", 3)
    assert (cache.get("a"), cache.get("b"), cache.get("c"), len(cache)) == (1, None, 3, 2)
    cache.put("a", 4)  # a new value under a kept key replaces it and drops nothing
    assert (cache.get("a"), cache.get("c"), len(cache)) == (4, 3, 2)
    cache.clear()
    assert (cache.get("a"), len(cache)) == (None, 0)


def test_the_cache_holds_its_bound_under_concurrent_use():
    cache = BoundedCache(4)

    def work(worker: int) -> None:
        for step in range(200):
            key = (worker * step) % 16
            cache.put(key, key)
            found = cache.get(key)
            assert found is None or found == key

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(work, range(THREADS)))
    assert len(cache) <= 4


def test_the_look_schema_cache_is_the_shared_bounded_cache():
    assert isinstance(looks_service._schema_cache, BoundedCache)
