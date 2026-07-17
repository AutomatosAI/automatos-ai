"""PRD-221 S9 — Auto's Read digest generation (pure: LLM + cache mocked).

Locks the economics (one LLM call per state_hash, not per pageview), the
never-500 fallback, that the digest names the blocked item, and that the
cache key is workspace-isolated.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import services.digest_service as ds  # noqa: E402


class _FakeRedis:
    """Dict-backed get/setex — mirrors CacheService.redis."""
    def __init__(self):
        self.store = {}
    def get(self, key):
        return self.store.get(key)
    def setex(self, key, ttl, value):
        self.store[key] = value


class _LLM:
    def __init__(self, text="Your workspace looks healthy."):
        self.text = text
        self.calls = 0
    async def generate_response(self, messages):
        self.calls += 1
        return MagicMock(content=self.text)


def _mock_snapshot(monkeypatch, items, stats=None):
    import services.workspace_digest as wd
    svc = MagicMock()
    svc.get_stats.return_value = stats or {
        "working_now": 1, "completed_today": 0,
        "needs_attention": 0, "channels_live": 0, "period": "1d",
    }
    svc.get_feed.return_value = {"items": items, "total": len(items), "limit": 50, "offset": 0}
    monkeypatch.setattr(wd, "ActivityService", lambda db, ws: svc)


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_digest_cache_hit_skips_llm(monkeypatch):
    _mock_snapshot(monkeypatch, [{"id": "1", "type": "mission", "status": "running", "name": "X"}])
    redis = _FakeRedis()
    llm = _LLM()
    first = _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: llm))
    second = _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: llm))
    assert llm.calls == 1  # second call served from cache
    assert first["text"] == second["text"]
    assert first["generated_at"] == second["generated_at"]  # same cached object


def test_digest_regenerates_on_hash_change(monkeypatch):
    redis = _FakeRedis()
    llm = _LLM()
    _mock_snapshot(monkeypatch, [{"id": "1", "type": "mission", "status": "running", "name": "X"}])
    _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: llm))
    # state changes → new hash → new LLM call
    _mock_snapshot(monkeypatch, [{"id": "1", "type": "mission", "status": "failed", "name": "X"}])
    _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: llm))
    assert llm.calls == 2


def test_digest_llm_failure_falls_back_never_500(monkeypatch):
    _mock_snapshot(
        monkeypatch,
        [{"id": "1", "type": "mission", "status": "failed", "name": "Outreach",
          "error_message": "No email account connected"}],
        stats={"working_now": 0, "completed_today": 0, "needs_attention": 1,
               "channels_live": 0, "period": "1d"},
    )

    class _BoomLLM:
        async def generate_response(self, messages):
            raise RuntimeError("model down")

    redis = _FakeRedis()
    result = _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: _BoomLLM()))
    # 200-equivalent: a dict with fallback text that names the blocked item
    assert "Outreach" in result["text"]
    assert "No email account connected" in result["text"]
    assert result["needs_attention_count"] == 1
    # fallback is NOT cached, so it self-heals on the next call
    assert redis.store == {}


def test_digest_names_blocked_item_via_llm(monkeypatch):
    _mock_snapshot(
        monkeypatch,
        [{"id": "1", "type": "mission", "status": "failed", "name": "Outreach",
          "error_message": "No email account connected"}],
    )
    redis = _FakeRedis()
    llm = _LLM(text="One item needs attention: Outreach is blocked — no email account connected.")
    result = _run(ds.generate_digest(MagicMock(), "ws-a", redis_client=redis, llm_factory=lambda: llm))
    assert "Outreach" in result["text"]
    assert result["state_hash"]


def test_digest_cache_key_is_workspace_isolated():
    k1 = ds._cache_key("ws-a", "hash1")
    k2 = ds._cache_key("ws-b", "hash1")
    assert k1 != k2
    assert "ws-a" in k1 and "ws-b" in k2
