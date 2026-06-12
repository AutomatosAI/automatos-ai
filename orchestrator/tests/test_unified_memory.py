"""
Integration tests for UnifiedMemoryService, ContextRouter, and MemoryNamespace.

Tests L1 session lifecycle, L2 CRUD + decay, L3 cache hit,
Context Router signal routing, and MemoryNamespace correctness.

All external dependencies (Redis, Mem0, Postgres) are mocked.

Uses importlib to load modules directly, avoiding the pgvector dependency
pulled in by modules/memory/__init__.py.
"""

import importlib.util
import inspect
import json
import math
import pathlib
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Direct module loading (bypasses modules.memory.__init__ pgvector import)
# ---------------------------------------------------------------------------

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pre-register a mock config module so `from config import config` works
# inside UnifiedMemoryService methods when loaded via importlib.
_mock_config_obj = MagicMock()
_mock_config_obj.MEMORY_SESSION_TTL_SECONDS = 86400
_mock_config_obj.MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS = 3600
_mock_config_obj.MEMORY_CACHE_TTL_SECONDS = 300
_mock_config_obj.MEMORY_DECAY_RATE = 0.004  # PRD-154 S3: week-scale (was 0.1/hr)
_mock_config_obj.MEMORY_DECAY_ARCHIVE_THRESHOLD = 0.3
_mock_config_obj.MEMORY_DECAY_BATCH_SIZE = 100
_mock_config_obj.MEMORY_PROMOTION_MIN_IMPORTANCE = 0.7
_mock_config_obj.MEMORY_PROMOTION_MIN_ACCESS_COUNT = 3
_mock_config_obj.MEMORY_PROMOTION_BATCH_SIZE = 50

_config_mod = types.ModuleType("config")
_config_mod.config = _mock_config_obj

# Install the mock ``config`` for the importlib loads below, then restore so the
# collection of sibling test modules never sees this fake (PRD-142 W2-S2b): a
# pathless ``config`` stub left in sys.modules shadows the real config package
# for every file collected afterwards. The loaded services import config lazily
# inside their methods, so the autouse fixture re-installs it for each test.
_config_snapshot = sys.modules.get("config")
sys.modules["config"] = _config_mod

_ums_mod = _load("unified_memory_service", "modules/memory/unified_memory_service.py")
_cr_mod = _load("context_router", "modules/memory/context_router.py")

if _config_snapshot is None:
    sys.modules.pop("config", None)
else:
    sys.modules["config"] = _config_snapshot

MemoryNamespace = _ums_mod.MemoryNamespace
SessionMemory = _ums_mod.SessionMemory
UnifiedMemoryService = _ums_mod.UnifiedMemoryService
ContextRouter = _cr_mod.ContextRouter
ContextSignals = _cr_mod.ContextSignals
ContextBundle = _cr_mod.ContextBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WS_ID = str(uuid.uuid4())
AGENT_ID = 42
CONV_ID = "conv-test-001"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton + install the mock ``config`` for the duration of the test."""
    _prev_config = sys.modules.get("config")
    sys.modules["config"] = _config_mod
    UnifiedMemoryService._instance = None
    try:
        yield
    finally:
        UnifiedMemoryService._instance = None
        if _prev_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = _prev_config


@pytest.fixture
def mock_mem0():
    # Mem0Client is async (PRD-141 US-003): add/search/get_all/delete are
    # coroutines, so they must be AsyncMock to be awaitable. api_url stays a
    # plain attribute (read synchronously by is_mem0_configured).
    m = MagicMock()
    m.api_url = "http://mem0.test"
    m.add = AsyncMock(return_value={"success": True, "id": "mem-001"})
    m.search = AsyncMock(
        return_value=[{"id": "m1", "memory": "user likes dark mode", "score": 0.9}]
    )
    m.get_all = AsyncMock(return_value=[{"id": "m1", "memory": "user likes dark mode"}])
    m.delete = AsyncMock(return_value=True)
    return m


@pytest.fixture
def mock_redis_conn():
    conn = MagicMock()
    conn.get.return_value = None
    conn.setex.return_value = True
    conn.delete.return_value = 1
    conn.scan.return_value = (0, [])
    conn.scan_iter.return_value = iter([])
    return conn


@pytest.fixture
def mock_redis_client(mock_redis_conn):
    client = MagicMock()
    client.get_redis.return_value = mock_redis_conn
    return client


@pytest.fixture
def service(mock_mem0, mock_redis_client):
    svc = UnifiedMemoryService.__new__(UnifiedMemoryService)
    svc._mem0 = mock_mem0
    svc._redis_client_getter = lambda: mock_redis_client
    UnifiedMemoryService._instance = svc
    return svc


# ===========================================================================
# 1. MemoryNamespace
# ===========================================================================


class TestMemoryNamespace:

    def test_workspace(self):
        ns = MemoryNamespace(workspace_id="ws-123")
        assert ns.workspace() == "mem:ws-123"

    def test_agent(self):
        assert MemoryNamespace("ws-1").agent(42) == "mem:ws-1:agent:42"

    def test_recipe(self):
        ns = MemoryNamespace("ws-1")
        assert ns.recipe(10) == "mem:ws-1:recipe:10"
        assert ns.recipe("tpl-5") == "mem:ws-1:recipe:tpl-5"

    def test_recipe_agent(self):
        assert MemoryNamespace("ws-1").recipe_agent(10, 42) == "mem:ws-1:recipe:10:agent:42"

    def test_workflow(self):
        assert MemoryNamespace("ws-1").workflow(99) == "mem:ws-1:workflow:99"

    def test_daily(self):
        assert MemoryNamespace("ws-1").daily() == "mem:ws-1:daily"

    def test_session(self):
        assert MemoryNamespace("ws-1").session("c1") == "mem:session:ws-1:c1"

    def test_cache_key(self):
        ns = MemoryNamespace("ws-1")
        assert ns.cache_key(42, "abc") == "mem:cache:ws-1:42:abc"
        assert ns.cache_key(None, "abc") == "mem:cache:ws-1:global:abc"

    def test_cache_pattern(self):
        assert MemoryNamespace("ws-1").cache_pattern() == "mem:cache:ws-1:*"

    def test_resolve_with_agent(self):
        assert MemoryNamespace("ws-1").resolve(42) == "mem:ws-1:agent:42"

    def test_resolve_without_agent(self):
        assert MemoryNamespace("ws-1").resolve() == "mem:ws-1"

    def test_frozen(self):
        ns = MemoryNamespace("ws-1")
        with pytest.raises(AttributeError):
            ns.workspace_id = "changed"


# ===========================================================================
# 2. SessionMemory serialisation
# ===========================================================================


class TestSessionMemory:

    def test_round_trip(self):
        original = SessionMemory(
            summary="pricing talk",
            exchange_count=3,
            last_updated="2026-03-12T10:00:00+00:00",
            ended=False,
        )
        restored = SessionMemory.from_json(original.to_json())
        assert restored.summary == original.summary
        assert restored.exchange_count == 3
        assert restored.ended is False

    def test_from_json_ignores_unknown_keys(self):
        # PRD-159 removed decisions/action_items; pre-existing Redis payloads
        # that still carry them must load cleanly (tolerant deserialisation).
        import json as _json
        raw = _json.dumps({
            "summary": "x", "exchange_count": 1, "ended": False,
            "decisions": ["old"], "action_items": ["legacy"],
        })
        s = SessionMemory.from_json(raw)
        assert s.summary == "x"
        assert s.exchange_count == 1
        assert not hasattr(s, "decisions")

    def test_defaults(self):
        s = SessionMemory()
        assert s.summary == ""
        assert s.exchange_count == 0
        assert s.ended is False


# ===========================================================================
# 3. L1 Session Lifecycle
# ===========================================================================


class TestL1Session:

    @pytest.mark.asyncio
    async def test_get_session_none_when_empty(self, service, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        assert await service.get_session(WS_ID, CONV_ID) is None

    @pytest.mark.asyncio
    async def test_get_session_returns_data(self, service, mock_redis_conn):
        session = SessionMemory(summary="test", exchange_count=2)
        mock_redis_conn.get.return_value = session.to_json()

        result = await service.get_session(WS_ID, CONV_ID)
        assert result is not None
        assert result.summary == "test"
        assert result.exchange_count == 2

    @pytest.mark.asyncio
    async def test_update_session_creates_new(self, service, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        await service.update_session(WS_ID, CONV_ID, "Hello", "Hi!")

        key, ttl, payload = mock_redis_conn.setex.call_args[0]
        assert key == f"mem:session:{WS_ID}:{CONV_ID}"
        assert ttl == 86400
        stored = SessionMemory.from_json(payload)
        assert stored.exchange_count == 1
        assert "Hello" in stored.summary

    @pytest.mark.asyncio
    async def test_update_session_appends(self, service, mock_redis_conn):
        existing = SessionMemory(summary="prior", exchange_count=1)
        mock_redis_conn.get.return_value = existing.to_json()
        await service.update_session(WS_ID, CONV_ID, "Q2", "A2")

        stored = SessionMemory.from_json(mock_redis_conn.setex.call_args[0][2])
        assert stored.exchange_count == 2
        assert "Q2" in stored.summary

    @pytest.mark.asyncio
    async def test_redis_failure_returns_none(self, service, mock_redis_conn):
        mock_redis_conn.get.side_effect = ConnectionError("down")
        assert await service.get_session(WS_ID, CONV_ID) is None

    @pytest.mark.asyncio
    async def test_redis_none_client_returns_none(self, service):
        service._redis_client_getter = lambda: None
        assert await service.get_session(WS_ID, CONV_ID) is None


# ===========================================================================
# 4. L3 Long-term Memory
# ===========================================================================


class TestL3:

    @pytest.mark.asyncio
    async def test_store_long_term(self, service, mock_mem0):
        result = await service.store_long_term(WS_ID, "dark mode", agent_id=AGENT_ID)
        assert result["success"] is True
        assert mock_mem0.add.call_args[1]["user_id"] == f"mem:{WS_ID}:agent:{AGENT_ID}"

    @pytest.mark.asyncio
    async def test_store_long_term_with_category(self, service, mock_mem0):
        await service.store_long_term(WS_ID, "coffee", category="pref", metadata={"src": "chat"})
        meta = mock_mem0.add.call_args[1]["metadata"]
        assert meta["category"] == "pref"
        assert meta["src"] == "chat"

    @pytest.mark.asyncio
    async def test_search_cache_miss(self, service, mock_mem0, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        results = await service.search_long_term(WS_ID, "dark mode")
        assert len(results) == 1
        mock_mem0.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_cache_hit(self, service, mock_mem0, mock_redis_conn):
        cached = [{"id": "c1", "memory": "cached", "score": 0.8}]
        mock_redis_conn.get.return_value = json.dumps(cached)

        results = await service.search_long_term(WS_ID, "dark mode")
        assert results[0]["id"] == "c1"
        mock_mem0.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_search_agent_namespace(self, service, mock_mem0, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        await service.search_long_term(WS_ID, "test", agent_id=AGENT_ID)
        assert mock_mem0.search.call_args[1]["user_id"] == f"mem:{WS_ID}:agent:{AGENT_ID}"

    @pytest.mark.asyncio
    async def test_search_workspace_namespace(self, service, mock_mem0, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        await service.search_long_term(WS_ID, "test")
        assert mock_mem0.search.call_args[1]["user_id"] == f"mem:{WS_ID}"

    @pytest.mark.asyncio
    async def test_get_all(self, service, mock_mem0):
        results = await service.get_all_memories(WS_ID)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete(self, service, mock_mem0):
        # PRD-154 S3: delete now routes through the bulk endpoint, so the service
        # resolves the namespace user_id and passes mem0 a memory_ids list.
        assert await service.delete_memory("mem-001", workspace_id=WS_ID) is True
        mock_mem0.delete.assert_called_once_with(
            memory_ids=["mem-001"], user_id=f"mem:{WS_ID}", workspace_id=WS_ID
        )

    @pytest.mark.asyncio
    async def test_store_long_term_infer_false(self, service, mock_mem0):
        # Curated facts must not be re-extracted by Mem0's personal-info prompt.
        await service.store_long_term(WS_ID, "distilled durable fact")
        assert mock_mem0.add.call_args[1].get("infer") is False

    @pytest.mark.asyncio
    async def test_store_long_term_messages_infer_false(self, service, mock_mem0):
        await service.store_long_term_messages(
            f"mem:{WS_ID}:recipe:1", [{"role": "user", "content": "x"}]
        )
        assert mock_mem0.add.call_args[1].get("infer") is False

    @pytest.mark.asyncio
    async def test_store_daily_log_infer_false(self, service, mock_mem0):
        await service.store_daily_log(WS_ID, "5 tasks done", metadata={"date": "2026-06-10", "type": "s"})
        assert mock_mem0.add.call_args[1].get("infer") is False

    @pytest.mark.asyncio
    async def test_store_two_tier_infer_false(self, service, mock_mem0):
        await service.store_two_tier(WS_ID, [{"role": "user", "content": "x"}], agent_id=1, tier="global")
        assert mock_mem0.add.call_args[1].get("infer") is False

    def test_all_curated_writers_grep_infer_false(self):
        # Grep-able guard (PRD-154 S3 AC): every curated L3 writer passes infer=False.
        for name in ("store_long_term", "store_long_term_messages", "store_daily_log", "store_two_tier"):
            src = inspect.getsource(getattr(UnifiedMemoryService, name))
            assert "infer=False" in src, f"{name} must pass infer=False to mem0.add"

    @pytest.mark.asyncio
    async def test_recall_touches_each_short_term_result(self, service):
        # touch_short_term had ZERO callers; recall must now bump access_count so
        # the L2->L3 promotion threshold becomes reachable.
        rows = [{"id": "r1", "content": "a"}, {"id": "r2", "content": "b"}]
        service._search_short_term_sync = lambda *a, **k: rows
        touched: list = []

        async def _fake_touch(memory_id):
            touched.append(memory_id)
            return True

        service.touch_short_term = _fake_touch
        out = await service.search_short_term(WS_ID, "anything")

        assert out == rows
        assert set(touched) == {"r1", "r2"}

    @pytest.mark.asyncio
    async def test_store_failure(self, service, mock_mem0):
        mock_mem0.add.side_effect = Exception("boom")
        result = await service.store_long_term(WS_ID, "x")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_search_failure(self, service, mock_mem0, mock_redis_conn):
        mock_redis_conn.get.return_value = None
        mock_mem0.search.side_effect = Exception("boom")
        assert await service.search_long_term(WS_ID, "x") == []

    @pytest.mark.asyncio
    async def test_is_mem0_configured(self, service, mock_mem0):
        assert service.is_mem0_configured is True
        mock_mem0.api_url = None
        assert service.is_mem0_configured is False


# ===========================================================================
# 5. Sleep-time consolidation (PRD-159 S4) — replaces the dead L1→L2 session
#    decision scan. Contradiction-based invalidation is the primary lifecycle:
#    near-duplicates merge into one canonical (rest deleted), contradictions
#    supersede by recency+confidence. The pure algorithm is covered in
#    test_memory_consolidation.py; here we test the service applies the plan.
# ===========================================================================


class TestSleepTimeConsolidation:

    @pytest.mark.asyncio
    async def test_merges_dups_via_delete(self, service):
        mems = [
            {"id": "1", "memory": "InBuildUK is a UK smoke ventilation contractor.",
             "created_at": "2026-06-10", "metadata": {"importance": 0.9}},
            {"id": "2", "memory": "InBuildUK is a UK smoke ventilation contractor.",
             "created_at": "2026-06-01", "metadata": {"importance": 0.4}},
            {"id": "3", "memory": "The user prefers British spelling.",
             "created_at": "2026-06-05", "metadata": {"importance": 0.7}},
        ]
        deleted = []

        async def _del(mid, ws, agent_id=None):
            deleted.append(mid)
            return True

        with patch.object(service, "get_all_memories", new_callable=AsyncMock) as mock_get, \
             patch.object(service, "delete_memory", side_effect=_del):
            mock_get.return_value = mems
            result = await service.run_sleep_time_consolidation(workspace_id=WS_ID)

        # 1 & 2 are duplicates → canonical "1" (higher importance) kept, "2" deleted.
        assert result["workspaces_processed"] == 1
        assert result["merged"] == 1
        assert deleted == ["2"]

    @pytest.mark.asyncio
    async def test_empty_workspace_is_noop(self, service):
        with patch.object(service, "get_all_memories", new_callable=AsyncMock) as mock_get, \
             patch.object(service, "delete_memory", new_callable=AsyncMock) as mock_del:
            mock_get.return_value = []
            result = await service.run_sleep_time_consolidation(workspace_id=WS_ID)

        assert result["merged"] == 0
        assert result["superseded"] == 0
        mock_del.assert_not_called()


# ===========================================================================
# 6. store_exchange — RETIRED in PRD-142 W3-S7 (G12 dual L2 write collapsed).
#    The canonical L2 write per chat turn is now ``store_transcript`` (PRD-131d
#    Phase 3), called from SmartMemoryManager.store_conversation. No
#    production caller of UnifiedMemoryService.store_exchange remains; the
#    method was deleted in the same commit as this test block.
# ===========================================================================


# ===========================================================================
# 7. Context Router Signal Detection
# ===========================================================================


class TestContextSignals:

    def _analyze(self, query):
        router = ContextRouter()
        return router.analyze_query(query)

    def test_temporal_last_week(self):
        s = self._analyze("What did we discuss last week?")
        assert s.is_temporal is True
        assert s.temporal_window is not None

    def test_temporal_yesterday(self):
        assert self._analyze("What happened yesterday?").is_temporal is True

    def test_temporal_n_days_ago(self):
        s = self._analyze("What was decided 3 days ago?")
        assert s.is_temporal is True
        assert s.temporal_window is not None

    def test_temporal_recently(self):
        assert self._analyze("Have we talked about this recently?").is_temporal is True

    def test_personal_my_name(self):
        assert self._analyze("What is my name?").is_personal_fact is True

    def test_personal_i_prefer(self):
        assert self._analyze("I prefer dark mode").is_personal_fact is True

    def test_personal_remember_when(self):
        assert self._analyze("Do you remember when we set up the API?").is_personal_fact is True

    def test_session_just_discussed(self):
        assert self._analyze("As we just discussed, let's finalize").is_session_continuation is True

    def test_session_earlier_in_conversation(self):
        assert self._analyze("Earlier in this conversation you mentioned X").is_session_continuation is True

    def test_knowledge_find_doc(self):
        assert self._analyze("Find the document about onboarding").is_knowledge_query is True

    def test_knowledge_policy(self):
        assert self._analyze("What's our policy on remote work?").is_knowledge_query is True

    def test_live_data_mrr(self):
        assert self._analyze("What's our current MRR?").is_live_data is True

    def test_live_data_how_many_users(self):
        assert self._analyze("How many users signed up last month?").is_live_data is True

    def test_live_data_stats(self):
        assert self._analyze("Show me the latest stats").is_live_data is True

    def test_no_signals(self):
        s = self._analyze("Help me write a JSON parser")
        assert not any([s.is_temporal, s.is_personal_fact, s.is_session_continuation,
                        s.is_knowledge_query, s.is_live_data])

    def test_empty_input(self):
        s = self._analyze("")
        assert s.is_temporal is False

    def test_multiple_signals(self):
        s = self._analyze("What was my preference last week?")
        assert s.is_temporal is True
        assert s.is_personal_fact is True

    def test_frozen(self):
        s = self._analyze("test")
        with pytest.raises(AttributeError):
            s.is_temporal = True


# ===========================================================================
# 8. Context Bundle
# ===========================================================================


class TestContextBundle:

    def test_frozen(self):
        b = ContextBundle()
        with pytest.raises(AttributeError):
            b.session_summary = "tampered"

    def test_defaults(self):
        b = ContextBundle()
        assert b.session_summary == ""
        assert b.long_term_memories == ()
        assert b.temporal_results == ()
        assert b.total_tokens_estimate == 0


# ===========================================================================
# 9. Ebbinghaus Decay Formula
# ===========================================================================


class TestDecayFormula:

    # Mirror config.MEMORY_DECAY_RATE — PRD-154 S3 retuned it to week-scale
    # (0.1/hr archived importance-0.8 in ~15h). Threshold unchanged at 0.3.
    RATE = 0.004
    THRESHOLD = 0.3

    def _retention(self, hours, importance=0.5, access_count=0, rate=None):
        rate = self.RATE if rate is None else rate
        return math.exp(-rate * hours) * (
            1.0 + 0.5 * importance + 0.1 * min(access_count, 10)
        )

    def test_fresh_high_retention(self):
        assert self._retention(1) > 0.9

    def test_importance_boost(self):
        # Even after a full week, higher importance retains more.
        week = 7 * 24
        assert self._retention(week, importance=0.9) > self._retention(week, importance=0.2)

    def test_access_boost(self):
        week = 7 * 24
        assert self._retention(week, access_count=5) > self._retention(week, access_count=0)

    def test_access_capped_at_10(self):
        assert self._retention(24, access_count=10) == self._retention(24, access_count=100)

    def test_high_importance_survives_one_week(self):
        # PRD-154 S3 AC: an importance-0.8 row stays above the archive threshold
        # after 7 days under the new rate — and the OLD 0.1/hr rate would not.
        assert self._retention(7 * 24, importance=0.8) >= self.THRESHOLD
        assert self._retention(7 * 24, importance=0.8, rate=0.1) < self.THRESHOLD

    def test_eventually_archives_on_a_week_scale(self):
        # Low-importance memory still decays — but over weeks, not hours.
        first_below = next(
            (d for d in range(1, 120) if self._retention(d * 24, importance=0.1) < self.THRESHOLD),
            None,
        )
        assert first_below is not None and first_below >= 7


# ===========================================================================
# 10. Daily Logs
# ===========================================================================


class TestDailyLogs:

    @pytest.mark.asyncio
    async def test_store_daily_log(self, service, mock_mem0):
        result = await service.store_daily_log(WS_ID, "5 tasks done", metadata={"date": "2026-03-12"})
        assert result["success"] is True
        assert mock_mem0.add.call_args[1]["user_id"] == f"mem:{WS_ID}:daily"

    @pytest.mark.asyncio
    async def test_get_all_daily_logs(self, service, mock_mem0):
        results = await service.get_all_daily_logs(WS_ID)
        assert len(results) == 1
        assert mock_mem0.get_all.call_args[1]["user_id"] == f"mem:{WS_ID}:daily"


# ===========================================================================
# 11. Singleton
# ===========================================================================


class TestSingleton:

    def test_same_instance(self, mock_mem0, mock_redis_client):
        # Bypass __init__ to avoid real imports
        svc = UnifiedMemoryService.__new__(UnifiedMemoryService)
        svc._mem0 = mock_mem0
        svc._redis_client_getter = lambda: mock_redis_client
        UnifiedMemoryService._instance = svc

        a = UnifiedMemoryService.get_instance()
        b = UnifiedMemoryService.get_instance()
        assert a is b
        assert a is svc

    def test_reset_clears(self, mock_mem0, mock_redis_client):
        svc1 = UnifiedMemoryService.__new__(UnifiedMemoryService)
        svc1._mem0 = mock_mem0
        svc1._redis_client_getter = lambda: mock_redis_client
        UnifiedMemoryService._instance = svc1

        a = UnifiedMemoryService.get_instance()
        UnifiedMemoryService.reset_instance()

        svc2 = UnifiedMemoryService.__new__(UnifiedMemoryService)
        svc2._mem0 = mock_mem0
        svc2._redis_client_getter = lambda: mock_redis_client
        UnifiedMemoryService._instance = svc2

        b = UnifiedMemoryService.get_instance()
        assert a is not b
