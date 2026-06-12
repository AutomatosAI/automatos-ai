import importlib.util
import pathlib
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


# Loaded by path (mirroring test_mem0_async_client.py) so monkeypatch targets the
# exact module objects the client uses.
mem0_client_module = _load_module(
    "memory_test_mem0_client",
    "modules/memory/integrations/mem0_client.py",
)
workspace_handlers_module = _load_module(
    "memory_test_workspace_handlers",
    "modules/tools/discovery/handlers_workspace.py",
)

Mem0Client = mem0_client_module.Mem0Client
get_memory_stats = workspace_handlers_module.get_memory_stats


class _FakeResponse:
    """Stand-in for httpx.Response — only the attributes the client reads."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_mem0_search_sends_search_query_and_prefers_score(monkeypatch):
    """Mem0Client.search (async httpx) GETs with a ``search_query`` param and
    re-ranks results by score so the strongest match leads regardless of recency.
    """
    captured = {}

    async def fake_request(self, method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
            payload={
                "items": [
                    {
                        "id": "recent-low",
                        "content": "recent but weak",
                        # Above the PRD-159 S3 relevance floor (0.3) so this test
                        # exercises score ORDERING, not floor filtering (which has
                        # its own suite: test_recall_relevance_floor).
                        "score": 0.45,
                        "created_at": "2026-03-10T00:00:00Z",
                    },
                    {
                        "id": "older-strong",
                        "content": "older but relevant",
                        "score": 0.9,
                        "created_at": "2026-03-09T00:00:00Z",
                    },
                ]
            }
        )

    monkeypatch.setattr(httpx.AsyncClient, "request", fake_request)

    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")
    results = await client.search(query="pricing", user_id="ws_123", limit=2)

    assert captured["method"] == "GET"
    assert captured["kwargs"]["params"]["user_id"] == "ws_123"
    assert captured["kwargs"]["params"]["search_query"] == "pricing"
    assert results[0]["id"] == "older-strong"
    assert results[1]["id"] == "recent-low"
    await client.aclose()


@pytest.mark.asyncio
async def test_platform_memory_stats_marks_partial_results(monkeypatch):
    """get_memory_stats scans at most 10 agents and flags partial results.

    The handler resolves the shared UnifiedMemoryService, fetches global + the
    first 10 agents' memories, and reports scanned_agents / total_agents so a
    workspace with >10 agents is marked ``partial``.
    """
    service = MagicMock()
    # is_mem0_configured is a property on the real service → plain bool here.
    service.is_mem0_configured = True

    async def fake_get_all_memories(workspace_id, agent_id=None, limit=200):
        if agent_id is None:
            return [{"memory": "global memory"}]
        return [{"memory": f"memory for agent {agent_id}"}]

    service.get_all_memories = AsyncMock(side_effect=fake_get_all_memories)

    # The handler imports get_unified_memory_service() *inside* the function from
    # the canonical module, so patch the seam there (not on the by-path-loaded
    # handler module). Otherwise the real singleton runs and the result depends
    # on whether MEM0_API_URL is configured — non-deterministic across envs.
    import modules.memory.unified_memory_service as ums
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)

    # 12 workspace agents → scan caps at 10, partial=True.
    db = MagicMock()
    query_result = MagicMock()
    query_result.filter.return_value.all.return_value = [
        (i, f"Agent {i}") for i in range(1, 13)
    ]
    db.query.return_value = query_result

    result = await get_memory_stats(db, "workspace", {})

    assert result["success"] is True
    assert result["partial"] is True
    assert result["scanned_agents"] == 10
    assert result["total_agents"] == 12
