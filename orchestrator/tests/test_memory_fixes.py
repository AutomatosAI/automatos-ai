import importlib.util
import pathlib
import sys
import types
import uuid
from unittest.mock import MagicMock

import pytest


_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module(module_name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(module_name, _ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


mem0_client_module = _load_module(
    "memory_test_mem0_client",
    "modules/memory/integrations/mem0_client.py",
)
platform_executor_module = _load_module(
    "memory_test_platform_executor",
    "modules/tools/discovery/platform_executor.py",
)

Mem0Client = mem0_client_module.Mem0Client
PlatformActionExecutor = platform_executor_module.PlatformActionExecutor


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


def test_mem0_search_sends_search_query_and_prefers_score(monkeypatch):
    captured = {}

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return _FakeResponse(
            payload={
                "items": [
                    {
                        "id": "recent-low",
                        "content": "recent but weak",
                        "score": 0.2,
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

    monkeypatch.setattr(
        mem0_client_module.requests,
        "request",
        fake_request,
    )

    client = Mem0Client(api_url="http://mem0.test", api_key="test-key")
    results = client.search(query="pricing", user_id="ws_123", limit=2)

    assert captured["method"] == "GET"
    assert captured["kwargs"]["params"]["user_id"] == "ws_123"
    assert captured["kwargs"]["params"]["search_query"] == "pricing"
    assert results[0]["id"] == "older-strong"
    assert results[1]["id"] == "recent-low"
@pytest.mark.asyncio
async def test_platform_memory_stats_marks_partial_results(monkeypatch):
    fake_client = MagicMock()
    fake_client.api_url = "http://mem0.test"

    def fake_get_all(user_id, limit=200):
        if user_id == "ws_workspace":
            return [{"memory": "global memory"}]
        return [{"memory": f"memory for {user_id}"}]

    fake_client.get_all.side_effect = fake_get_all

    modules_pkg = types.ModuleType("modules")
    modules_pkg.__path__ = []
    memory_pkg = types.ModuleType("modules.memory")
    memory_pkg.__path__ = []
    integrations_pkg = types.ModuleType("modules.memory.integrations")
    integrations_pkg.__path__ = []
    mem0_stub = types.ModuleType("modules.memory.integrations.mem0_client")
    mem0_stub.Mem0Client = lambda: fake_client

    monkeypatch.setitem(sys.modules, "modules", modules_pkg)
    monkeypatch.setitem(sys.modules, "modules.memory", memory_pkg)
    monkeypatch.setitem(sys.modules, "modules.memory.integrations", integrations_pkg)
    monkeypatch.setitem(sys.modules, "modules.memory.integrations.mem0_client", mem0_stub)

    db = MagicMock()
    query_result = MagicMock()
    query_result.filter.return_value.all.return_value = [
        (i, f"Agent {i}") for i in range(1, 13)
    ]
    db.query.return_value = query_result

    executor = PlatformActionExecutor(db=db, workspace_id="workspace")
    result = await executor._get_memory_stats({})

    assert result["success"] is True
    assert result["partial"] is True
    assert result["scanned_agents"] == 10
    assert result["total_agents"] == 12
