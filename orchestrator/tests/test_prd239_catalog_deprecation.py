"""PRD-239 S5 — a model the route no longer offers stops being offered: the
OpenRouter projection deprecates rows missing from the cache (as the NVIDIA
sync does), and the agent model-config endpoint refuses a deprecated route with
the reason. Pure units."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.openrouter_cache import OpenRouterModelCache  # noqa: E402
from core.services.provider_catalog_sync import ProviderCatalogSync  # noqa: E402

WS = uuid4()


class _CacheQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _UpdateQuery:
    def __init__(self, record):
        self._record = record

    def filter(self, *clauses):
        self._record["filtered"] = True
        return self

    def update(self, values, synchronize_session=None):
        self._record["values"] = values
        return 3


class _DB:
    def __init__(self, rows, record):
        self.rows = rows
        self.record = record
        self.commits = 0

    def query(self, model):
        return _CacheQuery(self.rows) if model is OpenRouterModelCache else _UpdateQuery(self.record)

    def commit(self):
        self.commits += 1


def _cached(model_id):
    return SimpleNamespace(
        model_id=model_id, provider=model_id.split("/")[0], display_name=model_id, description="",
        context_length=1000, max_completion_tokens=100, prompt_cost=0.0, completion_cost=0.0,
        supports_tools=True, supports_vision=False, supports_streaming=True, category="c", tags=[],
    )


def test_projection_deprecates_openrouter_routes_the_cache_no_longer_lists(monkeypatch):
    record = {}
    sync = ProviderCatalogSync(_DB([_cached("moonshotai/kimi-k3")], record))
    upserts = []
    monkeypatch.setattr(sync, "_upsert_route", lambda route, model_id, values: upserts.append((route, model_id)))
    out = sync.project_openrouter_cache()
    assert upserts == [("openrouter", "moonshotai/kimi-k3")]
    assert out == {"provider": "openrouter", "rows": 1, "deprecated": 3}
    assert record["values"] == {"status": "deprecated"} and record["filtered"] is True


def test_an_empty_cache_deprecates_nothing():
    record = {}
    sync = ProviderCatalogSync(_DB([], record))
    out = sync.project_openrouter_cache()
    assert out == {"provider": "openrouter", "rows": 0, "deprecated": 0} and "values" not in record


# ── the endpoint ─────────────────────────────────────────────────────────────

class _AgentQuery:
    def __init__(self, agent):
        self._agent = agent

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._agent


class _AgentDB:
    def __init__(self, agent):
        self.agent = agent
        self.rolled_back = 0

    def query(self, model):
        return _AgentQuery(self.agent)

    def commit(self):
        raise AssertionError("a refused model must not be written")

    def rollback(self):
        self.rolled_back += 1


def test_model_config_refuses_a_deprecated_route_with_the_reason(monkeypatch):
    from fastapi import HTTPException

    import api.agent_endpoints as ep
    import api.llm_marketplace as mk

    monkeypatch.setattr(
        mk, "_get_or_create_from_cache",
        lambda db, model_id, provider=None: SimpleNamespace(
            status="deprecated", serving_provider="openrouter", display_name="DeepSeek Coder", provider="deepseek",
        ),
    )
    db = _AgentDB(SimpleNamespace(id=57, name="Researcher", is_system_agent=False, model_config={}))
    ctx = SimpleNamespace(workspace_id=WS)
    with pytest.raises(HTTPException) as info:
        asyncio.run(ep.update_agent_model_config(57, {"model_id": "deepseek/deepseek-coder", "provider": "openrouter"}, ctx, db))
    assert info.value.status_code == 422
    assert "DeepSeek Coder is no longer offered by OpenRouter" in str(info.value.detail)
