"""PRD-236 W1 — the catalogue knows who serves what.

Pure tests (no Postgres, no network): fake sessions stand in for the ORM, the
NVIDIA list is stubbed at the httpx boundary. What is pinned:

- the migration chains onto the PRD-234 S1a head and the head guard follows;
- pricing tiers derive from the row price; a route key is "provider:model_id";
- `_find_route` honours the caller's route, then OpenRouter's row, then any;
- `_model_to_out` labels the route, marks free routes and missing keys;
- the NVIDIA sync borrows metadata from OpenRouter's row for the same vendor id
  (alias-aware), prices the route at zero, skips non-chat ids, records a job;
- `check_model_for_agent` judges the tagged route only;
- `UsageTracker` prices the route that served the call;
- `installed-ids` returns per-route keys; the manifest lists the new endpoints.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from core.llm import providers as reg


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


def _row(**kw):
    base = dict(
        id=1, provider="moonshotai", serving_provider="openrouter", model_id="moonshotai/kimi-k3",
        display_name="Kimi K3", model_family="moonshotai", description="d", context_window=1048576,
        max_output_tokens=32768, input_cost_per_1k_tokens=0.003, output_cost_per_1k_tokens=0.015,
        capabilities={}, recommended_for=[], supports_functions=True, supports_vision=False,
        supports_streaming=True, status="active", sourcing="aggregator", category="premium",
        tags=["function-calling"], is_featured=False, is_default=False, requires_plan=None,
        install_count=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class _Q:
    """A query whose filters are evaluated in Python over a row list.

    Only the predicate shapes used by the code under test are understood:
    ``Column == value`` comparisons on ``model_id`` / ``serving_provider`` /
    ``workspace_id`` / ``status``. Everything else is a no-op filter.
    """

    def __init__(self, rows):
        self._rows = list(rows)

    def _apply(self, clause):
        try:
            col = clause.left.name
            val = clause.right.value
        except Exception:
            return self._rows
        return [r for r in self._rows if getattr(r, col, None) == val]

    def filter(self, *clauses):
        rows = self._rows
        for c in clauses:
            self._rows = rows = _Q(rows)._apply(c)
        return self

    def join(self, *_a, **_k):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)

    def order_by(self, *_a):
        return self

    def distinct(self):
        return self


class _Session:
    def __init__(self, rows_by_model=None, cache_rows=None, jobs=None):
        self.rows = rows_by_model or []
        self.cache = cache_rows or []
        self.added = []
        self.executed = []
        self.commits = 0

    def query(self, model, *rest):
        name = getattr(model, "__name__", "") or getattr(getattr(model, "class_", None), "__name__", "")
        if "Cache" in name:
            return _Q(self.cache)
        if "Job" in name:
            return _Q([])
        return _Q(self.rows)

    def add(self, obj):
        self.added.append(obj)

    def execute(self, stmt, *a, **k):
        self.executed.append(stmt)

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def flush(self):
        pass

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# S1.1 — migration + head guard
# --------------------------------------------------------------------------- #


def test_migration_chains_onto_prd234_head_and_guard_follows():
    versions = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    src = (versions / "prd236_w1_serving_provider.py").read_text()
    assert 'revision = "prd236_w1_serving_provider"' in src
    assert 'down_revision = "prd234_s1a_cli_hosts_runtime_ref"' in src
    assert "uq_llm_models_provider_model" in src and "serving_provider" in src
    assert "new_column_name=\"sourcing\"" in src  # PRD-223 Q1 executed
    guard = (Path(__file__).resolve().parent / "test_prd209_alembic_single_head.py").read_text()
    assert 'EXPECTED_HEAD = "prd236_w1_serving_provider"' in guard


def test_orm_row_is_keyed_by_route():
    from core.models.core import LLMModel

    cols = {c.name for c in LLMModel.__table__.columns}
    assert {"serving_provider", "sourcing", "external_id"} <= cols
    assert "tier" not in cols
    uniques = [tuple(c.name for c in u.columns) for u in LLMModel.__table__.constraints if u.__class__.__name__ == "UniqueConstraint"]
    assert ("serving_provider", "model_id") in uniques
    assert not LLMModel.__table__.c.model_id.unique


# --------------------------------------------------------------------------- #
# S1.3 — marketplace helpers
# --------------------------------------------------------------------------- #


def test_price_tier_and_route_key():
    from api.llm_marketplace import price_tier_for, route_key

    assert price_tier_for(0) == "free"
    assert price_tier_for(0.0004) == "budget"
    assert price_tier_for(0.003) == "mid"
    assert price_tier_for(0.0031) == "premium"
    assert route_key("nvidia", "moonshotai/kimi-k3") == "nvidia:moonshotai/kimi-k3"


def test_find_route_prefers_the_callers_route_then_openrouter_then_any():
    from api.llm_marketplace import _find_route

    nv = _row(id=2, serving_provider="nvidia", input_cost_per_1k_tokens=0, output_cost_per_1k_tokens=0)
    orr = _row(id=1)
    db = _Session(rows_by_model=[nv, orr])
    assert _find_route(db, "moonshotai/kimi-k3", "nvidia") is nv
    assert _find_route(db, "moonshotai/kimi-k3", "NVIDIA") is nv  # alias-normalised
    assert _find_route(db, "moonshotai/kimi-k3") is orr           # OpenRouter first
    assert _find_route(_Session(rows_by_model=[nv]), "moonshotai/kimi-k3") is nv  # any
    assert _find_route(db, "moonshotai/kimi-k3", "deepseek") is None


def test_model_to_out_labels_the_route_free_and_missing_key():
    from api.llm_marketplace import _model_to_out

    nv = _row(id=2, serving_provider="nvidia", sourcing="hosted_open",
              input_cost_per_1k_tokens=0, output_cost_per_1k_tokens=0)
    out = _model_to_out(nv, installed_row_ids={2}, available_providers={"openrouter"})
    assert out.serving_provider == "nvidia" and out.serving_provider_label == "NVIDIA"
    assert out.route_label == "Kimi K3 · NVIDIA"
    assert out.vendor == "moonshotai" and out.provider == "moonshotai"
    assert out.is_free and out.price_tier == "free"
    assert out.is_installed is True
    assert out.key_available is False
    assert "trial" in (out.terms_note or "").lower()

    orr = _model_to_out(_row(), installed_row_ids=set(), available_providers={"openrouter"})
    assert orr.route_label == "Kimi K3 · OpenRouter" and not orr.is_free
    assert orr.price_tier == "mid" and orr.key_available is True and orr.is_installed is False


def test_installed_ids_returns_route_keys():
    import api.llm_marketplace as mp

    class _WSQ:
        def __init__(self, rows):
            self.rows = rows
        def filter(self, *a):
            return self
        def all(self):
            return self.rows

    class _DB:
        def query(self, *cols):
            # first call: WorkspaceModel.model_id ; second: (serving_provider, model_id)
            if len(cols) == 1:
                return _WSQ([(1,), (2,)])
            return _WSQ([("openrouter", "moonshotai/kimi-k3"), ("nvidia", "moonshotai/kimi-k3")])

    ctx = SimpleNamespace(workspace_id=uuid4())
    payload = asyncio.run(mp.get_installed_ids(ctx=ctx, db=_DB()))
    assert payload["model_ids"] == ["moonshotai/kimi-k3"]
    assert payload["routes"] == ["nvidia:moonshotai/kimi-k3", "openrouter:moonshotai/kimi-k3"]


# --------------------------------------------------------------------------- #
# S1.2 — NVIDIA catalogue sync
# --------------------------------------------------------------------------- #


def _cache_row(model_id="moonshotai/kimi-k3", **kw):
    base = dict(
        model_id=model_id, provider=model_id.split("/")[0], display_name="Kimi K3", description="Moonshot's K3",
        context_length=1048576, max_completion_tokens=32768, prompt_cost=0.000003, completion_cost=0.000015,
        supports_tools=True, supports_vision=False, supports_streaming=True, category="premium",
        tags=["function-calling"], status="active",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_nvidia_values_borrow_metadata_and_price_zero():
    from core.services.provider_catalog_sync import ProviderCatalogSync

    sync = ProviderCatalogSync(_Session(cache_rows=[_cache_row()]))
    values = sync._values_for_nvidia("moonshotai/kimi-k3", sync._borrow("moonshotai/kimi-k3"))
    assert values["display_name"] == "Kimi K3" and values["context_window"] == 1048576
    assert values["supports_functions"] is True
    assert values["input_cost_per_1k_tokens"] == 0.0 and values["output_cost_per_1k_tokens"] == 0.0
    assert values["sourcing"] == "hosted_open" and values["provider"] == "moonshotai"
    assert {"free", "nvidia", "function-calling"} <= set(values["tags"])
    assert values["external_id"] == "moonshotai/kimi-k3"


def test_nvidia_borrow_is_alias_aware_and_falls_back_to_defaults():
    from core.services.provider_catalog_sync import ProviderCatalogSync

    sync = ProviderCatalogSync(_Session(cache_rows=[_cache_row("deepseek/deepseek-v4-pro-0813", display_name="DeepSeek V4 Pro")]))
    borrowed = sync._borrow("deepseek-ai/deepseek-v4-pro-0813")  # NVIDIA vendor slug differs
    assert borrowed is not None and borrowed.display_name == "DeepSeek V4 Pro"

    values = sync._values_for_nvidia("nvidia/nemotron-3-super-120b-a12b", None)
    assert values["display_name"] == "Nemotron 3 Super 120b A12b"
    assert values["context_window"] == 0 and values["category"] == "free"
    assert values["input_cost_per_1k_tokens"] == 0.0 and values["sourcing"] == "hosted_open"


def test_non_chat_ids_are_skipped():
    from core.services.provider_catalog_sync import _NON_CHAT_ID

    for skipped in ("nvidia/nemotron-3-embed-1b", "nvidia/nemotron-4-340b-reward", "nvidia/nemotron-parse",
                    "nvidia/llama-3.1-nemotron-safety-guard-8b-v3", "google/deplot", "adept/fuyu-8b"):
        assert _NON_CHAT_ID.search(skipped), skipped
    for kept in ("moonshotai/kimi-k3", "deepseek-ai/deepseek-v4-pro-0813", "openai/gpt-oss-20b",
                 "nvidia/nemotron-3-ultra-550b-a55b"):
        assert not _NON_CHAT_ID.search(kept), kept


def test_fetch_nvidia_ids_uses_the_registry_base_url(monkeypatch):
    from core.services import provider_catalog_sync as pcs

    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"id": "moonshotai/kimi-k3"}, {"id": "nvidia/nemotron-3-embed-1b"}, {"id": "moonshotai/kimi-k3"}]}

    class _Client:
        def __init__(self, timeout=None):
            seen["timeout"] = timeout
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url, headers=None):
            seen["url"] = url
            seen["headers"] = headers
            return _Resp()

    monkeypatch.setattr(pcs.httpx, "Client", _Client)
    ids = pcs.ProviderCatalogSync._fetch_nvidia_ids(None)
    assert seen["url"] == "https://integrate.api.nvidia.com/v1/models" and seen["headers"] == {}
    assert ids == ["moonshotai/kimi-k3", "nvidia/nemotron-3-embed-1b"]  # de-duplicated, sorted
    pcs.ProviderCatalogSync._fetch_nvidia_ids("nvapi-x")
    assert seen["headers"] == {"Authorization": "Bearer nvapi-x"}


def test_sync_nvidia_upserts_routes_and_records_a_job(monkeypatch):
    from core.services import provider_catalog_sync as pcs

    db = _Session(cache_rows=[_cache_row()])
    sync = pcs.ProviderCatalogSync(db)
    monkeypatch.setattr(pcs.ProviderCatalogSync, "_fetch_nvidia_ids", staticmethod(
        lambda api_key=None: ["moonshotai/kimi-k3", "nvidia/nemotron-3-embed-1b", "openai/gpt-oss-20b"]))
    upserts = []
    monkeypatch.setattr(sync, "_upsert_route", lambda sp, mid, values: upserts.append((sp, mid, values)))

    result = sync.sync_nvidia()
    assert result["status"] == "completed"
    assert result["models_synced"] == 2 and result["skipped_non_chat"] == 1 and result["borrowed_metadata"] == 1
    assert [(sp, mid) for sp, mid, _ in upserts] == [("nvidia", "moonshotai/kimi-k3"), ("nvidia", "openai/gpt-oss-20b")]
    job = db.added[0]
    assert job.job_type == "nvidia_sync" and job.status == "completed" and job.models_synced == 2
    assert job.job_metadata["skipped_non_chat"] == 1


def test_sync_dispatch_refuses_unsyncable_providers():
    from core.services.provider_catalog_sync import ProviderCatalogSync, SYNCABLE_PROVIDERS

    assert SYNCABLE_PROVIDERS == ("openrouter", "nvidia")
    with pytest.raises(ValueError):
        ProviderCatalogSync(_Session()).sync("cohere")


def test_upsert_route_targets_the_route_constraint():
    from core.services.provider_catalog_sync import ProviderCatalogSync

    db = _Session()
    ProviderCatalogSync(db)._upsert_route("nvidia", "moonshotai/kimi-k3", {
        "provider": "moonshotai", "display_name": "Kimi K3", "context_window": 1, "max_output_tokens": 1,
        "input_cost_per_1k_tokens": 0.0, "output_cost_per_1k_tokens": 0.0, "status": "active",
        "sourcing": "hosted_open", "tags": [], "capabilities": {}, "recommended_for": [],
    })
    stmt = db.executed[0]
    compiled = str(stmt.compile(compile_kwargs={"literal_binds": False}))
    assert "ON CONFLICT ON CONSTRAINT uq_llm_models_provider_model DO UPDATE" in compiled
    assert "install_count" in compiled and "serving_provider" in compiled


# --------------------------------------------------------------------------- #
# S1.5/S2.1 — governance and pricing judge the tagged route
# --------------------------------------------------------------------------- #


def test_check_model_for_agent_judges_only_the_tagged_route(monkeypatch):
    from core.llm import model_policy

    quarantined_nv = SimpleNamespace(approval_status="quarantined", approved_roles=None,
                                     serving_provider="nvidia", model_id="moonshotai/kimi-k3", workspace_id="ws")
    approved_or = SimpleNamespace(approval_status="approved", approved_roles=None,
                                  serving_provider="openrouter", model_id="moonshotai/kimi-k3", workspace_id="ws")
    db = _Session(rows_by_model=[quarantined_nv, approved_or])
    monkeypatch.setattr(model_policy, "check_orchestrator_model", lambda m: (True, "ok"))

    allowed, reason = model_policy.check_model_for_agent(db, "ws", "moonshotai/kimi-k3", orchestrator_seat=False, provider="nvidia")
    assert not allowed and "quarantined" in reason
    allowed, _ = model_policy.check_model_for_agent(db, "ws", "moonshotai/kimi-k3", orchestrator_seat=False, provider="openrouter")
    assert allowed


def test_usage_tracker_prices_the_route_that_served_the_call(monkeypatch):
    from core.llm.usage_tracker import UsageTracker

    nv = _row(id=2, serving_provider="nvidia", input_cost_per_1k_tokens=0, output_cost_per_1k_tokens=0, sourcing="hosted_open")
    orr = _row(id=1)
    captured = []

    class _S(_Session):
        def add(self, obj):
            captured.append(obj)

    monkeypatch.setitem(sys.modules, "core.database.database",
                        types.SimpleNamespace(SessionLocal=lambda: _S(rows_by_model=[nv, orr])))
    UsageTracker.track(workspace_id=uuid4(), model_id="moonshotai/kimi-k3", provider="nvidia",
                       input_tokens=1000, output_tokens=1000)
    UsageTracker.track(workspace_id=uuid4(), model_id="moonshotai/kimi-k3", provider="openrouter",
                       input_tokens=1000, output_tokens=1000)
    assert captured[0].provider == "nvidia" and captured[0].total_cost == 0.0 and captured[0].tier == "hosted_open"
    assert captured[1].provider == "openrouter" and captured[1].total_cost == pytest.approx(0.018)


def test_route_manifest_lists_the_catalog_and_sync_endpoints():
    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    routes = manifest["routes"]
    for entry in (
        {"method": "GET", "path": "/api/marketplace/llm/catalog"},
        {"method": "GET", "path": "/api/marketplace/llm/sync/status"},
        {"method": "POST", "path": "/api/marketplace/llm/sync/{provider}"},
    ):
        assert entry in routes, entry
    assert manifest["route_count"] == len(routes)
