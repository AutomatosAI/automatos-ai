"""#829 — the owner's own Anthropic key refreshes the direct Anthropic models.

``POST /api/marketplace/llm/sync/anthropic`` answered "'anthropic' has no
catalogue sync. Syncable: openrouter, nvidia", so a local install still offered
claude-sonnet-4-5-20250929 as its newest direct model.

Real Postgres (``llm_models`` with its route constraint; the cache and sync-job
tables, which the test database lacks, made from their models as F141 does).
Anthropic's Models API is answered by an ``httpx.MockTransport``; the key
resolver is stubbed. What is pinned:

- the sync reads every page (``after_id`` = the previous ``last_id``) with the
  workspace's key and the ``anthropic-version`` header;
- a listed id becomes an Anthropic route; the API's context window, output cap
  and vision flag refresh an existing row, whose price and description stay;
- a new row borrows price and description from the OpenRouter twin, and with no
  twin and no API numbers starts at 0 tokens and no price;
- an id Anthropic no longer lists goes ``deprecated``, but an undated alias of a
  listed id (``claude-sonnet-4-5`` of ``claude-sonnet-4-5-20250929``) stays
  active: Anthropic still answers it, and a deprecated route stops routing;
- a call on an unpriced route (NULL input and output price) is priced from the
  fallbacks, never booked at $0 from the row; an explicit 0 stays 0; the
  OpenRouter-cache fallback also tries a direct Anthropic id's OpenRouter twin;
- no key, a refused key or no answer is a clear message through the sync
  endpoint (502), never the key itself; the job is recorded failed (on a
  recording session: a failed sync rolls back, which would end ``db_session``).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from sqlalchemy import text

KEY = "sk-ant-829-owners-key"
EXISTING = "claude-sonnet-829-1-20250101"
NEW_WITH_TWIN = "claude-opus-829-2"
NEW_BARE = "claude-haiku-829-3"
GONE = "claude-instant-829-0"
ALIAS = "claude-sonnet-829-1"  # EXISTING without its date: Anthropic answers it, the API doesn't list it


@pytest.fixture
def catalog(db_session):
    """The real llm_models; the cache and sync-job tables as temp tables (F141's fixture)."""
    from sqlalchemy.schema import CreateTable

    from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob

    for model in (OpenRouterModelCache, OpenRouterSyncJob):
        ddl = str(CreateTable(model.__table__).compile(dialect=db_session.bind.dialect))
        db_session.execute(text(ddl.replace("CREATE TABLE", "CREATE TEMP TABLE", 1)))
    return db_session


@pytest.fixture
def owners_key(monkeypatch):
    """The key resolver answers with the workspace's own Anthropic key; calls are recorded."""
    from core.llm import key_resolver

    asked = []

    def _resolve(db, provider, workspace_id=None, agent_name=""):
        asked.append((provider, workspace_id))
        return key_resolver.ResolvedKey(api_key=KEY, source="byok", is_byok=True, provider=provider)

    monkeypatch.setattr(key_resolver, "resolve_provider_key", _resolve)
    return asked


def _anthropic_answers(monkeypatch, handler):
    """Every ``httpx.Client`` the sync opens talks to ``handler`` instead of the network."""
    from core.services import anthropic_catalog_sync as acs

    real_client = httpx.Client
    monkeypatch.setattr(acs.httpx, "Client",
                        lambda timeout=None: real_client(transport=httpx.MockTransport(handler), timeout=timeout))


def _model(model_id, name, max_input=None, max_out=None, vision=None):
    caps = None if vision is None else {"image_input": {"supported": vision}, "pdf_input": {"supported": True}}
    return {"type": "model", "id": model_id, "display_name": name, "created_at": "2026-07-24T00:00:00Z",
            "max_input_tokens": max_input, "max_tokens": max_out, "capabilities": caps}


PAGE_1 = {"data": [_model(EXISTING, "Claude Sonnet 829.1", 1_000_000, 128_000, True),
                   _model(NEW_WITH_TWIN, "Claude Opus 829.2", 1_000_000, 128_000, True)],
          "has_more": True, "first_id": EXISTING, "last_id": NEW_WITH_TWIN}
PAGE_2 = {"data": [_model(NEW_BARE, "Claude Haiku 829.3")],
          "has_more": False, "first_id": NEW_BARE, "last_id": NEW_BARE}


def _paged(requests):
    def handler(request):
        requests.append(request)
        after = request.url.params.get("after_id")
        return httpx.Response(200, json=PAGE_2 if after == NEW_WITH_TWIN else PAGE_1)
    return handler


def _seed(db):
    db.execute(text(
        "INSERT INTO llm_models (provider, serving_provider, model_id, display_name, description, context_window, "
        "max_output_tokens, input_cost_per_1k_tokens, output_cost_per_1k_tokens, supports_vision, status) "
        "VALUES ('anthropic', 'anthropic', :m, 'Old name', 'Seeded by hand', 200000, 8192, 0.003, 0.015, false, "
        "'active')"), {"m": EXISTING})
    db.execute(text(
        "INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
        "max_output_tokens, status) VALUES ('anthropic', 'anthropic', :m, :m, 100000, 4096, 'active')"), {"m": GONE})
    db.execute(text(
        "INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
        "max_output_tokens, status) VALUES ('anthropic', 'anthropic', :m, :m, 200000, 8192, 'active')"), {"m": ALIAS})
    from core.models.openrouter_cache import OpenRouterModelCache

    db.add(OpenRouterModelCache(
        model_id="anthropic/claude-opus-829.2", display_name="Anthropic: Claude Opus 829.2", provider="anthropic",
        description="OpenRouter's words for Opus 829.2", prompt_cost=0.000005, completion_cost=0.000025,
        context_length=200_000, max_completion_tokens=64_000, supports_tools=True, supports_vision=True,
        category="premium", tags=["function-calling"], status="active",
    ))
    db.flush()


def _row(db, model_id):
    return db.execute(text(
        "SELECT display_name, description, context_window, max_output_tokens, input_cost_per_1k_tokens, "
        "output_cost_per_1k_tokens, supports_vision, supports_functions, sourcing, status FROM llm_models "
        "WHERE serving_provider = 'anthropic' AND model_id = :m"), {"m": model_id}).mappings().first()


def _sync_endpoint(db, workspace_id):
    from api.llm_marketplace import sync_provider

    return asyncio.run(sync_provider("anthropic", ctx=NS(workspace_id=workspace_id), db=db))


def _last_job(db):
    return db.execute(text("SELECT status FROM openrouter_sync_jobs WHERE job_type = 'anthropic_sync' "
                           "ORDER BY id DESC LIMIT 1")).mappings().first()


# ── the sync ────────────────────────────────────────────────────────────────

def test_the_owners_key_reads_every_page_and_the_models_land_in_the_catalogue(catalog, owners_key, monkeypatch):
    _seed(catalog)
    requests = []
    _anthropic_answers(monkeypatch, _paged(requests))
    ws = uuid4()

    result = _sync_endpoint(catalog, ws)

    assert owners_key == [("anthropic", ws)]
    assert len(requests) == 2
    assert all(str(r.url).startswith("https://api.anthropic.com/v1/models") for r in requests)
    assert all(r.headers["x-api-key"] == KEY and r.headers["anthropic-version"] == "2023-06-01" for r in requests)
    assert "after_id" not in requests[0].url.params and requests[1].url.params["after_id"] == NEW_WITH_TWIN
    assert (result["status"], result["models_synced"], result["listed"]) == ("completed", 3, 3)

    existing = _row(catalog, EXISTING)              # refreshed from the API; price and words kept
    assert (existing["display_name"], existing["context_window"], existing["max_output_tokens"]) == (
        "Claude Sonnet 829.1", 1_000_000, 128_000)
    assert existing["supports_vision"] is True and existing["status"] == "active"
    assert (existing["input_cost_per_1k_tokens"], existing["output_cost_per_1k_tokens"]) == (0.003, 0.015)
    assert existing["description"] == "Seeded by hand"

    twin = _row(catalog, NEW_WITH_TWIN)             # new: the API's numbers, OpenRouter's price
    assert (twin["context_window"], twin["max_output_tokens"]) == (1_000_000, 128_000)
    assert twin["input_cost_per_1k_tokens"] == pytest.approx(0.005)
    assert twin["output_cost_per_1k_tokens"] == pytest.approx(0.025)
    assert twin["description"] == "OpenRouter's words for Opus 829.2"
    assert twin["supports_functions"] is True and twin["sourcing"] == "direct" and twin["status"] == "active"

    bare = _row(catalog, NEW_BARE)                  # new, nothing to borrow, nulls from the API
    assert (bare["display_name"], bare["context_window"], bare["max_output_tokens"]) == ("Claude Haiku 829.3", 0, 0)
    assert bare["input_cost_per_1k_tokens"] is None and bare["output_cost_per_1k_tokens"] is None

    assert _row(catalog, GONE)["status"] == "deprecated"
    assert _row(catalog, ALIAS)["status"] == "active"
    assert result["deprecated"] >= 1
    assert _last_job(catalog)["status"] == "completed"


def test_anthropic_is_syncable_and_shows_when_it_was_last_synced(catalog, owners_key, monkeypatch):
    from api.llm_marketplace import sync_status

    _anthropic_answers(monkeypatch, _paged([]))
    assert "anthropic" in asyncio.run(sync_status(db=catalog))["syncable"]
    assert asyncio.run(sync_status(db=catalog))["last_synced"]["anthropic"] is None
    _sync_endpoint(catalog, uuid4())
    assert asyncio.run(sync_status(db=catalog))["last_synced"]["anthropic"] is not None


# ── what the owner is told when it cannot sync ──────────────────────────────
# A failed sync rolls its session back; inside ``db_session`` that would end the
# test's own transaction, so these run on a session that records instead.

class _Recorded:
    """A session that records the job rows and refuses any catalogue query."""

    def __init__(self):
        self.added = []
        self.rollbacks = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def rollback(self):
        self.rollbacks += 1

    def query(self, *_a):
        raise AssertionError("a failed sync must not touch the catalogue")


def test_no_key_says_so_and_calls_nobody(monkeypatch):
    from core.llm import key_resolver

    monkeypatch.setattr(key_resolver, "resolve_provider_key", lambda *a, **k: None)
    called = []
    _anthropic_answers(monkeypatch, _paged(called))
    db = _Recorded()

    with pytest.raises(HTTPException) as refused:
        _sync_endpoint(db, uuid4())

    assert refused.value.status_code == 502
    assert "No Anthropic API key is available to this workspace" in refused.value.detail
    assert called == []
    job = db.added[-1]
    assert (job.job_type, job.status) == ("anthropic_sync", "failed")


@pytest.mark.parametrize("status, said", [(401, "Anthropic refused the API key (HTTP 401)"),
                                          (500, "Anthropic's Models API answered HTTP 500")])
def test_a_refused_key_or_a_broken_answer_is_worded_without_the_key(owners_key, monkeypatch, status, said):
    _anthropic_answers(monkeypatch, lambda request: httpx.Response(status, json={"type": "error"}))
    db = _Recorded()

    with pytest.raises(HTTPException) as failed:
        _sync_endpoint(db, uuid4())

    assert failed.value.status_code == 502
    assert said in failed.value.detail and KEY not in failed.value.detail
    assert db.added[-1].status == "failed" and KEY not in str(db.added[-1].error_details)


def test_no_answer_from_anthropic_is_worded(owners_key, monkeypatch):
    def unreachable(request):
        raise httpx.ConnectError("name resolution failed", request=request)

    _anthropic_answers(monkeypatch, unreachable)
    with pytest.raises(HTTPException) as failed:
        _sync_endpoint(_Recorded(), uuid4())
    assert "Could not reach Anthropic's Models API (ConnectError)" in failed.value.detail


# ── an unpriced route is never booked at $0 from its row ────────────────────

def _priced_route(db, provider, model_id, price_in, price_out):
    db.execute(text(
        "INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
        "max_output_tokens, input_cost_per_1k_tokens, output_cost_per_1k_tokens, status) "
        "VALUES ('anthropic', :p, :m, :m, 200000, 8192, :i, :o, 'active')"),
        {"p": provider, "m": model_id, "i": price_in, "o": price_out})


def test_an_unpriced_route_falls_through_to_the_estimate_and_an_explicit_zero_stays(catalog):
    from core.llm.usage_tracker import resolve_price

    unpriced = "claude-sonnet-4-829-unpriced"            # the static map knows "claude-sonnet-4"
    _priced_route(catalog, "anthropic", unpriced, None, None)
    _priced_route(catalog, "openrouter", unpriced, None, None)   # the any-model stage skips it too
    price = resolve_price(catalog, unpriced, "anthropic")
    assert price["source"] not in ("route", "model")
    assert price["input_per_1k"] > 0 and price["output_per_1k"] > 0

    free = "claude-sonnet-4-829-free"
    _priced_route(catalog, "anthropic", free, 0.0, 0.0)
    price = resolve_price(catalog, free, "anthropic")
    assert (price["source"], price["input_per_1k"], price["output_per_1k"]) == ("route", 0.0, 0.0)


def test_an_unpriced_anthropic_route_is_priced_from_its_openrouter_twin(catalog):
    from core.llm.usage_tracker import resolve_price
    from core.models.openrouter_cache import OpenRouterModelCache

    unpriced = "claude-opus-829-x"                  # no static-map key matches it
    _priced_route(catalog, "anthropic", unpriced, None, None)
    catalog.add(OpenRouterModelCache(
        model_id="anthropic/claude-opus-829-x", display_name="Anthropic: Claude Opus 829 X", provider="anthropic",
        prompt_cost=0.000004, completion_cost=0.00002, status="active",
    ))
    catalog.flush()

    price = resolve_price(catalog, unpriced, "anthropic")

    assert price["source"] == "catalogue"
    assert price["input_per_1k"] == pytest.approx(0.004) and price["output_per_1k"] == pytest.approx(0.02)


@pytest.mark.parametrize("model_id, twins", [
    ("claude-opus-5", ["anthropic/claude-opus-5"]),
    ("claude-sonnet-4-5-20250929", ["anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4-5-20250929"]),
    ("anthropic/claude-opus-5", []),                 # already an OpenRouter id
    ("gpt-4o", []),                                  # not an Anthropic id
])
def test_the_openrouter_ids_tried_for_a_direct_anthropic_id(model_id, twins):
    from core.llm.anthropic_ids import openrouter_twin_ids

    assert openrouter_twin_ids(model_id) == twins


# ── an undated alias is not a retired model ─────────────────────────────────

@pytest.mark.parametrize("model_id, listed, alias", [
    ("claude-sonnet-4-5", ["claude-sonnet-4-5-20250929"], True),
    ("claude-3-5-sonnet-latest", ["claude-3-5-sonnet-20241022"], True),
    ("claude-opus-4-1", ["claude-opus-4-1-20250805", "claude-opus-5"], True),
    ("claude-3-opus-20240229", ["claude-opus-4-1-20250805"], False),     # retired: no listed id is its alias
    ("claude-opus-4", ["claude-opus-4-1-20250805"], False),              # 4-1 is not 4 + a date
    ("claude-3-5-sonnet-latest", ["claude-3-5-haiku-20241022"], False),
    ("claude-sonnet-4-5", ["claude-sonnet-4-5-2025"], False),            # not 8 digits
])
def test_an_alias_of_a_listed_id(model_id, listed, alias):
    from core.services.anthropic_catalog_sync import is_alias_of_listed

    assert is_alias_of_listed(model_id, listed) is alias


# ── Anthropic's id → OpenRouter's id for the same model ─────────────────────

@pytest.mark.parametrize("anthropic_id, openrouter_id", [
    ("claude-sonnet-4-5-20250929", "anthropic/claude-sonnet-4.5"),
    ("claude-3-5-sonnet-20241022", "anthropic/claude-3.5-sonnet"),
    ("claude-3-haiku-20240307", "anthropic/claude-3-haiku"),
    ("claude-opus-4-8", "anthropic/claude-opus-4.8"),
    ("claude-opus-5", "anthropic/claude-opus-5"),
])
def test_the_openrouter_twin_id(anthropic_id, openrouter_id):
    from core.llm.anthropic_ids import openrouter_twin_id

    assert openrouter_twin_id(anthropic_id) == openrouter_id
