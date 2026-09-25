"""F141 (refresh-3 retest) — a provider's refusal of a model is remembered, and
chat routing and the ASSIGN lane stop sending work to that agent.

Chat 9c44148f was routed to BEANCOUNTER (agent 302). OpenRouter answered its
model with HTTP 400 "anthropic/claude-sonnet-4-20250514 is not a valid model ID",
and nothing remembered it: every turn routed there would fail the same way. The
catalog could not warn either: the OpenRouter sync only ever upserted, so a model
OpenRouter stopped listing stayed active and PRD-239 S5's deprecation never fired.

- Only a definitive refusal counts: a 400/404 whose answer names the model and
  says the provider does not offer it. Never a feature, policy or access
  refusal, a bare 404, a rate limit, a 5xx or a timeout.
- The LLM manager records it, whatever lane the call came from. The route row
  goes deprecated for every workspace. The agent carries model_config.unavailable
  when the refused id is its own model (BEANCOUNTER's id has no route row).
- Chat routing hands such a turn to Auto, naming the agent and model; the ASSIGN
  lane refuses the task with the same words. A checked model write drops it.
- The sync marks a model OpenRouter no longer lists inactive, unless the fetch
  was partial; a listed model is active again, and the projection follows.
"""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

from core.llm.clients.openai_compatible_client import ProviderModelUnavailableError
from core.models import Agent

DEAD = "anthropic/claude-sonnet-4-20250514"
LIVE = "anthropic/claude-sonnet-4"
BEANCOUNTERS_400 = f"{DEAD} is not a valid model ID"
REFUSED_TASK = (f"BEANCOUNTER's model {DEAD} is not available from its provider, so it cannot take this "
                "task. Pick another model in its Model tab, or give the task to another agent.")


class _ProviderError(Exception):
    """What the OpenAI SDK raises: the HTTP status and the provider's error body."""

    def __init__(self, status, message):
        super().__init__(f"Error code: {status} - {{'error': {{'message': {message!r}, 'code': {status}}}}}")
        self.status_code = status
        self.body = {"message": message, "code": status}


def _refused_for_good(exc, model):
    from core.llm.clients.openai_compatible_client import _refused_for_good as refused_for_good

    return refused_for_good(exc, model)


# ── what counts as a refusal for good ───────────────────────────────────────

@pytest.mark.parametrize("status, said", [
    (400, BEANCOUNTERS_400),                                   # chat 9c44148f
    (404, f"No endpoints found for {DEAD}."),                  # OpenRouter, a delisted model
    (404, f"Model not found: {DEAD}"),
])
def test_a_refusal_naming_the_model_as_not_offered_counts(status, said):
    assert _refused_for_good(_ProviderError(status, said), DEAD) is True


@pytest.mark.parametrize("status, said", [
    (404, "No endpoints found that support tool use. To learn more about provider routing, "
          "visit: https://openrouter.ai/docs/provider-routing"),
    (404, f"No endpoints found for {DEAD} that support tool use."),
    (404, "No endpoints found that can handle the requested parameters."),
    (404, f"No endpoints found for {DEAD} with context length 400000."),
    (404, "No endpoints found that support image input"),
    (404, "No endpoints found matching your data policy (Free model publication). "
          "Configure: https://openrouter.ai/settings/privacy"),
    (404, "No allowed providers are available for the selected model."),
    (404, "Not Found"),                                         # a wrong base URL
    (400, f"The model `{DEAD}` does not exist or you do not have access to it."),
    (429, f"Rate limit exceeded for {DEAD}"),
    (502, BEANCOUNTERS_400),                                    # never a 5xx
    (400, f"The model {DEAD} is not available right now"),     # the loose marker
])
def test_a_feature_policy_access_bare_or_transient_answer_never_counts(status, said):
    assert _refused_for_good(_ProviderError(status, said), DEAD) is False


def test_a_timeout_never_counts():
    assert _refused_for_good(TimeoutError(f"Request timed out: {DEAD}"), DEAD) is False


def _client(model=DEAD):
    from core.llm import providers as registry
    from core.llm.clients.openai_compatible_client import OpenAICompatibleProvider

    client = object.__new__(OpenAICompatibleProvider)
    client.spec, client.config = registry.get_spec("openrouter"), NS(model=model)
    return client


def test_beancounters_refusal_is_typed_with_its_route_and_marked_for_good():
    typed = _client()._classified(_ProviderError(400, BEANCOUNTERS_400))
    assert isinstance(typed, ProviderModelUnavailableError)
    assert (typed.provider, typed.model, typed.definitive, typed.said) == (
        "openrouter", DEAD, True, BEANCOUNTERS_400)
    assert str(typed) == (f"OpenRouter does not offer the model '{DEAD}' (any more). "
                          "Pick another model in the agent's Model tab.")


def test_openrouters_404_for_a_delisted_model_is_typed_too():
    """"No endpoints found for <model>." never says "model", so it was not typed."""
    typed = _client()._classified(_ProviderError(404, f"No endpoints found for {DEAD}."))
    assert isinstance(typed, ProviderModelUnavailableError) and typed.definitive is True


def test_a_loose_refusal_still_words_the_chat_but_is_never_remembered():
    typed = _client()._classified(_ProviderError(400, f"The model {DEAD} is not available right now"))
    assert isinstance(typed, ProviderModelUnavailableError) and typed.definitive is False


# ── the LLM call layer records it, for any lane ─────────────────────────────

def test_the_llm_manager_records_a_refusal_for_good_with_the_agent_that_asked(monkeypatch):
    from core.llm import model_refusals
    from core.llm.manager import LLMManager

    recorded = []
    monkeypatch.setattr(model_refusals, "record_model_refusal", lambda **kwargs: recorded.append(kwargs))

    def _refusing(definitive):
        class _Provider:
            async def generate_response(self, messages, tools=None):
                raise ProviderModelUnavailableError("OpenRouter does not offer the model", provider="openrouter",
                                                    model=DEAD, definitive=definitive, said=BEANCOUNTERS_400)
        return _Provider()

    manager = object.__new__(LLMManager)
    manager.config, manager._tracking_ctx = NS(model=DEAD), {"agent_id": 302}
    monkeypatch.setattr(manager, "_ensure_provider_initialized", lambda: None, raising=False)
    monkeypatch.setattr(manager, "_track_usage", lambda *args, **kwargs: None, raising=False)
    for definitive in (True, False):
        manager.provider = _refusing(definitive)
        with pytest.raises(ProviderModelUnavailableError):
            asyncio.run(LLMManager.generate_response(manager, [{"role": "user", "content": "How many subscribers?"}]))
    assert recorded == [{"agent_id": 302, "provider": "openrouter", "model": DEAD, "said": BEANCOUNTERS_400}]


# ── the record, on the real schema ──────────────────────────────────────────

class _Borrowed:
    """The writer opens its own session; here it borrows the test's, which rolls back."""

    def __init__(self, session):
        self._session = session

    def execute(self, *args, **kwargs):
        return self._session.execute(*args, **kwargs)

    def commit(self):
        self._session.flush()

    def rollback(self):
        pass

    def close(self):
        pass


@pytest.fixture
def db(db_session, monkeypatch):
    """The real session, with a route catalog this test controls."""
    from core.database import database

    db_session.execute(text("CREATE TEMP TABLE llm_models (LIKE public.llm_models INCLUDING DEFAULTS)"))
    monkeypatch.setattr(database, "SessionLocal", lambda: _Borrowed(db_session))
    return db_session


def _route(db, model_id, status="active", provider="openrouter"):
    db.execute(text("INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
                    "max_output_tokens, status) VALUES (:vendor, :provider, :m, :m, 200000, 8000, :status)"),
               {"vendor": model_id.split("/")[0], "provider": provider, "m": model_id, "status": status})


def _status(db, model_id, provider="openrouter"):
    return db.execute(text("SELECT status FROM llm_models WHERE serving_provider = :p AND model_id = :m"),
                      {"p": provider, "m": model_id}).scalar()


def _agent(db, ws, name, model_config, configuration=None):
    agent = Agent(name=name, agent_type="chatbot", description="", status="active",
                  configuration=configuration or {}, model_config=model_config, workspace_id=ws,
                  created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _model_config(db, agent_id):
    return db.execute(text("SELECT model_config FROM agents WHERE id = :id"), {"id": agent_id}).scalar()


def _stamped(model_id=DEAD):
    return {"model_id": model_id, "provider": "openrouter", "called": model_id, "said": BEANCOUNTERS_400,
            "at": "2026-09-25 09:03:27"}


def test_the_refusal_stamps_the_agent_whose_own_model_it_is(db, seed_workspace):
    from core.llm.model_refusals import record_model_refusal

    ws = UUID(seed_workspace())
    beancounter = _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "temperature": 0.7})
    record_model_refusal(agent_id=beancounter.id, provider="openrouter", model=DEAD, said=BEANCOUNTERS_400)
    config = _model_config(db, beancounter.id)
    stamp = config["unavailable"]
    assert {key: stamp[key] for key in ("model_id", "provider", "called", "said")} == {
        "model_id": DEAD, "provider": "openrouter", "called": DEAD, "said": BEANCOUNTERS_400}
    assert stamp["at"] and config["temperature"] == 0.7 and config["model_id"] == DEAD


def test_a_refused_route_goes_deprecated_for_everyone_and_only_that_route(db, seed_workspace):
    from core.llm.model_refusals import record_model_refusal

    _route(db, "moonshotai/kimi-k3")
    _route(db, "moonshotai/kimi-k3", provider="nvidia")
    record_model_refusal(agent_id=None, provider="openrouter", model="moonshotai/kimi-k3",
                         said="No endpoints found for moonshotai/kimi-k3.")
    assert _status(db, "moonshotai/kimi-k3") == "deprecated"
    assert _status(db, "moonshotai/kimi-k3", provider="nvidia") == "active"


def test_a_bare_id_the_factory_prefixed_is_still_the_agents_own(db, seed_workspace):
    from core.llm.model_refusals import record_model_refusal

    ws = UUID(seed_workspace())
    writer = _agent(db, ws, "WRITER", {"provider": "openrouter", "model_id": "gpt-4o"})
    record_model_refusal(agent_id=writer.id, provider="openrouter", model="openai/gpt-4o",
                         said="openai/gpt-4o is not a valid model ID")
    assert _model_config(db, writer.id)["unavailable"]["model_id"] == "gpt-4o"


def test_a_refusal_of_some_other_model_leaves_the_agent_alone(db, seed_workspace):
    """A trial pin, a fallback or a tier override called a model that is not the agent's."""
    from core.llm.model_refusals import record_model_refusal

    ws = UUID(seed_workspace())
    ops = _agent(db, ws, "OPS", {"provider": "openrouter", "model_id": LIVE})
    record_model_refusal(agent_id=ops.id, provider="openrouter", model="google/gemini-2.0-flash-001",
                         said="google/gemini-2.0-flash-001 is not a valid model ID")
    assert "unavailable" not in _model_config(db, ops.id)


# ── what routing and the ASSIGN lane read ───────────────────────────────────

def test_the_reader_names_the_agent_and_the_model(db):
    from core.llm.model_refusals import unavailable_reason

    beancounter = NS(name="BEANCOUNTER", configuration={},
                     model_config={"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    assert unavailable_reason(db, beancounter) == f"BEANCOUNTER's model {DEAD} is not available from its provider"


def test_a_new_model_voids_the_stamp(db):
    from core.llm.model_refusals import unavailable_reason

    _route(db, LIVE)
    repaired = NS(name="BEANCOUNTER", configuration={},
                  model_config={"provider": "openrouter", "model_id": LIVE, "unavailable": _stamped()})
    assert unavailable_reason(db, repaired) is None


def test_the_route_the_agent_names_being_deprecated_is_enough(db):
    from core.llm.model_refusals import unavailable_reason

    _route(db, "openai/gpt-4-32k", status="deprecated")
    agent = NS(name="COUNTINGHOUSE", configuration={}, model_config={"provider": "openrouter", "model_id": "openai/gpt-4-32k"})
    assert unavailable_reason(db, agent) == "COUNTINGHOUSE's model openai/gpt-4-32k is not available from its provider"


def test_a_legacy_vendor_provider_is_never_re_resolved(db):
    """The factory re-routes (anthropic, anthropic/claude-sonnet-4) through OpenRouter at
    call time; the reader never guesses that route. Its own stamp covers it after one refusal."""
    from core.llm.model_refusals import unavailable_reason

    _route(db, LIVE, status="deprecated")
    legacy = NS(name="WRITER", configuration={}, model_config={"provider": "anthropic", "model_id": LIVE})
    assert unavailable_reason(db, legacy) is None


def test_a_session_agent_or_one_on_the_default_is_never_judged(db):
    from core.llm.model_refusals import unavailable_reason

    session = NS(name="NEWSROOM", configuration={"runtime": "cli"},
                 model_config={"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    assert unavailable_reason(db, session) is None
    assert unavailable_reason(db, NS(name="SHOPKEEPER", configuration={}, model_config=None)) is None


def _envelope(source=None):
    from core.models.routing import ChannelSource, RequestEnvelope

    return RequestEnvelope(source=source or ChannelSource.CHATBOT, content="How many subscribers do I have?",
                           workspace_id=UUID("00000000-0000-0000-0000-0000000000c1"))


def _to(agent_id):
    from core.models.routing import RoutingDecision

    return RoutingDecision(route_type="agent", agent_id=agent_id, confidence=0.9,
                           reasoning="LLM classification (confidence=0.90)")


def test_a_chat_turn_routed_to_a_refused_agent_goes_to_auto(db, seed_workspace):
    from core.routing.engine import UniversalRouter

    ws = UUID(seed_workspace())
    beancounter = _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    out = UniversalRouter(db, cache=None)._chat_never_routes_to_a_refused_model(_envelope(), _to(beancounter.id))
    assert (out.route_type, out.agent_id) == ("orchestrate", None)
    assert out.reasoning == (f"BEANCOUNTER's model {DEAD} is not available from its provider, so Auto takes the "
                             "turn (was: LLM classification (confidence=0.90))")


def test_other_sources_and_answering_agents_are_left_alone(db, seed_workspace):
    from core.models.routing import ChannelSource
    from core.routing.engine import UniversalRouter

    ws = UUID(seed_workspace())
    _route(db, LIVE)
    beancounter = _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    shopkeeper = _agent(db, ws, "SHOPKEEPER", {"provider": "openrouter", "model_id": LIVE})
    router = UniversalRouter(db, cache=None)
    telegram = _to(beancounter.id)
    assert router._chat_never_routes_to_a_refused_model(_envelope(ChannelSource.TELEGRAM), telegram) is telegram
    answering = _to(shopkeeper.id)
    assert router._chat_never_routes_to_a_refused_model(_envelope(), answering) is answering


def test_every_routing_decision_leaves_through_both_guards(monkeypatch):
    from core.routing.engine import UniversalRouter

    router = UniversalRouter(NS(), cache=None)
    decision, seen = _to(302), []

    async def _tiers(envelope):
        return decision

    monkeypatch.setattr(router, "_route_through_tiers", _tiers)
    monkeypatch.setattr(router, "_chat_never_routes_to_a_session", lambda envelope, d: seen.append("F071") or d)
    monkeypatch.setattr(router, "_chat_never_routes_to_a_refused_model",
                        lambda envelope, d: seen.append("F141") or d)
    assert asyncio.run(router.route(_envelope())) is decision and seen == ["F071", "F141"]


def test_a_task_is_never_filed_for_an_agent_whose_model_was_refused(db, seed_workspace):
    from core.models.core import BoardTask
    from modules.tools.discovery.handlers_board_tasks import create_board_task

    ws = UUID(seed_workspace())
    _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    reply = asyncio.run(create_board_task(db, ws, {"title": "Chase the café invoices", "description": "Four overdue.",
                                                   "assigned_agent_name": "BEANCOUNTER"}))
    assert reply == {"success": False, "error": REFUSED_TASK}
    assert db.query(BoardTask).filter(BoardTask.workspace_id == ws).count() == 0


def test_nor_is_a_task_assigned_to_it(db, seed_workspace):
    from core.models.core import BoardTask
    from modules.tools.discovery.handlers_board_tasks import assign_board_task

    ws = UUID(seed_workspace())
    _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    task = BoardTask(workspace_id=ws, title="Chase the café invoices")
    db.add(task)
    db.flush()
    reply = asyncio.run(assign_board_task(db, ws, {"task_id": task.id, "agent_name": "BEANCOUNTER"}))
    assert reply == {"success": False, "error": REFUSED_TASK}
    assert task.assigned_agent_id is None


# ── a checked model write drops the stamp ───────────────────────────────────

def test_update_agent_to_a_catalog_model_drops_the_stamp(db, seed_workspace):
    from modules.tools.discovery.handlers_agents import update_agent

    ws = UUID(seed_workspace())
    _route(db, LIVE)
    beancounter = _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    reply = asyncio.run(update_agent(db, ws, {"agent_id": beancounter.id, "model_id": LIVE}))
    assert reply["success"] is True
    assert beancounter.model_config == {"provider": "openrouter", "model_id": LIVE}


def test_the_model_tabs_save_drops_it_too(db, seed_workspace):
    from api.agent_endpoints import update_agent_model_config

    ws = UUID(seed_workspace())
    _route(db, LIVE)
    beancounter = _agent(db, ws, "BEANCOUNTER", {"provider": "anthropic", "model_id": DEAD, "unavailable": _stamped()})
    echoed = {"provider": "openrouter", "model_id": LIVE, "temperature": 0.7, "unavailable": _stamped()}
    reply = asyncio.run(update_agent_model_config(beancounter.id, echoed, ctx=NS(workspace_id=ws), db=db))
    assert reply["model_config"] == {"provider": "openrouter", "model_id": LIVE, "temperature": 0.7}


# ── the sync: a model OpenRouter stops listing is retired, and can come back ─

@pytest.fixture
def catalog(db_session):
    """The real llm_models (the projection's ON CONFLICT names its constraint), and the
    cache and sync-job tables, which the test database lacks, made from their models."""
    from sqlalchemy.schema import CreateTable

    from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob

    for model in (OpenRouterModelCache, OpenRouterSyncJob):
        ddl = str(CreateTable(model.__table__).compile(dialect=db_session.bind.dialect))
        db_session.execute(text(ddl.replace("CREATE TABLE", "CREATE TEMP TABLE", 1)))
    return db_session


def _listing(count):
    return [{"id": f"f141/model-{i}", "name": f"F141 Model {i}", "context_length": 8000,
             "pricing": {"prompt": "0.000001", "completion": "0.000002"}} for i in range(count)]


def _sync(db, monkeypatch, models):
    from core.services.openrouter_sync_service import OpenRouterSyncService

    service = OpenRouterSyncService(db)
    monkeypatch.setattr(service, "_fetch_models", lambda: models)
    return service.run_full_sync()


def _cache_status(db, model_id):
    return db.execute(text("SELECT status FROM openrouter_models_cache WHERE model_id = :m"), {"m": model_id}).scalar()


def test_a_model_openrouter_stops_listing_is_retired_and_comes_back_when_listed(catalog, monkeypatch):
    from core.services.provider_catalog_sync import ProviderCatalogSync

    assert _sync(catalog, monkeypatch, _listing(10))["models_delisted"] == 0
    ProviderCatalogSync(catalog).project_openrouter_cache()
    assert _status(catalog, "f141/model-9") == "active"
    result = _sync(catalog, monkeypatch, _listing(9))                     # 9 of 10: not a partial answer
    assert (result["status"], result["models_delisted"], result["delisting_held"]) == ("completed", 1, False)
    assert _cache_status(catalog, "f141/model-9") == "inactive" and _cache_status(catalog, "f141/model-8") == "active"
    ProviderCatalogSync(catalog).project_openrouter_cache()
    assert _status(catalog, "f141/model-9") == "deprecated" and _status(catalog, "f141/model-8") == "active"

    _sync(catalog, monkeypatch, _listing(10))                              # listed again
    assert _cache_status(catalog, "f141/model-9") == "active"
    ProviderCatalogSync(catalog).project_openrouter_cache()
    assert _status(catalog, "f141/model-9") == "active"


def test_a_partial_answer_retires_nothing_and_says_what_it_would_have(catalog, monkeypatch, caplog):
    _sync(catalog, monkeypatch, _listing(10))
    with caplog.at_level(logging.WARNING, logger="core.services.openrouter_sync_service"):
        result = _sync(catalog, monkeypatch, _listing(5))
    assert (result["status"], result["models_delisted"], result["delisting_held"]) == ("completed", 0, True)
    assert all(_cache_status(catalog, f"f141/model-{i}") == "active" for i in range(10))
    assert any("a partial answer retires nothing" in r.getMessage()
               and "f141/model-5, f141/model-6, f141/model-7, f141/model-8, f141/model-9" in r.getMessage()
               for r in caplog.records)


def test_an_empty_fetch_retires_nothing(catalog, monkeypatch):
    _sync(catalog, monkeypatch, _listing(3))
    assert _sync(catalog, monkeypatch, [])["models_delisted"] == 0
    assert _cache_status(catalog, "f141/model-0") == "active"


def test_a_write_path_never_revives_a_delisted_model(catalog, monkeypatch):
    from api.llm_marketplace import _get_or_create_from_cache

    _sync(catalog, monkeypatch, _listing(10))
    _sync(catalog, monkeypatch, _listing(9))
    assert _get_or_create_from_cache(catalog, "f141/model-9", "openrouter") is None
    assert _get_or_create_from_cache(catalog, "f141/model-8", "openrouter") is not None
