"""F141 (refresh-3 retest) — a chat write of an agent's model is checked the way
the Model tab checks it.

Chat 9c44148f was routed to BEANCOUNTER (agent 302). OpenRouter answered its
model, 'anthropic/claude-sonnet-4-20250514', with HTTP 400 "is not a valid model
ID": it never offered that id (its Claude Sonnet 4 is 'anthropic/claude-sonnet-4').
The id came through platform_update_agent, which stored any string and guessed
the provider by substring ('claude' → anthropic). PRD-223 W1 closed that hole in
create_agent only, and our own tool copy gave that id as its example. Now
update_agent refuses an id the catalog lacks, and a route its provider no
longer offers, before it changes anything; the policy gate runs; the ROUTE is
stored as the provider. create_agent treats a retired route like an unknown id.
Local census, 2026-09-25: 1 of 65 agents had a model no catalog route serves,
agent 302.
"""
from __future__ import annotations

import asyncio
import json
from uuid import UUID

import pytest
from sqlalchemy import text

from core.llm.defaults import DEFAULT_LLM_MODEL
from core.models import Agent
from modules.tools.discovery.handlers_agents import create_agent, update_agent

DEAD = "anthropic/claude-sonnet-4-20250514"
LIVE = "anthropic/claude-sonnet-4"
RETIRED = "openai/gpt-4-32k"
KEEPS = "Use a model that platform_list_workspace_models lists."


@pytest.fixture
def db(db_session):
    """The real session, with a catalog this test controls (rolled back with it).
    The test database has no openrouter_models_cache, so its model's DDL makes one."""
    from sqlalchemy.schema import CreateTable

    from core.models.openrouter_cache import OpenRouterModelCache

    db_session.execute(text("CREATE TEMP TABLE llm_models (LIKE public.llm_models INCLUDING DEFAULTS)"))
    cache = str(CreateTable(OpenRouterModelCache.__table__).compile(dialect=db_session.bind.dialect))
    db_session.execute(text(cache.replace("CREATE TABLE", "CREATE TEMP TABLE", 1)))
    return db_session


def _route(db, model_id, status="active"):
    db.execute(text("INSERT INTO llm_models (provider, serving_provider, model_id, display_name, context_window, "
                    "max_output_tokens, status) VALUES (:vendor, 'openrouter', :m, :m, 200000, 8000, :status)"),
               {"vendor": model_id.split("/")[0], "m": model_id, "status": status})


def _agent(db, ws, name="BEANCOUNTER"):
    agent = Agent(name=name, agent_type="chatbot", description="", status="active", configuration={},
                  model_config={"provider": "openrouter", "model_id": DEFAULT_LLM_MODEL, "temperature": 0.7},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _update(db, ws, **params):
    return asyncio.run(update_agent(db, ws, params))


# ── update_agent ────────────────────────────────────────────────────────────

def test_an_id_the_catalog_lacks_is_refused_and_nothing_changes(db, seed_workspace):
    ws = UUID(seed_workspace())
    _route(db, LIVE)
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, model_id=DEAD, description="Counts the beans.")
    assert reply == {"success": False,
                     "error": f"Model '{DEAD}' is not in the catalog, so 'BEANCOUNTER' keeps {DEFAULT_LLM_MODEL}. {KEEPS}"}
    assert agent.model_config["model_id"] == DEFAULT_LLM_MODEL and agent.description == ""


def test_a_catalog_model_is_stored_with_its_route_as_the_provider(db, seed_workspace):
    ws = UUID(seed_workspace())
    _route(db, LIVE)
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, model_id=LIVE)
    assert reply["success"] is True and reply["changes"] == [f"model -> '{LIVE}'"]
    assert (agent.model_config["provider"], agent.model_config["model_id"]) == ("openrouter", LIVE)


def test_a_route_its_provider_no_longer_offers_is_refused(db, seed_workspace):
    ws = UUID(seed_workspace())
    _route(db, RETIRED, status="deprecated")
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, model_id=RETIRED)
    assert reply == {"success": False, "error": (f"Model '{RETIRED}' is no longer offered by OpenRouter, "
                                                 f"so 'BEANCOUNTER' keeps {DEFAULT_LLM_MODEL}. {KEEPS}")}
    assert agent.model_config["model_id"] == DEFAULT_LLM_MODEL


def test_an_id_only_in_the_openrouter_cache_gets_its_route_as_on_the_model_tab(db, seed_workspace):
    ws = UUID(seed_workspace())
    db.execute(text("INSERT INTO openrouter_models_cache (model_id, display_name, provider, status) "
                    "VALUES ('anthropic/claude-sonnet-4.5', 'Anthropic: Claude Sonnet 4.5', 'anthropic', 'active')"))
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, model_id="anthropic/claude-sonnet-4.5")
    assert reply["success"] is True
    assert agent.model_config["provider"] == "openrouter"


def test_the_policy_gate_runs_for_an_update(db, seed_workspace, monkeypatch):
    import core.llm.model_policy as policy

    monkeypatch.setattr(policy, "check_model_for_agent",
                        lambda *args, **kwargs: (False, f"model '{LIVE}' is quarantined in this workspace"))
    ws = UUID(seed_workspace())
    _route(db, LIVE)
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, model_id=LIVE, new_name="COUNTINGHOUSE")
    assert reply == {"success": False, "error": f"Model rejected: model '{LIVE}' is quarantined in this workspace"}
    assert agent.name == "BEANCOUNTER"


def test_a_temperature_change_alone_never_asks_the_catalog(db, seed_workspace, monkeypatch):
    import api.llm_marketplace as marketplace

    monkeypatch.setattr(marketplace, "_get_or_create_from_cache",
                        lambda *args, **kwargs: pytest.fail("a temperature change consulted the catalog"))
    ws = UUID(seed_workspace())
    agent = _agent(db, ws)
    reply = _update(db, ws, agent_id=agent.id, temperature=0.2)
    assert reply["success"] is True
    assert agent.model_config == {"provider": "openrouter", "model_id": DEFAULT_LLM_MODEL, "temperature": 0.2}


# ── create_agent ────────────────────────────────────────────────────────────

def test_create_agent_puts_a_retired_route_on_the_default_and_says_so(db, seed_workspace):
    ws = UUID(seed_workspace())
    _route(db, RETIRED, status="deprecated")
    reply = asyncio.run(create_agent(db, ws, {"name": "Counting House", "model_id": RETIRED}))
    assert reply["success"] is True and reply["agent"]["model_id"] == DEFAULT_LLM_MODEL
    assert reply["model_note"] == (f"Model '{RETIRED}' is no longer offered by OpenRouter — the agent uses the "
                                   f"workspace default ({DEFAULT_LLM_MODEL}) instead.")


# ── the copy that taught the id ─────────────────────────────────────────────

@pytest.mark.parametrize("name", ["platform_create_agent", "platform_update_agent", "platform_install_model"])
def test_the_tool_copy_names_a_model_openrouter_offers(name):
    from modules.tools.discovery.action_registry import get_action_registry

    action = get_action_registry().get(name)
    copy = json.dumps([action.description, action.parameters, action.examples])
    assert "claude-sonnet-4-20250514" not in copy and "Defaults to 'gpt-4o'" not in copy
    assert DEFAULT_LLM_MODEL in copy
