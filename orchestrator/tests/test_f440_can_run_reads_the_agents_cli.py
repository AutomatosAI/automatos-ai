"""PRD-256 N17-8 (F440): whether a session agent can run is read from the CLI it runs on.

Night 16: WRITER (#58, runtime cli, provider claude, model sonnet) was said to be unable to run
minutes after finishing two cards: "Nothing is serving Anthropic", "runs on gpt-4o through
OpenRouter… no host is serving OpenRouter". The can-run check read the listing's ``provider``,
which named the model route (``model_config``) first; a session agent never uses it, the CLI
host runs ``configuration.provider`` and ``.model``. These run Auto's agent tools against
Postgres (rolled back), with a CLI host online that serves claude, codex and copilot.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

SESSION = {"runtime": "cli", "provider": "claude", "model": "sonnet"}
CODEX = {"runtime": "cli", "provider": "codex"}
ANTHROPIC_ROUTE = {"provider": "anthropic", "model_id": "claude-sonnet-4-20250514"}
WRITERS_ROUTE = {"provider": "openrouter", "model_id": "openai/gpt-4o"}   # #58's dormant route
SERVED = ("claude", "codex", "copilot")
TEAM_SIZE = 65
_CREDIT_FAILED_SQL = text(
    "INSERT INTO board_tasks (workspace_id, title, status, error_message, completed_at) "
    "VALUES (CAST(:w AS uuid), 'Break-even on the gift box', 'failed', :e, :at)")


@pytest.fixture
def team(db_session, seed_workspace, monkeypatch):
    """A workspace whose CLI host is online and serves ``SERVED``; ``serve`` changes what it serves."""
    from core.models import Agent
    from services import cli_host_service, cli_ticket_lane

    ws = UUID(seed_workspace())

    def serve(*clis):
        monkeypatch.setattr(cli_host_service, "serving_providers", lambda db, workspace_id: list(clis))

    def agent(name, configuration, route, description="Writes the newsletter."):
        made = Agent(name=name, agent_type="custom", description=description, status="active",
                     configuration=dict(configuration), model_config=dict(route), workspace_id=ws, created_by="platform",
                     owner_type="workspace", owner_id=str(ws))
        db_session.add(made)
        db_session.flush()
        return made.id

    def credit_out():
        from core.llm.credit import OUT_OF_CREDIT_TEXT

        db_session.execute(_CREDIT_FAILED_SQL, {"w": str(ws), "e": OUT_OF_CREDIT_TEXT,
                                                "at": datetime.now(timezone.utc)})

    monkeypatch.setattr(cli_ticket_lane, "host_online", lambda db, workspace_id: True)
    serve(*SERVED)
    return NS(db=db_session, ws=ws, agent=agent, serve=serve, credit_out=credit_out)


def _list(team):
    from modules.tools.discovery.handlers_agents import list_agents

    return asyncio.run(list_agents(team.db, team.ws, {}))


def _listed(team):
    return {a["name"]: a for a in _list(team)["agents"]}


def _got(team, agent_id):
    from modules.tools.discovery.handlers_agents import get_agent

    return asyncio.run(get_agent(team.db, team.ws, {"agent_id": agent_id}))["agent"]


# ── a session agent's dormant model route never stops it ────────────────────

def test_an_active_session_agent_on_an_anthropic_route_can_run(team):
    writer = team.agent("WRITER", SESSION, ANTHROPIC_ROUTE)

    for entry in (_listed(team)["WRITER"], _got(team, writer)):
        assert entry["can_run"] is True and "why" not in entry


def test_writers_openrouter_route_does_not_stop_it(team):
    """#58's shape on night 16: a Claude Code agent with gpt-4o through OpenRouter left in model_config."""
    writer = team.agent("WRITER", SESSION, WRITERS_ROUTE)

    for entry in (_listed(team)["WRITER"], _got(team, writer)):
        assert (entry["can_run"], entry["provider"]) == (True, "claude") and "why" not in entry


def test_an_agent_whose_cli_no_host_serves_still_says_so(team):
    team.serve("claude")
    coder = team.agent("CODER", CODEX, ANTHROPIC_ROUTE)

    for entry in (_listed(team)["CODER"], _got(team, coder)):
        assert entry["can_run"] is False and "runs codex" in entry["why"]
        assert "anthropic" not in entry["why"].lower()


def test_auto_and_the_agents_page_agree(team):
    from core.models import Agent
    from services.agent_availability import CREDIT_OUT, why_unavailable

    team.serve("claude", "copilot")
    writer = team.agent("WRITER", SESSION, WRITERS_ROUTE)
    coder = team.agent("CODER", CODEX, ANTHROPIC_ROUTE)
    analyst = team.agent("ANALYST", {}, WRITERS_ROUTE)
    team.credit_out()

    told = {a["id"]: a["why"] for a in _list(team)["agents"] if "why" in a}
    rows = team.db.query(Agent).filter(Agent.id.in_([writer, coder, analyst])).all()

    assert told == why_unavailable(team.db, team.ws, rows)
    assert set(told) == {coder, analyst} and told[analyst] == CREDIT_OUT


# ── the listing names a session agent's CLI ─────────────────────────────────

def test_a_session_agents_listing_names_its_cli(team):
    writer = team.agent("WRITER", SESSION, ANTHROPIC_ROUTE)
    analyst = team.agent("ANALYST", {}, WRITERS_ROUTE)
    listed = _listed(team)

    for entry in (listed["WRITER"], _got(team, writer)):
        assert (entry["provider"], entry["model_id"], entry["runtime"]) == ("claude", "sonnet", "cli")
        assert (entry["cli_provider"], entry["cli_model"]) == ("claude", "sonnet")
        assert "anthropic" not in json.dumps(entry, default=str).lower()
    for entry in (listed["ANALYST"], _got(team, analyst)):
        assert (entry["provider"], entry["model_id"], entry["runtime"]) == ("openrouter", "openai/gpt-4o", "api")
        assert "cli_provider" not in entry and "cli_model" not in entry


def test_a_shortened_roster_still_names_each_cli(team):
    for i in range(TEAM_SIZE):
        team.agent(f"WRITER {i:02d}", SESSION, ANTHROPIC_ROUTE if i % 2 else WRITERS_ROUTE, description="d" * 200)

    result = _list(team)
    shown = json.dumps(result, default=str).lower()

    assert "note" in result and len(result["agents"]) == TEAM_SIZE
    assert all(isinstance(a, dict) and a["provider"] == "claude" for a in result["agents"])
    assert "anthropic" not in shown and "openrouter" not in shown


def test_a_failed_configuration_read_gives_no_reason(team, monkeypatch):
    """With no CLI host online a session agent whose CLI was read would say NO_HOST; one whose
    configuration could not be read is given no reason rather than one from its model route."""
    from modules.tools.discovery import agent_made_by
    from services import cli_ticket_lane

    def unreadable(db, workspace_id, ids):
        raise RuntimeError("the configuration could not be read")

    writer = team.agent("WRITER", SESSION, ANTHROPIC_ROUTE)
    monkeypatch.setattr(cli_ticket_lane, "host_online", lambda db, workspace_id: False)
    monkeypatch.setattr(agent_made_by, "_made", unreadable)

    for entry in (_listed(team)["WRITER"], _got(team, writer)):
        assert entry["can_run"] is True and "why" not in entry and "cli_provider" not in entry


# ── echoing what the listing says into an update changes nothing ────────────

def test_echoing_the_listed_provider_into_update_is_a_no_op(team, monkeypatch):
    from config import config
    from core.models import Agent
    from modules.tools.discovery.handlers_agents import update_agent

    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True, raising=False)
    writer = team.agent("WRITER", SESSION, ANTHROPIC_ROUTE)
    listed = _got(team, writer)["provider"]

    echoed = asyncio.run(update_agent(team.db, team.ws, {"agent_id": writer, "provider": listed}))
    refused = asyncio.run(update_agent(team.db, team.ws, {"agent_id": writer, "provider": "anthropic"}))

    assert listed == "claude"
    assert echoed["success"] is False and echoed["error"].startswith("Nothing was changed")
    assert refused["success"] is False and "configuration.provider must be one of" in refused["error"]
    team.db.expire_all()
    row = team.db.get(Agent, writer)
    assert (row.configuration, row.model_config) == (SESSION, ANTHROPIC_ROUTE)
