"""PRD-256 FX-016 (H, F396): Auto can make a Claude session agent.

Night 12: asked for "a proper Claude session", Auto made api agents on the default model:
platform_create_agent wrote ``configuration={}`` and neither the create nor the update tool
had a runtime, so CLUB DESK and MARKET-MANAGER were switched by hand (the two api agents
Auto made graded 1-3). Both tools now take ``runtime`` ('api' | 'cli') and, for a cli agent,
``provider`` and ``model``, written into ``Agent.configuration`` under the keys the CLI host
reads. A create that names none takes DEFAULT_AGENT_RUNTIME; an update's card says
'runtime: api → cli' and the owner's click writes it; an unknown runtime is refused.

These run Auto's tools as the platform executor binds them (``PLATFORM_HANDLERS``).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from core.cli_runtime import CONFIG_MODEL_KEY, CONFIG_PROVIDER_KEY, CONFIG_RUNTIME_KEY, RUNTIME_CLI
from modules.tools.discovery import owner_only

ORCH = Path(__file__).resolve().parents[1]
CREATE, UPDATE, GET, LIST = ("platform_create_agent", "platform_update_agent", "platform_get_agent",
                             "platform_list_agents")
SESSION_ASK = {"name": "CLUB DESK", "description": "Runs the club desk.", "runtime": "cli", "model": "sonnet"}


@pytest.fixture
def session_mode(monkeypatch):
    """The local edition with session mode on (CLI_RUNTIME_ENABLED), the default runtime 'api'."""
    from config import config

    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", True, raising=False)
    monkeypatch.setattr(config, "DEFAULT_AGENT_RUNTIME", "api", raising=False)
    return config


@pytest.fixture
def ws(db_session, seed_workspace):
    return UUID(seed_workspace())


def _tool(db, ws, action, params):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    return asyncio.run(PLATFORM_HANDLERS[action](db, ws, params))


def _agent(db, ws, name, configuration=None):
    from core.models.core import Agent

    agent = Agent(name=name, agent_type="chatbot", description="", status="active",
                  configuration=configuration or {}, model_config={"model_id": "claude-sonnet-5-5"},
                  workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db.add(agent)
    db.flush()
    return agent


def _row(db, ws, name):
    from core.models.core import Agent

    return db.query(Agent).filter(Agent.workspace_id == ws, Agent.name == name).all()


def _owners_chat():
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}


# ── platform_create_agent ────────────────────────────────────────────────────────────

def test_create_with_runtime_cli_writes_the_configuration_the_host_reads(db_session, ws, session_mode):
    from core.cli_runtime import runtime_kind_of
    from services.cli_host_service import host_health

    reply = _tool(db_session, ws, CREATE, dict(SESSION_ASK))

    assert reply["success"] is True, reply
    (agent,) = _row(db_session, ws, "CLUB DESK")
    assert agent.configuration == {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: "claude",
                                   CONFIG_MODEL_KEY: "sonnet"}
    assert runtime_kind_of(agent.configuration) == RUNTIME_CLI
    assert host_health(db_session, ws)["cli_agents"] == 1                    # the host's own reader counts it
    assert reply["agent"]["runtime"] == RUNTIME_CLI and reply["agent"]["cli_model"] == "sonnet"
    assert "Claude Code session" in reply["message"]
    assert reply["model_note"] is None                                       # 'sonnet' never met the API catalog


def test_create_cli_names_its_cli_and_leaves_the_model_to_it(db_session, ws, session_mode):
    reply = _tool(db_session, ws, CREATE, {"name": "CODE DESK", "runtime": "CLI", "provider": "codex"})

    assert reply["success"] is True, reply
    (agent,) = _row(db_session, ws, "CODE DESK")
    assert agent.configuration == {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: "codex"}
    assert "the CLI's default model" in reply["message"]


def test_the_default_runtime_applies_when_the_call_names_none(db_session, ws, session_mode):
    made_api = _tool(db_session, ws, CREATE, {"name": "API DESK"})
    assert made_api["success"] is True and _row(db_session, ws, "API DESK")[0].configuration == {}

    session_mode.DEFAULT_AGENT_RUNTIME = "cli"
    made_cli = _tool(db_session, ws, CREATE, {"name": "SESSION DESK"})

    assert made_cli["success"] is True, made_cli
    assert _row(db_session, ws, "SESSION DESK")[0].configuration == {CONFIG_RUNTIME_KEY: RUNTIME_CLI,
                                                                    CONFIG_PROVIDER_KEY: "claude"}
    named_api = _tool(db_session, ws, CREATE, {"name": "STILL API", "runtime": "api"})
    assert named_api["success"] is True and _row(db_session, ws, "STILL API")[0].configuration == {}


def test_the_default_is_api_and_a_named_setting():
    from config import Config

    assert Config.DEFAULT_AGENT_RUNTIME in ("api", "cli")
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text())
    assert "DEFAULT_AGENT_RUNTIME" in surface["settings"]
    assert 'os.getenv("DEFAULT_AGENT_RUNTIME", "api")' in (ORCH / "config.py").read_text()


def test_an_unknown_runtime_is_refused_and_nothing_is_made(db_session, ws, session_mode):
    reply = _tool(db_session, ws, CREATE, {"name": "GPU DESK", "runtime": "gpu"})

    assert reply["success"] is False and "Unknown runtime 'gpu'" in reply["error"]
    assert "Nothing was created" in reply["error"] and _row(db_session, ws, "GPU DESK") == []

    session_mode.DEFAULT_AGENT_RUNTIME = "gpu"
    by_default = _tool(db_session, ws, CREATE, {"name": "GPU DESK"})
    assert by_default["success"] is False and "DEFAULT_AGENT_RUNTIME" in by_default["error"]


def test_a_cli_agent_is_refused_where_session_mode_is_off(db_session, ws, session_mode):
    """The hosted edition (no CLI_RUNTIME_ENABLED): the rule the agents API applies."""
    session_mode.CLI_RUNTIME_ENABLED = False

    reply = _tool(db_session, ws, CREATE, dict(SESSION_ASK))

    assert reply["success"] is False and "CLI_RUNTIME_ENABLED" in reply["error"]
    assert _row(db_session, ws, "CLUB DESK") == []


def test_a_model_the_cli_does_not_take_is_refused(db_session, ws, session_mode):
    reply = _tool(db_session, ws, CREATE, {**SESSION_ASK, "model": "gpt-4o"})

    assert reply["success"] is False and "'gpt-4o' is not a claude model" in reply["error"]
    assert _row(db_session, ws, "CLUB DESK") == []


# ── platform_update_agent: the card, then the click ──────────────────────────────────

class _Executor:
    """PlatformActionExecutor._run_cleared's shape, the gates cleared, the real handler run."""

    def __init__(self, db, ws):
        self.db, self.ws = db, ws

    @owner_only.asks_the_owner_first
    async def _run_cleared(self, action_name, params, caller_context, cleared, handler):
        return await handler(self.db, self.ws, params)


def _from_the_chat(db, ws, params):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    run = _Executor(db, ws)._run_cleared(UPDATE, dict(params), _owners_chat(), None, PLATFORM_HANDLERS[UPDATE])
    return asyncio.run(run)


def test_update_api_to_cli_raises_the_card_and_the_click_writes_it(db_session, ws, session_mode):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    desk = _agent(db_session, ws, "MARKET-MANAGER")
    params = {"agent_id": desk.id, "runtime": "cli", "model": "opus"}

    ask = _from_the_chat(db_session, ws, params)

    assert ask["requires_confirmation"] is True and ask["owner_only"] is True
    asked = ask["question_md"]
    assert "- runtime: api → cli" in asked
    assert "- session CLI: (empty) → claude" in asked and "- session model: (empty) → opus" in asked
    assert desk.configuration == {}                                           # nothing changed yet

    grant_grant(db_session.get(ApprovalGrant, ask["grant_id"]), granted_by="user:7")
    db_session.flush()
    done = _from_the_chat(db_session, ws, params)

    assert done["success"] is True and done["approved_via_grant_id"] == ask["grant_id"], done
    assert "runtime: api → cli" in done["changes"]
    assert desk.configuration == {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: "claude",
                                  CONFIG_MODEL_KEY: "opus"}


def test_an_update_the_click_could_not_do_is_refused_before_the_card(db_session, ws, session_mode):
    from core.models.approval_grants import ApprovalGrant

    desk = _agent(db_session, ws, "MARKET-MANAGER")
    unknown = _from_the_chat(db_session, ws, {"agent_id": desk.id, "runtime": "gpu"})
    session_mode.CLI_RUNTIME_ENABLED = False
    hosted = _from_the_chat(db_session, ws, {"agent_id": desk.id, "runtime": "cli"})

    assert unknown["success"] is False and "Unknown runtime 'gpu'" in unknown["error"]
    assert hosted["success"] is False and "CLI_RUNTIME_ENABLED" in hosted["error"]
    assert not unknown.get("requires_confirmation") and not hosted.get("requires_confirmation")
    assert db_session.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == ws).count() == 0
    assert desk.configuration == {}


def test_back_to_api_drops_the_session_keys_and_keeps_the_rest(db_session, ws, session_mode):
    desk = _agent(db_session, ws, "CLUB DESK", {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: "claude",
                                               CONFIG_MODEL_KEY: "sonnet", "working_directory": "/srv/club"})

    reply = _tool(db_session, ws, UPDATE, {"agent_id": desk.id, "runtime": "api"})

    assert reply["success"] is True and "runtime: cli → api" in reply["changes"], reply
    assert desk.configuration == {CONFIG_RUNTIME_KEY: "api", "working_directory": "/srv/club"}


def test_a_session_agents_model_changes_alone(db_session, ws, session_mode):
    from modules.tools.discovery.card_question import platform_question

    desk = _agent(db_session, ws, "CLUB DESK", {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: "claude",
                                               CONFIG_MODEL_KEY: "sonnet"})
    asked = platform_question(db_session, ws, UPDATE, {"agent_id": desk.id, "model": "opus"}, "change an agent")
    assert "- session model: sonnet → opus" in asked and "runtime:" not in asked

    reply = _tool(db_session, ws, UPDATE, {"agent_id": desk.id, "model": "opus"})

    assert reply["success"] is True and reply["changes"] == ["session model: sonnet → opus"], reply
    assert desk.configuration[CONFIG_MODEL_KEY] == "opus"


def test_the_runtime_and_other_fields_change_together_and_a_refusal_writes_neither(db_session, ws, session_mode):
    desk = _agent(db_session, ws, "CLUB DESK")

    refused = _tool(db_session, ws, UPDATE, {"agent_id": desk.id, "runtime": "cli", "reports_to_id": desk.id})
    assert refused["success"] is False and desk.configuration == {}          # a manager of itself is refused

    reply = _tool(db_session, ws, UPDATE, {"agent_id": desk.id, "runtime": "cli", "description": "Bookings."})
    assert reply["success"] is True and reply["changes"][0] == "description updated", reply
    assert "runtime: api → cli" in reply["changes"] and desk.description == "Bookings."
    assert desk.configuration[CONFIG_RUNTIME_KEY] == RUNTIME_CLI


def test_an_update_never_reaches_another_workspaces_agent(db_session, ws, seed_workspace, session_mode):
    theirs = _agent(db_session, UUID(seed_workspace()), "BRAVO DESK")

    reply = _tool(db_session, ws, UPDATE, {"agent_id": theirs.id, "runtime": "cli"})

    assert reply["success"] is False and theirs.configuration == {}


def test_an_unknown_runtime_is_refused_on_update(db_session, ws, session_mode):
    desk = _agent(db_session, ws, "CLUB DESK")

    reply = _tool(db_session, ws, UPDATE, {"agent_id": desk.id, "runtime": "gpu"})

    assert reply["success"] is False and "Nothing was changed" in reply["error"] and desk.configuration == {}


def test_an_api_agents_update_is_as_before():
    """No runtime named and not a session agent: the call reaches update_agent as it was
    sent; a named 'api' passes the API model's provider through."""
    from types import SimpleNamespace as NS

    from modules.tools.discovery.agent_runtime import _for_the_handler, update_plan

    api = NS(configuration={})
    assert update_plan(api, {"agent_id": 1, "model_id": "x", "provider": "openrouter", "model": "y"}) is None
    planned = update_plan(api, {"agent_id": 1, "runtime": "api", "provider": "openrouter"})
    assert _for_the_handler({"agent_id": 1, "runtime": "api", "provider": "openrouter"}, planned) == {
        "agent_id": 1, "provider": "openrouter"}


# ── get and list say how it runs; the schemas take it ─────────────────────────────────

def test_get_and_list_report_the_runtime(db_session, ws, session_mode):
    made = _tool(db_session, ws, CREATE, dict(SESSION_ASK))
    agent_id = made["agent"]["id"]

    got = _tool(db_session, ws, GET, {"agent_id": agent_id})
    listed = _tool(db_session, ws, LIST, {})

    assert got["agent"]["runtime"] == RUNTIME_CLI
    assert next(a for a in listed["agents"] if a["id"] == agent_id)["runtime"] == RUNTIME_CLI


@pytest.mark.parametrize("action", [CREATE, UPDATE])
def test_both_schemas_take_the_runtime_and_say_what_a_claude_session_is(action):
    from modules.tools.discovery import get_action_registry

    tool = get_action_registry().get(action)
    properties = tool.parameters["properties"]

    assert properties["runtime"]["enum"] == ["api", "cli"]
    said = properties["runtime"]["description"]
    assert "'A Claude session'" in said and "'Claude Code'" in said and "'runs on my machine'" in said
    assert "provider 'claude'" in said
    assert "claude" in properties["provider"]["description"] and "codex" in properties["provider"]["description"]
    assert "'sonnet'" in properties["model"]["description"]
    assert not {"runtime", "provider", "model"} & set(tool.accepts)
    assert "name" in properties if action == CREATE else "agent_id" in properties


def test_the_generated_seed_is_untouched():
    seed = (ORCH / "core" / "seeds" / "platform-management-skill.md").read_text()
    assert "runs on my machine" not in seed
