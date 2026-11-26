"""Ticket #371: platform_get_agent reads an agent's public id, runtime profile and memory profile.

Auto could not say how Phylax Guide (agent 48, a lean agent the Phylax website calls by its
public UUID) was set up: the lookup carried neither the UUID nor the two profiles its Model tab
sets. It now carries ``public_id``, ``runtime_profile`` and ``memory_profile``, read as the
Model tab reads them; a widget turn still sees only who the agent is. The handler tests run
against Postgres (rolled back).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery.agent_profiles import profiles_of

GUIDE_ID = UUID("6f1c2a4e-8d3b-4f7a-9c21-5e0b7d4a3f18")


def _agent(configuration, public_id=GUIDE_ID, system=False):
    return NS(configuration=configuration, public_id=public_id, is_system_agent=system)


# ── how each field is read ───────────────────────────────────────────────────

def test_a_lean_agent_with_its_own_memory_profile_names_all_three():
    got = profiles_of(_agent({"runtime_profile": "lean", "memory_profile": " security-guide "}))

    assert got == {"public_id": str(GUIDE_ID), "runtime_profile": "lean", "memory_profile": "security-guide"}


@pytest.mark.parametrize("configuration", [None, {}, {"runtime_profile": "full"}, {"memory_profile": "  "}],
                         ids=["none", "empty", "full", "blank-memory"])
def test_an_agent_with_nothing_set_is_full_on_its_workspaces_memory(configuration):
    got = profiles_of(_agent(configuration))

    assert (got["runtime_profile"], got["memory_profile"]) == ("full", None)


def test_a_system_agent_is_never_lean_whatever_its_configuration_says():
    assert profiles_of(_agent({"runtime_profile": "lean"}, system=True))["runtime_profile"] == "full"


def test_an_agent_with_no_public_id_says_none():
    assert profiles_of(_agent({}, public_id=None))["public_id"] is None


# ── through platform_get_agent ───────────────────────────────────────────────

@pytest.fixture
def guide(db_session, seed_workspace):
    from core.models import Agent

    ws = UUID(seed_workspace())
    made = Agent(name="Phylax Guide", agent_type="custom", description="Answers Ask Phylax.", status="active",
                 configuration={"runtime_profile": "lean", "memory_profile": "general"},
                 model_config={"provider": "anthropic", "model_id": "claude-haiku-5-5"},
                 public_id=uuid4(), workspace_id=ws, created_by="platform", owner_type="workspace",
                 owner_id=str(ws))
    db_session.add(made)
    db_session.flush()
    return NS(db=db_session, ws=ws, agent=made)


def _lookup(guide):
    from modules.tools.discovery.handlers_agents import get_agent

    return asyncio.run(get_agent(guide.db, guide.ws, {"agent_id": guide.agent.id}))


def test_the_lookup_names_the_public_id_and_both_profiles(guide):
    got = _lookup(guide)["agent"]

    assert (got["public_id"], got["runtime_profile"], got["memory_profile"]) == (
        str(guide.agent.public_id), "lean", "general")


def test_the_lookup_changes_nothing_on_the_agent(guide):
    before = (dict(guide.agent.configuration), dict(guide.agent.model_config), guide.agent.public_id)

    _lookup(guide)
    guide.db.refresh(guide.agent)

    assert (guide.agent.configuration, guide.agent.model_config, guide.agent.public_id) == before


def test_a_widget_turn_still_sees_only_who_the_agent_is(guide):
    from core.security.surface import WIDGET, turn_surface

    with turn_surface(WIDGET, ("chat", "agents:read")):
        got = _lookup(guide)["agent"]

    assert set(got) == {"name", "description", "status"}
