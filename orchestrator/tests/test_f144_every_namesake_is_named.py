"""F144 (night 4) — refusing a namesake names every active agent with the name.

Asked to re-plan "following what I said", Auto called platform_create_agent
three times and made a third WRITER, COUNTINGHOUSE and OPS (309–311; the same
trio on 24 Sep, 306–308). F134 refuses a name an active agent already has, but
named only the first holder, so Auto could not tell which WRITER was meant. The
refusal now lists every active namesake by id, with its team and job title when
set, and returns all their ids.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

from core.models import Agent
from modules.tools.discovery.handlers_agents import create_agent


def _agent(db, ws, name, *, team=None, job_title=None, status="active"):
    agent = Agent(name=name, agent_type="chatbot", description="", status=status, configuration={},
                  model_config=None, workspace_id=ws, created_by="test", owner_type="workspace",
                  owner_id=str(ws), team=team, job_title=job_title)
    db.add(agent)
    db.flush()
    return agent.id


def test_every_active_namesake_is_named_with_its_team_and_job(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    first = _agent(db_session, ws, "WRITER", team="Front of house", job_title="Café notes")
    second = _agent(db_session, ws, "writer ")
    third = _agent(db_session, ws, "WRITER", job_title="Newsletter")
    _agent(db_session, ws, "WRITER", status="inactive")
    reply = asyncio.run(create_agent(db_session, ws, {"name": "Writer"}))
    assert reply == {
        "success": False,
        "existing_agent_id": first,
        "existing_agent_ids": [first, second, third],
        "error": (f"3 active agents are already called 'WRITER' (id {first}, Front of house, Café notes; "
                  f"id {second}; id {third}, Newsletter). Use one of them, or give the new one a different name."),
    }


def test_one_namesake_reads_as_before(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    ops = _agent(db_session, ws, "OPS")
    reply = asyncio.run(create_agent(db_session, ws, {"name": "ops"}))
    assert reply["existing_agent_ids"] == [ops]
    assert reply["error"] == (f"An active agent is already called 'OPS' (id {ops}). "
                              "Use that agent, or give the new one a different name.")
