"""PRD-256 N15-6 (F427): whether an agent can run comes from what it did, not its flag alone.

Night 15, A371: "RESEARCHER and WRITER can't run right now… no connected machine serves
either", while both had finished work for the owner that night; WRITER's record still said
inactive. These run Auto's agent tools and the claim against Postgres (rolled back).
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text

CLI = {"runtime": "cli", "provider": "claude", "model": "sonnet"}
_AGENT_SQL = text(
    "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
    "VALUES (:n, 'custom', CAST(:w AS uuid), :s, CAST(:c AS json), 'workspace') RETURNING id")
_TASK_SQL = text(
    "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, source_type, attempts, "
    "started_at, completed_at) VALUES (CAST(:w AS uuid), :t, :s, :a, 'user', 0, :started, :done) RETURNING id")


@pytest.fixture
def night(db_session, seed_workspace):
    """Night 15: WRITER (a Claude Code agent) flagged inactive, no CLI host online, and the
    workspace's other agents; another workspace with its own runs."""
    ws, elsewhere = UUID(seed_workspace()), UUID(seed_workspace())
    now = datetime.now(timezone.utc)

    def agent(name, configuration, status="inactive", workspace=ws):
        return db_session.execute(_AGENT_SQL, {"n": name, "w": str(workspace), "s": status,
                                               "c": json.dumps(configuration)}).scalar()

    def ran(agent_id, minutes_ago, status="review", workspace=ws, title="The Guji blog draft"):
        at = now - timedelta(minutes=minutes_ago)
        done = None if status in ("in_progress", "assigned") else at
        return db_session.execute(_TASK_SQL, {"w": str(workspace), "t": title, "s": status, "a": agent_id,
                                              "started": at - timedelta(minutes=5), "done": done}).scalar()

    return NS(db=db_session, ws=ws, elsewhere=elsewhere, agent=agent, ran=ran,
              writer=agent("WRITER", CLI), researcher=agent("RESEARCHER", CLI))


def _listed(night):
    from modules.tools.discovery.handlers_agents import list_agents

    return {a["name"]: a for a in asyncio.run(list_agents(night.db, night.ws, {}))["agents"]}


def _got(night, agent_id):
    from modules.tools.discovery.handlers_agents import get_agent

    return asyncio.run(get_agent(night.db, night.ws, {"agent_id": agent_id}))["agent"]


# ── an agent flagged inactive that ran in the last hour can run ─────────────

def test_an_inactive_agent_with_a_run_in_the_last_hour_is_reported_as_able_to_run(night):
    from services.agent_availability import why_unavailable
    from core.models import Agent

    night.ran(night.writer, 20)                                  # finished into Review 20 minutes ago
    night.ran(night.researcher, 40, status="done", title="Westmoor invoice")
    agents = night.db.query(Agent).filter(Agent.id.in_([night.writer, night.researcher])).all()

    assert why_unavailable(night.db, night.ws, agents) == {}
    listed = _listed(night)
    for name in ("WRITER", "RESEARCHER"):
        assert listed[name]["can_run"] is True and "why" not in listed[name]
        assert listed[name]["status"] == "active" and listed[name]["last_run"]


def test_platform_get_agent_says_whether_it_can_run_and_its_last_run(night):
    night.ran(night.writer, 5, status="in_progress")             # running now

    got = _got(night, night.writer)

    assert (got["can_run"], got["status"]) == (True, "active") and got["last_run"]
    assert "why" not in got


def test_without_a_recent_run_the_no_host_line_and_the_flag_stand(night):
    from services.agent_availability import NO_HOST

    night.ran(night.writer, 3 * 60)                              # three hours ago: outside the window
    night.ran(night.writer, 10, status="failed")                 # a failed run shows nothing
    night.ran(night.writer, 10, status="assigned")               # nor a ticket still waiting

    writer = _listed(night)["WRITER"]
    assert (writer["can_run"], writer["why"], writer["status"]) == (False, NO_HOST, "inactive")
    assert "last_run" not in writer
    got = _got(night, night.writer)
    assert (got["can_run"], got["why"], got["status"]) == (False, NO_HOST, "inactive")


def test_another_workspaces_run_never_counts(night):
    night.ran(night.writer, 10, workspace=night.elsewhere)

    assert _listed(night)["WRITER"]["can_run"] is False


def test_an_api_agents_inactive_flag_is_real_and_stays(night):
    """The agent factory refuses an inactive API agent: its flag is never shown as active."""
    analyst = night.agent("Shopify Business Analyst", {"model": "anthropic/claude-sonnet-4"})
    night.ran(analyst, 10)

    listed = _listed(night)["Shopify Business Analyst"]
    assert listed["status"] == "inactive" and listed["last_run"]


def test_the_window_is_config(night, monkeypatch):
    from config import config

    night.ran(night.writer, 20)
    monkeypatch.setattr(config, "AGENT_RECENT_RUN_WINDOW_SECONDS", 10 * 60)

    assert _listed(night)["WRITER"]["can_run"] is False


# ── the flag is refreshed when the agent runs ───────────────────────────────

def _flag(night, agent_id):
    return night.db.execute(text("SELECT status FROM agents WHERE id = :i"), {"i": agent_id}).scalar()


def test_a_cli_hosts_claim_refreshes_the_inactive_flag(night):
    from services import board_dispatcher

    ticket = night.ran(night.writer, 0, status="assigned")
    other = night.agent("Their Writer", CLI, workspace=night.elsewhere)

    claimed = board_dispatcher.claim_tasks(night.db, worker_id="cli-host:n15-6", limit=5, lease_seconds=60,
                                           runtime=board_dispatcher.RUNTIME_CLI, workspace_id=night.ws)

    assert [t.id for t in claimed] == [ticket]
    assert _flag(night, night.writer) == "active"
    assert _flag(night, night.researcher) == "inactive" and _flag(night, other) == "inactive"


def test_the_api_lanes_claim_leaves_the_flag_alone(night):
    from services import board_dispatcher

    analyst = night.agent("Shopify Business Analyst", {"model": "anthropic/claude-sonnet-4"})
    night.ran(analyst, 0, status="assigned")

    board_dispatcher.claim_tasks(night.db, worker_id="loop", limit=5, lease_seconds=60, workspace_id=night.ws)

    assert _flag(night, analyst) == "inactive"


def test_a_claim_that_takes_nothing_changes_nothing(night):
    from services import board_dispatcher

    claimed = board_dispatcher.claim_tasks(night.db, worker_id="cli-host:n15-6", limit=5, lease_seconds=60,
                                           runtime=board_dispatcher.RUNTIME_CLI, workspace_id=night.ws)

    assert claimed == [] and _flag(night, night.writer) == "inactive"


# ── a shortened listing keeps the last run ──────────────────────────────────

def test_a_shortened_listing_keeps_each_agents_last_run():
    from services.agent_roster_fit import every_agent_fits

    team = [{"id": 400 + i, "name": "WRITER", "status": "active", "runtime": "cli", "can_run": True,
             "description": "d" * 600, "last_run": "2026-10-14T00:40:00+00:00"} for i in range(40)]

    fitted = every_agent_fits({"success": True, "agents": team})

    assert all(isinstance(a, dict) and a["last_run"] for a in fitted["agents"])
