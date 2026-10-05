"""#831 — ``agents.team``/``agents.reports_to_id`` were only ever returned by
GET /api/agents/org-chart, and only ever writable through Auto's tools, with
no validation on the write (any string became the team; ``reports_to_id`` was
cast straight to ``int`` with no check it was a real agent, in the same
workspace, or not a cycle/self-report).

This suite proves, against the real tables (no mocks for the write paths):
  - ``AgentResponse`` now carries ``team``/``reports_to_id`` (_build_agent_response
    and the Pydantic models).
  - ``AgentCreate``/``AgentUpdate`` accept them.
  - the shared validator (``services.agent_org_fields``) used by BOTH the REST
    API (``api/agents.py``) and Auto's tool handler
    (``modules/tools/discovery/handlers_agents.py``) rejects a cross-workspace
    manager, a self-report, and a cycle — and the tool handler (exercised here
    with a real db session, as the existing agent-handler tests do) actually
    enforces it on create/update, which it did not before this change.

Every test below fails without the fix: before it, ``AgentResponse`` has no
``team``/``reports_to_id`` fields at all, and the handler's old
``agent.reports_to_id = int(params["reports_to_id"])`` never raised on a
cross-workspace id, self-id, or a cycle.
"""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# ---------------------------------------------------------------------------
# Stub transitive jwt/clerk deps so api.agents imports cleanly in CI without
# the full venv (same pattern as test_agents_api_plugins.py).
# ---------------------------------------------------------------------------
_stubs = {}
for mod_name in ["jwt", "jwt.algorithms", "jwt.exceptions", "core.auth.clerk", "core.auth.hybrid"]:
    if mod_name not in sys.modules:
        stub = ModuleType(mod_name)
        stub.get_request_context_hybrid = MagicMock()
        stub.get_clerk_auth = MagicMock()
        stub.decode = MagicMock()
        stub.DecodeError = Exception
        stub.ExpiredSignatureError = Exception
        sys.modules[mod_name] = stub
        _stubs[mod_name] = stub

from api.agents import _apply_agent_org_update, _build_agent_response, _resolve_agent_org_fields  # noqa: E402
from core.models import Agent  # noqa: E402
from core.models.core import AgentCreate, AgentResponse, AgentUpdate  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from modules.tools.discovery.handlers_agents import create_agent, update_agent  # noqa: E402
from services.agent_org_fields import AgentOrgFieldError, normalized_team_or_none, validate_manager  # noqa: E402

for _k in _stubs:  # PRD-142 W2-S2b: drop stubs so a later real import isn't shadowed
    sys.modules.pop(_k, None)


def _agent(db, ws, name="Agent", **kw):
    agent = Agent(
        name=name, agent_type="chatbot", description="", status="active", configuration={},
        workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), **kw,
    )
    db.add(agent)
    db.flush()
    return agent


def _create(db, ws, **params):
    return asyncio.run(create_agent(db, ws, params))


def _update(db, ws, **params):
    return asyncio.run(update_agent(db, ws, params))


# ── AgentResponse / _build_agent_response ───────────────────────────────────

def test_agent_response_model_has_team_and_reports_to_id_fields():
    fields = AgentResponse.model_fields
    assert "team" in fields and fields["team"].default is None
    assert "reports_to_id" in fields and fields["reports_to_id"].default is None


def test_agent_create_and_update_accept_team_and_reports_to_id():
    created = AgentCreate(name="Ops Bot", agent_type="chatbot", team="Engineering", reports_to_id=7)
    assert created.team == "Engineering" and created.reports_to_id == 7
    updated = AgentUpdate(team="Growth", reports_to_id=3)
    assert updated.team == "Growth" and updated.reports_to_id == 3


def test_build_agent_response_carries_team_and_manager():
    agent = MagicMock()
    agent.id, agent.name, agent.description = 42, "Ops Bot", "desc"
    agent.agent_type, agent.status, agent.configuration = "chatbot", "active", {}
    agent.priority_level, agent.max_concurrent_tasks, agent.auto_start = "medium", 5, False
    agent.tags, agent.created_by, agent.performance_metrics = [], "api", {}
    agent.created_at = agent.updated_at = datetime(2026, 1, 1)
    agent.model_config, agent.skills, agent.assigned_plugins = {}, [], []
    agent.workspace_id = uuid4()
    agent.public_id = agent.slug = agent.required_role = agent.marketplace_category = None
    agent.model_usage_stats = agent.voice_profile_id = None
    agent.is_system_agent = False
    agent.job_title = "SRE"
    agent.team = "Engineering & DevOps"
    agent.reports_to_id = 11

    db = MagicMock()
    tools_q = MagicMock()
    tools_q.filter.return_value.all.return_value = []
    db.query.side_effect = [tools_q]

    resp = _build_agent_response(agent, db)
    assert resp.team == "Engineering & DevOps"
    assert resp.reports_to_id == 11


# ── services.agent_org_fields ───────────────────────────────────────────────

def test_normalized_team_or_none_trims_and_clears():
    assert normalized_team_or_none("  Engineering & DevOps  ") == "Engineering & DevOps"
    assert normalized_team_or_none("") is None
    assert normalized_team_or_none("   ") is None
    assert normalized_team_or_none(None) is None


def test_validate_manager_clears_on_falsy_without_a_query():
    db = MagicMock()
    db.query.side_effect = AssertionError("a falsy reports_to_id must never query the db")
    assert validate_manager(db, uuid4(), agent_id=1, reports_to_id=None) is None
    assert validate_manager(db, uuid4(), agent_id=1, reports_to_id=0) is None


def test_validate_manager_rejects_self(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws, name="Solo")
    with pytest.raises(AgentOrgFieldError, match="cannot report to itself"):
        validate_manager(db_session, ws, agent_id=agent.id, reports_to_id=agent.id)


def test_validate_manager_rejects_a_manager_outside_the_workspace(db_session, seed_workspace):
    ws_a = UUID(seed_workspace())
    ws_b = UUID(seed_workspace())
    reportee = _agent(db_session, ws_a, name="Reportee")
    manager_elsewhere = _agent(db_session, ws_b, name="Manager")
    with pytest.raises(AgentOrgFieldError, match="not found in this workspace"):
        validate_manager(db_session, ws_a, agent_id=reportee.id, reports_to_id=manager_elsewhere.id)


def test_validate_manager_rejects_a_manager_that_does_not_exist(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    reportee = _agent(db_session, ws, name="Reportee")
    with pytest.raises(AgentOrgFieldError, match="not found in this workspace"):
        validate_manager(db_session, ws, agent_id=reportee.id, reports_to_id=999999)


def test_validate_manager_rejects_a_cycle(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    root = _agent(db_session, ws, name="Root")
    mid = _agent(db_session, ws, name="Mid", reports_to_id=root.id)
    leaf = _agent(db_session, ws, name="Leaf", reports_to_id=mid.id)
    # Root -> Mid -> Leaf already; pointing Root at Leaf would close the loop.
    with pytest.raises(AgentOrgFieldError, match="reporting cycle"):
        validate_manager(db_session, ws, agent_id=root.id, reports_to_id=leaf.id)


def test_validate_manager_accepts_a_same_workspace_manager(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    reportee = _agent(db_session, ws, name="Reportee")
    assert validate_manager(db_session, ws, agent_id=reportee.id, reports_to_id=manager.id) == manager.id


# ── Auto's tool handler (modules/tools/discovery/handlers_agents.py) ───────

def test_update_agent_rejects_a_cross_workspace_manager_and_changes_nothing(db_session, seed_workspace):
    ws_a = UUID(seed_workspace())
    ws_b = UUID(seed_workspace())
    reportee = _agent(db_session, ws_a, name="Reportee")
    manager_elsewhere = _agent(db_session, ws_b, name="Manager")
    reply = _update(db_session, ws_a, agent_id=reportee.id, reports_to_id=manager_elsewhere.id)
    assert reply == {"success": False, "error": f"Manager agent {manager_elsewhere.id} was not found in this workspace."}
    assert reportee.reports_to_id is None


def test_update_agent_rejects_self_report(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws, name="Solo")
    reply = _update(db_session, ws, agent_id=agent.id, reports_to_id=agent.id)
    assert reply == {"success": False, "error": "An agent cannot report to itself."}


def test_update_agent_rejects_a_cycle(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    root = _agent(db_session, ws, name="Root")
    leaf = _agent(db_session, ws, name="Leaf", reports_to_id=root.id)
    reply = _update(db_session, ws, agent_id=root.id, reports_to_id=leaf.id)
    assert reply == {"success": False, "error": "That manager would create a reporting cycle."}
    assert root.reports_to_id is None


def test_update_agent_sets_a_normalised_team_and_a_valid_manager(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    reportee = _agent(db_session, ws, name="Reportee")
    reply = _update(db_session, ws, agent_id=reportee.id, team="  Engineering  ", reports_to_id=manager.id)
    assert reply["success"] is True
    assert reportee.team == "Engineering"
    assert reportee.reports_to_id == manager.id
    assert f"reports_to_id -> {manager.id}" in reply["changes"]


def test_update_agent_clears_team_and_manager_on_blank_and_zero(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    reportee = _agent(db_session, ws, name="Reportee", team="Engineering", reports_to_id=manager.id)
    reply = _update(db_session, ws, agent_id=reportee.id, team="   ", reports_to_id=0)
    assert reply["success"] is True
    assert reportee.team is None
    assert reportee.reports_to_id is None


def test_create_agent_rejects_a_manager_in_another_workspace(db_session, seed_workspace):
    ws_a = UUID(seed_workspace())
    ws_b = UUID(seed_workspace())
    manager_elsewhere = _agent(db_session, ws_b, name="Manager")
    reply = _create(db_session, ws_a, name="New Hire", reports_to_id=manager_elsewhere.id)
    assert reply == {"success": False, "error": f"Manager agent {manager_elsewhere.id} was not found in this workspace."}


def test_create_agent_sets_validated_team_and_manager(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    reply = _create(db_session, ws, name="New Hire", team="  Growth & Marketing  ", reports_to_id=manager.id)
    assert reply["success"] is True
    created = db_session.query(Agent).filter(Agent.id == reply["agent"]["id"]).first()
    assert created.team == "Growth & Marketing"
    assert created.reports_to_id == manager.id


# ── api/agents.py helpers (the REST path, same shared validator) ───────────

def test_api_resolve_agent_org_fields_normalises_and_validates(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    agent_data = AgentCreate(name="New Hire", agent_type="chatbot", team=" Support ", reports_to_id=manager.id)
    clean_team, manager_id = _resolve_agent_org_fields(db_session, ws, agent_data)
    assert clean_team == "Support"
    assert manager_id == manager.id


def test_api_resolve_agent_org_fields_422_on_cross_workspace_manager(db_session, seed_workspace):
    ws_a = UUID(seed_workspace())
    ws_b = UUID(seed_workspace())
    manager_elsewhere = _agent(db_session, ws_b, name="Manager")
    agent_data = AgentCreate(name="New Hire", agent_type="chatbot", reports_to_id=manager_elsewhere.id)
    with pytest.raises(HTTPException) as exc_info:
        _resolve_agent_org_fields(db_session, ws_a, agent_data)
    assert exc_info.value.status_code == 422
    assert "not found in this workspace" in exc_info.value.detail


def test_api_apply_agent_org_update_422_on_self_report(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    agent = _agent(db_session, ws, name="Solo")
    agent_update = AgentUpdate(reports_to_id=agent.id)
    with pytest.raises(HTTPException) as exc_info:
        _apply_agent_org_update(db_session, ws, agent, agent_update)
    assert exc_info.value.status_code == 422
    assert "cannot report to itself" in exc_info.value.detail


def test_api_apply_agent_org_update_sets_team_and_manager(db_session, seed_workspace):
    ws = UUID(seed_workspace())
    manager = _agent(db_session, ws, name="Manager")
    reportee = _agent(db_session, ws, name="Reportee")
    agent_update = AgentUpdate(team="  Engineering  ", reports_to_id=manager.id)
    _apply_agent_org_update(db_session, ws, reportee, agent_update)
    assert reportee.team == "Engineering"
    assert reportee.reports_to_id == manager.id
