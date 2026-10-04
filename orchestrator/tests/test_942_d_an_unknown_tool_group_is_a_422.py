"""#942 (D): an unknown session tool group is a 422, on the agent update route and
on ``GET /api/agents/{id}?groups=``; the agent page reads the groups payload.

Only the catalogue's ids are stored, so ``enabled`` can only ever hold ids that
``available`` lists, and every ``fix.enable_group`` is one of them too.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api import agent_session_tools as agent_tools
from api import agents as agents_api
from core.models import AgentResponse, AgentUpdate

CTX = SimpleNamespace(workspace_id="ws-shop")


class _DB:
    """Answers the route's one agent lookup; anything after it is a failure here."""

    def __init__(self, agent=None):
        self.agent = agent
        self.commits = 0

    def query(self, *_args, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def options(self, *_args, **_kwargs):
        return self

    def first(self):
        return self.agent

    def commit(self):
        self.commits += 1


def _agent(**config):
    return SimpleNamespace(id=268, name="Business Analyst", workspace_id="ws-shop",
                           configuration={"runtime": "cli", **config})


def test_the_update_route_refuses_an_unknown_group_with_422_and_saves_nothing():
    db = _DB(_agent())
    update = AgentUpdate(configuration={"session_tool_groups": ["data", "spreadsheets"]})
    with pytest.raises(HTTPException) as refused:
        asyncio.run(agents_api.update_agent(268, update, ctx=CTX, db=db))
    assert refused.value.status_code == 422
    assert "spreadsheets" in str(refused.value.detail) and "data, graph, documents" in str(refused.value.detail)
    assert db.commits == 0


def test_a_list_of_known_groups_and_an_empty_list_pass():
    for value in (["data", "graph"], [], None):
        agent_tools.reject_unknown_tool_groups({"session_tool_groups": value})
    agent_tools.reject_unknown_tool_groups({})
    with pytest.raises(HTTPException) as refused:
        agent_tools.reject_unknown_tool_groups({"session_tool_groups": "data"})   # a string is not a list
    assert refused.value.status_code == 422


def test_get_with_an_unknown_preview_group_is_a_422():
    with pytest.raises(HTTPException) as refused:
        asyncio.run(agents_api.get_agent(268, groups="data,nope", ctx=CTX, db=_DB(_agent())))
    assert refused.value.status_code == 422 and "nope" in str(refused.value.detail)


def test_the_preview_parameter_reads_absent_empty_and_a_list():
    assert agent_tools.requested_groups(None) is None
    assert agent_tools.requested_groups("") == []                                  # core only
    assert agent_tools.requested_groups("graph, data") == ["data", "graph"]        # display order


def _base(agent):
    now = datetime.now(timezone.utc)
    return AgentResponse(id=agent.id, name=agent.name, description=None, agent_type="custom", status="active",
                         configuration=agent.configuration, priority_level="medium", max_concurrent_tasks=5,
                         auto_start=False, created_at=now, updated_at=now)


def test_the_agent_detail_carries_the_groups_payload(monkeypatch):
    async def no_gaps(db, workspace_id, enabled):
        return []

    monkeypatch.setattr(agent_tools, "workspace_gaps", no_gaps)
    agent = _agent(session_tool_groups=["graph", "data"])
    detail = asyncio.run(agent_tools.agent_detail(_base(agent), agent, None, None, []))
    payload = detail.session_tool_groups
    assert payload["enabled"] == ["data", "graph"] and payload["is_default"] is False
    assert [g["id"] for g in payload["available"]] == ["data", "graph", "documents", "playbooks", "reports",
                                                       "missions"]
    assert payload["available"][3]["tools"] == ["list_playbooks", "get_playbook", "run_playbook"]
    assert set(payload["enabled"]) <= {g["id"] for g in payload["available"]}

    default = _agent()
    preview = asyncio.run(agent_tools.agent_detail(_base(default), default, None, ["reports"], []))
    assert preview.session_tool_groups["enabled"] == ["reports"] and preview.session_tool_groups["is_default"]
