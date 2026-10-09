"""Marketplace /items/{item_id} routes act on the item of the type they are given.

Agents and playbooks live in two tables with separate id sequences. The routes
took a bare id and looked in ``agents`` only (approve, delete) or first
(detail, toggle-featured), so deleting playbook #5 from the admin Playbooks tab
deleted marketplace agent #5 when one existed. Each route now requires
``type`` ('agent' | 'recipe') and touches only that table.
"""
from __future__ import annotations

import asyncio
import random
import uuid
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from core.auth.dependencies import UserContext

ADMIN = NS(user=UserContext(id="u-admin", role="admin", system_role="super_admin"), workspace_id=None)
# Explicit ids far above any sequence value, shared by one agent and one playbook.
_SHARED_ID = random.randint(1_500_000_000, 2_000_000_000)


@pytest.fixture
def twins(db_session):
    """A marketplace agent and a marketplace playbook with the same id."""
    from core.models.core import Agent, WorkflowTemplate

    agent = Agent(id=_SHARED_ID, name=f"Agent twin {uuid.uuid4().hex[:6]}", agent_type="custom", description="",
                  status="active", created_by="marketplace", owner_type="marketplace", owner_id="marketplace",
                  is_approved=False, is_featured=False)
    playbook = WorkflowTemplate(id=_SHARED_ID, name=f"Playbook twin {uuid.uuid4().hex[:6]}",
                                template_id=f"mkt-{uuid.uuid4().hex[:8]}", description="Drafts posts",
                                owner_type="marketplace", owner_id="marketplace", is_approved=False,
                                is_featured=False, install_count=0, created_by="marketplace", tags=[],
                                template_definition={"steps": []})
    db_session.add_all([agent, playbook])
    db_session.flush()
    return NS(db=db_session, agent=agent, playbook=playbook)


def _run(coro):
    return asyncio.run(coro)


def test_approving_a_playbook_leaves_the_agent_with_its_id_alone(twins):
    from api.marketplace import approve_marketplace_item

    result = _run(approve_marketplace_item(_SHARED_ID, type="recipe", ctx=ADMIN, db=twins.db))

    twins.db.refresh(twins.agent)
    twins.db.refresh(twins.playbook)
    assert result["success"] is True and "Playbook" in result["message"]
    assert twins.playbook.is_approved is True
    assert twins.agent.is_approved is False


def test_deleting_a_playbook_leaves_the_agent_with_its_id_alone(twins):
    from api.marketplace import delete_marketplace_item
    from core.models.core import Agent, WorkflowTemplate

    _run(delete_marketplace_item(_SHARED_ID, type="recipe", ctx=ADMIN, db=twins.db))

    assert twins.db.get(WorkflowTemplate, _SHARED_ID) is None
    assert twins.db.get(Agent, _SHARED_ID) is not None


def test_deleting_an_agent_leaves_the_playbook_with_its_id_alone(twins):
    from api.marketplace import delete_marketplace_item
    from core.models.core import Agent, WorkflowTemplate

    _run(delete_marketplace_item(_SHARED_ID, type="agent", ctx=ADMIN, db=twins.db))

    assert twins.db.get(Agent, _SHARED_ID) is None
    assert twins.db.get(WorkflowTemplate, _SHARED_ID) is not None


def test_toggle_featured_flips_only_the_named_type(twins):
    from api.marketplace import toggle_featured

    result = _run(toggle_featured(_SHARED_ID, type="recipe", ctx=ADMIN, db=twins.db))

    twins.db.refresh(twins.agent)
    twins.db.refresh(twins.playbook)
    assert result["is_featured"] is True
    assert twins.playbook.is_featured is True and twins.agent.is_featured is False


def test_detail_returns_the_playbook_when_asked_for_one(twins):
    from api.marketplace import get_item

    twins.agent.is_approved = True
    twins.playbook.is_approved = True
    twins.db.flush()

    detail = _run(get_item(_SHARED_ID, type="recipe", ctx=ADMIN, db=twins.db))

    assert detail.type == "recipe" and detail.name == twins.playbook.name


def test_detail_hides_an_unapproved_item(twins):
    from api.marketplace import get_item

    with pytest.raises(HTTPException) as caught:
        _run(get_item(_SHARED_ID, type="recipe", ctx=ADMIN, db=twins.db))
    assert caught.value.status_code == 404


def test_every_item_route_requires_the_type():
    from api.marketplace import router

    paths = {("GET", "/api/marketplace/items/{item_id}"), ("POST", "/api/marketplace/items/{item_id}/approve"),
             ("POST", "/api/marketplace/items/{item_id}/toggle-featured"), ("DELETE", "/api/marketplace/items/{item_id}")}
    for route in router.routes:
        for method in route.methods:
            if (method, route.path) in paths:
                required = {p.alias for p in route.dependant.query_params if p.required}
                assert "type" in required, f"{method} {route.path} must require ?type="
