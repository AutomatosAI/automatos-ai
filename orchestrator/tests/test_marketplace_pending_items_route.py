"""GET /api/marketplace/items/pending reaches its handler and names each creator.

The route was declared after ``/items/{item_id}``, so ``pending`` was parsed as an
item id and every request was rejected with a 422. Behind that, the handler read
``User.full_name``, a column the model does not have (it is ``name``), so it would
have raised as soon as it became reachable (#1058).

These go through the router over HTTP on purpose: calling the handler directly
would not notice the route being shadowed.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.auth.dependencies import RequestContext, UserContext

PENDING_URL = "/api/marketplace/items/pending"


def _client(db, system_role: str = "super_admin") -> TestClient:
    """A client on the marketplace router, signed in with ``system_role``."""
    from api.marketplace import router
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=uuid.uuid4(), user=UserContext(id="u-test", system_role=system_role), auth_type="clerk")
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


@pytest.fixture
def submit_agent(db_session):
    """Factory: an unapproved marketplace agent submitted by a new user."""
    from core.models.core import Agent, User

    def _submit(creator_name):
        tag = uuid.uuid4().hex[:8]
        creator = User(username=f"creator-{tag}", email=f"creator-{tag}@example.com", name=creator_name)
        db_session.add(creator)
        db_session.flush()
        agent = Agent(name=f"Pending agent {tag}", agent_type="custom", description="", status="active",
                      created_by="marketplace", owner_type="marketplace", owner_id="marketplace",
                      is_approved=False, is_featured=False, original_creator_id=creator.id)
        db_session.add(agent)
        db_session.flush()
        return agent, creator

    return _submit


def _pending_entry(response, agent_id: int) -> dict:
    """The pending-list entry for ``agent_id``; the list may hold other agents too."""
    assert response.status_code == 200, response.text
    return next(item for item in response.json()["items"] if item["id"] == agent_id)


def test_pending_lists_an_unapproved_agent_with_its_creator_name(db_session, submit_agent):
    agent, creator = submit_agent("Ada Creator")

    entry = _pending_entry(_client(db_session).get(PENDING_URL), agent.id)

    assert entry["type"] == "agent" and entry["name"] == agent.name
    assert entry["creator"] == "Ada Creator"


def test_pending_falls_back_to_the_creator_email_when_they_have_no_name(db_session, submit_agent):
    agent, creator = submit_agent(None)

    entry = _pending_entry(_client(db_session).get(PENDING_URL), agent.id)

    assert entry["creator"] == creator.email


def test_pending_is_refused_to_a_caller_who_is_not_an_admin(db_session):
    response = _client(db_session, system_role="user").get(PENDING_URL)

    assert response.status_code == 403
