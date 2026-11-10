"""PRD-206 Q7 on ``GET /api/memory`` and ``GET /api/memory/search``: a private memory is its owner's.

The resume route and the chat's memory section already keep a ``scope: private`` memory to its
``owner`` (``user:<users.id>``); list and search now do too. A member sees the workspace's
memories and their own private ones. A caller who can't be resolved to a person sees no private
ones. Real users rows (the Clerk lane resolves through them); the memory store is a fake.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from api import widget_memory


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def people(new_session):
    """Two people with Clerk ids; their ids, swept on teardown."""
    s = new_session()
    made = []
    for name in ("priya", "sam"):
        clerk = f"user_{name}_{uuid.uuid4().hex[:8]}"
        pk = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                       {"u": clerk, "e": f"{clerk}@example.com", "c": clerk}).scalar()
        made.append((pk, clerk))
    s.commit()
    s.close()
    yield made
    s = new_session.sweep()
    s.execute(text("DELETE FROM users WHERE id = ANY(:ids)"), {"ids": [pk for pk, _ in made]})
    s.commit()
    s.close()


def _rows(priya, sam):
    return [
        {"id": "shared", "memory": "Acme renews in March.", "metadata": {"scope": "workspace"}},
        {"id": "legacy", "memory": "Acme's account manager is Priya.", "metadata": {}},
        {"id": "priyas", "memory": "Priya prefers email.", "metadata": {"scope": "private", "owner": f"user:{priya}"}},
        {"id": "sams", "memory": "Sam is on leave in May.", "metadata": {"scope": "private", "owner": f"user:{sam}"}},
        {"id": "ownerless", "memory": "No one's.", "metadata": {"scope": "private"}},
    ]


@pytest.fixture
def store(monkeypatch, people):
    (priya, _), (sam, _) = people

    class _Store:
        async def get_all_memories(self, workspace_id, limit):
            return _rows(priya, sam)

        async def search_long_term(self, workspace_id, query, limit):
            return _rows(priya, sam)

    monkeypatch.setattr(widget_memory, "_get_memory_service", lambda: _Store())


def _ctx(user_id, clerk=None):
    return SimpleNamespace(workspace_id=uuid.uuid4(), auth_type="clerk",
                           user=SimpleNamespace(id=user_id, clerk_user_id=clerk, email=None))


def _listed(new_session, ctx):
    s = new_session()
    try:
        return [m.id for m in asyncio.run(widget_memory.list_memories(ctx=ctx, limit=50, db=s)).memories]
    finally:
        s.close()


def _found(new_session, ctx):
    s = new_session()
    try:
        return [m.id for m in asyncio.run(widget_memory.search_memories(q="Acme", ctx=ctx, limit=20, db=s)).results]
    finally:
        s.close()


def test_a_member_sees_the_workspaces_memories_and_only_their_own_private_ones(new_session, people, store):
    (priya, priya_clerk), (sam, sam_clerk) = people
    assert _listed(new_session, _ctx(priya_clerk, priya_clerk)) == ["shared", "legacy", "priyas"]
    assert _listed(new_session, _ctx(sam_clerk, sam_clerk)) == ["shared", "legacy", "sams"]
    assert _found(new_session, _ctx(priya_clerk, priya_clerk)) == ["shared", "legacy", "priyas"]


def test_the_local_editions_integer_user_is_its_own_viewer(new_session, people, store):
    (priya, _), _ = people
    assert _listed(new_session, _ctx(priya)) == ["shared", "legacy", "priyas"]


def test_a_caller_who_is_not_a_known_person_sees_no_private_memory(new_session, store):
    for ctx in (_ctx("user_nobody", "user_nobody"), _ctx("api_key")):
        assert _listed(new_session, ctx) == ["shared", "legacy"]
        assert _found(new_session, ctx) == ["shared", "legacy"]
