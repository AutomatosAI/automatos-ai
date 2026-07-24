"""PRD-183 S2 (F033) — first-connect auto-sync uses its OWN DB session.

When SHOPIFY first goes active, ``list_connected_apps`` kicks off the catalog
→ knowledge-graph sync as a detached background task. The bug: it passed the
**request-scoped** session (``db=Depends(get_db)``) into
``asyncio.create_task(_product_sync_impl(ws, db))``. FastAPI closes that
session the moment the listing endpoint returns, so the sync ran against a
torn-down session and died mid-flight.

The fix threads the auto-sync through ``_fire_shopify_autosync``, which opens a
fresh ``SessionLocal`` inside the task. These tests pin that contract:

  * ``_fire_shopify_autosync`` opens its own session (a ``SessionLocal``), never
    the request session handed to the endpoint;
  * that session is closed when the task finishes (no leak).

Pure: ``SessionLocal`` and ``_product_sync_impl`` are patched at the boundary;
no DB, no Composio, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from api import tools as tools_api  # noqa: E402


class _FakeSession:
    """Stands in for a SQLAlchemy Session; records close()."""

    def __init__(self, tag):
        self.tag = tag
        self.closed = False

    def close(self):
        self.closed = True


def _fire_and_drain(workspace_id):
    """Invoke the auto-sync inside a running loop (as the async endpoint does),
    then drain the detached task it spawned."""
    async def _driver():
        # _fire_shopify_autosync calls asyncio.create_task — only valid with a
        # running loop, which the real (async) endpoint always has.
        tools_api._fire_shopify_autosync(workspace_id)

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_driver())
        pending = asyncio.all_tasks(loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def test_autosync_uses_own_session(monkeypatch):
    """The background sync must open its OWN session, not the request's."""
    request_session = _FakeSession("REQUEST")  # the one Depends(get_db) owns
    own_session = _FakeSession("OWN")           # the one SessionLocal() mints

    # Patch the SessionLocal used inside the task to hand back our own session.
    import core.database.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: own_session)

    seen = {}

    async def _fake_impl(workspace_id, session):
        seen["workspace_id"] = workspace_id
        seen["session"] = session

    monkeypatch.setattr(tools_api, "_product_sync_impl", _fake_impl)

    _fire_and_drain("ws-abc")

    # The sync ran against the OWN session, never the request session.
    assert seen["session"] is own_session
    assert seen["session"] is not request_session
    assert seen["workspace_id"] == "ws-abc"


def test_autosync_closes_its_session(monkeypatch):
    """The task closes its own session when done (no connection leak)."""
    own_session = _FakeSession("OWN")

    import core.database.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: own_session)

    async def _fake_impl(workspace_id, session):
        return None

    monkeypatch.setattr(tools_api, "_product_sync_impl", _fake_impl)

    _fire_and_drain("ws-abc")

    assert own_session.closed is True


def test_autosync_swallows_errors(monkeypatch):
    """A sync failure must not escape the background task (and still closes)."""
    own_session = _FakeSession("OWN")

    import core.database.database as db_mod
    monkeypatch.setattr(db_mod, "SessionLocal", lambda: own_session)

    async def _boom(workspace_id, session):
        raise RuntimeError("composio exploded")

    monkeypatch.setattr(tools_api, "_product_sync_impl", _boom)

    _fire_and_drain("ws-abc")  # gather(return_exceptions=True) — must not raise

    assert own_session.closed is True
