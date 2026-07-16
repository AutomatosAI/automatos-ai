"""PRD-204 S11 -- the watchlist API (api/watches.py).

The HTTP read/cancel surface over the watch registry. Locks the S11
guarantees:

* list is workspace-scoped and live-only by default (``include_closed`` /
  ``status`` widen it);
* detail carries the S9 serializer shape plus recent events newest-first;
* cross-workspace, unknown, and malformed ids all read as 404 (no DB-cast
  500s);
* cancel closes a live watch, refuses a closed one (422).

Auth rides the app-level dependency overrides: the hybrid context is stubbed
to an anonymous principal (the local-edition posture -- owner of its resolved
workspace), so the ``require_workspace_permission`` gates run for real and
grant through the PRD-195 matrix.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from core.database.database import get_database_url


# ---------------------------------------------------------------------------
# DB fixtures (house idiom of test_prd204_watch_tools)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1 FROM watches LIMIT 1"))
            c.execute(text("SELECT 1 FROM watch_events LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"watchlist API suite needs a reachable Postgres with schema: {exc}")
    yield eng
    eng.dispose()


def _seed_workspace(new_session, name: str) -> str:
    ws_id = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text(
            "INSERT INTO workspaces (id, name) "
            "VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": ws_id, "n": name},
    )
    s.commit()
    s.close()
    return ws_id


def _drop_workspace(new_session, ws_id: str) -> None:
    s = new_session.sweep()
    for stmt in (
        "DELETE FROM watch_events WHERE watch_id IN "
        "(SELECT id FROM watches WHERE workspace_id = CAST(:w AS uuid))",
        "DELETE FROM watches WHERE workspace_id = CAST(:w AS uuid)",
        "DELETE FROM workspaces WHERE id = CAST(:w AS uuid)",
    ):
        s.execute(text(stmt), {"w": ws_id})
    s.commit()
    s.close()


@pytest.fixture
def workspace(new_session):
    ws_id = _seed_workspace(new_session, "prd204-watchlist-api")
    yield ws_id
    _drop_workspace(new_session, ws_id)


@pytest.fixture
def other_workspace(new_session):
    ws_id = _seed_workspace(new_session, "prd204-watchlist-api-other")
    yield ws_id
    _drop_workspace(new_session, ws_id)


# ---------------------------------------------------------------------------
# App factory -- real router + real permission gate, stubbed hybrid context
# ---------------------------------------------------------------------------


def _make_client(new_session, ws_id: str):
    from api.watches import router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)

    session = new_session()

    def _override_db():
        yield session

    ctx = RequestContext(
        workspace_id=uuid.UUID(ws_id),
        user=UserContext(),
        auth_type="anonymous",
    )
    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app), session


def _create_watch(session, ws_id: str, *, title: str = "Watch me") -> str:
    from services.watch_service import WatchService

    watch = WatchService.create_watch(
        session,
        workspace_id=ws_id,
        watch_type="mission",
        target_type="mission",
        target_id=str(uuid.uuid4()),
        title=title,
    )
    session.commit()
    return str(watch.id)


# ---------------------------------------------------------------------------
# List
# ---------------------------------------------------------------------------


def test_list_is_workspace_scoped(new_session, workspace, other_workspace):
    client, session = _make_client(new_session, workspace)
    try:
        mine = _create_watch(session, workspace, title="Mine")
        _create_watch(session, other_workspace, title="Not mine")

        resp = client.get("/api/v1/watches")
        assert resp.status_code == 200
        body = resp.json()
        assert [w["id"] for w in body["watches"]] == [mine]
        assert body["total"] == 1
        # The S9 serializer shape rides through unchanged.
        row = body["watches"][0]
        for key in ("status", "policy", "final_score_display", "action_budget"):
            assert key in row
    finally:
        session.close()


def test_list_live_only_unless_widened(new_session, workspace):
    from services.watch_service import WatchService

    client, session = _make_client(new_session, workspace)
    try:
        live_id = _create_watch(session, workspace, title="Live")
        closed_id = _create_watch(session, workspace, title="Closed")
        WatchService.cancel_watch(session, workspace, closed_id)
        session.commit()

        default = client.get("/api/v1/watches").json()
        assert [w["id"] for w in default["watches"]] == [live_id]

        widened = client.get("/api/v1/watches?include_closed=true").json()
        assert {w["id"] for w in widened["watches"]} == {live_id, closed_id}

        filtered = client.get("/api/v1/watches?status=cancelled").json()
        assert [w["id"] for w in filtered["watches"]] == [closed_id]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Detail
# ---------------------------------------------------------------------------


def test_detail_returns_recent_events_newest_first(new_session, workspace):
    from core.models.watches import WatchEvent

    client, session = _make_client(new_session, workspace)
    try:
        watch_id = _create_watch(session, workspace)
        session.add(
            WatchEvent(
                watch_id=uuid.UUID(watch_id),
                event_type="terminal",
                summary="Target finished",
                event_key="terminal:mission:x",
                created_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            )
        )
        session.commit()

        resp = client.get(f"/api/v1/watches/{watch_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["watch"]["id"] == watch_id
        assert "lineage" in body["watch"]
        types = [e["event_type"] for e in body["recent_events"]]
        assert types == ["terminal", "created"]
    finally:
        session.close()


def test_detail_404_for_cross_workspace_unknown_and_malformed(
    new_session, workspace, other_workspace
):
    client, session = _make_client(new_session, workspace)
    try:
        foreign_id = _create_watch(session, other_workspace)

        assert client.get(f"/api/v1/watches/{foreign_id}").status_code == 404
        assert client.get(f"/api/v1/watches/{uuid.uuid4()}").status_code == 404
        assert client.get("/api/v1/watches/not-a-uuid").status_code == 404
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------


def test_cancel_closes_then_refuses(new_session, workspace):
    client, session = _make_client(new_session, workspace)
    try:
        watch_id = _create_watch(session, workspace)

        resp = client.post(f"/api/v1/watches/{watch_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["watch"]["status"] == "cancelled"

        again = client.post(f"/api/v1/watches/{watch_id}/cancel")
        assert again.status_code == 422

        assert client.post(f"/api/v1/watches/{uuid.uuid4()}/cancel").status_code == 404
    finally:
        session.close()
