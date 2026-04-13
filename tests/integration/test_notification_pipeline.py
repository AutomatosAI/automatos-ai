"""PRD-128 US-010: End-to-end notification pipeline smoke test.

Exercises the full round-trip:

1. ``NotificationDispatcher.dispatch('heartbeat_complete', ...)`` inserts a row.
2. ``GET /api/notifications/unread-count`` reports 1.
3. ``POST /api/notifications/{id}/read`` clears it — unread-count becomes 0.
4. ``POST /api/notifications/{id}/dismiss`` hides it from
   ``GET /api/notifications``.

The test drives the dispatcher and the API handler coroutines directly against
a stateful ``FakeDB`` that implements just enough SQL pattern matching to
persist notifications across dispatch → list → update calls. This keeps the
smoke test hermetic (no Postgres, no FastAPI TestClient) while still hitting
the real production code paths.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import UUID, uuid4

# core.database.database imports config.py which requires these.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

# Ensure orchestrator/ is on sys.path so imports resolve the same way they do
# inside the orchestrator process.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCHESTRATOR_ROOT = _REPO_ROOT / "orchestrator"
if str(_ORCHESTRATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCHESTRATOR_ROOT))

import importlib.util  # noqa: E402

import pytest  # noqa: E402

from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.services.notification_dispatcher import NotificationDispatcher  # noqa: E402

# `tests/api/` exists as a sibling package and shadows `orchestrator/api/` on
# sys.path during pytest collection — so `import api.notifications` finds the
# wrong package. Load the module directly by file path to bypass the collision.
_notifications_path = _ORCHESTRATOR_ROOT / "api" / "notifications.py"
_spec = importlib.util.spec_from_file_location(
    "orchestrator_api_notifications", _notifications_path
)
assert _spec and _spec.loader, f"cannot load {_notifications_path}"
notifications_api = importlib.util.module_from_spec(_spec)
# Pydantic v2 resolves forward refs via sys.modules — register before exec.
sys.modules["orchestrator_api_notifications"] = notifications_api
_spec.loader.exec_module(notifications_api)


# ---------------------------------------------------------------- FakeDB


class _FakeResult:
    """Minimal SQLAlchemy ``Result`` stand-in."""

    def __init__(
        self,
        rows: Optional[list] = None,
        scalar: Optional[int] = None,
        rowcount: int = 0,
    ) -> None:
        self._rows = rows or []
        self._scalar = scalar
        self.rowcount = rowcount

    def fetchall(self) -> list:
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def first(self):
        return self._rows[0] if self._rows else None

    def scalar_one(self) -> int:
        return int(self._scalar or 0)

    def __iter__(self):
        return iter(self._rows)


class _FakeRow:
    """Row whose ``_mapping`` and ``__getitem__`` both expose a dict."""

    def __init__(self, data: dict) -> None:
        self._mapping = data

    def __getitem__(self, key):
        return self._mapping[key]


class FakeDB:
    """Stateful in-memory stand-in for a SQLAlchemy Session.

    Handles only the SQL patterns emitted by ``NotificationDispatcher`` and
    ``api.notifications``. Any unexpected SQL text returns an empty result —
    if a new pattern shows up, the test will fail cleanly on an assertion
    rather than a silent miss.
    """

    def __init__(self, workspace_id: UUID) -> None:
        self.workspace_id = workspace_id
        self.notifications: dict[str, dict[str, Any]] = {}
        self.commits = 0
        self.rollbacks = 0

    # --- Session API ---------------------------------------------------

    def execute(self, sql_obj, params=None):  # noqa: C901 — dispatch table
        sql = str(sql_obj)
        params = params or {}

        # Dispatcher: preferences lookup — empty list triggers in_app default.
        if "FROM notification_preferences" in sql and "SELECT" in sql:
            return _FakeResult(rows=[])

        # Dispatcher: insert a notification row.
        if "INSERT INTO notifications" in sql:
            row_id = uuid4()
            self.notifications[str(row_id)] = {
                "id": row_id,
                "workspace_id": self.workspace_id,
                "user_id": params.get("user_id"),
                "event_type": params.get("event_type"),
                "title": params.get("title"),
                "message": params.get("message"),
                "link_type": params.get("link_type"),
                "link_id": params.get("link_id"),
                "agent_id": params.get("agent_id"),
                "agent_name": params.get("agent_name"),
                "status": params.get("status") or "ok",
                "read_at": None,
                "dismissed_at": None,
                "created_at": datetime.utcnow(),
            }
            return _FakeResult(rowcount=1)

        # API: list notifications.
        if "SELECT id, workspace_id, user_id, event_type" in sql:
            rows = [
                _FakeRow(n)
                for n in self._visible_rows(params)
                if n["dismissed_at"] is None
                and (not self._unread_only(sql) or n["read_at"] is None)
            ]
            rows.sort(key=lambda r: r["created_at"], reverse=True)
            limit = params.get("limit")
            offset = params.get("offset") or 0
            if limit is not None:
                rows = rows[offset : offset + limit]
            return _FakeResult(rows=rows)

        # API: count queries (list total + unread-count).
        if "SELECT COUNT(*) FROM notifications" in sql:
            matches = [
                n
                for n in self._visible_rows(params)
                if n["dismissed_at"] is None
                and (not self._unread_only(sql) or n["read_at"] is None)
            ]
            return _FakeResult(scalar=len(matches))

        # API: mark a single notification read.
        if "UPDATE notifications" in sql and "SET read_at = NOW()" in sql:
            if ":id" in sql:
                row = self.notifications.get(str(params.get("id")))
                if row and row["read_at"] is None:
                    row["read_at"] = datetime.utcnow()
                    return _FakeResult(rowcount=1)
                return _FakeResult(rowcount=0)
            # read-all
            marked = 0
            for row in self.notifications.values():
                if row["read_at"] is None and row["dismissed_at"] is None:
                    row["read_at"] = datetime.utcnow()
                    marked += 1
            return _FakeResult(rowcount=marked)

        # API: dismiss a notification.
        if "UPDATE notifications" in sql and "SET dismissed_at = NOW()" in sql:
            row = self.notifications.get(str(params.get("id")))
            if row and row["dismissed_at"] is None:
                row["dismissed_at"] = datetime.utcnow()
                return _FakeResult(rowcount=1)
            return _FakeResult(rowcount=0)

        # API: existence probe after a zero-rowcount update.
        if "SELECT 1 FROM notifications" in sql:
            row = self.notifications.get(str(params.get("id")))
            return _FakeResult(rows=[_FakeRow({"one": 1})] if row else [])

        return _FakeResult()

    def query(self, _model):  # noqa: D401 — mimics Session.query
        q = MagicMock()
        q.filter.return_value = q
        q.first.return_value = None
        return q

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    # --- helpers -------------------------------------------------------

    def _visible_rows(self, params: dict) -> list[dict]:
        ws_param = params.get("workspace_id")
        return [
            n
            for n in self.notifications.values()
            if str(n["workspace_id"]) == str(ws_param)
        ]

    @staticmethod
    def _unread_only(sql: str) -> bool:
        return "read_at IS NULL" in sql


# ------------------------------------------------------------- fixtures


@pytest.fixture
def workspace_id() -> UUID:
    return uuid4()


@pytest.fixture
def fake_db(workspace_id: UUID) -> FakeDB:
    return FakeDB(workspace_id)


@pytest.fixture
def ctx(workspace_id: UUID) -> RequestContext:
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="user_smoke", email="smoke@test.invalid"),
        auth_type="api_key",
    )


# ------------------------------------------------------------------ test


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_notification_pipeline_smoke(
    fake_db: FakeDB, workspace_id: UUID, ctx: RequestContext
) -> None:
    """Dispatch → list → unread count → read → dismiss round-trip."""

    # 1. Dispatch a heartbeat_complete event. With zero configured
    #    preferences the dispatcher falls back to a single in_app insert.
    dispatcher = NotificationDispatcher(fake_db, workspace_id)
    dispatch_result = _run(
        dispatcher.dispatch(
            event_type="heartbeat_complete",
            title="Heartbeat finished",
            message="3 of 3 tasks passed",
            link_type="heartbeat",
            link_id="hb-001",
            agent_id=42,
            agent_name="Sentinel",
            status="ok",
        )
    )
    assert dispatch_result == {"dispatched_to": ["in_app"]}

    # One row should now exist in the fake notifications table.
    assert len(fake_db.notifications) == 1
    stored = next(iter(fake_db.notifications.values()))
    assert stored["workspace_id"] == workspace_id
    assert stored["event_type"] == "heartbeat_complete"
    assert stored["title"] == "Heartbeat finished"
    assert stored["link_type"] == "heartbeat"
    notif_id = stored["id"]

    # Dispatcher must not commit — caller owns the transaction.
    assert fake_db.commits == 0

    # 2. GET /api/notifications/unread-count → 1
    unread = _run(notifications_api.unread_count(ctx=ctx, db=fake_db))
    assert unread.success is True
    assert unread.count == 1

    # 3. GET /api/notifications → the row is visible.
    listing = _run(
        notifications_api.list_notifications(
            limit=20, offset=0, unread_only=False, ctx=ctx, db=fake_db
        )
    )
    assert listing.success is True
    assert listing.total == 1
    assert len(listing.notifications) == 1
    row = listing.notifications[0]
    assert row.event_type == "heartbeat_complete"
    assert row.title == "Heartbeat finished"
    assert row.link_type == "heartbeat"
    assert row.agent_name == "Sentinel"

    # 4. POST /api/notifications/{id}/read — success, then count is 0.
    mark = _run(notifications_api.mark_read(notification_id=notif_id, ctx=ctx, db=fake_db))
    assert mark.success is True

    unread_after = _run(notifications_api.unread_count(ctx=ctx, db=fake_db))
    assert unread_after.count == 0

    # The row is still listed (read, not dismissed).
    listing_after_read = _run(
        notifications_api.list_notifications(
            limit=20, offset=0, unread_only=False, ctx=ctx, db=fake_db
        )
    )
    assert listing_after_read.total == 1
    assert listing_after_read.notifications[0].read_at is not None

    # 5. POST /api/notifications/{id}/dismiss hides it from subsequent lists.
    dismiss = _run(
        notifications_api.dismiss(notification_id=notif_id, ctx=ctx, db=fake_db)
    )
    assert dismiss.success is True

    listing_after_dismiss = _run(
        notifications_api.list_notifications(
            limit=20, offset=0, unread_only=False, ctx=ctx, db=fake_db
        )
    )
    assert listing_after_dismiss.total == 0
    assert listing_after_dismiss.notifications == []
