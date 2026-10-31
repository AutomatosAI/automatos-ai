"""#1060 — verifying or deleting a training example stays inside its source.

``SQLExampleStore.mark_verified`` and ``delete_example`` raise ``ValueError``
when the row is gone and neither route caught it, so a stale Training Examples
panel answered 500. Neither route used the ``source_id`` path parameter either:
example 42 of source 2 could be verified or deleted through source 1 in the
same workspace. Both now look the example up with the id + workspace + source
filter the sibling ``update_training_example`` uses, and answer 404 with the
same message when it is absent or belongs to another source.

Pure: the router is mounted on a bare app with the DB and auth dependencies
overridden; no database.
"""
from __future__ import annotations

import operator
import os
import sys
from pathlib import Path
from uuid import uuid4

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import database_knowledge as database_api  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.database.database import get_db  # noqa: E402

WORKSPACE_ID = uuid4()
SOURCE_ID = 1
OTHER_SOURCE_ID = 2
EXAMPLE_ID = 42
PREFIX = "/api/knowledge/sources/database"
MISSING_DETAIL = "Training example not found"


class _Example:
    """The columns the training-example routes and store touch."""

    def __init__(self, *, example_id: int, workspace_id, source_id: int):
        self.id = example_id
        self.workspace_id = workspace_id
        self.database_source_id = source_id
        self.question = "how many orders this month?"
        self.sql = "SELECT count(*) FROM orders"
        self.tables_used = []
        self.is_verified = False
        self.verification_source = None
        self.usage_count = 0
        self.last_used_at = None
        self.extra_metadata = {}
        self.updated_at = None


def _matches(row: _Example, criterion) -> bool:
    """Evaluate one equality filter the way SQLAlchemy would for ``row``.

    ``str`` on both sides mirrors the DB's coercion of the store's string
    workspace id against the route's ``UUID`` (both name the same workspace).
    """
    key = getattr(getattr(criterion, "left", None), "key", None)
    assert key is not None and criterion.operator is operator.eq, (
        f"the fake only understands equality filters, got {criterion!r}"
    )
    return str(getattr(row, key)) == str(criterion.right.value)


class _Query:
    """A chainable query over an in-memory list of examples."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *criteria):
        return _Query([row for row in self._rows if all(_matches(row, c) for c in criteria)])

    def first(self):
        return self._rows[0] if self._rows else None


class _Db:
    """The session surface the routes and SQLExampleStore use."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.deleted = []
        self.commits = 0

    def query(self, _model):
        return _Query(self.rows)

    def delete(self, row):
        self.rows.remove(row)
        self.deleted.append(row)

    def commit(self):
        self.commits += 1


def _example(*, example_id: int = EXAMPLE_ID, source_id: int = SOURCE_ID) -> _Example:
    return _Example(example_id=example_id, workspace_id=WORKSPACE_ID, source_id=source_id)


def _client(rows):
    app = FastAPI()
    app.include_router(database_api.router)
    ctx = RequestContext(
        workspace_id=WORKSPACE_ID,
        user=UserContext(id="u-1", system_role="super_admin"),
        auth_type="clerk",
    )
    db = _Db(rows)
    app.dependency_overrides[database_api.get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app), db


def _verify_url(source_id: int = SOURCE_ID, example_id: int = EXAMPLE_ID) -> str:
    return f"{PREFIX}/{source_id}/examples/{example_id}/verify"


def _example_url(source_id: int = SOURCE_ID, example_id: int = EXAMPLE_ID) -> str:
    return f"{PREFIX}/{source_id}/examples/{example_id}"


# --- missing rows answer 404, not 500 ----------------------------------------

def test_verifying_a_missing_example_is_404():
    client, _ = _client([])
    response = client.put(_verify_url())
    assert response.status_code == 404
    assert response.json()["detail"] == MISSING_DETAIL


def test_deleting_a_missing_example_is_404():
    client, _ = _client([])
    response = client.delete(_example_url())
    assert response.status_code == 404
    assert response.json()["detail"] == MISSING_DETAIL


# --- an example of another source is out of reach ----------------------------

def test_verifying_an_example_of_another_source_is_404_and_leaves_it_alone():
    example = _example(source_id=OTHER_SOURCE_ID)
    client, _ = _client([example])
    response = client.put(_verify_url(source_id=SOURCE_ID))
    assert response.status_code == 404
    assert response.json()["detail"] == MISSING_DETAIL
    assert example.is_verified is False


def test_deleting_an_example_of_another_source_is_404_and_leaves_it_alone():
    example = _example(source_id=OTHER_SOURCE_ID)
    client, db = _client([example])
    response = client.delete(_example_url(source_id=SOURCE_ID))
    assert response.status_code == 404
    assert response.json()["detail"] == MISSING_DETAIL
    assert db.deleted == []
    assert db.rows == [example]


# --- the source's own example still works -----------------------------------

def test_verifying_the_sources_own_example_still_works():
    example = _example()
    client, db = _client([example])
    response = client.put(_verify_url())
    assert response.status_code == 200
    assert response.json() == {"success": True, "example_id": EXAMPLE_ID, "verified": True}
    assert example.is_verified is True
    assert example.verification_source == "user"
    assert db.commits == 1


def test_deleting_the_sources_own_example_still_works():
    example = _example()
    client, db = _client([example])
    response = client.delete(_example_url())
    assert response.status_code == 200
    assert response.json() == {"success": True, "deleted_id": EXAMPLE_ID}
    assert db.rows == []
    assert db.deleted == [example]
    assert db.commits == 1
