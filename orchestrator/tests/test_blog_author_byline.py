"""Blog author byline resolution.

The authenticated create-post endpoint used to read ``ctx.user.display_name``,
but ``UserContext`` has no such attribute (it carries ``id``/``email``/role only).
Every authenticated create therefore raised ``AttributeError`` and returned 500 —
manual blog posting was 100% broken in production.

The byline is *public* (widget endpoints expose ``author_name`` on every post),
so we deliberately do NOT fall back to the user's email (PII leak). Instead the
public byline is the workspace/brand name (e.g. "InBuildUK"), falling back to a
neutral "Workspace Author" when the name is missing or blank.

These tests pin ``_resolve_author_name`` with a MagicMock DB session — no real
Postgres, no network.
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# core.database.database resolves a DB URL at import time; never connects here.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

from api.blog import _resolve_author_name  # noqa: E402


def _db_returning(scalar_value):
    """MagicMock session whose query(...).filter(...).scalar() yields a value."""
    db = MagicMock()
    db.query.return_value.filter.return_value.scalar.return_value = scalar_value
    return db


def test_resolve_author_name_uses_workspace_name():
    db = _db_returning("InBuildUK")
    assert _resolve_author_name(db, uuid4()) == "InBuildUK"


def test_resolve_author_name_falls_back_when_missing():
    # No workspace row / NULL name → neutral byline, never an empty string.
    db = _db_returning(None)
    assert _resolve_author_name(db, uuid4()) == "Workspace Author"


def test_resolve_author_name_falls_back_on_blank():
    # A blank/whitespace name must not become the public byline.
    db = _db_returning("   ")
    assert _resolve_author_name(db, uuid4()) == "Workspace Author"


def test_resolve_author_name_strips_surrounding_whitespace():
    db = _db_returning("  InBuildUK  ")
    assert _resolve_author_name(db, uuid4()) == "InBuildUK"


def test_resolve_author_name_never_leaks_pii():
    # Guard the security intent: the resolver only ever touches the workspace
    # name, so an email can never become the public byline.
    db = _db_returning("user@example.com")
    # (This only happens if a workspace is literally named after an email; the
    # point is the resolver has no access path to ctx.user.email at all.)
    result = _resolve_author_name(db, uuid4())
    assert result == "user@example.com"  # returned verbatim — it's the ws name
    # The function signature takes only (db, workspace_id) — no user object.
    import inspect
    params = list(inspect.signature(_resolve_author_name).parameters)
    assert params == ["db", "workspace_id"]
