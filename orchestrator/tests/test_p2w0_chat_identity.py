"""PRD-185 S6: real chat identity — get_user_id must prefer the authenticated
principal, not a hardcoded id=1.

``get_user_id`` ignored the request context and returned id=1, so chats.user_id,
message saves, vote-ownership checks, and the PRD-163 mid-chat mission approval
(``_driving_clerk`` derives from ``user_id``) all mis-attributed to user 1. The
fix threads ``ctx.user.id``; the id=1/default lookup remains only for genuinely
principal-less system paths.

Pure unit test — no DB / network (db is mocked; the request context is a stub).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _get_user_id():
    try:
        from api.chat import get_user_id
    except Exception as e:  # env without the heavy router deps
        pytest.skip(f"api.chat not importable in this env: {e}")
    return get_user_id


def test_prefers_authenticated_principal():
    get_user_id = _get_user_id()
    db = MagicMock()
    ctx = SimpleNamespace(user=SimpleNamespace(id=42))
    assert get_user_id(db, ctx) == 42
    # A logged-in caller must NEVER fall through to the id=1 default lookup.
    db.execute.assert_not_called()


def test_falls_back_when_no_principal():
    get_user_id = _get_user_id()
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (7,)
    # ctx=None (system path) and ctx.user=None both fall back cleanly, no crash.
    assert get_user_id(db, None) == 7
    assert get_user_id(db, SimpleNamespace(user=None)) == 7


def test_resolves_clerk_string_principal_to_integer_pk():
    """Regression — prod chat outage (POST /api/chat -> 500).

    ``UserContext.id`` carries the Clerk subject STRING (``user_xxx``) / email,
    NOT the integer ``users.id``. The S6 change returned it verbatim, so a Clerk
    string was written into the INTEGER ``chats.user_id`` column:
    ``invalid input syntax for type integer: "user_38Z...""``. get_user_id must
    resolve the principal to the integer PK via ``clerk_user_id``.
    """
    get_user_id = _get_user_id()
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = (99,)
    ctx = SimpleNamespace(
        user=SimpleNamespace(
            id="user_38Z4SP1ttmy9Sk3wf79XgQLS8H1",
            clerk_user_id="user_38Z4SP1ttmy9Sk3wf79XgQLS8H1",
            email="pilot@example.com",
        )
    )
    result = get_user_id(db, ctx)
    assert result == 99
    # Must be the integer PK from the lookup — never the raw Clerk string.
    assert isinstance(result, int)
    db.execute.assert_called()
