"""``core.auth.actor.resolve_internal_user_id``: the principal as an integer ``users.id``.

These cases moved here from ``test_harness_api.py`` when PRD-263 US-047 deleted the
HARNESS approve route and the ``api.harness`` alias they called through; the resolver
itself is unchanged. Clerk id first, then email, and None (fail-closed) when no
``users`` row matches or there is no principal.
"""
from __future__ import annotations

import types
from uuid import UUID

from core.auth.actor import resolve_internal_user_id
from core.auth.dependencies import RequestContext, UserContext

_WS_ID = UUID("00000000-0000-0000-0000-000000000001")


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._result


class _FakeDB:
    """Minimal stand-in for the Session: only ``query(User).filter().first()`` is used."""

    def __init__(self, user=None):
        self._user = user

    def query(self, _model):
        return _FakeQuery(self._user)


def _ctx(*, clerk=None, email=None):
    return RequestContext(
        workspace_id=_WS_ID,
        user=UserContext(id=clerk or email, email=email, clerk_user_id=clerk),
    )


def _user(uid):
    return types.SimpleNamespace(id=uid)


def test_resolve_user_id_via_clerk():
    assert resolve_internal_user_id(_FakeDB(user=_user(42)), _ctx(clerk="clerk_admin")) == 42


def test_resolve_returns_none_when_no_match():
    # No matching users row → None → a caller gating on identity refuses.
    assert resolve_internal_user_id(_FakeDB(user=None), _ctx(clerk="ghost", email="x@y.z")) is None


def test_resolve_returns_none_without_user():
    assert resolve_internal_user_id(_FakeDB(user=_user(1)), RequestContext(workspace_id=_WS_ID, user=None)) is None
