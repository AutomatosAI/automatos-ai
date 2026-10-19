"""Clerk production cutover: sign-in never moves a ``users`` row, and never 500s.

Moving Clerk from its development instance to production gives every returning
person a new Clerk user ID and the same email ("Automatos billing options", 19 Oct
2026, section 4, bug 2). ``users.email`` and ``users.username`` are unique, so
provisioning's insert (``ON CONFLICT (clerk_user_id) DO NOTHING``) raised on the
email, was rolled back, the read-back by the new Clerk ID found nothing and
``uid_row[0]`` crashed: a 500 for every returning user.

The first fix relinked the row by email at sign-in. That was held on PR #1198: a
login-time email match can rebind an existing account to a new Clerk user. Sign-in
now finds a row by Clerk ID or inserts one; an email already held by another row is
a clear 409 and nothing moves. Returning users are relinked once by the cutover
step (``core/auth/clerk_cutover.py``, tested in ``test_clerk_cutover_relink.py``).

These tests pin ``core/auth/user_link.py`` and the resolver. The real-Postgres
journey is ``test_clerk_cutover_returning_user_realdb.py``.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock, patch  # noqa: E402
from uuid import uuid4  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import core.auth.hybrid as hybrid  # noqa: E402
import core.auth.user_link as user_link  # noqa: E402
from core.auth.hybrid import _resolve_workspace_for_clerk_user  # noqa: E402
from core.auth.user_link import (  # noqa: E402
    LINK_CONFLICT_MESSAGE,
    ensure_user_row,
    user_id_for_invitee,
)

OLD_CLERK_ID = "user_dev_2xOldDevInstance"
NEW_CLERK_ID = "user_prod_2yNewProdInstance"
EMAIL = "returning@example.test"


class _Result:
    def __init__(self, rows, rowcount=0):
        self._rows = rows
        self.rowcount = rowcount

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _UsersTable:
    """The ``users`` statements user_link sends, answered from a list of dicts.

    The unique columns are clerk_user_id, email and username, as in
    ``core/models/core.py`` ``User`` (case-sensitive, as Postgres compares them).
    Any other statement fails the test.
    """

    def __init__(self, *rows):
        self.rows = [dict(row) for row in rows]
        self.statements = []
        self.commits = 0

    def execute(self, clause, params=None):
        sql = " ".join(str(clause).split())
        self.statements.append(sql)
        params = params or {}
        if sql.startswith("SELECT id FROM users WHERE clerk_user_id = :cid"):
            return _Result([(r["id"],) for r in self.rows if r["clerk_user_id"] == params["cid"]])
        if sql.startswith("SELECT id, clerk_user_id FROM users WHERE LOWER(email) = LOWER(:email)"):
            hits = sorted((r for r in self.rows if r["email"].lower() == params["email"].lower()), key=lambda r: r["id"])
            return _Result([(r["id"], r["clerk_user_id"]) for r in hits])
        if sql == "UPDATE users SET clerk_user_id = :cid WHERE id = :id AND clerk_user_id IS NULL":
            return self._claim(params)
        if sql.startswith("INSERT INTO users"):
            return self._insert(sql, params)
        raise AssertionError(f"unexpected SQL: {sql}")

    def _claim(self, params):
        hits = [r for r in self.rows if r["id"] == params["id"] and r["clerk_user_id"] is None]
        for row in hits:
            row["clerk_user_id"] = params["cid"]
        return _Result([], rowcount=len(hits))

    def _insert(self, sql, params):
        assert sql.endswith("ON CONFLICT DO NOTHING"), "the insert must skip on ANY unique conflict"
        taken = any(
            r["clerk_user_id"] == params["cid"] or r["email"] == params["email"] or r["username"] == params["username"]
            for r in self.rows
        )
        if not taken:
            self.rows.append({
                "id": max([r["id"] for r in self.rows], default=0) + 1,
                "clerk_user_id": params["cid"],
                "email": params["email"],
                "username": params["username"],
            })
        return _Result([])

    def commit(self):
        self.commits += 1

    def inserts(self):
        return [s for s in self.statements if s.startswith("INSERT INTO users")]

    def updates(self):
        return [s for s in self.statements if s.startswith("UPDATE users")]


def _returning_user(**overrides):
    row = {"id": 7, "clerk_user_id": OLD_CLERK_ID, "email": EMAIL, "username": EMAIL}
    row.update(overrides)
    return row


def _conflict(call):
    with pytest.raises(HTTPException) as raised:
        call()
    assert raised.value.status_code == 409
    assert raised.value.detail == LINK_CONFLICT_MESSAGE
    return raised.value


# ── ensure_user_row: the sign-in path ────────────────────────────────────

def test_a_known_clerk_id_resolves_without_inserting():
    db = _UsersTable(_returning_user(clerk_user_id=NEW_CLERK_ID))
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL) == 7
    assert len(db.statements) == 1


def test_a_new_clerk_id_with_a_taken_email_is_a_409_and_moves_nothing():
    """The old 500, and the held relink: the row stays on its Clerk ID."""
    db = _UsersTable(_returning_user())
    _conflict(lambda: ensure_user_row(db, NEW_CLERK_ID, email=EMAIL, name="Returning"))
    assert db.rows == [_returning_user()], "the existing row is not rebound to the new Clerk ID"
    assert db.inserts() == [] and db.updates() == []


def test_the_taken_email_is_matched_case_insensitively():
    """users.email is unique but case-sensitive: without this a second row would be made."""
    db = _UsersTable(_returning_user(email="Returning@Example.TEST", username="Returning@Example.TEST"))
    _conflict(lambda: ensure_user_row(db, NEW_CLERK_ID, email=EMAIL))
    assert len(db.rows) == 1 and db.rows[0]["clerk_user_id"] == OLD_CLERK_ID
    assert db.inserts() == []


def test_a_brand_new_person_still_gets_a_row():
    db = _UsersTable(_returning_user(email="someone-else@example.test", username="someone-else@example.test"))
    new_id = ensure_user_row(db, NEW_CLERK_ID, email=EMAIL, name="New")
    assert new_id == 8
    assert db.rows[-1] == {"id": 8, "clerk_user_id": NEW_CLERK_ID, "email": EMAIL, "username": EMAIL}
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL) == 8, "a second call finds the same row"
    assert len(db.inserts()) == 1


def test_a_token_without_an_email_gets_the_pending_placeholder():
    db = _UsersTable()
    assert ensure_user_row(db, NEW_CLERK_ID, email=None) == 1
    assert db.rows[0]["email"] == f"{NEW_CLERK_ID}@pending"
    assert db.rows[0]["username"] == NEW_CLERK_ID
    assert not any("LOWER(email)" in s for s in db.statements)


def test_a_username_held_by_another_row_is_a_clear_409_not_a_500():
    db = _UsersTable(_returning_user(email="other@example.test", username=EMAIL))
    _conflict(lambda: ensure_user_row(db, NEW_CLERK_ID, email=EMAIL))
    assert "users" not in LINK_CONFLICT_MESSAGE and "clerk" not in LINK_CONFLICT_MESSAGE.lower()


def test_sign_in_has_no_relink_by_email():
    """The held design is gone, not just unused."""
    assert not hasattr(user_link, "relink_user_by_email")
    assert not hasattr(hybrid, "relink_user_by_email")
    assert not hasattr(hybrid, "_relinked_workspace_for_clerk_user")


# ── the resolver: a returning person is never rebound at sign-in ─────────

@patch("core.auth.hybrid._has_pending_invitations", return_value=False)
@patch("core.auth.hybrid._accessible_workspace_for_clerk_user", return_value=None)
def test_the_resolver_gives_a_returning_person_a_409_not_their_row(accessible, pending):
    """Before the cutover step relinks them, a new Clerk ID reaches provisioning and stops."""
    db = _UsersTable(_returning_user())
    _conflict(lambda: _resolve_workspace_for_clerk_user(db, clerk_user_id=NEW_CLERK_ID, org_id=None, email=EMAIL))
    accessible.assert_called_once_with(db, NEW_CLERK_ID)
    assert db.rows == [_returning_user()]
    assert db.updates() == [] and db.commits == 0


@patch("core.auth.hybrid._provision_new_user_workspace")
@patch("core.auth.hybrid._has_pending_invitations", return_value=False)
@patch("core.auth.hybrid._accessible_workspace_for_clerk_user", return_value=None)
def test_the_resolver_still_provisions_a_genuinely_new_person(accessible, pending, provision):
    new_workspace = uuid4()
    provision.return_value = new_workspace
    db = MagicMock()

    resolved = _resolve_workspace_for_clerk_user(db, clerk_user_id=NEW_CLERK_ID, org_id=None, email=EMAIL, name="New")

    assert resolved == new_workspace
    db.commit.assert_not_called()
    provision.assert_called_once_with(db, NEW_CLERK_ID, email=EMAIL, name="New")


# ── user_id_for_invitee: accepting an invitation ─────────────────────────

def test_an_invitee_with_a_known_clerk_id_uses_their_row():
    db = _UsersTable(_returning_user(clerk_user_id=NEW_CLERK_ID))
    assert user_id_for_invitee(db, NEW_CLERK_ID, EMAIL) == 7
    assert db.updates() == [] and db.inserts() == []


def test_an_invitee_claims_an_unlinked_row_whatever_the_case_of_its_email():
    """team.py matched User.email == the lowercased email, so a mixed-case row got a twin."""
    db = _UsersTable(_returning_user(clerk_user_id=None, email="Returning@Example.TEST"))
    assert user_id_for_invitee(db, NEW_CLERK_ID, EMAIL) == 7
    assert db.rows[0]["clerk_user_id"] == NEW_CLERK_ID
    assert db.inserts() == []
    assert db.commits == 0, "the caller commits with the membership"


def test_an_invitee_never_takes_a_row_that_belongs_to_another_clerk_id():
    db = _UsersTable(_returning_user(email="Returning@Example.TEST"))
    _conflict(lambda: user_id_for_invitee(db, NEW_CLERK_ID, EMAIL))
    assert db.rows[0]["clerk_user_id"] == OLD_CLERK_ID
    assert db.inserts() == [] and db.updates() == []


def test_a_new_invitee_gets_one_row():
    db = _UsersTable()
    assert user_id_for_invitee(db, NEW_CLERK_ID, EMAIL, name="New") == 1
    assert db.rows == [{"id": 1, "clerk_user_id": NEW_CLERK_ID, "email": EMAIL, "username": EMAIL}]
