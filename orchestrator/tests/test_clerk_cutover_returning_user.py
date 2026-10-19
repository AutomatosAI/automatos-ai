"""Clerk production cutover: a returning person signs in with a NEW Clerk user ID.

Moving Clerk from its development instance to production gives every returning
person a new Clerk user ID and the same email ("Automatos billing options", 19 Oct
2026, section 4, bug 2). ``users.email`` and ``users.username`` are unique, so
provisioning's insert (``ON CONFLICT (clerk_user_id) DO NOTHING``) raised on the
email, was rolled back, the read-back by the new Clerk ID found nothing and
``uid_row[0]`` crashed: a 500 for every returning user.

These tests pin the fix in ``core/auth/user_link.py`` (find by Clerk ID, else
relink by email, else insert) and its wiring into the workspace resolver. The
real-Postgres version of the same journey is
``test_clerk_cutover_returning_user_realdb.py``.
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

from core.auth.hybrid import _resolve_workspace_for_clerk_user  # noqa: E402
from core.auth.user_link import (  # noqa: E402
    LINK_CONFLICT_MESSAGE,
    ensure_user_row,
    relink_user_by_email,
)

OLD_CLERK_ID = "user_dev_2xOldDevInstance"
NEW_CLERK_ID = "user_prod_2yNewProdInstance"
EMAIL = "returning@example.test"


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _UsersTable:
    """The ``users`` statements user_link sends, answered from a list of dicts.

    The unique columns are clerk_user_id, email and username, as in
    ``core/models/core.py`` ``User``. Any other statement fails the test.
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
        if sql.startswith("UPDATE users SET clerk_user_id = :cid WHERE id = :id"):
            for row in self.rows:
                if row["id"] == params["id"]:
                    row["clerk_user_id"] = params["cid"]
            return _Result([])
        if sql.startswith("INSERT INTO users"):
            return self._insert(sql, params)
        raise AssertionError(f"unexpected SQL: {sql}")

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


def _returning_user(**overrides):
    row = {"id": 7, "clerk_user_id": OLD_CLERK_ID, "email": EMAIL, "username": EMAIL}
    row.update(overrides)
    return row


# ── ensure_user_row ──────────────────────────────────────────────────────

def test_a_known_clerk_id_resolves_without_relinking_or_inserting():
    db = _UsersTable(_returning_user(clerk_user_id=NEW_CLERK_ID))
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL) == 7
    assert len(db.statements) == 1


def test_a_returning_person_with_a_new_clerk_id_keeps_their_row_relinked():
    """The 500: a new Clerk ID with an email that already has a row."""
    db = _UsersTable(_returning_user())
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL, name="Returning") == 7
    assert db.rows == [_returning_user(clerk_user_id=NEW_CLERK_ID)]
    assert db.inserts() == [], "no second users row is attempted"


def test_the_relink_matches_the_email_case_insensitively():
    db = _UsersTable(_returning_user(email="Returning@Example.TEST"))
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL) == 7
    assert db.rows[0]["clerk_user_id"] == NEW_CLERK_ID


def test_a_brand_new_person_still_gets_a_row():
    db = _UsersTable(_returning_user(email="someone-else@example.test", username="someone-else@example.test"))
    new_id = ensure_user_row(db, NEW_CLERK_ID, email=EMAIL, name="New")
    assert new_id == 8
    assert db.rows[-1] == {"id": 8, "clerk_user_id": NEW_CLERK_ID, "email": EMAIL, "username": EMAIL}
    assert ensure_user_row(db, NEW_CLERK_ID, email=EMAIL) == 8, "a second call finds the same row"
    assert len(db.inserts()) == 1


def test_a_token_without_an_email_gets_the_pending_placeholder_and_no_relink():
    db = _UsersTable()
    assert ensure_user_row(db, NEW_CLERK_ID, email=None) == 1
    assert db.rows[0]["email"] == f"{NEW_CLERK_ID}@pending"
    assert db.rows[0]["username"] == NEW_CLERK_ID
    assert not any("LOWER(email)" in s for s in db.statements)


def test_a_username_held_by_another_row_is_a_clear_409_not_a_500():
    db = _UsersTable(_returning_user(email="other@example.test", username=EMAIL))
    with pytest.raises(HTTPException) as raised:
        ensure_user_row(db, NEW_CLERK_ID, email=EMAIL)
    assert raised.value.status_code == 409
    assert raised.value.detail == LINK_CONFLICT_MESSAGE
    assert "users" not in LINK_CONFLICT_MESSAGE and "clerk" not in LINK_CONFLICT_MESSAGE.lower()


# ── relink_user_by_email ─────────────────────────────────────────────────

def test_relink_does_nothing_without_an_email_or_a_matching_row():
    db = _UsersTable(_returning_user())
    assert relink_user_by_email(db, NEW_CLERK_ID, None) is None
    assert relink_user_by_email(db, NEW_CLERK_ID, "") is None
    assert relink_user_by_email(db, NEW_CLERK_ID, "nobody@example.test") is None
    assert db.rows[0]["clerk_user_id"] == OLD_CLERK_ID


def test_relink_is_not_committed_by_itself():
    db = _UsersTable(_returning_user())
    relink_user_by_email(db, NEW_CLERK_ID, EMAIL)
    assert db.commits == 0


# ── the resolver: a returning person lands in the workspace they had ─────

@patch("core.auth.hybrid._provision_new_user_workspace")
@patch("core.auth.hybrid._has_pending_invitations", return_value=True)
@patch("core.auth.hybrid.relink_user_by_email", return_value=7)
@patch("core.auth.hybrid._accessible_workspace_for_clerk_user")
def test_the_resolver_relinks_then_returns_the_workspace_they_had(accessible, relink, pending, provision):
    """Ahead of the pending-invitation gate, as it was on the old Clerk ID."""
    their_workspace = uuid4()
    accessible.side_effect = [None, their_workspace]
    db = MagicMock()

    resolved = _resolve_workspace_for_clerk_user(db, clerk_user_id=NEW_CLERK_ID, org_id=None, email=EMAIL)

    assert resolved == their_workspace
    relink.assert_called_once_with(db, NEW_CLERK_ID, EMAIL)
    db.commit.assert_called_once()
    pending.assert_not_called()
    provision.assert_not_called()


@patch("core.auth.hybrid._provision_new_user_workspace")
@patch("core.auth.hybrid._has_pending_invitations", return_value=False)
@patch("core.auth.hybrid.relink_user_by_email", return_value=None)
@patch("core.auth.hybrid._accessible_workspace_for_clerk_user", return_value=None)
def test_the_resolver_still_provisions_a_genuinely_new_person(accessible, relink, pending, provision):
    new_workspace = uuid4()
    provision.return_value = new_workspace
    db = MagicMock()

    resolved = _resolve_workspace_for_clerk_user(db, clerk_user_id=NEW_CLERK_ID, org_id=None, email=EMAIL, name="New")

    assert resolved == new_workspace
    db.commit.assert_not_called()
    provision.assert_called_once_with(db, NEW_CLERK_ID, email=EMAIL, name="New")
