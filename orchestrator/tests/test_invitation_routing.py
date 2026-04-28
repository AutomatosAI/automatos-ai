"""Regression pins: invitee → inviter's workspace routing.

Bug (2026-04-28): A new user invited to an existing workspace ended up in an
auto-provisioned personal workspace. Two cooperating defects:

1. ``_resolve_workspace_for_clerk_user`` only considered ``owner_id`` — never
   memberships — so members never resolved to their joined workspace.
2. ``_provision_new_user_workspace`` ran silently inside auth resolution,
   beating the explicit ``/accept-invitation`` flow.

Fix: membership-aware resolver + pending-invitation gate around auto-provision.
"""

from __future__ import annotations

import os

# hybrid.py imports SessionLocal at module load, which requires Postgres
# credentials. This test never touches a real DB — seed minimum env vars so
# the import side-effects succeed (matches the pattern in test_prd128_default_prefs.py).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock, patch  # noqa: E402
from uuid import uuid4  # noqa: E402

from core.auth.hybrid import (  # noqa: E402
    _has_pending_invitations,
    _resolve_workspace_for_clerk_user,
)


def _exec_returning(*fetchone_values):
    """Build a db.execute mock that returns the supplied rows in order.

    Each call to ``db.execute(...).fetchone()`` pops the next value off the
    queue. Test ordering matches the SQL call sequence in the resolver.
    """
    queue = list(fetchone_values)

    def _execute(*_args, **_kwargs):
        result = MagicMock()
        result.fetchone.return_value = queue.pop(0) if queue else None
        return result

    db = MagicMock()
    db.execute.side_effect = _execute
    return db


# ── _has_pending_invitations ─────────────────────────────────────────────

def test_pending_invitations_false_when_email_missing():
    db = MagicMock()
    assert _has_pending_invitations(db, None) is False
    assert _has_pending_invitations(db, "") is False
    db.execute.assert_not_called()


def test_pending_invitations_true_when_row_exists():
    db = _exec_returning((1,))
    assert _has_pending_invitations(db, "alice@example.com") is True


def test_pending_invitations_false_when_no_row():
    db = _exec_returning(None)
    assert _has_pending_invitations(db, "alice@example.com") is False


# ── _resolve_workspace_for_clerk_user ────────────────────────────────────

def test_resolver_returns_membership_workspace_when_no_org():
    """The invitee bug: user is a *member* (not owner) of inviter's workspace.

    The new query joins through workspace_members so this user resolves to
    the joined workspace instead of falling through to auto-provision.
    """
    inviter_workspace = uuid4()
    db = _exec_returning((inviter_workspace,))

    resolved = _resolve_workspace_for_clerk_user(
        db,
        clerk_user_id="user_clerk_123",
        org_id=None,
        email="invitee@example.com",
    )

    assert resolved == inviter_workspace


def test_resolver_returns_org_workspace_first():
    """Org workspace short-circuits before any user lookup."""
    org_workspace = uuid4()
    db = _exec_returning((org_workspace,))

    resolved = _resolve_workspace_for_clerk_user(
        db,
        clerk_user_id="user_clerk_123",
        org_id="org_clerk_456",
        email="member@example.com",
    )

    assert resolved == org_workspace
    # Only the org-lookup query should have run
    assert db.execute.call_count == 1


@patch("core.auth.hybrid._provision_new_user_workspace")
def test_resolver_skips_provision_when_pending_invitation(mock_provision):
    """The auto-provision race fix: pending invite blocks silent provisioning."""
    db = _exec_returning(
        None,   # no membership
        (1,),   # pending invitation row exists
    )

    resolved = _resolve_workspace_for_clerk_user(
        db,
        clerk_user_id="user_clerk_new",
        org_id=None,
        email="pending@example.com",
    )

    assert resolved is None
    mock_provision.assert_not_called()


@patch("core.auth.hybrid._provision_new_user_workspace")
def test_resolver_provisions_when_no_invitation(mock_provision):
    """Genuine first-time signups (no invite) still auto-provision."""
    new_workspace = uuid4()
    mock_provision.return_value = new_workspace

    db = _exec_returning(
        None,   # no membership
        None,   # no pending invitation
    )

    resolved = _resolve_workspace_for_clerk_user(
        db,
        clerk_user_id="user_clerk_brandnew",
        org_id=None,
        email="firsttimer@example.com",
        name="First Timer",
    )

    assert resolved == new_workspace
    mock_provision.assert_called_once()


def test_resolver_returns_none_for_anonymous_no_workspace():
    """Without clerk_user_id, resolver short-circuits to None."""
    db = _exec_returning()  # no calls expected

    resolved = _resolve_workspace_for_clerk_user(
        db,
        clerk_user_id=None,
        org_id=None,
        email=None,
    )

    assert resolved is None
    db.execute.assert_not_called()
