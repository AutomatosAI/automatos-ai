"""F152 — an invitation never makes a second owner.

A workspace has one owner. invite_member_to_workspace, which the REST invite
route and the platform_invite_member tool share, refuses the owner role in any
case or spacing and accepts only admin, editor or viewer. Accepting an
invitation checks its stored role again, and a role change never makes an
owner either.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

REFUSED = "Nobody is invited as the owner: invite them as an admin, editor or viewer."


def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


@pytest.fixture
def cafe(db_session, seed_workspace):
    db = db_session
    ws = UUID(seed_workspace())
    owner, admin = _user(db, "gerard"), _user(db, "priya")
    for user, role in ((owner, "owner"), (admin, "admin")):
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})
    return NS(db=db, ws=ws, owner=owner, admin=admin)


def _invite_over_rest(cafe, role):
    """The admin calls POST /api/workspaces/{id}/team/invite (members:invite)."""
    import api.team as team

    request = team.InviteMemberRequest(email=f"cofounder-{uuid4().hex[:6]}@example.com", role=role)
    clerk = NS(create_user_invitation=AsyncMock(return_value={"id": "inv_test"}))
    with patch.object(team, "_resolve_internal_user_id", return_value=cafe.admin), \
            patch("core.auth.clerk.get_clerk_auth", return_value=clerk):
        return asyncio.run(team.invite_member(workspace_id=str(cafe.ws), request=request,
                                              ctx=NS(user=NS(email="priya@harbourline.test")), db=cafe.db))


def _invitations(cafe, role):
    return cafe.db.execute(text("SELECT count(*) FROM workspace_invitations "
                                "WHERE workspace_id = CAST(:ws AS uuid) AND lower(trim(role)) = :role"),
                           {"ws": str(cafe.ws), "role": role}).scalar()


@pytest.mark.parametrize("role", ["owner", " Owner "])
def test_an_admin_cannot_invite_an_owner_over_rest(cafe, role):
    with pytest.raises(HTTPException) as refused:
        _invite_over_rest(cafe, role)
    assert (refused.value.status_code, refused.value.detail) == (400, REFUSED)
    assert _invitations(cafe, "owner") == 0


def test_an_admin_invites_an_editor_over_rest(cafe):
    assert _invite_over_rest(cafe, "editor").role == "editor"
    assert _invitations(cafe, "editor") == 1


def test_the_chat_tool_is_refused_the_same_way(cafe):
    from modules.tools.discovery.handlers_members import invite_member

    reply = asyncio.run(invite_member(cafe.db, cafe.ws, {"email": "cofounder@example.com", "role": "owner",
                                                         "_driving_user_id": cafe.owner}))
    assert reply == {"success": False, "error": REFUSED}
    assert _invitations(cafe, "owner") == 0


def _owners(cafe):
    return cafe.db.execute(text("SELECT count(*) FROM workspace_members WHERE workspace_id = CAST(:ws AS uuid) "
                                "AND role = 'owner' AND is_active"), {"ws": str(cafe.ws)}).scalar()


def test_a_role_change_never_makes_an_owner(cafe):
    import api.team as team_api

    editor = _user(cafe.db, "sam")
    row = cafe.db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                               "VALUES (CAST(:ws AS uuid), :user, 'editor', TRUE) RETURNING id"),
                          {"ws": str(cafe.ws), "user": editor}).scalar()
    with patch.object(team_api, "_resolve_internal_user_id", return_value=cafe.owner), \
            pytest.raises(HTTPException) as refused:
        asyncio.run(team_api.update_member_role(workspace_id=str(cafe.ws), member_id=row, ctx=NS(), db=cafe.db,
                                                request=team_api.UpdateMemberRoleRequest(role="owner")))
    assert (refused.value.status_code, refused.value.detail) == (
        400, "A workspace has one owner: make them an admin, editor or viewer.")
    assert _owners(cafe) == 1


def test_an_owner_invitation_stored_before_the_fix_is_not_accepted(cafe):
    import api.team as team_api
    from core.workspaces.invitations import WorkspaceInvitation

    invitation = WorkspaceInvitation(workspace_id=cafe.ws, email="cofounder@example.com", role="owner",
                                     token=uuid4().hex, invited_by=cafe.admin,
                                     expires_at=datetime.utcnow() + timedelta(days=7))
    cafe.db.add(invitation)
    cafe.db.flush()
    clerk_user = {"clerk_user_id": f"user_{uuid4().hex[:8]}", "email": "cofounder@example.com"}
    with patch.object(team_api, "_verify_clerk_user_only", return_value=clerk_user), \
            pytest.raises(HTTPException) as refused:
        asyncio.run(team_api.accept_invitation(payload=team_api.AcceptInvitationRequest(token=invitation.token),
                                               request=NS(), db=cafe.db))
    assert refused.value.status_code == 400
    assert _owners(cafe) == 1
