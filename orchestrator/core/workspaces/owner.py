"""The workspace's owner: the person a call acts for when no person makes it.

An agent's tool call, Auto's chat turn and a playbook step carry no signed-in
user. Whatever they do in a workspace is done for that workspace's owner, the
principal Auto acts for (``handlers_members`` attributes its audit rows the same
way). F344: a document an agent generates prints the owner's name in its
``{{user.name}}`` chip, as it does when the owner generates it.

Every lookup starts from the one workspace row the caller already holds, so the
owner found is that workspace's and never another's.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import UUID

from config import config

OWNER_ROLE = "owner"


def owner_member_user_id(db: Any, workspace_id: UUID) -> Optional[int]:
    """The ``users.id`` of the workspace's active owner member, or ``None``."""
    from core.workspaces.models import WorkspaceMember

    owner = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace_id,
            WorkspaceMember.role == OWNER_ROLE,
            WorkspaceMember.is_active == True,  # noqa: E712
        )
        .first()
    )
    return owner.user_id if owner else None


def workspace_owner(db: Any, workspace: Any) -> Optional[Any]:
    """The owner's ``users`` row for ``workspace`` (a ``Workspace`` row), or ``None``.

    The workspace's ``owner_id``, else its active owner member. The local edition
    seeds neither (one operator, no accounts): there the owner is the operator,
    found by ``config.LOCAL_OPERATOR_EMAIL`` as the local sign-in finds them.
    """
    from core.models.core import User

    if workspace is None:
        return None
    owner_id = getattr(workspace, "owner_id", None) or owner_member_user_id(db, workspace.id)
    if owner_id:
        return db.query(User).filter(User.id == owner_id).first()
    if config.IS_LOCAL_EDITION:
        return db.query(User).filter(User.email == config.LOCAL_OPERATOR_EMAIL).first()
    return None


__all__ = ["OWNER_ROLE", "owner_member_user_id", "workspace_owner"]
