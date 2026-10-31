"""PRD-266 US-011b (§4.4, K8, D16) — who in a workspace may be pushed an event, read at send time.

The fan-out (``fanout.deliver``) calls ``recipients`` when the push is sent, not when the
event happened, so a member removed (or made a viewer) in between receives nothing.

* ``needs_answer`` (a question, an approval, a Mission plan) goes to the people who may answer
  it: owner, admin or editor (``core.auth.workspace_admin.WORKSPACE_ANSWERER_ROLES``, W1
  US-006a's one list; D16). A viewer never.
* ``review`` goes to the people who may move a ticket on: their role holds
  ``missions:update`` (``modules.policy.roles.workspace_has_permission``).
* ``auto_message`` goes to the person Auto wrote to, and only while they still belong to the
  workspace; without a named person, to nobody.
* ``suggestion`` (PRD-268 K9) goes to the person the suggestion is for, by the same rule as
  ``auto_message``: only while they still belong; without a named person, to nobody.

A workspace's people are its active ``workspace_members`` rows plus ``workspaces.owner_id``
(always an owner), read by ``users.id`` in one query; a deactivated account receives
nothing. On a local install the operator
(``config.LOCAL_OPERATOR_EMAIL``'s ``users`` row) owns every workspace, as the local sign-in
treats them. ``core.auth.workspace_permission.resolve_workspace_role`` reads one Clerk caller
from a request; a send-time list has no request, hence this query.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Callable, Dict, Mapping, Optional, Set

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.auth.workspace_admin import WORKSPACE_ANSWERER_ROLES
from modules.policy.roles import WorkspaceRole, workspace_has_permission
from services.push.payload import NEEDS_ANSWER, SUGGESTION

REVIEW = "review"
AUTO_MESSAGE = "auto_message"
REVIEW_PERMISSION = "missions:update"
OWNER_ROLE = WorkspaceRole.OWNER.value
# The push types for the one person they name: without a person they reach nobody.
NAMED_PERSON_TYPES = frozenset({AUTO_MESSAGE, SUGGESTION})

# Everyone who belongs to :ws with the role they hold there, or only :uid when it is given.
# The owner column and the local operator (when :local) count as owners. A deactivated
# account (users.is_active false) is nobody. Bound parameters only.
_PEOPLE_SQL = text(
    "SELECT wm.user_id AS user_id, wm.role AS role FROM workspace_members wm "
    "JOIN users u ON u.id = wm.user_id AND u.is_active IS NOT FALSE "
    "WHERE wm.workspace_id = CAST(:ws AS uuid) AND wm.is_active = true "
    "  AND (CAST(:uid AS integer) IS NULL OR wm.user_id = CAST(:uid AS integer)) "
    "UNION ALL "
    "SELECT w.owner_id, :owner FROM workspaces w "
    "JOIN users u ON u.id = w.owner_id AND u.is_active IS NOT FALSE "
    "WHERE w.id = CAST(:ws AS uuid) "
    "  AND (CAST(:uid AS integer) IS NULL OR w.owner_id = CAST(:uid AS integer)) "
    "UNION ALL "
    "SELECT u.id, :owner FROM users u "
    "WHERE CAST(:local AS boolean) AND u.email = :operator AND u.is_active IS NOT FALSE "
    "  AND (CAST(:uid AS integer) IS NULL OR u.id = CAST(:uid AS integer))"
)


def _answers(role: str) -> bool:
    return role in WORKSPACE_ANSWERER_ROLES


def _reviews(role: str) -> bool:
    return workspace_has_permission(role, REVIEW_PERMISSION)


def _belongs(role: str) -> bool:
    """Any role at all: the person is still in the workspace."""
    return True


# Each push type's rule over one role (the keys are payload.PUSH_TYPE_EVENTS' push types).
ROLE_RULES: Mapping[str, Callable[[str], bool]] = MappingProxyType({
    NEEDS_ANSWER: _answers,
    REVIEW: _reviews,
    AUTO_MESSAGE: _belongs,
    SUGGESTION: _belongs,
})


def _roles_by_person(db: Session, workspace_id: Any, user_id: Optional[int]) -> Dict[int, Set[str]]:
    """Every role each person holds in the workspace (an owner may also hold a member row)."""
    operator = str(config.LOCAL_OPERATOR_EMAIL or "").strip()
    rows = db.execute(_PEOPLE_SQL, {
        "ws": str(workspace_id),
        "uid": user_id,
        "owner": OWNER_ROLE,
        "local": bool(config.IS_LOCAL_EDITION and operator),
        "operator": operator,
    }).fetchall()
    roles: Dict[int, Set[str]] = {}
    for person, role in rows:
        roles.setdefault(int(person), set()).add(str(role or ""))
    return roles


def recipients(db: Session, workspace_id: Any, push_type: str, user_id: Optional[int] = None) -> list[int]:
    """The ``users.id`` of everyone in the workspace this push type may reach, sorted.

    With ``user_id``: that person only, if they still belong and their role allows it.
    ``auto_message`` or ``suggestion`` without a person, and an unknown push type, reach nobody.
    """
    rule = ROLE_RULES.get(push_type)
    if rule is None or (push_type in NAMED_PERSON_TYPES and user_id is None):
        return []
    roles = _roles_by_person(db, workspace_id, user_id)
    return sorted(person for person, held in roles.items() if any(rule(role) for role in held))


__all__ = ["AUTO_MESSAGE", "NAMED_PERSON_TYPES", "REVIEW", "REVIEW_PERMISSION", "ROLE_RULES", "recipients"]
