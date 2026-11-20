"""One way to ask who a person is, and which workspaces they hold (PRD-277 US-202).

Every membership and role read here is keyed on the person's own ``users.id``, so it answers
the same for any way of signing in. A request names its person one of two ways today:

* ``by_user_pk``: the lanes that already carry the ``users.id`` (a paired device, the local
  operator; an Automatos-issued session later);
* ``by_clerk_id``: the Clerk lane. Its Clerk user ID becomes the ``users.id`` inside the same
  statement (``core.auth.clerk_person.CLERK_PERSON_PK_SQL``), so each check makes the round
  trips it always made: no lookup first, then a read.

A statement names its person with ``PERSON_SLOT``. ``person_statement`` fills the slot with one
of the two fixed expressions above, never with a value: every value is a bound parameter.

The rules stay with the callers (``core/auth/workspace_permission.py``,
``core/auth/workspace_admin.py``, ``core/auth/hybrid.py``): which lane may be read at all, the
owner fallback, the roles that qualify. This module only asks the database.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from sqlalchemy import bindparam, text
from sqlalchemy.sql.elements import TextClause

from core.auth.clerk_person import CLERK_ID_PARAM, CLERK_PERSON_PK_SQL

PERSON_SLOT = "{person}"
USER_PK_PARAM = "uid"
USER_PK_SQL = ":" + USER_PK_PARAM
_PERSON_SQL = frozenset({USER_PK_SQL, CLERK_PERSON_PK_SQL})
# Bound as a list: ``IN :roles`` expands to one parameter per role.
ROLES_PARAM = "roles"

_MEMBER_ROLE_SQL = (
    "SELECT wm.role FROM workspace_members wm "
    "WHERE wm.workspace_id = :ws AND wm.user_id = {person} AND wm.is_active = true LIMIT 1"
)
_OWNS_SQL = "SELECT 1 FROM workspaces w WHERE w.id = :ws AND w.owner_id = {person} LIMIT 1"
# The owner of ``:ws``, or an active member of it in one of ``:roles``.
_OWNS_OR_HOLDS_SQL = (
    "SELECT 1 FROM users u "
    "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws "
    "LEFT JOIN workspace_members wm ON wm.user_id = u.id "
    "  AND wm.workspace_id = :ws AND wm.is_active = true "
    "  AND wm.role IN :roles "
    "WHERE u.id = {person} AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
    "LIMIT 1"
)
# The owner of ``:ws_id``, or an active member of it in any role.
_MAY_OPEN_SQL = (
    "SELECT 1 FROM users u "
    "LEFT JOIN workspaces w ON w.owner_id = u.id AND w.id = :ws_id "
    "LEFT JOIN workspace_members wm ON wm.user_id = u.id AND wm.workspace_id = :ws_id AND wm.is_active = true "
    "WHERE u.id = {person} AND (w.id IS NOT NULL OR wm.id IS NOT NULL) "
    "LIMIT 1"
)
# A workspace the person can open (owned, or an active membership), owner-biased: owned or
# owner-role first, then the most recent join, then the newest workspace.
_DEFAULT_WORKSPACE_SQL = (
    "SELECT w.id FROM users u "
    "JOIN workspaces w "
    "  ON w.owner_id = u.id "
    "  OR EXISTS ("
    "    SELECT 1 FROM workspace_members wm "
    "    WHERE wm.workspace_id = w.id "
    "      AND wm.user_id = u.id "
    "      AND wm.is_active = true "
    "  ) "
    "LEFT JOIN workspace_members wm2 "
    "  ON wm2.workspace_id = w.id AND wm2.user_id = u.id AND wm2.is_active = true "
    "WHERE u.id = {person} "
    "  AND w.is_active = true "
    "  AND w.deleted_at IS NULL "
    "ORDER BY "
    "  CASE WHEN w.owner_id = u.id OR COALESCE(wm2.role, '') = 'owner' THEN 0 ELSE 1 END, "
    "  wm2.joined_at DESC NULLS LAST, "
    "  w.created_at DESC "
    "LIMIT 1"
)


# PRD-277 US-501: every usable workspace the person can open (owned, or an active membership),
# with the role the permission gate uses (core/auth/workspace_permission.py ``_role_of``): the
# active member row's role as it is, else ``owner`` for the workspace's owner with no member row.
# In the default workspace's order: owned or owner-role first, then the most recent join, then the
# newest workspace.
_WORKSPACES_OF_SQL = (
    "SELECT w.id, w.name, CASE WHEN wm.id IS NULL THEN 'owner' ELSE wm.role END AS role FROM users u "
    "JOIN workspaces w "
    "  ON w.owner_id = u.id "
    "  OR EXISTS ("
    "    SELECT 1 FROM workspace_members m "
    "    WHERE m.workspace_id = w.id AND m.user_id = u.id AND m.is_active = true "
    "  ) "
    "LEFT JOIN workspace_members wm "
    "  ON wm.workspace_id = w.id AND wm.user_id = u.id AND wm.is_active = true "
    "WHERE u.id = {person} "
    "  AND w.is_active IS NOT FALSE "
    "  AND w.deleted_at IS NULL "
    "ORDER BY "
    "  CASE WHEN w.owner_id = u.id OR COALESCE(wm.role, '') = 'owner' THEN 0 ELSE 1 END, "
    "  wm.joined_at DESC NULLS LAST, "
    "  w.created_at DESC "
    "LIMIT :limit"
)
WORKSPACES_LISTED_MAX = 200


@dataclass(frozen=True)
class HeldWorkspace:
    """A workspace a person can open, and their role in it (None when the member row holds none)."""

    id: str
    name: str
    role: Optional[str]


@dataclass(frozen=True)
class Person:
    """A person as a statement names them: an expression that is their ``users.id``, and the
    one bound value it reads."""

    pk_sql: str
    params: Mapping[str, Any]


def by_user_pk(user_pk: int) -> Person:
    """The person whose ``users.id`` the request already carries."""
    return Person(USER_PK_SQL, MappingProxyType({USER_PK_PARAM: user_pk}))


def by_clerk_id(clerk_user_id: str) -> Person:
    """The person a Clerk user ID names: their ``users.id``, read in the same statement."""
    return Person(CLERK_PERSON_PK_SQL, MappingProxyType({CLERK_ID_PARAM: clerk_user_id}))


@lru_cache(maxsize=None)
def _statement(template: str, pk_sql: str, expanding: Tuple[str, ...]) -> TextClause:
    if pk_sql not in _PERSON_SQL:
        raise ValueError("a person is named by users.id or by Clerk user ID, nothing else")
    clause = text(template.replace(PERSON_SLOT, pk_sql))
    return clause.bindparams(*(bindparam(name, expanding=True) for name in expanding)) if expanding else clause


def person_statement(template: str, person: Person, expanding: Tuple[str, ...] = (),
                     **params: Any) -> Tuple[TextClause, Dict[str, Any]]:
    """``(statement, parameters)`` for ``template`` with its ``{person}`` slot naming ``person``."""
    return _statement(template, person.pk_sql, expanding), {**params, **person.params}


def member_row(db: Any, person: Person, workspace_id: Any) -> Optional[Any]:
    """The person's active ``workspace_members`` row for this workspace (``row[0]`` is its role), or None.

    The row, not just its role: a found row whose role is NULL is still a found row, and the
    caller's rule ends the lookup on it (``workspace_permission._role_of``).
    """
    return db.execute(*person_statement(_MEMBER_ROLE_SQL, person, ws=str(workspace_id))).fetchone()


def owns_workspace(db: Any, person: Person, workspace_id: Any) -> bool:
    """True when ``workspaces.owner_id`` is the person (the legacy owner, with no member row)."""
    return bool(db.execute(*person_statement(_OWNS_SQL, person, ws=str(workspace_id))).fetchone())


def owns_or_holds_role(db: Any, person: Person, workspace_id: Any, roles: Sequence[str]) -> bool:
    """True when the person owns this workspace, or is an active member of it in one of ``roles``."""
    statement, params = person_statement(_OWNS_OR_HOLDS_SQL, person, (ROLES_PARAM,),
                                         ws=str(workspace_id), roles=list(roles))
    return bool(db.execute(statement, params).fetchone())


def may_open_workspace(db: Any, person: Person, workspace_id: Any) -> bool:
    """True when the person owns this workspace or is an active member of it."""
    return bool(db.execute(*person_statement(_MAY_OPEN_SQL, person, ws_id=str(workspace_id))).fetchone())


def default_workspace(db: Any, person: Person) -> Optional[Any]:
    """The workspace a person lands in when the request names none (see ``_DEFAULT_WORKSPACE_SQL``)."""
    row = db.execute(*person_statement(_DEFAULT_WORKSPACE_SQL, person)).fetchone()
    return row[0] if row and row[0] else None


def workspaces_of(db: Any, person: Person, limit: int = WORKSPACES_LISTED_MAX) -> Tuple[HeldWorkspace, ...]:
    """Every usable workspace the person can open, with their role in each (see ``_WORKSPACES_OF_SQL``)."""
    rows = db.execute(*person_statement(_WORKSPACES_OF_SQL, person, limit=limit)).fetchall()
    return tuple(HeldWorkspace(id=str(row[0]), name=row[1] or "", role=row[2]) for row in rows)


__all__ = [
    "PERSON_SLOT",
    "WORKSPACES_LISTED_MAX",
    "HeldWorkspace",
    "Person",
    "by_clerk_id",
    "by_user_pk",
    "default_workspace",
    "may_open_workspace",
    "member_row",
    "owns_or_holds_role",
    "owns_workspace",
    "person_statement",
    "workspaces_of",
]
