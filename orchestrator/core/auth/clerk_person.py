"""Where a Clerk user ID meets a person's own ``users.id`` (PRD-277 US-202).

``users.clerk_user_id`` is Clerk's column. Outside the Clerk modules in ``core/auth/`` and the
sign-in link (``core/auth/user_link.py``) no SQL names it: code that holds only a Clerk user ID
asks this module for the person, and reads everything else by ``users.id``.
``tests/test_prd277_us202_no_clerk_id_sql.py`` names the exempt files one by one and fails when
SQL anywhere else names the column.

Two ways across, both by the person's row:

* a statement that must stay one round trip names the person with
  ``CLERK_PERSON_PK_SQL``, the person's ``users.id`` as a subquery
  (``core.auth.person.by_clerk_id``), or with ``clerk_person_pk_expression`` in an ORM filter;
* code that wants the person first asks ``clerk_person_row`` (a text read) or
  ``clerk_person_query`` (an ORM query of the entity it names).

The way back, ``clerk_id_row`` and ``clerk_id_query``, is for code that must hand Clerk its own
ID (deleting the Clerk user) or match a row recorded under it (a mission's ``created_by``).

Each helper keeps the statement shape its callers had: the same SQL text, or the same
``db.query(entity).filter(...)`` chain. A caller moved onto it makes the reads it always made.
Every read here names at most one person, and the same one, even when two ``users`` rows carry
the same Clerk user ID: the row with the lowest ``users.id`` wins (``ORDER BY id LIMIT 1``). The
model declares ``users.clerk_user_id`` unique, but no migration creates that constraint, so a
database built by migrations may not have it (#1654). A bare scalar subquery over two rows makes
Postgres raise "more than one row returned by a subquery", which would fail every request that
person makes. The lowest id is the row made first: the original account, the one their
workspaces and memberships were made under, while a later duplicate holds nothing. Sign-in
(``core/auth/user_link.py``) picks the same row.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Optional

from sqlalchemy import select, text

# The bound parameter that carries a Clerk user ID in every statement below.
CLERK_ID_PARAM = "cid"
# The person a Clerk user ID names, as their ``users.id``: NULL when no row carries it, so a
# comparison with it matches nothing (a person with no row has no role, as before). Never more
# than one row: the lowest ``users.id`` wins when two rows share the Clerk user ID (see above).
CLERK_PERSON_PK_SQL = "(SELECT id FROM users WHERE clerk_user_id = :cid ORDER BY id LIMIT 1)"

# The same read under each parameter name a caller already binds: ``cid`` everywhere, and
# ``value`` for ``core.auth.principal.resolve_user_pk``, whose parameters a test pins.
PRINCIPAL_PARAM = "value"
_PK_BY_CLERK_ID = MappingProxyType({
    CLERK_ID_PARAM: text("SELECT id FROM users WHERE clerk_user_id = :cid ORDER BY id LIMIT 1"),
    PRINCIPAL_PARAM: text("SELECT id FROM users WHERE clerk_user_id = :value ORDER BY id LIMIT 1"),
})
_CLERK_ID_BY_PK = text("SELECT clerk_user_id FROM users WHERE id = :id")


def clerk_person_row(db: Any, clerk_user_id: Optional[str], param: str = CLERK_ID_PARAM) -> Any:
    """``(users.id,)`` of the person this Clerk user ID names, or None. One text read, the
    Clerk user ID bound as ``param`` (``cid`` or ``value``; any other name is a ``KeyError``)."""
    return db.execute(_PK_BY_CLERK_ID[param], {param: clerk_user_id}).fetchone()


def clerk_person_query(db: Any, entity: Any, clerk_user_id: Optional[str]) -> Any:
    """``db.query(entity)`` narrowed to the person this Clerk user ID names.

    Narrowed by ``users.id`` to ``clerk_person_pk_expression``, so it holds at most one row, the
    same person every other read here names, and keeps the ``query(...).filter(...)`` shape.
    The caller asks for what it needs (``.first()``, ``.scalar()``). The model is imported
    here, at call time, as each caller imported it.
    """
    from core.models.core import User

    return db.query(entity).filter(User.id == clerk_person_pk_expression(clerk_user_id))


def clerk_person_pk_expression(clerk_user_id: str) -> Any:
    """The person's ``users.id`` as a scalar subquery, for an ORM filter keyed on ``users.id``.

    At most one row, the lowest ``users.id``, as ``CLERK_PERSON_PK_SQL``.
    """
    from core.models.core import User

    return (
        select(User.id)
        .where(User.clerk_user_id == clerk_user_id)
        .order_by(User.id)
        .limit(1)
        .scalar_subquery()
    )


def clerk_id_row(db: Any, user_pk: Optional[int]) -> Any:
    """``(clerk_user_id,)`` of this person, or None. One text read."""
    return db.execute(_CLERK_ID_BY_PK, {"id": user_pk}).fetchone()


def clerk_id_query(db: Any, user_pk: Any, *also: Any) -> Any:
    """``db.query(User.clerk_user_id, *also)`` for this person, keyed on ``users.id``."""
    from core.models.core import User

    return db.query(User.clerk_user_id, *also).filter(User.id == user_pk)


__all__ = [
    "CLERK_ID_PARAM",
    "CLERK_PERSON_PK_SQL",
    "PRINCIPAL_PARAM",
    "clerk_id_query",
    "clerk_id_row",
    "clerk_person_pk_expression",
    "clerk_person_query",
    "clerk_person_row",
]
