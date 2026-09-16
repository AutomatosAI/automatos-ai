"""Resolve the request principal to the integer ``users.id`` primary key.

``UserContext.id`` is NOT the ``users.id`` PK. The Clerk lane binds it to the
Clerk subject string (``user_xxx``) or the email, and the local-operator lane
(PRD-233 S6) binds it to the operator's EMAIL on purpose — every
``clerk_user_id == ctx.user.id`` site in the tree would 500 on an integer.
Any code that hands ``ctx.user.id`` to an INTEGER column (``chats.user_id``,
``User.id == …``) therefore raises ``psycopg2 InvalidTextRepresentation`` — the
2026-07-09 chat outage (#513) and the 2026-09-11 Template Studio "Failed to
fetch" (``GET /api/documents/variables``) are the same bug.

This is the ONE resolver. Lanes that already carry the PK (an int ``id``, or
the local operator's ``raw_claims["user_id"]``) resolve without a query; the
Clerk lane looks the row up by ``clerk_user_id`` and then by email.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import text


# The only lane that stashes the integer PK on ``raw_claims`` — the local
# operator context in ``core/auth/hybrid.py`` (``"source": "local_operator"``).
# The Clerk lane's ``raw_claims`` is the RAW JWT payload; a claim that merely
# happens to be called ``user_id`` there must never be trusted as our PK.
LOCAL_OPERATOR_CLAIM_SOURCE = "local_operator"


def _claimed_pk(user: Any) -> Optional[int]:
    """The integer PK the local-operator lane stashed on ``raw_claims``, else None."""
    claims = getattr(user, "raw_claims", None)
    if not isinstance(claims, dict) or claims.get("source") != LOCAL_OPERATOR_CLAIM_SOURCE:
        return None
    claimed = claims.get("user_id")
    return claimed if isinstance(claimed, int) and not isinstance(claimed, bool) else None


def _lookup(db: Any, column: str, value: str) -> Optional[int]:
    row = db.execute(
        text(f"SELECT id FROM users WHERE {column} = :value LIMIT 1"),  # noqa: S608 — column is a literal below
        {"value": value},
    ).fetchone()
    return int(row[0]) if row else None


def resolve_user_pk(db: Any, ctx: Any) -> Optional[int]:
    """Integer ``users.id`` for the request principal, or ``None``.

    ``None`` means "no resolvable person" (anonymous / system / API-key
    callers, or a principal whose row is missing). Callers that need a default
    user for principal-less paths (chat) add that themselves — a document
    render must NOT silently attribute ``{{user.*}}`` to user 1.
    """
    user = getattr(ctx, "user", None) if ctx is not None else None
    if user is None:
        return None

    uid = getattr(user, "id", None)
    if isinstance(uid, int) and not isinstance(uid, bool):
        return uid

    claimed = _claimed_pk(user)
    if claimed is not None:
        return claimed

    clerk_uid = getattr(user, "clerk_user_id", None) or (uid if isinstance(uid, str) else None)
    if clerk_uid:
        found = _lookup(db, "clerk_user_id", clerk_uid)
        if found is not None:
            return found

    email = getattr(user, "email", None)
    if email:
        return _lookup(db, "email", email)
    return None


__all__ = ["LOCAL_OPERATOR_CLAIM_SOURCE", "resolve_user_pk"]
