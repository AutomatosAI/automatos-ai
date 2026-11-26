"""Find or create the ``users`` row behind a sign-in, never by moving someone else's.

Two ways in, one rule. A Clerk user is found by ``clerk_user_id`` (``ensure_user_row``). A
person who signs in through an organisation's own identity provider is found by the
``(issuer, subject)`` pair in ``user_identities`` (``link_identity``, PRD-277 US-201).
Neither ever matches a person on email.

``users`` rows are found by ``clerk_user_id``. ``users.email`` is unique, and so is
``users.username`` (provisioning sets it to the email). Provisioning used to insert
with ``ON CONFLICT (clerk_user_id) DO NOTHING``, swallow the email
``IntegrityError``, read the row back by the new Clerk ID, find nothing and crash
on ``uid_row[0]``: a 500 for anyone whose email already had a row under another
Clerk ID (every returning user after the Clerk production cutover).

No request path ever attaches an existing row to a different identity: not
sign-in, not accepting an invitation, and not a row that has no Clerk ID yet. The
email in a caller's token is not proof that they own a row with that email. A row
is found by Clerk ID or a new one is inserted; when the email or username is
already held by another row, the caller gets a clear 409, never a 500, and no row
changes (on the identity path, ``IdentityLinkRefused``). Returning users are moved
to their production Clerk IDs once, by the cutover's operator step
(``scripts/clerk_cutover_relink.py``, ``core.auth.clerk_cutover``), which matches on
the import's own ID link and a verified email and records every move.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from core.models.user_identities import ISSUER_MAX, SUBJECT_MAX

logger = logging.getLogger(__name__)

# The email a users row gets when the Clerk token carried none.
PENDING_EMAIL_TEMPLATE = "{clerk_user_id}@pending"

LINK_CONFLICT_MESSAGE = (
    "This sign-in matches an existing account that is not linked to it. "
    "Contact Automatos support."
)

# PRD-277 US-201: what a person whose identity cannot be linked is told. It names no issuer,
# subject or email: the detail goes to the server log as a users.id.
IDENTITY_LINK_REFUSED_MESSAGE = (
    "This sign-in matches an existing account that is not linked to it. "
    "Ask your administrator to resolve it."
)
# The refusal's reason, for the sign-in audit row (PRD-277 §5.12, auth:sign_in_refused).
REFUSED_EMAIL_CONFLICT = "email_conflict"
# The person is already linked under the token's ``sub``, and the subject is now read from another claim.
REFUSED_SUBJECT_CLAIM_CHANGED = "subject_claim_changed"

# The email and username a person gets when the identity provider sent no email: one per
# (issuer, subject), so two first sign-ins for the same pair meet on the same unique value.
PENDING_IDENTITY_EMAIL_TEMPLATE = "identity-{digest}@pending"
# The domain every placeholder email ends with (``PENDING_EMAIL_TEMPLATE`` too). An email from a
# provider in this domain is no email at all, so no identity can hold another's placeholder.
PENDING_EMAIL_DOMAIN = "@pending"
PENDING_IDENTITY_DIGEST_LENGTH = 32
# users.email, users.username and users.name are VARCHAR(255).
PROFILE_VALUE_MAX = 255
# Postgres's SQLSTATE for a row a unique index refuses.
UNIQUE_VIOLATION = "23505"


def user_id_for_clerk_id(db, clerk_user_id: str) -> Optional[int]:
    """The ``users.id`` linked to this Clerk user ID, or None: the lowest one, should two rows
    carry it, as ``core.auth.clerk_person`` reads it."""
    row = db.execute(
        text("SELECT id FROM users WHERE clerk_user_id = :cid ORDER BY id LIMIT 1"),
        {"cid": clerk_user_id},
    ).fetchone()
    return row[0] if row else None


def _row_with_email(db, email: str):
    """``(id, clerk_user_id)`` of the oldest row with this email, case-insensitive, or None."""
    return db.execute(
        text(
            "SELECT id, clerk_user_id FROM users "
            "WHERE LOWER(email) = LOWER(:email) ORDER BY id LIMIT 1"
        ),
        {"email": email},
    ).fetchone()


def _link_conflict(clerk_user_id: str, reason: str) -> HTTPException:
    logger.error("No users row for clerk_user_id=%s: %s", clerk_user_id, reason)
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=LINK_CONFLICT_MESSAGE)


def ensure_user_row(
    db, clerk_user_id: str, email: Optional[str] = None, name: Optional[str] = None,
) -> int:
    """The ``users.id`` for this Clerk user: found by Clerk ID, or inserted.

    A row that already has the email (case-insensitive: the unique index is not,
    so ``Jane@Example.com`` would otherwise get a second row) belongs to someone
    else, or to a returning user the cutover step has not relinked yet, whether or
    not it has a Clerk ID: the caller gets a 409 with a clear message, never a 500,
    and no row changes. Idempotent, and safe when two first requests race: the
    insert skips on ANY unique conflict (Clerk ID, email or username) and the row
    is read back by Clerk ID; nothing found then (another row holds the username)
    is the same 409.
    """
    existing = user_id_for_clerk_id(db, clerk_user_id)
    if existing is not None:
        return existing
    if email:
        holder = _row_with_email(db, email)
        if holder:
            raise _link_conflict(clerk_user_id, f"users.id={holder[0]} already has its email")
    db.execute(
        text(
            "INSERT INTO users (username, email, clerk_user_id, name, is_active) "
            "VALUES (:username, :email, :cid, :name, true) "
            "ON CONFLICT DO NOTHING"
        ),
        {
            "username": email or clerk_user_id,
            "email": email or PENDING_EMAIL_TEMPLATE.format(clerk_user_id=clerk_user_id),
            "cid": clerk_user_id,
            "name": name,
        },
    )
    created = user_id_for_clerk_id(db, clerk_user_id)
    if created is not None:
        return created
    raise _link_conflict(clerk_user_id, "another row holds its email or username")


def user_id_for_invitee(
    db, clerk_user_id: str, email: Optional[str], name: Optional[str] = None,
) -> int:
    """The ``users.id`` that accepts an invitation for this Clerk user: ``ensure_user_row``.

    Found by Clerk ID, else a new row is inserted. A row that already has the
    email (case-insensitive) is never attached to this Clerk ID, even one with no
    Clerk ID yet: the invitee gets the sign-in path's 409 and nothing moves. Not
    committed: the caller commits with the membership.
    """
    return ensure_user_row(db, clerk_user_id, email=email, name=name)


# --------------------------------------------------------------------------- (issuer, subject)


class IdentityLinkRefused(Exception):
    """A sign-in that could only be linked to a person by taking another person's row.

    ``message`` is safe to show the person: it names no issuer, subject or email. ``reason``
    is a short code for the audit row (``REFUSED_EMAIL_CONFLICT``). No row was changed.
    """

    def __init__(self, reason: str, message: str = IDENTITY_LINK_REFUSED_MESSAGE) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class LinkedPerson:
    """The person an ``(issuer, subject)`` pair is linked to; ``created`` when this call made them."""

    user_id: int
    created: bool


def _check_identity(issuer: str, subject: str) -> None:
    """Refuse an issuer or subject that is not a non-empty string ``user_identities`` can hold.

    Nothing is trimmed, folded or shortened: a value is stored and compared exactly as given.
    The message never repeats the value.
    """
    for label, value, limit in (("issuer", issuer, ISSUER_MAX), ("subject", subject, SUBJECT_MAX)):
        if not isinstance(value, str) or not value:
            raise ValueError(f"an identity's {label} must be a non-empty string")
        if "\x00" in value or len(value.encode("utf-8")) > limit:
            raise ValueError(f"an identity's {label} is longer than {limit} bytes or holds a NUL")


def _check_profile(email: Optional[str], name: Optional[str]) -> None:
    """Refuse an email or name that is not a string a ``users`` row can hold."""
    for label, value in (("email", email), ("name", name)):
        if value is None:
            continue
        if not isinstance(value, str) or "\x00" in value or len(value) > PROFILE_VALUE_MAX:
            raise ValueError(f"a person's {label} must be a string of at most {PROFILE_VALUE_MAX} characters")


def _usable_email(email: Optional[str]) -> Optional[str]:
    """The provider's email as the new person's profile, or None when it is not an email.

    Taken as given, never trimmed or folded. Blank (empty or only whitespace) or in the
    placeholder domain is None. Leading or trailing whitespace is refused: a verified claim
    never carries it, so it is a bug in the caller.
    """
    if email is None or not email.strip():
        return None
    if email != email.strip():
        raise ValueError("a person's email must not start or end with whitespace")
    if email.lower().endswith(PENDING_EMAIL_DOMAIN):
        return None
    return email


def _check_email_verified(email_verified: bool) -> None:
    """``email_verified`` is the provider's own yes or no: a ``bool``, never a string or None."""
    if not isinstance(email_verified, bool):
        raise TypeError("email_verified must be True or False, as the provider's verified token states it")


def _person_email(email: Optional[str], email_verified: bool) -> Optional[str]:
    """The address the new person gets: the provider's email only when the provider verified it.

    An unverified email is kept nowhere, so it can never hold an address its owner signs in
    with later. Padded emails are refused whether verified or not (``_usable_email``).
    """
    usable = _usable_email(email)
    return usable if email_verified else None


def _is_unique_violation(error: IntegrityError) -> bool:
    """Whether a unique index refused the write (psycopg2's ``pgcode``, psycopg's ``sqlstate``)."""
    orig = error.orig
    return (getattr(orig, "pgcode", None) or getattr(orig, "sqlstate", None)) == UNIQUE_VIOLATION


def _pending_identity_email(issuer: str, subject: str) -> str:
    """The placeholder email for a pair whose provider sent none; the same pair, the same value."""
    pair = f"{len(issuer)}:{issuer}{subject}".encode("utf-8")
    digest = hashlib.sha256(pair).hexdigest()[:PENDING_IDENTITY_DIGEST_LENGTH]
    return PENDING_IDENTITY_EMAIL_TEMPLATE.format(digest=digest)


def _identity_refused(reason: str, detail: str) -> IdentityLinkRefused:
    logger.warning("Identity link refused (%s): %s", reason, detail)
    return IdentityLinkRefused(reason)


def _linked_or_refused(db, issuer: str, subject: str, detail: str) -> LinkedPerson:
    """The person the pair is linked to by now (a request for it committed meanwhile), or refuse.

    Read again before any refusal, so the same person's second request is never told that their
    own row is someone else's.
    """
    linked = user_id_for_identity(db, issuer, subject)
    if linked is not None:
        return LinkedPerson(user_id=linked, created=False)
    raise _identity_refused(REFUSED_EMAIL_CONFLICT, detail)


def refuse_if_linked_before(db, *, issuer: str, subject: str, linked_before_as: str) -> None:
    """``IdentityLinkRefused`` (``REFUSED_SUBJECT_CLAIM_CHANGED``) for a first sign-in of a person
    already linked as ``(issuer, linked_before_as)``; nothing when ``(issuer, subject)`` is linked.

    ``linked_before_as`` is the token's ``sub``, given only when the subject is read from another
    claim (``OIDC_CLAIM_SUBJECT``): a person linked under it signed in before that setting changed,
    so this sign-in must not make them a second person. Logged at error, naming the setting and no
    value. Only a change away from ``sub`` is seen: a change back to ``sub``, or between two other
    claims, is not. Not called at all when the subject is ``sub``, so that runs no query for it.
    """
    if user_id_for_identity(db, issuer, subject) is not None:
        return
    if user_id_for_identity(db, issuer, linked_before_as) is None:
        return
    logger.error("Identity link refused (%s): this person is linked under the token's sub, and "
                 "OIDC_CLAIM_SUBJECT now names another claim", REFUSED_SUBJECT_CLAIM_CHANGED)
    raise IdentityLinkRefused(REFUSED_SUBJECT_CLAIM_CHANGED)


def user_id_for_identity(db, issuer: str, subject: str) -> Optional[int]:
    """The ``users.id`` linked to exactly this ``(issuer, subject)``, or None.

    Compared exactly, as stored: no case folding, no trimming, no prefix match.
    """
    _check_identity(issuer, subject)
    row = db.execute(
        text("SELECT user_id FROM user_identities WHERE issuer = :issuer AND subject = :subject"),
        {"issuer": issuer, "subject": subject},
    ).fetchone()
    return int(row[0]) if row else None


def _insert_person_with_link(
    db, issuer: str, subject: str, email: Optional[str], name: Optional[str],
) -> Optional[int]:
    """Insert a new person and their link in one savepoint; None when a unique index refused it.

    On a refusal the savepoint is rolled back, so neither row is left behind and the caller's
    transaction is still usable. Any other integrity error is raised.
    """
    held = email or _pending_identity_email(issuer, subject)
    try:
        with db.begin_nested():
            user_id = db.execute(
                text(
                    "INSERT INTO users (username, email, name, is_active, created_at, updated_at) "
                    "VALUES (:username, :email, :name, true, now(), now()) RETURNING id"
                ),
                {"username": held, "email": held, "name": name},
            ).scalar_one()
            db.execute(
                text("INSERT INTO user_identities (user_id, issuer, subject) VALUES (:uid, :issuer, :subject)"),
                {"uid": user_id, "issuer": issuer, "subject": subject},
            )
    except IntegrityError as refused:
        if not _is_unique_violation(refused):
            raise
        logger.info("Identity link: a unique index refused the new person; reading the link again")
        return None
    return int(user_id)


def link_identity(
    db, *, issuer: str, subject: str, email_verified: bool,
    email: Optional[str] = None, name: Optional[str] = None,
) -> LinkedPerson:
    """The person signed in as ``(issuer, subject)``: found by that exact pair, or created.

    ``issuer`` and ``subject`` must already be verified: checking the ID token they came from
    (its signature, issuer, audience, times and nonce) is the caller's job (PRD-277 Wave 4).
    They are the only identifier. ``email`` and ``name`` are only the new person's profile:
    a person is never found by email, and a repeat sign-in changes no row, whatever its email.

    ``email_verified`` is required, with no default, so every caller states it: it is the
    provider's own statement that ``email`` is verified, which the caller takes from the
    verified ID token (``email_verified``). The email becomes the person's address, and is
    checked against other people's addresses, only when it is True. When it is False, or there
    is no email, the person gets the per-identity placeholder, no address check runs, and
    nothing of the unverified email is kept. A value that is not a ``bool`` is a ``TypeError``
    before any query.

    A new person gets a ``users`` row and a link, and nothing more: no workspace, no
    membership, no role (``system_role`` stays NULL). Never a token. A linked person is
    returned whatever their ``is_active``: refusing a deactivated person is the sign-in's check.

    An email that is blank or in the placeholder domain (``PENDING_EMAIL_DOMAIN``) counts as no
    email; one with leading or trailing whitespace is a ``ValueError`` before any query.

    When another row already holds the email (case-insensitive) or the username, the pair is
    read again: a link made meanwhile (the same person's other request committed) is returned,
    so a person is never refused because of their own row; otherwise this raises
    ``IdentityLinkRefused`` and no row changes. Two first sign-ins for the same pair racing:
    the unique indexes decide. Both find no link; the first to insert holds the unique keys
    until it commits; the second's insert waits on them, then is refused, its savepoint rolls
    back, and it reads the first's committed link and returns that person (``created`` False).
    One person, one link, no error. Not committed: the caller commits with what it writes next.
    """
    _check_identity(issuer, subject)
    _check_email_verified(email_verified)
    _check_profile(email, name)
    address = _person_email(email, email_verified)
    existing = user_id_for_identity(db, issuer, subject)
    if existing is not None:
        return LinkedPerson(user_id=existing, created=False)
    if address:
        holder = _row_with_email(db, address)
        if holder:
            return _linked_or_refused(db, issuer, subject, f"users.id={holder[0]} already has its email")
    created = _insert_person_with_link(db, issuer, subject, address, name)
    if created is not None:
        return LinkedPerson(user_id=created, created=True)
    return _linked_or_refused(db, issuer, subject, "another row holds its email or username")
