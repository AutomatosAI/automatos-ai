"""PRD-277 US-402 (§5.2, §5.11): a sign-in's state, nonce and PKCE verifier, kept ten minutes, used once.

They live in Postgres, in ``oidc_pending_sign_ins``, as the device pairing codes do in theirs: a row
holding digests and an expiry, ended by the one call that uses it.

* ``begin`` makes the three values for one sign-in: ``state`` and ``nonce`` of ``STATE_BYTES`` and
  ``NONCE_BYTES`` random bytes, and a PKCE verifier of ``VERIFIER_BYTES`` (86 characters, within RFC
  7636's 43 to 128). Its row keeps only the SHA-256 of the state and of the nonce
  (``core.auth.pairing.hash_secret``), the verifier, where the person returns, and the expiry,
  ``SIGN_IN_TTL_SECONDS`` (ten minutes) on. It returns the state, the nonce and the S256 challenge,
  once, for the redirect to the provider. Before writing, it deletes up to ``SWEEP_BATCH`` rows whose
  time is up, so sign-ins nobody finished do not pile up. Committed.
* ``claim`` ends the sign-in a callback's state names in ONE statement, ``DELETE … RETURNING``,
  committed at once, before the code is exchanged. Two callbacks with one state: the second's delete
  waits on the first's row lock, then finds the row gone. Exactly one wins; the other gets ``None``,
  as an unknown, used or expired state does (an expired row is deleted all the same). The digest
  finds the row and the state is then confirmed in constant time (``secrets_match``).

Nothing here logs a value. A database error is rolled back and raised as ``PendingSignInFailed``
outside the handler, so the driver's text, which carries the statement's values, is neither its
cause nor its context.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional, TypeVar
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.auth.pairing import hash_secret, secrets_match
from core.models.oidc_pending_sign_ins import RETURN_TO_MAX, OidcPendingSignIn

logger = logging.getLogger(__name__)

T = TypeVar("T")

SIGN_IN_TTL_SECONDS = 600
STATE_BYTES = 32
NONCE_BYTES = 32
VERIFIER_BYTES = 64
SWEEP_BATCH = 100
# A state this server made is 43 URL-safe characters; anything far longer is not one.
STATE_MAX_LENGTH = 256
FAILED_MESSAGE = "The sign-in could not be stored. Try again."

_SWEEP = text(
    "DELETE FROM oidc_pending_sign_ins WHERE id IN ("
    "SELECT id FROM oidc_pending_sign_ins WHERE expires_at <= :now ORDER BY expires_at LIMIT :batch)"
)
_CLAIM = text(
    "DELETE FROM oidc_pending_sign_ins WHERE state_hash = :state_hash "
    "RETURNING state_hash, nonce_hash, code_verifier, return_to, expires_at"
)


class PendingSignInFailed(RuntimeError):
    """The sign-in's row could not be written or ended; nothing was kept. Its text never holds a value."""


@dataclass(frozen=True)
class StartedSignIn:
    """A new sign-in's values for the redirect, each in clear exactly once, never in a repr."""

    state: str = field(repr=False)
    nonce: str = field(repr=False)
    code_challenge: str = field(repr=False)
    expires_at: datetime


@dataclass(frozen=True)
class ClaimedSignIn:
    """What the callback needs of the sign-in it ended."""

    nonce_hash: str = field(repr=False)
    code_verifier: str = field(repr=False)
    return_to: str


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def code_challenge(verifier: str) -> str:
    """RFC 7636 §4.2, S256: BASE64URL(SHA256(verifier)), without padding."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _committed(db: Session, work: Callable[[], T]) -> T:
    """``work()``, committed; on a database error, roll back and raise ``PendingSignInFailed``."""
    failure: Optional[str] = None
    try:
        done = work()
        db.commit()
    except SQLAlchemyError as error:
        failure = type(error).__name__
    if failure is None:
        return done
    db.rollback()
    logger.error("A started sign-in could not be stored or ended (%s)", failure)
    raise PendingSignInFailed(FAILED_MESSAGE)


def begin(db: Session, return_to: str, now: Optional[datetime] = None) -> StartedSignIn:
    """Start one sign-in: its row written and committed, its values returned once."""
    if not isinstance(return_to, str) or not return_to or len(return_to) > RETURN_TO_MAX:
        raise ValueError(f"return_to must be a path of at most {RETURN_TO_MAX} characters")
    moment = now or _now()
    state, nonce = secrets.token_urlsafe(STATE_BYTES), secrets.token_urlsafe(NONCE_BYTES)
    verifier = secrets.token_urlsafe(VERIFIER_BYTES)
    expires_at = moment + timedelta(seconds=SIGN_IN_TTL_SECONDS)

    def write() -> None:
        db.execute(_SWEEP, {"now": moment, "batch": SWEEP_BATCH})
        db.add(OidcPendingSignIn(id=uuid4(), state_hash=hash_secret(state), nonce_hash=hash_secret(nonce),
                                 code_verifier=verifier, return_to=return_to, created_at=moment,
                                 expires_at=expires_at))
        db.flush()

    _committed(db, write)
    return StartedSignIn(state=state, nonce=nonce, code_challenge=code_challenge(verifier), expires_at=expires_at)


def _plausible_state(state: Any) -> bool:
    return isinstance(state, str) and 0 < len(state) <= STATE_MAX_LENGTH and state.isascii()


def claim(db: Session, state: Any, now: Optional[datetime] = None) -> Optional[ClaimedSignIn]:
    """End the sign-in ``state`` names and return what the callback needs, or ``None``.

    ``None`` for a state that is unknown, already used, or older than ``SIGN_IN_TTL_SECONDS``.
    The row is deleted and committed before this returns, whatever it held.
    """
    if not _plausible_state(state):
        return None
    row = _committed(db, lambda: db.execute(_CLAIM, {"state_hash": hash_secret(state)}).fetchone())
    if row is None or not secrets_match(row.state_hash, state):
        return None
    if (now or _now()) >= _as_utc(row.expires_at):
        return None
    return ClaimedSignIn(nonce_hash=row.nonce_hash, code_verifier=row.code_verifier, return_to=row.return_to)


__all__ = [
    "NONCE_BYTES",
    "SIGN_IN_TTL_SECONDS",
    "STATE_BYTES",
    "SWEEP_BATCH",
    "VERIFIER_BYTES",
    "ClaimedSignIn",
    "PendingSignInFailed",
    "StartedSignIn",
    "begin",
    "claim",
    "code_challenge",
]
