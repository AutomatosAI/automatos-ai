"""PRD-277 US-402 (§5.2, §5.11): a sign-in's state, nonce and PKCE verifier, kept ten minutes, used once.

They live in Postgres, in ``oidc_pending_sign_ins``, as the device pairing codes do in theirs: a row
holding digests and an expiry, ended by the one call that uses it.

* ``begin`` makes the four values for one sign-in: ``state``, ``nonce`` and the browser ``binding``
  of ``STATE_BYTES`` random bytes each, and a PKCE verifier of ``VERIFIER_BYTES`` (86 characters,
  within RFC 7636's 43 to 128). The binding is the browser's own secret: it goes only into the
  ``__Host-`` sign-in cookie, never into an address, so a state seen in an address or a log is not
  enough to finish the sign-in in another browser. The row keeps only the SHA-256 of the state, the
  nonce and the binding (``core.auth.pairing.hash_secret``), the verifier, where the person returns,
  and the expiry, ``SIGN_IN_TTL_SECONDS`` (ten minutes) on. It returns the state, the nonce, the
  binding and the S256 challenge, once.
* ``begin`` also bounds the table, so callers nobody has authenticated cannot grow it, and so a
  flood costs the flooder rather than everyone else. In the one transaction that writes the new
  row it first deletes every row whose time is up, then:

  - the per-address bound: the row records ``client_hash``, the SHA-256 of the address that
    started it as the rate limits count it (one IPv4 address or one IPv6 /64,
    ``core.security.ip_bucket``; the address itself is never kept). One address holds at most
    ``OIDC_PENDING_SIGN_INS_PER_ADDRESS`` live rows: when it is at that limit, its OLDEST rows are
    deleted to make room. A person who clicks sign-in again and again loses only their own
    older attempts, and one address can never hold more than its share.
  - the ceiling: when the live rows overall are at ``OIDC_PENDING_SIGN_INS_MAX``, the oldest
    rows overall are deleted to make room. A start is never refused for want of room.

  What this does not do: a sign-in whose row was deleted to make room ends as an expired one
  does, and the person sees "sign-in expired, start again". With enough addresses a flood can
  still push real sign-ins out at the ceiling, but only by starting ``OIDC_PENDING_SIGN_INS_MAX``
  sign-ins inside the few seconds a real one takes to come back, and it can never stop a new
  sign-in from starting. Two starts at the same moment may each see room for one, so the table
  can pass the ceiling by the number of starts under way; the next start trims it again.
* ``claim`` ends the sign-in a callback's state and binding name in ONE statement, ``DELETE …
  RETURNING``, committed at once, before the code is exchanged. A callback whose binding is missing
  or another browser's deletes nothing, so the person's own browser can still finish. Two callbacks
  with one state: the second's delete waits on the first's row lock, then finds the row gone.
  Exactly one wins; the other gets ``None``, as an unknown, used or expired state does (an expired
  row is deleted all the same). The digests find the row; the state and the binding are then
  confirmed in constant time (``secrets_match``).

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
from typing import Any, Callable, Optional, Tuple, TypeVar
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.auth.pairing import hash_secret, secrets_match
from core.models.oidc_pending_sign_ins import RETURN_TO_MAX, OidcPendingSignIn
from core.security.ip_bucket import ip_rate_bucket

logger = logging.getLogger(__name__)

T = TypeVar("T")

SIGN_IN_TTL_SECONDS = 600
STATE_BYTES = 32
NONCE_BYTES = 32
VERIFIER_BYTES = 64
BINDING_BYTES = 32
# config.OIDC_PENDING_SIGN_INS_MAX's and _PER_ADDRESS's defaults, for values start-up would have refused.
PENDING_SIGN_INS_MAX_DEFAULT = 10_000
PENDING_SIGN_INS_PER_ADDRESS_DEFAULT = 10
# The bucket name an address is counted under before it is hashed (core.security.ip_bucket).
CLIENT_BUCKET_PREFIX = "oidc_pending"
# A state this server made is 43 URL-safe characters; anything far longer is not one.
STATE_MAX_LENGTH = 256
FAILED_MESSAGE = "The sign-in could not be stored. Try again."

_SWEEP = text("DELETE FROM oidc_pending_sign_ins WHERE expires_at <= :now")
# After the sweep every row is live. Each keeps the newest :keep rows (expires_at is created_at plus
# the one TTL, so newest by expiry is newest by start) and deletes the rest; ix_..._client_hash_expires_at
# serves the first, ix_..._expires_at the second.
_EVICT_ADDRESS = text(
    "DELETE FROM oidc_pending_sign_ins WHERE id IN (SELECT id FROM oidc_pending_sign_ins "
    "WHERE client_hash = :client_hash ORDER BY expires_at DESC, id DESC OFFSET :keep)"
)
_EVICT_OLDEST = text(
    "DELETE FROM oidc_pending_sign_ins WHERE id IN (SELECT id FROM oidc_pending_sign_ins "
    "ORDER BY expires_at DESC, id DESC OFFSET :keep)"
)
_CLAIM = text(
    "DELETE FROM oidc_pending_sign_ins WHERE state_hash = :state_hash AND binding_hash = :binding_hash "
    "RETURNING state_hash, binding_hash, nonce_hash, code_verifier, return_to, expires_at"
)


class PendingSignInFailed(RuntimeError):
    """The sign-in's row could not be written or ended; nothing was kept. Its text never holds a value."""


@dataclass(frozen=True)
class StartedSignIn:
    """A new sign-in's values for the redirect, each in clear exactly once, never in a repr."""

    state: str = field(repr=False)
    nonce: str = field(repr=False)
    binding: str = field(repr=False)
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


def _whole_number_or(value: Any, default: int) -> int:
    usable = isinstance(value, int) and not isinstance(value, bool) and value >= 1
    return value if usable else default


def pending_sign_ins_max() -> int:
    """``OIDC_PENDING_SIGN_INS_MAX``; its default when it is not a whole number of at least one."""
    return _whole_number_or(config.OIDC_PENDING_SIGN_INS_MAX, PENDING_SIGN_INS_MAX_DEFAULT)


def pending_sign_ins_per_address() -> int:
    """``OIDC_PENDING_SIGN_INS_PER_ADDRESS``; its default when it is not a whole number of at least one."""
    return _whole_number_or(config.OIDC_PENDING_SIGN_INS_PER_ADDRESS, PENDING_SIGN_INS_PER_ADDRESS_DEFAULT)


def client_hash(client_address: str) -> str:
    """The SHA-256 of ``client_address``'s rate-limit bucket (one IPv4 address or one IPv6 /64)."""
    return hash_secret(ip_rate_bucket(str(client_address or ""), CLIENT_BUCKET_PREFIX))


def _make_room(db: Session, moment: datetime, client: str) -> Tuple[int, int]:
    """Sweep, then delete this address's oldest rows past its limit less one, then the oldest overall
    past the ceiling less one. The two counts deleted; not committed (the insert commits them)."""
    db.execute(_SWEEP, {"now": moment})
    own = db.execute(_EVICT_ADDRESS, {"client_hash": client, "keep": pending_sign_ins_per_address() - 1})
    oldest = db.execute(_EVICT_OLDEST, {"keep": pending_sign_ins_max() - 1})
    return max(own.rowcount or 0, 0), max(oldest.rowcount or 0, 0)


def _log_room_made(own: int, oldest: int) -> None:
    """Counts only: never an address, a hash or a value."""
    if own:
        logger.info("A sign-in start ended %d older sign-ins of its own address (at most %d per address)",
                    own, pending_sign_ins_per_address())
    if oldest:
        logger.warning("A sign-in start ended the %d oldest sign-ins under way (at most %d at once)",
                       oldest, pending_sign_ins_max())


def begin(db: Session, return_to: str, client_address: str, now: Optional[datetime] = None) -> StartedSignIn:
    """Start one sign-in for ``client_address``: room made, its row written, all in one committed
    transaction; its values returned once."""
    if not isinstance(return_to, str) or not return_to or len(return_to) > RETURN_TO_MAX:
        raise ValueError(f"return_to must be a path of at most {RETURN_TO_MAX} characters")
    moment, client = now or _now(), client_hash(client_address)
    state, nonce = secrets.token_urlsafe(STATE_BYTES), secrets.token_urlsafe(NONCE_BYTES)
    binding, verifier = secrets.token_urlsafe(BINDING_BYTES), secrets.token_urlsafe(VERIFIER_BYTES)
    expires_at = moment + timedelta(seconds=SIGN_IN_TTL_SECONDS)

    def write() -> Tuple[int, int]:
        made = _make_room(db, moment, client)
        db.add(OidcPendingSignIn(id=uuid4(), state_hash=hash_secret(state), nonce_hash=hash_secret(nonce),
                                 binding_hash=hash_secret(binding), client_hash=client, code_verifier=verifier,
                                 return_to=return_to, created_at=moment, expires_at=expires_at))
        db.flush()
        return made

    _log_room_made(*_committed(db, write))
    return StartedSignIn(state=state, nonce=nonce, binding=binding, code_challenge=code_challenge(verifier),
                         expires_at=expires_at)


def _plausible(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= STATE_MAX_LENGTH and value.isascii()


def claim(db: Session, state: Any, binding: Any, now: Optional[datetime] = None) -> Optional[ClaimedSignIn]:
    """End the sign-in ``state`` and this browser's ``binding`` name; what the callback needs, or ``None``.

    ``None`` for a state that is unknown, already used, or older than ``SIGN_IN_TTL_SECONDS``, and
    for a binding that is missing or not the one this sign-in started with. A matched row is
    deleted and committed before this returns, whatever it held; an unmatched binding deletes nothing.
    """
    if not _plausible(state) or not _plausible(binding):
        return None
    found = {"state_hash": hash_secret(state), "binding_hash": hash_secret(binding)}
    row = _committed(db, lambda: db.execute(_CLAIM, found).fetchone())
    if row is None or not (secrets_match(row.state_hash, state) and secrets_match(row.binding_hash, binding)):
        return None
    if (now or _now()) >= _as_utc(row.expires_at):
        return None
    return ClaimedSignIn(nonce_hash=row.nonce_hash, code_verifier=row.code_verifier, return_to=row.return_to)


__all__ = [
    "BINDING_BYTES",
    "NONCE_BYTES",
    "SIGN_IN_TTL_SECONDS",
    "STATE_BYTES",
    "VERIFIER_BYTES",
    "ClaimedSignIn",
    "PendingSignInFailed",
    "StartedSignIn",
    "begin",
    "claim",
    "client_hash",
    "code_challenge",
    "pending_sign_ins_max",
    "pending_sign_ins_per_address",
]
