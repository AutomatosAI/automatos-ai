"""PRD-266 §4.10/§4.11 — the one pairing kernel.

CLI hosts (PRD-234), phones and paired browsers (PRD-266) and, next, PRD-257's
estate runners all pair the same way: an operator reads a short one-time code
off a screen, the device exchanges it for a long-lived token, and the server
keeps only the SHA-256 of either secret. These are the primitives that make
that exchange; each lane keeps its own rows, routes and token length.

Posture, for every caller:

* a code or token is returned in clear exactly once, by the call that mints it;
* only ``hash_secret`` of it is stored, and stored hashes are looked up by
  digest and then compared in constant time (``secrets_match``);
* nothing here logs, reads config or touches the database.

The digest is a plain SHA-256 hex (no salt, no key): every secret it hashes is
server-generated with high entropy, and the digest must be deterministic so a
presented token can be found by an indexed lookup. These are the exact digests
CLI hosts have stored since PRD-234, so existing rows keep resolving.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import Optional, Tuple

# No 0/O/1/I: a code is read off a screen and typed once.
PAIRING_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
PAIRING_CODE_LENGTH = 8
PAIRING_CODE_TTL_SECONDS = 600
# A code is shown as two groups (XXXX-XXXX) so it is easy to read aloud.
_CODE_GROUP_SEPARATOR = "-"


def hash_secret(value: Optional[str]) -> str:
    """The SHA-256 hex digest of ``value`` (``None`` hashes as the empty string)."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def normalize_pairing_code(code: Optional[str]) -> str:
    """A typed code in its canonical form, the form it was issued and hashed in.

    Trimmed, upper-cased, spaces and hyphens removed; a result of exactly
    ``PAIRING_CODE_LENGTH`` characters gets the group separator back after the
    first half. So ``abcdefgh``, ``ABCD EFGH`` and ``abcd-efgh`` all match a code
    issued as ``ABCD-EFGH``, and an issued code is its own canonical form: the
    hashes already stored keep matching. Any other length stays ungrouped and
    matches nothing.
    """
    bare = (code or "").strip().upper().replace(" ", "").replace(_CODE_GROUP_SEPARATOR, "")
    if len(bare) != PAIRING_CODE_LENGTH:
        return bare
    half = PAIRING_CODE_LENGTH // 2
    return f"{bare[:half]}{_CODE_GROUP_SEPARATOR}{bare[half:]}"


def new_pairing_code() -> str:
    """A fresh one-time code: ``PAIRING_CODE_LENGTH`` alphabet characters as ``XXXX-XXXX``."""
    raw = "".join(secrets.choice(PAIRING_ALPHABET) for _ in range(PAIRING_CODE_LENGTH))
    half = PAIRING_CODE_LENGTH // 2
    return f"{raw[:half]}{_CODE_GROUP_SEPARATOR}{raw[half:]}"


def mint_token(prefix: str, nbytes: int) -> Tuple[str, str]:
    """A new token and its digest: ``prefix + token_urlsafe(nbytes)``, ``hash_secret(token)``.

    The prefix names the lane a bearer belongs to (``amd_`` for a paired
    device); the empty prefix is a CLI host's token, unchanged since PRD-234.
    The caller stores the digest and returns the token once.
    """
    token = f"{prefix}{secrets.token_urlsafe(nbytes)}"
    return token, hash_secret(token)


def secrets_match(stored_hash: Optional[str], presented: Optional[str]) -> bool:
    """True iff ``presented`` hashes to ``stored_hash``, compared in constant time.

    An empty or missing value on either side never matches, so a row whose
    hash was cleared (revoked, used) can never be satisfied by an empty bearer.
    """
    if not stored_hash or not presented:
        return False
    return hmac.compare_digest(str(stored_hash).encode("utf-8"), hash_secret(presented).encode("utf-8"))


def within_rotation_overlap(valid_until: Optional[datetime], now: Optional[datetime] = None) -> bool:
    """Is a rotated-out token still inside its overlap window?

    True strictly before ``valid_until``; at or after it the previous token is
    dead (the JWT ``exp`` convention). ``None`` means no overlap was granted. A
    naive datetime (SQLite hands those back for a timezone-aware column) is
    read as UTC.
    """
    if valid_until is None:
        return False
    moment = _as_utc(now or datetime.now(timezone.utc))
    return moment < _as_utc(valid_until)


def _as_utc(value: datetime) -> datetime:
    """``value`` as a UTC-aware datetime (a naive value is taken to be UTC)."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
