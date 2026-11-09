"""
API Key Management Service.

Handles creation, validation, revocation, and listing of SDK API keys.
A new key is stored as a PBKDF2-HMAC-SHA256 hash; the plaintext key is
returned exactly once at creation time and never persisted.

CodeQL py/weak-sensitive-data-hashing: a plain SHA-256 of a password-class
secret is too fast to brute-force at scale. A key's own plaintext is never
stored, so an existing SHA-256 hash cannot be rehashed into PBKDF2 — only the
plaintext key owner could do that, by rotating it. ``validate_api_key``
therefore checks the new hash first, then the legacy SHA-256 hash, so a key
created before this change keeps working; ``create_api_key`` only ever
writes the new one.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Any, Optional
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from core.models.sdk_api_keys import SdkApiKey

# Fixed, purpose-bound salt: the same key always gives the same hash (it is looked
# up by it), and a stored hash is slow to test guesses against. A key is
# high-entropy (secrets.token_hex(32)); the cost (tens of ms) is paid once per
# validated request.
_FINGERPRINT_SALT = b"automatos/sdk-api-key/v1"
_FINGERPRINT_ITERATIONS = 100_000


def _hash_key(raw_key: str) -> str:
    """The key's PBKDF2-HMAC-SHA256 hash, hex: the form every new key is stored as."""
    digest = hashlib.pbkdf2_hmac("sha256", raw_key.encode(), _FINGERPRINT_SALT, _FINGERPRINT_ITERATIONS)
    return digest.hex()


def _legacy_hash_key(identity_text: str) -> str:
    """The plain SHA-256 hash: how a key created before the PBKDF2 switch is
    stored. Checked only by ``validate_api_key``, never written by
    ``create_api_key``.

    Not itself a weak password/secret hash: it recomputes a NON-SECRET lookup
    label to find an already-issued key's row, the same role a database index
    plays, not to protect the key's value (the slow PBKDF2 hash in
    ``_hash_key`` is what a brute-force attempt is actually priced against)."""
    return hashlib.sha256(identity_text.encode()).hexdigest()


def key_identity_digest(identity_text: str) -> str:
    """A fast, non-reversible label for ``identity_text``, for callers (the
    widget rate limiter) that need a stable per-key Redis bucket name — not a
    stored credential — and so must not pay PBKDF2's iteration cost on every
    request. Plain SHA-256 is the right tool here: nothing is verified
    against it. Same digest as ``_legacy_hash_key``; named for what it is
    used for."""
    return _legacy_hash_key(identity_text)


def _host_only(value: str) -> str:
    """Reduce an origin or allow-list pattern to its host[:port] form.

    Request origins arrive as bare hosts (``_extract_origin`` returns
    ``parsed.hostname``) while allow-list entries are commonly stored with a
    scheme (``https://*.myshopify.com``) and sometimes a trailing slash. A
    scheme-prefixed pattern can never ``fnmatch`` a bare host, so we strip the
    scheme and trailing slash from both sides before comparing.
    """
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.rstrip("/")


class ApiKeyService:
    """Manages SDK API keys, hashed with PBKDF2-HMAC-SHA256 (legacy SHA-256 keys still validate)."""

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------
    @staticmethod
    def create_api_key(
        db: Session,
        workspace_id: UUID,
        name: str,
        key_type: str = "public",
        permissions: Optional[list[str]] = None,
        allowed_domains: Optional[list[str]] = None,
        allowed_ips: Optional[list[str]] = None,
        rate_limit_requests: Optional[int] = None,
        rate_limit_tokens: Optional[int] = None,
        default_agent_id: Optional[int] = None,
        team: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Create a new API key for a workspace.

        Returns a dict that includes the full plaintext ``key`` — this is
        the **only** time the caller will ever see it.
        """
        secret = secrets.token_hex(32)
        prefix = "ak_pub_" if key_type == "public" else "ak_srv_"
        raw_key = f"{prefix}{secret}"
        key_prefix = raw_key[: len(prefix) + 4]  # e.g. "ak_pub_a1b2"
        key_hash = _hash_key(raw_key)

        record = SdkApiKey(
            id=uuid4(),
            workspace_id=workspace_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            key_type=key_type,
            permissions=permissions,
            allowed_domains=allowed_domains,
            allowed_ips=allowed_ips,
            rate_limit_requests=rate_limit_requests,
            rate_limit_tokens=rate_limit_tokens,
            default_agent_id=default_agent_id,
            team=team,
            expires_at=expires_at,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "id": str(record.id),
            "key": raw_key,
            "key_prefix": key_prefix,
            "name": record.name,
            "key_type": record.key_type,
            "permissions": record.permissions,
            "default_agent_id": record.default_agent_id,
            "team": record.team,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    @staticmethod
    def validate_api_key(db: Session, key: str) -> Optional[SdkApiKey]:
        """Validate a raw API key and return the matching record.

        Returns ``None`` when the key is unknown, revoked, or expired.
        On success the ``last_used_at`` timestamp is bumped.
        """
        # Checks both forms: a key created before the PBKDF2 switch is still
        # stored under its SHA-256 hash and must keep validating.
        candidate_hashes = (_hash_key(key), _legacy_hash_key(key))

        record: Optional[SdkApiKey] = (
            db.query(SdkApiKey)
            .filter(SdkApiKey.key_hash.in_(candidate_hashes))
            .first()
        )

        if record is None:
            return None

        if not record.is_active:
            return None

        now = datetime.now(timezone.utc)
        if record.expires_at is not None and record.expires_at < now:
            return None

        if record.key_hash != candidate_hashes[0]:
            # Lazy migration: this row still carries the legacy SHA-256 hash.
            # Upgrade it to PBKDF2 now that the plaintext key is in hand — the
            # same no-mass-reset approach a password-hash algorithm change
            # uses. Over time, fewer keys need the legacy check at all.
            record.key_hash = candidate_hashes[0]

        record.last_used_at = now
        db.commit()

        return record

    # ------------------------------------------------------------------
    # Revoke
    # ------------------------------------------------------------------
    @staticmethod
    def revoke_api_key(
        db: Session,
        key_id: UUID,
        workspace_id: UUID,
    ) -> bool:
        """Revoke an API key by setting ``is_active`` to False.

        The key must belong to the given *workspace_id* (workspace-scoped
        security).  Returns ``True`` on success, ``False`` if the key was
        not found or does not belong to the workspace.
        """
        record: Optional[SdkApiKey] = (
            db.query(SdkApiKey)
            .filter(
                SdkApiKey.id == key_id,
                SdkApiKey.workspace_id == workspace_id,
            )
            .first()
        )

        if record is None:
            return False

        record.is_active = False
        db.commit()

        return True

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    @staticmethod
    def list_api_keys(
        db: Session,
        workspace_id: UUID,
    ) -> list[dict[str, Any]]:
        """Return all API keys for a workspace.

        Keys are returned with only the ``key_prefix`` (first 8 chars)
        for display — the full key is never stored or returned here.
        """
        records = (
            db.query(SdkApiKey)
            .filter(SdkApiKey.workspace_id == workspace_id)
            .order_by(SdkApiKey.created_at.desc())
            .all()
        )

        return [
            {
                "id": str(r.id),
                "name": r.name,
                "key_prefix": f"{r.key_prefix}...",
                "key_type": r.key_type,
                "permissions": r.permissions,
                "default_agent_id": r.default_agent_id,
                "team": getattr(r, "team", None),
                "is_active": r.is_active,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                "last_used_at": r.last_used_at.isoformat() if r.last_used_at else None,
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Domain check
    # ------------------------------------------------------------------
    @staticmethod
    def check_domain(api_key: SdkApiKey, origin: str) -> bool:
        """Check whether *origin* is allowed by the key's domain list.

        An empty or ``None`` ``allowed_domains`` list means **all**
        origins are permitted.  Wildcard patterns (e.g. ``*.example.com``)
        are supported via :func:`fnmatch`.
        """
        if not api_key.allowed_domains:
            return True

        candidate = _host_only(origin)
        for pattern in api_key.allowed_domains:
            if fnmatch(candidate, _host_only(pattern)):
                return True

        return False

    @staticmethod
    def origin_allowed_by_any_key(db: Session, origin: str) -> bool:
        """True when *origin* matches the ``allowed_domains`` of ANY active
        public key (widget CORS preflight fallback, PRD-TUTOR-LIVE S0).

        A preflight carries no Authorization header, so the CORS middleware
        cannot resolve which key the actual request will present — it can
        only ask whether the origin is explicitly allowlisted by *some*
        active public key, via the same :meth:`check_domain` matcher the
        request-time check uses. Keys with an EMPTY ``allowed_domains`` (the
        "any origin" opt-in honoured by :meth:`check_domain`) are
        deliberately excluded here: counting them would reflect CORS headers
        for every origin on earth the moment one unrestricted key exists.
        A green preflight grants nothing by itself — ``widget_auth`` still
        enforces the presented key's own allowlist on the request proper.
        """
        keys = (
            db.query(SdkApiKey)
            .filter(
                SdkApiKey.is_active.is_(True),
                SdkApiKey.key_type == "public",
                SdkApiKey.allowed_domains.isnot(None),
            )
            .all()
        )
        return any(
            key.allowed_domains and ApiKeyService.check_domain(key, origin)
            for key in keys
        )

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------
    @staticmethod
    def check_permissions(api_key: SdkApiKey, permission: str) -> bool:
        """Check whether *api_key* grants the requested *permission*.

        PRD-195 S1 (P2-14 / F042): delegates to the one authority —
        ``modules.policy.roles.has_permission``. **Empty or ``None``
        permissions grant NOTHING** (least privilege); an explicit ``"*"``
        element is honoured as a deliberate full grant. The legacy
        "empty = unrestricted" god-key semantic is deleted — all three key
        checkers (service, widget, board) now share this rule.
        """
        from modules.policy.roles import has_permission

        return has_permission(api_key.permissions, permission)
