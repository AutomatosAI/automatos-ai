"""
API Key Management Service.

Handles creation, validation, revocation, and listing of SDK API keys.
Keys are stored as SHA-256 hashes — the plaintext key is returned exactly
once at creation time and never persisted.
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


def _hash_key(raw_key: str) -> str:
    """Return the hex-encoded SHA-256 digest of *raw_key*."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


class ApiKeyService:
    """Manages SDK API keys with SHA-256 hashing."""

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
        key_hash = _hash_key(key)

        record: Optional[SdkApiKey] = (
            db.query(SdkApiKey)
            .filter(SdkApiKey.key_hash == key_hash)
            .first()
        )

        if record is None:
            return None

        if not record.is_active:
            return None

        now = datetime.now(timezone.utc)
        if record.expires_at is not None and record.expires_at < now:
            return None

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

        for pattern in api_key.allowed_domains:
            if fnmatch(origin, pattern):
                return True

        return False

    # ------------------------------------------------------------------
    # Permission check
    # ------------------------------------------------------------------
    @staticmethod
    def check_permissions(api_key: SdkApiKey, permission: str) -> bool:
        """Check whether *api_key* grants the requested *permission*.

        An empty or ``None`` ``permissions`` list means **all**
        permissions are granted (unrestricted key).
        """
        if not api_key.permissions:
            return True

        return permission in api_key.permissions
