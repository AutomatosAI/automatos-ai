"""The ``anthropic-workspace-id`` header for an Anthropic key that isn't scoped to a workspace.

An Anthropic API key created inside a Console workspace bills that workspace. A key
created at organization level is not scoped to one, and every request with it must
name the workspace in the ``anthropic-workspace-id`` header, or the API answers 400:
"This API key is not scoped to a workspace, so this request must include the
anthropic-workspace-id header" (Gerard, 9 Oct 2026: such a key failed validation in
Settings → API Keys and could never be used).

The workspace ID (``wrkspc_…``) is now saved with the key (``user_api_keys.
provider_workspace_id``), beside the key's SHA-256 fingerprint (``key_fingerprint``).
Keys reach the Anthropic client as plain strings by many paths (BYOK, the operator
workspace's key, the agent factory, missions, chat), so the header is added where
every path ends: ``workspace_for_key`` looks the workspace up by the key's
fingerprint when the client is built. One indexed row, and no other key is ever
decrypted. The operator's env key (``ANTHROPIC_API_KEY``) takes
``ANTHROPIC_WORKSPACE_ID``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Dict, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

PROVIDER = "anthropic"
WORKSPACE_HEADER = "anthropic-workspace-id"
WORKSPACE_ID_MAX_LENGTH = 64
_WORKSPACE_ID = re.compile(r"^wrkspc_[A-Za-z0-9]{6,57}$")
# A key's workspace rarely changes; a short, bounded cache keeps client construction off the database.
CACHE_SECONDS = 300.0
CACHE_MAX_KEYS = 256
_cache: Dict[str, Tuple[float, Optional[str]]] = {}


class WorkspaceIdRefused(ValueError):
    """The workspace ID can't be used. The message is written for the user."""


def takes_workspace_id(provider: Optional[str]) -> bool:
    """Does a key for this provider carry the workspace it bills?"""
    return (provider or "").lower() == PROVIDER


def clean_workspace_id(provider: Optional[str], raw: Optional[str]) -> Optional[str]:
    """The workspace ID to store with a key; None when blank.

    Raises ``WorkspaceIdRefused`` for a provider that takes none, or an ID that
    isn't an Anthropic workspace ID.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if not takes_workspace_id(provider):
        raise WorkspaceIdRefused(f"A {provider} key takes no workspace ID.")
    if not _WORKSPACE_ID.match(text):
        raise WorkspaceIdRefused(
            "That isn't an Anthropic workspace ID. It starts with wrkspc_ and is shown "
            "on the workspace's page in the Claude Console."
        )
    return text


def workspace_headers(workspace_id: Optional[str]) -> Dict[str, str]:
    """The header that names the workspace, or none."""
    return {WORKSPACE_HEADER: workspace_id} if workspace_id else {}


def key_fingerprint(api_key: str) -> str:
    """The key's SHA-256, hex: a provider key is high-entropy, so this names it without revealing it."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def workspace_for_key(api_key: Optional[str]) -> Optional[str]:
    """The Anthropic workspace saved with ``api_key``, or None when it is scoped already."""
    if not api_key:
        return None
    if config.ANTHROPIC_WORKSPACE_ID and api_key == config.ANTHROPIC_API_KEY:
        return config.ANTHROPIC_WORKSPACE_ID
    fingerprint = key_fingerprint(api_key)
    cached = _cache.get(fingerprint)
    if cached and time.monotonic() - cached[0] < CACHE_SECONDS:
        return cached[1]
    workspace_id = _stored_workspace(fingerprint)
    if len(_cache) >= CACHE_MAX_KEYS:
        _cache.clear()
    _cache[fingerprint] = (time.monotonic(), workspace_id)
    return workspace_id


def _stored_workspace(fingerprint: str) -> Optional[str]:
    """The workspace saved with the Anthropic key whose fingerprint this is, or None."""
    from core.database.database import SessionLocal
    from core.models.core import UserApiKey

    db = None
    try:
        db = SessionLocal()
        row = (
            db.query(UserApiKey.provider_workspace_id)
            .filter(UserApiKey.provider == PROVIDER, UserApiKey.key_fingerprint == fingerprint,
                    UserApiKey.provider_workspace_id.isnot(None))
            .first()
        )
        return row[0] if row else None
    except Exception:
        logger.exception("Anthropic workspace lookup failed; sending the key without a workspace header")
        return None
    finally:
        if db is not None:
            db.close()


def clear_cache() -> None:
    """Forget cached lookups (a key was saved or deleted)."""
    _cache.clear()


__all__ = [
    "WORKSPACE_HEADER", "WORKSPACE_ID_MAX_LENGTH", "WorkspaceIdRefused", "clean_workspace_id", "clear_cache",
    "key_fingerprint", "takes_workspace_id", "workspace_for_key", "workspace_headers",
]
