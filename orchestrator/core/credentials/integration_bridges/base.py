"""
Shared types for credential→execution-platform bridges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from uuid import UUID


@dataclass
class BridgeContext:
    """Everything a bridge needs to wire a saved credential to its execution platform."""

    workspace_id: UUID
    credential_id: int
    credential_type_name: str       # DB row name, e.g. "shopifyAccessTokenApi"
    decrypted_data: Dict[str, Any]  # plaintext field values from the saved credential


@dataclass
class BridgeResult:
    """
    What the bridge tells the caller (and ultimately the frontend) happened.

    status values:
      - "connected"        : platform connection is ACTIVE, ready for tool calls
      - "pending_oauth"    : oauth_redirect_url must be opened by the user to finish
      - "unsupported"      : credential type isn't usable with the platform (informational)
      - "bridge_error"     : bridge raised; error has details
    """

    status: str
    connection_id: Optional[str] = None
    auth_config_id: Optional[str] = None
    auth_scheme: Optional[str] = None
    oauth_redirect_url: Optional[str] = None
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> Dict[str, Any]:
        """Shape for persisting on composio_connections.connection_metadata."""
        return {
            k: v
            for k, v in {
                "auth_config_id": self.auth_config_id,
                "auth_scheme": self.auth_scheme,
                "credential_id": None,  # set by caller — bridges don't know the workspace pk
                "bridge_status": self.status,
            }.items()
            if v is not None
        }
