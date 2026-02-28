"""
Audit Service
=============

Structured audit logging for security-relevant events.

Emits JSON-formatted log records to a dedicated ``audit`` logger so they can
be routed independently of application logs (e.g. to a SIEM, separate file,
or log aggregator) via standard Python logging configuration.

Supported event types (OWASP A09:2021 -- Security Logging and Monitoring):
  - auth.failure        : failed authentication attempts
  - permission.change   : permission grants / revocations
  - credential.access   : credential resolve / decrypt operations
  - admin.action        : admin-level mutations (prompt changes, etc.)
  - setting.change      : system-settings modifications
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Dedicated audit logger -- intentionally NOT __name__ so callers can attach
# handlers to the well-known name "audit" without coupling to module paths.
# ---------------------------------------------------------------------------
_audit_logger = logging.getLogger("audit")


class AuditEventType(str, Enum):
    """Canonical event types for the audit trail."""

    AUTH_FAILURE = "auth.failure"
    PERMISSION_CHANGE = "permission.change"
    CREDENTIAL_ACCESS = "credential.access"
    ADMIN_ACTION = "admin.action"
    SETTING_CHANGE = "setting.change"


# Allowed event type values for fast membership testing.
_VALID_EVENT_TYPES: set[str] = {e.value for e in AuditEventType}


class AuditService:
    """
    Structured audit logger for security-relevant platform events.

    All output goes through Python's ``logging`` module under the logger
    name **audit**, making it trivial to route audit records to a dedicated
    sink (file, stdout, remote collector) without touching application code.

    Usage::

        from orchestrator.core.services.audit_service import audit_service

        await audit_service.log_event(
            event_type="credential.access",
            actor_id="user_abc123",
            resource_type="credential",
            resource_id="cred_xyz",
            workspace_id="ws_1",
            details={"action": "resolve"},
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def log_event(
        self,
        event_type: str,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        workspace_id: str,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a single audit event.

        Parameters
        ----------
        event_type:
            One of the ``AuditEventType`` values (e.g. ``"auth.failure"``).
            Unknown types are logged at WARNING level with an
            ``_warning`` field so they still appear in the audit trail
            rather than being silently dropped.
        actor_id:
            Identifier of the user or service performing the action.
            Use ``"anonymous"`` when the actor is unauthenticated.
        resource_type:
            Logical type of the resource being acted upon
            (e.g. ``"credential"``, ``"system_setting"``, ``"permission"``).
        resource_id:
            Unique identifier of the affected resource.
        workspace_id:
            Workspace scope for multi-tenant isolation.
        details:
            Arbitrary dict of additional context.  Must be
            JSON-serialisable.  The ``action`` key is extracted to
            the top-level ``action`` field when present.
        ip_address:
            Optional source IP of the request.

        Returns
        -------
        dict
            The assembled audit record (useful for testing / chaining).
        """

        details = dict(details) if details else {}

        # Build the canonical audit record.
        record: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor_id": actor_id,
            "action": details.pop("action", event_type.rsplit(".", 1)[-1]),
            "resource_type": resource_type,
            "resource_id": resource_id,
            "workspace_id": workspace_id,
        }

        if ip_address is not None:
            record["ip_address"] = ip_address

        if details:
            record["details"] = details

        # Determine log level.  Security-sensitive events get higher
        # severity so they surface in default logging configurations.
        level = self._level_for(event_type)

        # Warn on unrecognised event types -- but still log them.
        if event_type not in _VALID_EVENT_TYPES:
            record["_warning"] = "unrecognised_event_type"
            level = max(level, logging.WARNING)

        # Emit as a single JSON line for machine parsing.
        _audit_logger.log(level, json.dumps(record, default=str))

        return record

    # ------------------------------------------------------------------
    # Convenience helpers (thin wrappers around log_event)
    # ------------------------------------------------------------------

    async def log_auth_failure(
        self,
        actor_id: str,
        workspace_id: str,
        *,
        ip_address: Optional[str] = None,
        reason: str = "invalid_credentials",
    ) -> Dict[str, Any]:
        return await self.log_event(
            event_type=AuditEventType.AUTH_FAILURE,
            actor_id=actor_id,
            resource_type="auth",
            resource_id="login",
            workspace_id=workspace_id,
            details={"action": "authenticate", "reason": reason},
            ip_address=ip_address,
        )

    async def log_permission_change(
        self,
        actor_id: str,
        resource_id: str,
        workspace_id: str,
        *,
        action: str = "update",
        target_user_id: Optional[str] = None,
        permission: Optional[str] = None,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {"action": action}
        if target_user_id:
            details["target_user_id"] = target_user_id
        if permission:
            details["permission"] = permission
        return await self.log_event(
            event_type=AuditEventType.PERMISSION_CHANGE,
            actor_id=actor_id,
            resource_type="permission",
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details,
        )

    async def log_credential_access(
        self,
        actor_id: str,
        resource_id: str,
        workspace_id: str,
        *,
        action: str = "resolve",
    ) -> Dict[str, Any]:
        return await self.log_event(
            event_type=AuditEventType.CREDENTIAL_ACCESS,
            actor_id=actor_id,
            resource_type="credential",
            resource_id=resource_id,
            workspace_id=workspace_id,
            details={"action": action},
        )

    async def log_admin_action(
        self,
        actor_id: str,
        resource_type: str,
        resource_id: str,
        workspace_id: str,
        *,
        action: str = "update",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {"action": action}
        if extra:
            details.update(extra)
        return await self.log_event(
            event_type=AuditEventType.ADMIN_ACTION,
            actor_id=actor_id,
            resource_type=resource_type,
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details,
        )

    async def log_setting_change(
        self,
        actor_id: str,
        resource_id: str,
        workspace_id: str,
        *,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        is_sensitive: bool = False,
    ) -> Dict[str, Any]:
        details: Dict[str, Any] = {"action": "update"}
        if is_sensitive:
            # Never log raw secret values — record only that a change occurred
            details["old_value_present"] = old_value is not None
            details["new_value_present"] = new_value is not None
        else:
            if old_value is not None:
                details["old_value"] = old_value
            if new_value is not None:
                details["new_value"] = new_value
        return await self.log_event(
            event_type=AuditEventType.SETTING_CHANGE,
            actor_id=actor_id,
            resource_type="system_setting",
            resource_id=resource_id,
            workspace_id=workspace_id,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _level_for(event_type: str) -> int:
        """Map event types to appropriate log levels.

        auth.failure and credential.access are security-sensitive and
        logged at WARNING so they are visible in default configurations
        (OWASP A07:2021 -- Identification and Authentication Failures).
        """
        _levels = {
            AuditEventType.AUTH_FAILURE: logging.WARNING,
            AuditEventType.CREDENTIAL_ACCESS: logging.WARNING,
            AuditEventType.PERMISSION_CHANGE: logging.INFO,
            AuditEventType.ADMIN_ACTION: logging.INFO,
            AuditEventType.SETTING_CHANGE: logging.INFO,
        }
        return _levels.get(event_type, logging.INFO)


# ---------------------------------------------------------------------------
# Module-level singleton -- matches project convention (see monitoring_service)
# ---------------------------------------------------------------------------
audit_service = AuditService()


def get_audit_service() -> AuditService:
    """Return the module-level AuditService singleton."""
    return audit_service
