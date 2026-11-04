"""Composio rejecting the API key, as one typed error for the catalogue sync.

With a wrong ``COMPOSIO_KEY`` Composio answers every call ``401
APIKey_InvalidAPIKey``. The SDK raises that as ``composio_client.AuthenticationError``
(403: ``PermissionDeniedError``), both carrying ``status_code``; the REST calls see
the status on the response. The client used to log it and return an empty list, so
the sync job ended ``completed`` with 0 apps and the Tools page gave no reason
(4 Nov 2026). The sync path now raises :class:`ComposioAuthError` instead, and the
job fails with :data:`COMPOSIO_KEY_REJECTED_MESSAGE`.
"""
from __future__ import annotations

from typing import Optional

# Shown to the operator (sync job, Tools page, boot log): it names the setting to fix.
COMPOSIO_KEY_REJECTED_MESSAGE = "Composio rejected the API key. Check COMPOSIO_KEY."

# 401: the key is unknown or revoked. 403: the key may not use the API.
KEY_REJECTED_STATUS_CODES = frozenset({401, 403})


class ComposioAuthError(RuntimeError):
    """Composio refused the configured API key (HTTP 401 or 403)."""

    def __init__(self, status_code: int) -> None:
        super().__init__(COMPOSIO_KEY_REJECTED_MESSAGE)
        self.status_code = status_code


def raise_if_key_rejected(status_code: Optional[int]) -> None:
    """Raise :class:`ComposioAuthError` when an HTTP status is Composio refusing the key."""
    if status_code in KEY_REJECTED_STATUS_CODES:
        raise ComposioAuthError(status_code)


def raise_if_sdk_rejected_key(error: BaseException) -> None:
    """The same for an SDK exception: its HTTP errors carry the response's ``status_code``."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in KEY_REJECTED_STATUS_CODES:
        raise ComposioAuthError(status_code) from error
