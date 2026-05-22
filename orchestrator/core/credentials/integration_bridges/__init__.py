"""
Integration bridges: glue between the n8n-style Credential store and the
underlying execution platforms (Composio today, others later).

When a workspace owner saves a Credential of a known type (e.g.
shopifyAccessTokenApi), the matching bridge is dispatched to translate the
encrypted credential payload into a working backend connection — for Shopify
that means a Composio connected_account so all 394 SHOPIFY_* tools start
working immediately.

Adding a new bridge:
  1. Create core/credentials/integration_bridges/<integration>.py
  2. Decorate handler(s) with @register("<credential_type_name>")
  3. Return a BridgeResult — None / non-registered types are simply ignored.

The dispatcher is intentionally fail-soft: a bridge raising an exception
must NOT prevent the credential from being saved. We log + return a
"bridge_error" result so the UI can surface the issue.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional

from .base import BridgeContext, BridgeResult

logger = logging.getLogger(__name__)

_BRIDGES: Dict[str, Callable[[BridgeContext], BridgeResult]] = {}


def register(credential_type_name: str):
    """Decorator: register a bridge handler for a Credential type name."""
    def deco(fn: Callable[[BridgeContext], BridgeResult]):
        _BRIDGES[credential_type_name] = fn
        return fn
    return deco


def dispatch(ctx: BridgeContext) -> Optional[BridgeResult]:
    """
    Run the registered bridge for ctx.credential_type_name, if any.

    Returns None when no bridge is registered for the type — credential save
    proceeds as a passive store. Errors raised by bridges are caught and
    surfaced as BridgeResult(status="bridge_error", error=...).
    """
    handler = _BRIDGES.get(ctx.credential_type_name)
    if not handler:
        return None
    try:
        return handler(ctx)
    except Exception as e:  # noqa: BLE001 — bridges must never break save
        logger.exception(
            "Integration bridge raised for %s (workspace=%s, credential=%s)",
            ctx.credential_type_name, ctx.workspace_id, ctx.credential_id,
        )
        return BridgeResult(status="bridge_error", error=str(e))


# Import bridges so their @register decorators fire on module load.
from . import shopify  # noqa: E402,F401
