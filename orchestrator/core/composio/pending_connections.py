"""A pending connection settles from Composio's own record of the attempt (F258).

Connect makes a Composio sign-in link and leaves the workspace's row ``pending``.
Three listings settle such a row: ``GET /api/tools/connected`` (the Tools page),
``POST /api/tools/refresh-connections`` and ``GET /api/composio/connections``.
On 3 Oct 2026 FAL_AI and KIEAI stayed ``pending`` for good: Composio had marked
both attempts EXPIRED ("Authorization was started but not completed within 10
minutes"), and the listings only looked for an ACTIVE or INITIATED account, so an
ended attempt read as "not yet". Two of them also counted INITIATED as connected,
but Composio's INITIATED means the user hasn't finished, and only ACTIVE runs tools.

Each listing now settles the row from the app's attempts: an ACTIVE account means
``active``; otherwise a newest attempt that EXPIRED, FAILED or was made INACTIVE
means ``added`` (in the workspace, not connected, so Connect is offered again);
anything else, or no attempt yet, stays ``pending``.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)

PENDING = "pending"
CONNECTED = "active"
NOT_CONNECTED = "added"

# Composio's connected-account statuses (docs.composio.dev → Connected Accounts).
COMPOSIO_CONNECTED = "ACTIVE"
COMPOSIO_ENDED = frozenset({"EXPIRED", "FAILED", "INACTIVE"})


class Settled(NamedTuple):
    """A pending row after settling: its status and its Composio account id."""

    status: str
    connection_id: Optional[str]


def is_pending(connection: Dict[str, Any]) -> bool:
    """Whether a workspace connection row is waiting on a Composio sign-in."""
    return (connection.get("status") or "").lower() == PENDING


def app_attempts(client: Any, entity_id: str, app_name: str, auth_config_id: Optional[str]) -> List[Any]:
    """Composio's connected accounts for this workspace's app, newest first: under the
    auth config the connect flow stored when there is one (the exact key), else every
    account for the toolkit."""
    query: Dict[str, Any] = {
        "user_ids": [entity_id],
        "toolkit_slugs": [app_name.lower()],
        "order_by": "created_at",
        "order_direction": "desc",
    }
    if auth_config_id:
        query["auth_config_ids"] = [auth_config_id]
    items = getattr(client.composio.connected_accounts.list(**query), "items", None) or []
    return sorted(items, key=lambda account: str(getattr(account, "created_at", "") or ""), reverse=True)


def _status(account: Any) -> str:
    return str(getattr(account, "status", "") or "").upper()


def settle_pending(
    entity_manager: Any, client: Any, entity: Dict[str, Any], app_name: str, auth_config_id: Optional[str],
) -> Settled:
    """Settle one pending row from Composio's record, writing any change to the row."""
    attempts = app_attempts(client, entity["composio_entity_id"], app_name, auth_config_id)
    account = next((a for a in attempts if _status(a) == COMPOSIO_CONNECTED), None)
    if account is not None:
        entity_manager.update_connection_status(
            entity_id=entity["id"], app_name=app_name, status=CONNECTED, connection_id=account.id,
        )
        return Settled(CONNECTED, account.id)
    newest = attempts[0] if attempts else None
    if newest is not None and _status(newest) in COMPOSIO_ENDED:
        entity_manager.update_connection_status(
            entity_id=entity["id"], app_name=app_name, status=NOT_CONNECTED, connection_id=None,
        )
        logger.info(
            "[F258] %s: Composio's attempt %s ended %s (%s); Connect is offered again",
            app_name, newest.id, _status(newest), getattr(newest, "status_reason", None) or "no reason given",
        )
        return Settled(NOT_CONNECTED, None)
    return Settled(PENDING, None)
