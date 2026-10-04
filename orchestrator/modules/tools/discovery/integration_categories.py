"""F302 (night 9): an integration's actions are shown only while the workspace has it connected.

Night 9 ran on a coffee roaster's workspace whose shop data is a connected database
(harbourline_shop); no Shopify store is connected. Auto still held
platform_shopify_sync_catalog and platform_shopify_sync_status: both promoted, both in every
dispatcher enum and action catalog, and "the shop system" ranked them near the top. "Yes, check
the shop system." called platform_shopify_sync_catalog, and Auto answered "the Shopify account
isn't connected ... would you like me to help you connect your Shopify account?" (L91); a fresh
"how much Guji have we got right now?" called platform_shopify_sync_status and offered the same
(chat 1618ac59). The owner had to say "I don't use a Shopify connection here".

An action category that belongs to an integration now joins the hidden categories (PRD-251B
US-B106: off means invisible) while the workspace has none of the signals the platform reads
that integration from: the install path's workspace setting, an active credential of the
integration's type, or a live Composio connection for its app. A read that fails hides the
category: a gate that cannot decide must deny.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

READ_FAILED_LOG = "[integrations] workspace %s's %s connection could not be read; its actions are hidden"
# A Composio connection whose OAuth finished; "pending" counts once Composio gave it an id
# (EntityManager.get_connected_apps reads it the same way).
ACTIVE_STATUS = "active"
PENDING_STATUS = "pending"


@dataclass(frozen=True)
class ConnectionSignals:
    """Where the platform reads that a workspace has an integration connected."""

    app_name: str          # the Composio app, stored upper-case
    credential_type: str   # CredentialType.name of the integration's stored credential
    settings_key: str      # the workspace setting its own install path writes


# action category -> its integration's connection signals (the same signals
# services/sites.py reads to tag a workspace's Site as a store).
INTEGRATION_CATEGORIES: Mapping[str, ConnectionSignals] = {
    "shopify": ConnectionSignals(app_name="SHOPIFY", credential_type="shopifyAccessTokenApi",
                                 settings_key="shopify_domain"),
}


def all_integration_categories() -> Tuple[str, ...]:
    """Every integration-owned category: what a workspace that cannot be read is not shown."""
    return tuple(INTEGRATION_CATEGORIES)


def unconnected_categories(workspace: Any, db: Any = None) -> Tuple[str, ...]:
    """The integration categories ``workspace`` (a ``Workspace`` row, or None) has not
    connected, read through ``db`` or the row's own session. No workspace hides them all."""
    if workspace is None:
        return all_integration_categories()
    session = db if db is not None else _session_of(workspace)
    return tuple(
        category for category, signals in INTEGRATION_CATEGORIES.items()
        if not integration_connected(workspace, session, signals)
    )


def integration_connected(workspace: Any, db: Any, signals: ConnectionSignals) -> bool:
    """Whether ``workspace`` has the integration ``signals`` describes connected."""
    settings = getattr(workspace, "settings", None)
    if isinstance(settings, dict) and settings.get(signals.settings_key):
        return True
    if db is None:
        return False
    try:
        return _credential_held(db, workspace.id, signals) or _app_connected(db, workspace.id, signals)
    except Exception:  # noqa: BLE001 — a gate that cannot decide must deny
        logger.exception(READ_FAILED_LOG, getattr(workspace, "id", None), signals.app_name)
        return False


def _credential_held(db: Any, workspace_id: Any, signals: ConnectionSignals) -> bool:
    from core.models.credentials import Credential, CredentialType

    row = (
        db.query(Credential.id)
        .join(CredentialType, Credential.credential_type_id == CredentialType.id)
        .filter(
            Credential.workspace_id == workspace_id,
            Credential.is_active.is_(True),
            CredentialType.name == signals.credential_type,
        )
        .first()
    )
    return row is not None


def _app_connected(db: Any, workspace_id: Any, signals: ConnectionSignals) -> bool:
    from sqlalchemy import and_, or_

    from core.models.composio import ComposioConnection, ComposioEntity

    row = (
        db.query(ComposioConnection.id)
        .join(ComposioEntity, ComposioConnection.entity_id == ComposioEntity.id)
        .filter(
            ComposioEntity.workspace_id == workspace_id,
            ComposioConnection.app_name == signals.app_name,
            or_(
                ComposioConnection.status == ACTIVE_STATUS,
                and_(ComposioConnection.status == PENDING_STATUS, ComposioConnection.connection_id.isnot(None)),
            ),
        )
        .first()
    )
    return row is not None


def _session_of(workspace: Any) -> Optional[Any]:
    """The session a loaded row belongs to, or None for a detached or stand-in row."""
    from sqlalchemy.orm import object_session
    from sqlalchemy.orm.exc import UnmappedInstanceError

    try:
        return object_session(workspace)
    except UnmappedInstanceError:  # a stand-in row: its categories stay hidden
        return None
