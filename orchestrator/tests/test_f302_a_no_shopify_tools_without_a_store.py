"""F302 (night 9) — a workspace with no Shopify store is offered no Shopify action.

Night 9's workspace keeps its shop data in a connected database; no Shopify store is
connected. "Yes, check the shop system." called platform_shopify_sync_catalog and Auto offered
to "help you connect your Shopify account" (L91); another fresh chat called
platform_shopify_sync_status. Both actions sat in every dispatcher enum and catalog.

Now the ``shopify`` category is hidden (PRD-251B US-B106) until the workspace has a store by
any of the three signals the platform reads: the install path's ``shopify_domain`` setting,
an active Shopify credential, or a live Composio SHOPIFY connection. Real Postgres rows.
"""
from __future__ import annotations

from uuid import UUID, uuid4

from modules.tools.discovery.action_registry import get_action_registry
from modules.tools.discovery.hidden_categories import exclude_kwargs, hidden_categories_for_workspace

SHOPIFY_ACTIONS = {"platform_shopify_sync_catalog", "platform_shopify_sync_status"}


def _offered(ws, db):
    """What Auto's dispatcher enum and action catalog offer this workspace."""
    hidden = hidden_categories_for_workspace(ws, db)
    registry = get_action_registry()
    schema = registry.to_dispatcher_schema(exclude_promoted=False, **exclude_kwargs(hidden))
    enum = set(schema["function"]["parameters"]["properties"]["action"].get("enum", []))
    catalog = registry.build_prompt_summary(exclude_admin=True, exclude_promoted=False, **exclude_kwargs(hidden))
    return enum, catalog


def _shopify_shown(ws, db):
    enum, catalog = _offered(ws, db)
    in_enum = SHOPIFY_ACTIONS <= enum
    in_catalog = all(name in catalog for name in SHOPIFY_ACTIONS)
    assert in_enum == in_catalog, (in_enum, in_catalog)
    return in_enum


def test_a_workspace_with_no_store_is_offered_no_shopify_action(db_session, seed_workspace):
    ws = seed_workspace()
    assert "shopify" in hidden_categories_for_workspace(ws, db_session)
    enum, catalog = _offered(ws, db_session)
    assert not SHOPIFY_ACTIONS & enum
    assert not any(name in catalog for name in SHOPIFY_ACTIONS)
    assert "platform_query_data" in enum          # the shop's own database stays offered


def test_the_install_paths_store_setting_shows_them(db_session, seed_workspace):
    from core.models.workspaces import Workspace

    ws = seed_workspace()
    row = db_session.get(Workspace, UUID(ws))
    row.settings = {**(row.settings or {}), "shopify_domain": "harbourline.myshopify.com"}
    db_session.flush()
    assert _shopify_shown(ws, db_session)


def test_a_live_composio_store_connection_shows_them_and_a_dead_one_does_not(db_session, seed_workspace):
    from core.models.composio import ComposioConnection, ComposioEntity

    ws = seed_workspace()
    entity = ComposioEntity(workspace_id=UUID(ws), composio_entity_id=f"ws-{uuid4().hex}")
    db_session.add(entity)
    db_session.flush()
    connection = ComposioConnection(entity_id=entity.id, app_name="SHOPIFY", status="disconnected")
    db_session.add(connection)
    db_session.flush()
    assert not _shopify_shown(ws, db_session)
    connection.status = "pending"                  # OAuth started, never finished
    db_session.flush()
    assert not _shopify_shown(ws, db_session)
    connection.connection_id = "ca_harbourline"    # finished on Composio's side
    db_session.flush()
    assert _shopify_shown(ws, db_session)


def test_an_active_store_credential_shows_them(db_session, seed_workspace):
    from core.models.credentials import Credential, CredentialType

    ws = seed_workspace()
    kind = db_session.query(CredentialType).filter(CredentialType.name == "shopifyAccessTokenApi").first()
    if kind is None:
        kind = CredentialType(name="shopifyAccessTokenApi", display_name="Shopify", schema_definition=[])
        db_session.add(kind)
        db_session.flush()
    credential = Credential(name=f"store-{uuid4().hex[:6]}", workspace_id=UUID(ws), credential_type_id=kind.id,
                            encrypted_data="sealed", is_active=False)
    db_session.add(credential)
    db_session.flush()
    assert not _shopify_shown(ws, db_session)      # a switched-off credential is no store
    credential.is_active = True
    db_session.flush()
    assert _shopify_shown(ws, db_session)


def test_another_workspaces_store_shows_nothing_here(db_session, seed_workspace):
    from core.models.composio import ComposioConnection, ComposioEntity

    mine, theirs = seed_workspace(), seed_workspace()
    entity = ComposioEntity(workspace_id=UUID(theirs), composio_entity_id=f"ws-{uuid4().hex}")
    db_session.add(entity)
    db_session.flush()
    db_session.add(ComposioConnection(entity_id=entity.id, app_name="SHOPIFY", status="active"))
    db_session.flush()
    assert _shopify_shown(theirs, db_session)
    assert not _shopify_shown(mine, db_session)
