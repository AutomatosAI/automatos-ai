"""PRD-274 Wave 5f (decisions 27–31): Shopify is a product record.

The vertical's machine key is its record's (no record: 503), so an admin's key replacement takes effect at once;
a provision joins the shop's workspace to the product (a reinstalled shop's new workspace takes it over from the
old one), so the owner pays for it. The environment key and the boot seed that made the record are gone
(decision 30).
"""
from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from config import config
from core.models.products import Product
from integrations.provisioning import PROVISIONER_REGISTRY, provision_vertical
from services.products import mint_key

import integrations.shopify  # noqa: F401  # registers the Shopify provisioner



@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    Product.__table__.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL "
                          "DEFAULT true"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_workspaces_product_external_id ON workspaces "
                          "((settings->>'product_id'), (settings->>'source_external_id')) "
                          "WHERE settings->>'product_id' IS NOT NULL"))
    yield eng
    eng.dispose()


@pytest.fixture
def world(new_session, monkeypatch):
    """Products on, and the Shopify product record (moved aside and put back if one exists); everything the
    test made is swept after."""
    owner, tag, key = str(uuid.uuid4()), uuid.uuid4().hex[:8], mint_key()
    shopify = PROVISIONER_REGISTRY["shopify"]
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:o AS uuid), 'w5f-owner')"), {"o": owner})
    s.execute(text("UPDATE products SET slug = slug || '-aside-' || :t WHERE slug = 'shopify'"), {"t": tag})
    s.add(Product(slug="shopify", name="Shopify", owner_workspace_id=uuid.UUID(owner), key_hash=key.key_hash,
                  key_prefix=key.key_prefix, agent_slugs=list(shopify.agent_slugs),
                  ops_manager_slug=shopify.ops_manager_slug, key_type="public",
                  key_scopes=list(shopify.key_permissions), reuse_existing_key=False))
    s.commit()
    s.close()
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)
    made = []
    yield SimpleNamespace(owner=owner, key=key.plaintext, made=made, shop=lambda **kw: _shop(new_session, made, **kw))
    s = new_session.sweep()
    ids = made + [r[0] for r in s.execute(text("SELECT id::text FROM workspaces WHERE settings->>'product_id' IN "
                                                "(SELECT id::text FROM products WHERE slug = 'shopify')")).all()]
    for table in ("sdk_api_keys", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id::text = ANY(:ids)"), {"ids": ids})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM products WHERE slug = 'shopify'"))
    s.execute(text("UPDATE products SET slug = 'shopify' WHERE slug = 'shopify-aside-' || :t"), {"t": tag})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


def _shop(new_session, made, *, domain, active=True, days_old=0, generic=True):
    ws = str(uuid.uuid4())
    settings = {"vertical": "shopify", "shopify_domain": domain, **({"source_external_id": domain} if generic else {})}
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, settings, is_active, created_at) VALUES (CAST(:id AS uuid), :d, "
                   "CAST(:s AS jsonb), :a, now() - make_interval(days => :o))"),
              {"id": ws, "d": domain, "s": json.dumps(settings), "a": active, "o": days_old})
    s.commit()
    s.close()
    made.append(ws)
    return ws


def _settings(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT settings FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws}).scalar()
    finally:
        s.close()


def _record(new_session):
    s = new_session()
    try:
        return s.query(Product).filter(Product.slug == "shopify").first()
    finally:
        s.close()


def test_the_key_check_reads_the_record_so_a_replaced_key_takes_over(new_session, world):
    from api.shopify import _verify_internal_key
    from api.verticals import _verify_vertical_internal_key

    _verify_vertical_internal_key("shopify", authorization=f"Bearer {world.key}", db=None)
    _verify_internal_key(f"Bearer {world.key}", db=None)
    replaced = mint_key()
    s = new_session()
    s.execute(text("UPDATE products SET key_hash = :h WHERE slug = 'shopify'"), {"h": replaced.key_hash})
    s.commit()
    s.close()
    for stale in (world.key, "", "Bearer"):
        with pytest.raises(HTTPException) as refused:
            _verify_internal_key(f"Bearer {stale}", db=None)
        assert refused.value.status_code == 401
    _verify_vertical_internal_key("shopify", authorization=f"Bearer {replaced.plaintext}", db=None)


def test_a_provisioned_shop_joins_the_product_and_its_owner_pays(new_session, world):
    from services.credits.payer import paying_workspace

    s = new_session()
    try:
        made = provision_vertical(db=s, vertical="shopify", external_id="new.myshopify.com", name="New", metadata={})
        world.made.append(made["id"])
        assert made["is_new"] and made["key_minted"]
        assert _settings(new_session, made["id"])["product_id"] == str(_record(new_session).id)
        assert paying_workspace(s, made["id"]) == world.owner
    finally:
        s.close()


def test_a_reinstalled_shops_new_workspace_takes_the_product_over(new_session, world):
    old = world.shop(domain="back.myshopify.com", active=True)
    s = new_session()
    s.execute(text("UPDATE workspaces SET settings = settings || jsonb_build_object('product_id', :p, "
                   "'source_external_id', 'back.myshopify.com') WHERE id = CAST(:w AS uuid)"),
              {"p": str(_record(new_session).id), "w": old})
    s.execute(text("UPDATE workspaces SET is_active = false WHERE id = CAST(:w AS uuid)"), {"w": old})  # uninstalled
    s.commit()
    try:
        made = provision_vertical(db=s, vertical="shopify", external_id="back.myshopify.com", name="Back", metadata={})
        world.made.append(made["id"])
    finally:
        s.close()
    assert made["id"] != old and _settings(new_session, made["id"])["product_id"] == str(_record(new_session).id)
    assert "product_id" not in _settings(new_session, old)


def test_with_no_record_the_vertical_is_dark(new_session, world):
    from api.verticals import _verify_vertical_internal_key

    s = new_session()
    s.execute(text("UPDATE products SET slug = 'shopify-gone' WHERE slug = 'shopify'"))
    s.commit()
    try:
        with pytest.raises(HTTPException) as dark:
            _verify_vertical_internal_key("shopify", authorization=f"Bearer {world.key}", db=None)
        assert dark.value.status_code == 503
    finally:
        s.execute(text("UPDATE products SET slug = 'shopify' WHERE slug = 'shopify-gone'"))
        s.commit()
        s.close()


def test_no_environment_key_or_boot_seed_is_left():
    import inspect

    import integrations.vertical_products as vertical_products
    import main

    for name in ("SHOPIFY_INTERNAL_API_KEY", "BUDSTACKS_INTERNAL_API_KEY", "SHOPIFY_PRODUCT_OWNER_WORKSPACE_ID",
                 "BUDSTACKS_PRODUCT_OWNER_WORKSPACE_ID"):
        assert not hasattr(config, name), name
    assert not hasattr(vertical_products, "seed_vertical_products")
    assert "vertical_products" not in inspect.getsource(main._boot_phase_1_core)


def test_the_key_check_reads_on_the_requests_session_and_opens_no_other(new_session, world, monkeypatch):
    import core.database.database as database
    from api.verticals import _verify_vertical_internal_key

    monkeypatch.setattr(database, "SessionLocal", lambda: pytest.fail("the request's session should be used"))
    s = new_session()
    try:
        _verify_vertical_internal_key("shopify", authorization=f"Bearer {world.key}", db=s)
        with pytest.raises(HTTPException) as refused:
            _verify_vertical_internal_key("shopify", authorization="Bearer wrong", db=s)
        assert refused.value.status_code == 401
    finally:
        s.close()
