"""PRD-274 Wave 5f (decisions 27 and 28): Shopify moves onto a product record.

A boot seeds the record once products are on and both the Shopify internal key and the owner workspace are set:
the key's hash (never its value), the vertical's roster and minted key, and Shopify's mint-on-every-provision. Its
existing workspaces join it, one per shop: the live one when there is one. The vertical's key check then reads the
record, so an admin's key replacement takes effect, and a provision joins the shop's workspace to the product (a
reinstalled shop's new workspace takes it over from the old one), so the owner pays for it.
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
from integrations.vertical_products import seed_vertical_product
from services.products import key_hash, mint_key

import integrations.shopify  # noqa: F401  # registers the Shopify provisioner

ENV_KEY = "shpss-internal-" + uuid.uuid4().hex


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
    """Products on, the Shopify key and owner set, and no Shopify product or shop workspace before the test
    (any left are moved aside and put back); everything the test made is swept after."""
    owner, tag = str(uuid.uuid4()), uuid.uuid4().hex[:8]
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:o AS uuid), 'w5f-owner')"), {"o": owner})
    s.execute(text("UPDATE products SET slug = slug || '-aside-' || :t WHERE slug = 'shopify'"), {"t": tag})
    s.execute(text("UPDATE workspaces SET settings = settings || jsonb_build_object('vertical', 'aside-' || :t) "
                   "WHERE settings->>'vertical' = 'shopify' OR settings->>'source' = 'shopify'"), {"t": tag})
    s.commit()
    s.close()
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)
    monkeypatch.setattr(config, "SHOPIFY_INTERNAL_API_KEY", ENV_KEY)
    monkeypatch.setattr(config, "SHOPIFY_PRODUCT_OWNER_WORKSPACE_ID", owner)
    made = []
    yield SimpleNamespace(owner=owner, made=made, shop=lambda **kw: _shop(new_session, made, **kw))
    s = new_session.sweep()
    ids = made + [r[0] for r in s.execute(text("SELECT id::text FROM workspaces WHERE settings->>'product_id' IN "
                                                "(SELECT id::text FROM products WHERE slug = 'shopify')")).all()]
    for table in ("sdk_api_keys", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id::text = ANY(:ids)"), {"ids": ids})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM products WHERE slug = 'shopify'"))
    s.execute(text("UPDATE products SET slug = 'shopify' WHERE slug = 'shopify-aside-' || :t"), {"t": tag})
    s.execute(text("UPDATE workspaces SET settings = settings || '{\"vertical\": \"shopify\"}'::jsonb "
                   "WHERE settings->>'vertical' = 'aside-' || :t"), {"t": tag})
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


def _seed(new_session):
    s = new_session()
    try:
        return seed_vertical_product(s, "shopify", PROVISIONER_REGISTRY["shopify"])
    finally:
        s.close()


def _record(new_session):
    s = new_session()
    try:
        return s.query(Product).filter(Product.slug == "shopify").first()
    finally:
        s.close()


@pytest.mark.parametrize("setting, value, reason", [
    ("PRODUCTS_ENABLED", False, "products are off"),
    ("SHOPIFY_PRODUCT_OWNER_WORKSPACE_ID", "", "not set"),
    ("SHOPIFY_INTERNAL_API_KEY", "", "not set"),
    ("SHOPIFY_PRODUCT_OWNER_WORKSPACE_ID", str(uuid.uuid4()), "not a live workspace"),
])
def test_no_record_is_made_until_products_are_on_and_both_settings_name_something_real(
        new_session, world, monkeypatch, setting, value, reason):
    monkeypatch.setattr(config, setting, value)
    outcome = _seed(new_session)
    assert outcome["product"] is None and reason in outcome["reason"] and _record(new_session) is None


def test_the_record_is_seeded_from_the_keys_hash_and_the_verticals_roster(new_session, world):
    outcome = _seed(new_session)
    record = _record(new_session)
    shopify = PROVISIONER_REGISTRY["shopify"]
    assert outcome["product"] == str(record.id) and str(record.owner_workspace_id) == world.owner
    assert record.key_hash == key_hash(ENV_KEY) and record.key_prefix == "env" and ENV_KEY not in str(outcome)
    assert record.agent_slugs == shopify.agent_slugs and record.ops_manager_slug == shopify.ops_manager_slug
    assert (record.key_type, record.key_scopes, record.reuse_existing_key) == ("public", shopify.key_permissions, False)
    assert _seed(new_session) == {"vertical": "shopify", "product": str(record.id), "attached": 0}  # once


def test_existing_shops_join_one_workspace_per_shop_the_live_one_first(new_session, world):
    old = world.shop(domain="acme.myshopify.com", active=False, days_old=30)
    live = world.shop(domain="acme.myshopify.com", active=True, days_old=1)
    gone = world.shop(domain="gone.myshopify.com", active=False)
    early = world.shop(domain="early.myshopify.com", generic=False)  # made before the generic flow
    outcome = _seed(new_session)
    product = outcome["product"]
    assert outcome["attached"] == 3
    assert _settings(new_session, live)["product_id"] == product and "product_id" not in _settings(new_session, old)
    assert _settings(new_session, gone)["product_id"] == product
    assert _settings(new_session, early) == {"vertical": "shopify", "shopify_domain": "early.myshopify.com",
                                                  "product_id": product, "source_external_id": "early.myshopify.com"}
    assert _seed(new_session)["attached"] == 0


def test_the_key_check_reads_the_record_so_a_replaced_key_takes_over(new_session, world):
    from api.shopify import _verify_internal_key
    from api.verticals import _verify_vertical_internal_key

    _seed(new_session)
    _verify_vertical_internal_key("shopify", authorization=f"Bearer {ENV_KEY}")
    _verify_internal_key(f"Bearer {ENV_KEY}")
    replaced = mint_key()
    s = new_session()
    s.execute(text("UPDATE products SET key_hash = :h WHERE slug = 'shopify'"), {"h": replaced.key_hash})
    s.commit()
    s.close()
    for stale in (ENV_KEY, "", "Bearer"):
        with pytest.raises(HTTPException) as refused:
            _verify_internal_key(f"Bearer {stale}")
        assert refused.value.status_code == 401
    _verify_vertical_internal_key("shopify", authorization=f"Bearer {replaced.plaintext}")


def test_a_provisioned_shop_joins_the_product_and_its_owner_pays(new_session, world):
    from services.credits.payer import paying_workspace

    _seed(new_session)
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
    _seed(new_session)
    s = new_session()
    s.execute(text("UPDATE workspaces SET is_active = false WHERE id = CAST(:w AS uuid)"), {"w": old})  # uninstalled
    s.commit()
    try:
        made = provision_vertical(db=s, vertical="shopify", external_id="back.myshopify.com", name="Back", metadata={})
        world.made.append(made["id"])
    finally:
        s.close()
    assert made["id"] != old and _settings(new_session, made["id"])["product_id"] == str(_record(new_session).id)
    assert "product_id" not in _settings(new_session, old)


def test_the_boot_leader_seeds_it_and_a_failure_never_stops_the_boot(world, monkeypatch, caplog):
    import integrations.vertical_products as vertical_products

    ran = []
    monkeypatch.setattr(vertical_products, "seed_vertical_products", lambda db: ran.append(db) or [])
    vertical_products.seed_vertical_products_on_boot()
    assert len(ran) == 1

    def broken(db):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(vertical_products, "seed_vertical_products", broken)
    vertical_products.seed_vertical_products_on_boot()  # logged, never raised
    assert "the next boot tries again" in caplog.text
    import inspect

    import main

    assert "seed_vertical_products_on_boot()" in inspect.getsource(main._boot_phase_1_core)
