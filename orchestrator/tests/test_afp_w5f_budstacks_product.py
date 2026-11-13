"""PRD-274 Wave 5f (decisions 27 and 28): BudStacks moves onto a product record, keeping its behaviour.

The generic seed (``integrations/vertical_products``, tested in full with Shopify) makes BudStacks' record from its
internal key's hash and its own settings: no roster yet, a public ``chat`` key, and a re-provision that keeps a
live storefront's key. Its existing tenants join it, a provisioned tenant joins it and is paid for by the owner, a
re-provision mints no new key, and the key check reads the record.
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

import integrations.budstacks  # noqa: F401  # registers the BudStacks provisioner

ENV_KEY = "budstacks-internal-" + uuid.uuid4().hex
DOMAINS = {"domains": ["green.example"]}


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
    """Products on, the BudStacks key and owner set, and no BudStacks product or tenant workspace before the test
    (any left are moved aside and put back); everything the test made is swept after."""
    owner, tag = str(uuid.uuid4()), uuid.uuid4().hex[:8]
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:o AS uuid), 'w5f-bs-owner')"), {"o": owner})
    s.execute(text("UPDATE products SET slug = slug || '-aside-' || :t WHERE slug = 'budstacks'"), {"t": tag})
    s.execute(text("UPDATE workspaces SET settings = settings || jsonb_build_object('vertical', 'aside-' || :t) "
                   "WHERE settings->>'vertical' = 'budstacks' OR settings->>'source' = 'budstacks'"), {"t": tag})
    s.commit()
    s.close()
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)
    monkeypatch.setattr(config, "BUDSTACKS_INTERNAL_API_KEY", ENV_KEY)
    monkeypatch.setattr(config, "BUDSTACKS_PRODUCT_OWNER_WORKSPACE_ID", owner)
    made = []
    yield SimpleNamespace(owner=owner, made=made)
    s = new_session.sweep()
    ids = made + [r[0] for r in s.execute(text("SELECT id::text FROM workspaces WHERE settings->>'product_id' IN "
                                                "(SELECT id::text FROM products WHERE slug = 'budstacks')")).all()]
    s.execute(text("DELETE FROM sdk_api_keys WHERE workspace_id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM workspaces WHERE id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM products WHERE slug = 'budstacks'"))
    s.execute(text("UPDATE products SET slug = 'budstacks' WHERE slug = 'budstacks-aside-' || :t"), {"t": tag})
    s.execute(text("UPDATE workspaces SET settings = settings || '{\"vertical\": \"budstacks\"}'::jsonb "
                   "WHERE settings->>'vertical' = 'aside-' || :t"), {"t": tag})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


def _seed(new_session):
    s = new_session()
    try:
        return seed_vertical_product(s, "budstacks", PROVISIONER_REGISTRY["budstacks"])
    finally:
        s.close()


def _provision(new_session, world, tenant):
    s = new_session()
    try:
        made = provision_vertical(db=s, vertical="budstacks", external_id=tenant, name=tenant, metadata=DOMAINS)
    finally:
        s.close()
    world.made.append(made["id"])
    return made


def _product_id(new_session, ws):
    s = new_session()
    try:
        return s.execute(text("SELECT settings->>'product_id' FROM workspaces WHERE id = CAST(:w AS uuid)"),
                         {"w": ws}).scalar()
    finally:
        s.close()


def test_the_record_keeps_budstacks_own_settings(new_session, world):
    outcome = _seed(new_session)
    s = new_session()
    record = s.query(Product).filter(Product.slug == "budstacks").one()
    s.close()
    assert outcome["product"] == str(record.id) and str(record.owner_workspace_id) == world.owner
    assert record.key_hash == key_hash(ENV_KEY) and record.name == "BudStacks"
    assert (record.agent_slugs, record.ops_manager_slug, record.key_type, record.key_scopes) == ([], None, "public",
                                                                                                ["chat"])
    assert record.reuse_existing_key is True


def test_an_existing_tenant_joins_it(new_session, world):
    tenant = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, settings, is_active) VALUES (CAST(:id AS uuid), 'tenant-1', "
                   "CAST(:s AS jsonb), true)"),
              {"id": tenant, "s": json.dumps({"vertical": "budstacks", "budstacks_tenant_id": "tenant-1",
                                              "source_external_id": "tenant-1"})})
    s.commit()
    s.close()
    world.made.append(tenant)
    outcome = _seed(new_session)
    assert outcome["attached"] == 1 and _product_id(new_session, tenant) == outcome["product"]


def test_a_tenant_is_provisioned_into_it_paid_by_its_owner_and_keeps_its_key(new_session, world):
    from services.credits.payer import paying_workspace

    product = _seed(new_session)["product"]
    first = _provision(new_session, world, "tenant-2")
    again = _provision(new_session, world, "tenant-2")
    assert _product_id(new_session, first["id"]) == product and first["key_minted"]
    assert again["id"] == first["id"] and again["key_minted"] is False and again["api_key"] is None
    s = new_session()
    try:
        assert paying_workspace(s, first["id"]) == world.owner
    finally:
        s.close()


def test_the_key_check_reads_the_record(new_session, world):
    from api.verticals import _verify_vertical_internal_key

    _seed(new_session)
    _verify_vertical_internal_key("budstacks", authorization=f"Bearer {ENV_KEY}")
    replaced = mint_key()
    s = new_session()
    s.execute(text("UPDATE products SET key_hash = :h WHERE slug = 'budstacks'"), {"h": replaced.key_hash})
    s.commit()
    s.close()
    with pytest.raises(HTTPException) as refused:
        _verify_vertical_internal_key("budstacks", authorization=f"Bearer {ENV_KEY}")
    assert refused.value.status_code == 401
    _verify_vertical_internal_key("budstacks", authorization=f"Bearer {replaced.plaintext}")
