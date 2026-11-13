"""PRD-274 Wave 5f (decisions 27–31): BudStacks is a product record, keeping its behaviour.

A provisioned tenant joins the product and is paid for by its owner, a re-provision mints no new key (BudStacks'
``reuse_existing_key``), and the machine key is the record's.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from config import config
from core.models.products import Product
from integrations.provisioning import PROVISIONER_REGISTRY, provision_vertical
from services.products import mint_key

import integrations.budstacks  # noqa: F401  # registers the BudStacks provisioner

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
    """Products on, and the BudStacks product record (moved aside and put back if one exists); everything the
    test made is swept after."""
    owner, tag, key = str(uuid.uuid4()), uuid.uuid4().hex[:8], mint_key()
    budstacks = PROVISIONER_REGISTRY["budstacks"]
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:o AS uuid), 'w5f-bs-owner')"), {"o": owner})
    s.execute(text("UPDATE products SET slug = slug || '-aside-' || :t WHERE slug = 'budstacks'"), {"t": tag})
    s.add(Product(slug="budstacks", name="BudStacks", owner_workspace_id=uuid.UUID(owner), key_hash=key.key_hash,
                  key_prefix=key.key_prefix, agent_slugs=[], key_type="public",
                  key_scopes=list(budstacks.key_permissions), reuse_existing_key=True))
    s.commit()
    s.close()
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)
    made = []
    yield SimpleNamespace(owner=owner, key=key.plaintext, made=made)
    s = new_session.sweep()
    ids = made + [r[0] for r in s.execute(text("SELECT id::text FROM workspaces WHERE settings->>'product_id' IN "
                                                "(SELECT id::text FROM products WHERE slug = 'budstacks')")).all()]
    s.execute(text("DELETE FROM sdk_api_keys WHERE workspace_id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM workspaces WHERE id::text = ANY(:ids)"), {"ids": ids})
    s.execute(text("DELETE FROM products WHERE slug = 'budstacks'"))
    s.execute(text("UPDATE products SET slug = 'budstacks' WHERE slug = 'budstacks-aside-' || :t"), {"t": tag})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
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


def test_a_tenant_is_provisioned_into_it_paid_by_its_owner_and_keeps_its_key(new_session, world):
    from services.credits.payer import paying_workspace

    s = new_session()
    product = str(s.query(Product.id).filter(Product.slug == "budstacks").scalar())
    s.close()
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

    _verify_vertical_internal_key("budstacks", authorization=f"Bearer {world.key}")
    replaced = mint_key()
    s = new_session()
    s.execute(text("UPDATE products SET key_hash = :h WHERE slug = 'budstacks'"), {"h": replaced.key_hash})
    s.commit()
    s.close()
    with pytest.raises(HTTPException) as refused:
        _verify_vertical_internal_key("budstacks", authorization=f"Bearer {world.key}")
    assert refused.value.status_code == 401
    _verify_vertical_internal_key("budstacks", authorization=f"Bearer {replaced.plaintext}")
