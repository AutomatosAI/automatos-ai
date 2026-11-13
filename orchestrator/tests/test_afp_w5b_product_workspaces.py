"""PRD-274 Wave 5b (§5.1): a product finds or creates its customers' workspaces with its own key.

``POST /api/products/{id}/workspaces`` with the product's key: a new workspace (201) carries the product and the
customer's external id, starts with the product's roster and gets a key of the product's type and scopes; found
again (200), it is the same workspace, seeded once, and gets a new key only when the product doesn't reuse keys.
A product never reaches another's workspace, a closed workspace isn't reopened (409), a public key needs its
origins (422, nothing written), and two requests creating one workspace at once end with one.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.products import router
from config import config
from core.database.database import get_db
from core.models.products import Product
from services import product_workspaces
from services.products import mint_key


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


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)


@pytest.fixture
def world(new_session):
    """An owner workspace, a marketplace agent, and a product factory; everything swept after."""
    owner, agent = str(uuid.uuid4()), f"w5b-agent-{uuid.uuid4().hex[:8]}"
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w5b-owner')"), {"id": owner})
    s.execute(text("INSERT INTO agents (name, agent_type, owner_type, is_approved, slug) "
                   "VALUES ('Front desk', 'custom', 'marketplace', true, :slug)"), {"slug": agent})
    s.commit()
    s.close()
    made = []

    def product(**fields):
        key = mint_key()
        row = Product(**{"slug": f"w5b-{uuid.uuid4().hex[:8]}", "name": "Support desk",
                         "owner_workspace_id": uuid.UUID(owner), "key_hash": key.key_hash, "key_prefix": key.key_prefix,
                         "agent_slugs": [agent], "ops_manager_slug": agent, "key_type": "server",
                         "key_scopes": ["agents:chat"], **fields})
        s = new_session()
        s.add(row)
        s.commit()
        s.close()
        made.append(str(row.id))
        return SimpleNamespace(id=str(row.id), key=key.plaintext, row=row)

    yield SimpleNamespace(owner=owner, agent=agent, product=product)
    _sweep(new_session, owner, agent, made)


def _sweep(new_session, owner, agent, products):
    s = new_session.sweep()
    theirs = "SELECT id FROM workspaces WHERE settings->>'product_id' = ANY(:p)"
    for table in ("sdk_api_keys", "agents"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id IN ({theirs})"), {"p": products})  # noqa: S608
    s.execute(text(f"DELETE FROM workspaces WHERE id IN ({theirs})"), {"p": products})  # noqa: S608
    s.execute(text("DELETE FROM products WHERE owner_workspace_id = CAST(:o AS uuid)"), {"o": owner})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.execute(text("DELETE FROM agents WHERE slug = :a"), {"a": agent})
    s.commit()
    s.close()


def _client(new_session):
    app = FastAPI()
    app.include_router(router)

    def db():
        s = new_session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = db
    return TestClient(app)


def _find(client, product, key=None, **fields):
    body = {"external_id": "acme-store", "name": "Acme", **fields}
    headers = {"Authorization": f"Bearer {product.key if key is None else key}"}
    return client.post(f"/api/products/{product.id}/workspaces", json=body, headers=headers)


def _row(new_session, sql, **params):
    s = new_session()
    try:
        return s.execute(text(sql), params).all()
    finally:
        s.close()


def test_off_the_route_answers_404(new_session, world, monkeypatch):
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", False)
    assert _find(_client(new_session), world.product()).status_code == 404


def test_only_the_products_own_key_reaches_it(new_session, world):
    client, product, other = _client(new_session), world.product(), world.product()
    assert _find(client, product, key="").status_code == 401
    assert _find(client, product, key=other.key).status_code == 401
    assert _find(client, SimpleNamespace(id=str(uuid.uuid4()), key=product.key)).status_code == 401
    assert client.post(f"/api/products/{product.id}/workspaces", json={"external_id": "x", "name": "x"}).status_code == 401
    assert _row(new_session, "SELECT id FROM workspaces WHERE settings->>'product_id' = :p", p=product.id) == []


def test_a_new_workspace_carries_the_product_its_roster_and_a_key(new_session, world):
    product = world.product()
    made = _find(_client(new_session), product, metadata={"plan": "pro"})
    assert made.status_code == 201
    body = made.json()
    assert body["is_new"] is True and body["agents_installed"] == 1 and body["api_key"].startswith("ak_srv_")
    [(settings,)] = _row(new_session, "SELECT settings FROM workspaces WHERE id = CAST(:w AS uuid)", w=body["workspace_id"])
    assert settings == {"product_id": product.id, "source": "product", "source_external_id": "acme-store",
                        "product_metadata": {"plan": "pro"}}
    [(key_type, permissions)] = _row(new_session, "SELECT key_type, permissions FROM sdk_api_keys "
                                                  "WHERE workspace_id = CAST(:w AS uuid)", w=body["workspace_id"])
    assert key_type == "server" and permissions == ["agents:chat"]
    assert _row(new_session, "SELECT slug FROM agents WHERE workspace_id = CAST(:w AS uuid)",
                w=body["workspace_id"]) == [(world.agent,)]


def test_found_again_it_is_the_same_workspace_seeded_once_and_keeps_its_key(new_session, world):
    client, product = _client(new_session), world.product()
    first = _find(client, product).json()
    again = _find(client, product)
    assert again.status_code == 200
    assert again.json() == {**first, "api_key": None, "key_minted": False, "agents_installed": 0, "is_new": False}


def test_a_product_that_doesnt_reuse_keys_gets_a_new_one_on_every_find(new_session, world):
    client, product = _client(new_session), world.product(reuse_existing_key=False)
    first, again = _find(client, product).json(), _find(client, product).json()
    assert again["key_minted"] is True and again["api_key"] != first["api_key"]


def test_two_products_never_share_a_workspace(new_session, world):
    client = _client(new_session)
    one, two = _find(client, world.product()).json(), _find(client, world.product()).json()
    assert one["workspace_id"] != two["workspace_id"] and two["is_new"] is True


def test_a_public_key_needs_its_origins_and_nothing_is_written_without(new_session, world):
    client, product = _client(new_session), world.product(key_type="public")
    refused = _find(client, product)
    assert refused.status_code == 422 and "allowed_domains" in refused.text
    assert _row(new_session, "SELECT id FROM workspaces WHERE settings->>'product_id' = :p", p=product.id) == []
    made = _find(client, product, allowed_domains=["https://acme.example"])
    assert made.status_code == 201 and made.json()["api_key"].startswith("ak_pub_")


def test_a_closed_workspace_is_not_reopened_by_a_find(new_session, world):
    client, product = _client(new_session), world.product()
    made = _find(client, product).json()
    s = new_session()
    s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:w AS uuid)"), {"w": made["workspace_id"]})
    s.commit()
    s.close()
    closed = _find(client, product)
    assert closed.status_code == 409 and "closed" in closed.text


def test_of_two_requests_creating_one_workspace_the_second_finds_the_first(new_session, world, monkeypatch):
    client, product = _client(new_session), world.product()
    first = _find(client, product).json()
    real, calls = product_workspaces._theirs, []

    def missed_once(db, product_row, external_id):  # the other request's commit lands after this one looked
        calls.append(external_id)
        return None if len(calls) == 1 else real(db, product_row, external_id)

    monkeypatch.setattr(product_workspaces, "_theirs", missed_once)
    again = _find(client, product)
    assert again.status_code == 200 and again.json()["workspace_id"] == first["workspace_id"] and len(calls) == 2


def test_a_product_deleted_while_the_request_waited_is_gone(new_session, world):
    product = world.product()
    s = new_session()
    row = s.get(Product, uuid.UUID(product.id))
    other = new_session()
    other.execute(text("DELETE FROM products WHERE id = CAST(:p AS uuid)"), {"p": product.id})
    other.commit()
    other.close()
    with pytest.raises(product_workspaces.ProductGone):
        product_workspaces.find_or_create(s, row, external_id="acme-store", name="Acme", metadata={},
                                          allowed_domains=[])
    s.close()
    assert _row(new_session, "SELECT id FROM workspaces WHERE settings->>'product_id' = :p", p=product.id) == []


@pytest.mark.parametrize("fields", [{"external_id": "has space"}, {"external_id": ""}, {"name": ""},
                                    {"metadata": {"x": "y" * 9000}}])
def test_what_a_product_sends_is_checked(new_session, world, fields):
    assert _find(_client(new_session), world.product(), **fields).status_code == 422
