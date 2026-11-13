"""PRD-274 Wave 5a (§5.1, §8-3): a product's record, kept by a platform admin (``api/admin_products.py``).

An admin creates a product for an owner workspace and gets its key once; only the key's hash is stored. The
admin changes its name, roster, minted key and default retention, replaces its key, and deletes it while no
workspace carries it. Each change lands in the owner workspace's audit log, never with the key. The routes
are for platform admins only, and answer 404 while ``PRODUCTS_ENABLED`` is off. The owner workspace of a
product is never purged: the purge stops before it erases anything.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.admin_products import router
from config import config
from core.auth.dependencies import RequestContext, UserContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from services.products import OWNS_A_PRODUCT, key_hash, refused_while_it_owns_a_product


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url
    from core.models.products import Product

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    Product.__table__.create(eng, checkfirst=True)  # a test database built before this table
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL "
                          "DEFAULT true"))
    yield eng
    eng.dispose()


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", True)


@pytest.fixture
def owner(new_session):
    """A live workspace to own products, and a marketplace agent for rosters; both swept after."""
    ws, slug = str(uuid.uuid4()), f"w5a-agent-{uuid.uuid4().hex[:8]}"
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w5a-owner')"), {"id": ws})
    s.execute(text("INSERT INTO agents (name, agent_type, owner_type, is_approved, slug) "
                   "VALUES ('Front desk', 'custom', 'marketplace', true, :slug)"), {"slug": slug})
    s.commit()
    s.close()
    yield SimpleNamespace(id=ws, agent=slug)
    s = new_session.sweep()
    s.execute(text("DELETE FROM products WHERE owner_workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM audit_logs WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid) OR settings->>'product_of_test' = :id"),
              {"id": ws})
    s.execute(text("DELETE FROM agents WHERE slug = :slug"), {"slug": slug})
    s.commit()
    s.close()


def _client(new_session, system_role="super_admin"):
    app = FastAPI()
    app.include_router(router)
    ctx = RequestContext(workspace_id=uuid.uuid4(), auth_type="clerk",
                         user=UserContext(id="admin-1", email="admin@example.com", system_role=system_role))
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx

    def db():
        s = new_session()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = db
    return TestClient(app)


def _create(client, owner, **fields):
    body = {"slug": f"support-desk-{uuid.uuid4().hex[:6]}", "name": "Support desk", "owner_workspace_id": owner.id,
            "agent_slugs": [owner.agent], "ops_manager_slug": owner.agent, "key_scopes": ["agents:chat"], **fields}
    return client.post("/api/admin/products", json=body)


def _stored(new_session, product_id):
    s = new_session()
    try:
        return s.execute(text("SELECT key_hash, name, default_retention_days FROM products WHERE id = CAST(:id AS uuid)"),
                         {"id": product_id}).first()
    finally:
        s.close()


def _actions(new_session, owner):
    s = new_session()
    try:
        return s.execute(text("SELECT action, details::text FROM audit_logs WHERE workspace_id = CAST(:id AS uuid) "
                              "ORDER BY created_at"), {"id": owner.id}).all()
    finally:
        s.close()


def test_off_the_routes_answer_404(new_session, monkeypatch):
    monkeypatch.setattr(config, "PRODUCTS_ENABLED", False)
    assert _client(new_session).get("/api/admin/products").status_code == 404


def test_only_a_platform_admin_reaches_them(new_session, owner):
    client = _client(new_session, system_role="user")
    assert client.get("/api/admin/products").status_code == 403
    assert _create(client, owner).status_code == 403


def test_a_product_is_created_with_its_key_shown_once_and_only_its_hash_stored(new_session, owner):
    client = _client(new_session)
    made = _create(client, owner, default_retention_days=90)
    assert made.status_code == 201
    key, product = made.json()["key"], made.json()["product"]
    assert key.startswith("ak_prod_") and product["key_prefix"] == key[:12] and "key_hash" not in product
    assert product["agent_slugs"] == [owner.agent] and product["key_type"] == "server"
    assert _stored(new_session, product["id"]).key_hash == key_hash(key)
    shown = client.get(f"/api/admin/products/{product['id']}").json()
    assert shown["workspaces"] == 0 and "key" not in shown and "key_hash" not in shown
    [(action, details)] = _actions(new_session, owner)
    assert action == "product.created" and key not in details and key_hash(key) not in details


def test_a_slug_is_one_products(new_session, owner):
    client = _client(new_session)
    first = _create(client, owner).json()["product"]
    assert _create(client, owner, slug=first["slug"]).status_code == 409


@pytest.mark.parametrize("fields, why", [
    ({"key_scopes": ["admin:everything"]}, "Unknown scopes"),
    ({"agent_slugs": ["no-such-agent"], "ops_manager_slug": None}, "No approved marketplace agent"),
    ({"ops_manager_slug": "someone-else"}, "ops_manager_slug must be one of agent_slugs"),
    ({"default_retention_days": 0}, "greater than or equal to 1"),
    ({"slug": "Has Spaces"}, "pattern"),
])
def test_what_a_product_is_made_with_is_checked(new_session, owner, fields, why):
    answer = _create(_client(new_session), owner, **fields)
    assert answer.status_code == 422 and why in answer.text


def test_the_owner_is_a_live_workspace_and_not_a_products_own(new_session, owner):
    client = _client(new_session)
    assert _create(client, owner, owner_workspace_id=str(uuid.uuid4())).status_code == 422
    product = _create(client, owner).json()["product"]
    customer = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, settings) VALUES (CAST(:id AS uuid), 'w5a-customer', "
                   "CAST(:settings AS jsonb))"),
              {"id": customer, "settings": f'{{"product_id": "{product["id"]}", "product_of_test": "{owner.id}"}}'})
    s.commit()
    s.close()
    answer = _create(client, owner, owner_workspace_id=customer)
    assert answer.status_code == 422 and "can't own one" in answer.text


def test_an_admin_changes_what_the_body_names_and_never_the_slug_or_owner(new_session, owner):
    client = _client(new_session)
    product = _create(client, owner, default_retention_days=30).json()["product"]
    url = f"/api/admin/products/{product['id']}"
    changed = client.patch(url, json={"name": "Help desk", "default_retention_days": None})
    assert changed.status_code == 200 and changed.json()["name"] == "Help desk"
    assert changed.json()["key_scopes"] == ["agents:chat"] and changed.json()["default_retention_days"] is None
    assert client.patch(url, json={"slug": "other"}).status_code == 422
    assert client.patch(url, json={"owner_workspace_id": str(uuid.uuid4())}).status_code == 422
    assert client.patch(url, json={"name": None}).status_code == 422
    assert client.patch(url, json={"agent_slugs": []}).status_code == 422  # its ops manager would be gone
    assert _stored(new_session, product["id"]).name == "Help desk"
    assert [a for a, _ in _actions(new_session, owner)] == ["product.created", "product.updated"]


def test_a_replaced_key_stops_the_old_one_at_once(new_session, owner):
    client = _client(new_session)
    made = _create(client, owner).json()
    replaced = client.post(f"/api/admin/products/{made['product']['id']}/key")
    assert replaced.status_code == 200
    new_key = replaced.json()["key"]
    assert new_key != made["key"] and _stored(new_session, made["product"]["id"]).key_hash == key_hash(new_key)
    assert _actions(new_session, owner)[-1][0] == "product.key_replaced"


def test_a_product_is_deleted_only_while_no_workspace_carries_it(new_session, owner):
    client = _client(new_session)
    product = _create(client, owner).json()["product"]
    customer = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, settings, deleted_at) VALUES (CAST(:id AS uuid), 'w5a-closed', "
                   "CAST(:settings AS jsonb), now())"),  # closed, inside its grace: still the product's
              {"id": customer, "settings": f'{{"product_id": "{product["id"]}", "product_of_test": "{owner.id}"}}'})
    s.commit()
    refused = client.delete(f"/api/admin/products/{product['id']}")
    assert refused.status_code == 409
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": customer})
    s.commit()
    s.close()
    assert client.delete(f"/api/admin/products/{product['id']}").json()["deleted"] is True
    assert _stored(new_session, product["id"]) is None
    assert client.get(f"/api/admin/products/{product['id']}").status_code == 404
    assert _actions(new_session, owner)[-1][0] == "product.deleted"


def test_the_owner_of_a_product_is_never_purged(new_session, owner):
    _create(_client(new_session), owner)
    ran = []
    step = refused_while_it_owns_a_product(lambda workspace_id, result: ran.append(workspace_id) or True)
    result = SimpleNamespace(skipped_reason=None)
    assert step(owner.id, result) is False and result.skipped_reason == OWNS_A_PRODUCT and ran == []
    other = SimpleNamespace(skipped_reason=None)
    assert step(str(uuid.uuid4()), other) is True and other.skipped_reason is None and len(ran) == 1


def test_the_purge_stops_before_its_first_data_step_and_an_erase_says_why(new_session, owner, monkeypatch):
    from api.gdpr import public_erasure
    from services import workspace_purge

    _create(_client(new_session), owner)
    monkeypatch.setattr(workspace_purge, "_purge_workspace_s3", lambda *a: pytest.fail("S3 must not be touched"))
    s = new_session()
    s.execute(text("INSERT INTO jobs.jobs (id, kind, workspace_id, max_attempts) "
                   "VALUES (gen_random_uuid(), 'w5a.test', CAST(:id AS uuid), 1)"), {"id": owner.id})
    s.commit()
    result = SimpleNamespace(skipped_reason=None, s3_errors=[])
    assert workspace_purge._s3_purged(owner.id, result) is False and result.skipped_reason == OWNS_A_PRODUCT
    left = s.execute(text("DELETE FROM jobs.jobs WHERE workspace_id = CAST(:id AS uuid)"), {"id": owner.id}).rowcount
    s.commit()
    s.close()
    assert left == 1  # the guard runs before the jobs barrier: a refused purge stops nothing
    assert public_erasure({"sql": {"skipped_reason": OWNS_A_PRODUCT}})["sql"]["skipped_reason"] == OWNS_A_PRODUCT


def test_whether_a_find_mints_another_key_is_the_admins_to_set(new_session, owner):
    client = _client(new_session)
    product = _create(client, owner).json()["product"]
    assert product["reuse_existing_key"] is True
    changed = client.patch(f"/api/admin/products/{product['id']}", json={"reuse_existing_key": False})
    assert changed.json()["reuse_existing_key"] is False
    assert client.patch(f"/api/admin/products/{product['id']}", json={"reuse_existing_key": None}).status_code == 422
