"""PRD-274 Wave 5c (decision 25): a product lists, closes and restores its workspaces; a daily job purges them.

With the product's key: the list shows its own workspaces (closed ones when asked, with when each purge is
due); a close (202) stops the workspace at once, its keys included; a restore within the grace opens it again,
and after the grace is refused (410). Another product's workspace is never reached (404). Once a day,
``products.purge_closed`` purges every product workspace closed longer than the grace, and fails, for a
retry, when any purge was held.
"""
from __future__ import annotations

import asyncio
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
from core.services.api_key_service import ApiKeyService
from services import product_purge
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
    monkeypatch.setattr(config, "PRODUCT_CLOSE_GRACE_DAYS", 7)


@pytest.fixture
def world(new_session):
    """An owner workspace and a product factory; every workspace they make is swept after."""
    owner, made = str(uuid.uuid4()), []
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'w5c-owner')"), {"id": owner})
    s.commit()
    s.close()

    def product():
        key = mint_key()
        row = Product(slug=f"w5c-{uuid.uuid4().hex[:8]}", name="Support desk", owner_workspace_id=uuid.UUID(owner),
                      key_hash=key.key_hash, key_prefix=key.key_prefix, key_scopes=["agents:chat"])
        s = new_session()
        s.add(row)
        s.commit()
        s.close()
        made.append(str(row.id))
        return SimpleNamespace(id=str(row.id), key=key.plaintext)

    yield SimpleNamespace(owner=owner, product=product)
    s = new_session.sweep()
    theirs = "SELECT id FROM workspaces WHERE settings->>'product_id' = ANY(:p)"
    s.execute(text(f"DELETE FROM sdk_api_keys WHERE workspace_id IN ({theirs})"), {"p": made})  # noqa: S608
    s.execute(text(f"DELETE FROM workspaces WHERE id IN ({theirs})"), {"p": made})  # noqa: S608
    s.execute(text("DELETE FROM audit_logs WHERE workspace_id = CAST(:o AS uuid)"), {"o": owner})
    s.execute(text("DELETE FROM products WHERE owner_workspace_id = CAST(:o AS uuid)"), {"o": owner})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


@pytest.fixture
def client(new_session):
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


def _as(product):
    return {"Authorization": f"Bearer {product.key}"}


def _open(client, product, external_id="acme-store"):
    made = client.post(f"/api/products/{product.id}/workspaces", headers=_as(product),
                       json={"external_id": external_id, "name": external_id})
    assert made.status_code == 201
    return made.json()


def _url(product, workspace_id, tail=""):
    return f"/api/products/{product.id}/workspaces/{workspace_id}{tail}"


def _key_works(new_session, key):
    s = new_session()
    try:
        return ApiKeyService.validate_api_key(s, key) is not None
    finally:
        s.close()


def _closed_days_ago(new_session, workspace_id, days):
    s = new_session()
    s.execute(text("UPDATE workspaces SET deleted_at = (now() AT TIME ZONE 'utc') - make_interval(days => :d), "
                   "is_active = false WHERE id = CAST(:w AS uuid)"), {"d": days, "w": workspace_id})
    s.commit()
    s.close()


def test_the_list_shows_the_products_own_and_closed_ones_only_when_asked(client, world):
    product, other = world.product(), world.product()
    first, second = _open(client, product, "acme-1"), _open(client, product, "acme-2")
    _open(client, other, "acme-1")
    assert client.delete(_url(product, second["workspace_id"]), headers=_as(product)).status_code == 202
    listed = client.get(f"/api/products/{product.id}/workspaces", headers=_as(product)).json()["workspaces"]
    assert [w["workspace_id"] for w in listed] == [first["workspace_id"]] and listed[0]["status"] == "open"
    every = client.get(f"/api/products/{product.id}/workspaces?include_closed=true", headers=_as(product)).json()
    closed = every["workspaces"][1]
    assert [w["external_id"] for w in every["workspaces"]] == ["acme-1", "acme-2"]
    assert closed["status"] == "closed" and closed["purge_after"] > closed["closed_at"]
    assert client.get(f"/api/products/{product.id}/workspaces", headers=_as(other)).status_code == 401


def test_a_closed_workspace_stops_at_once_its_keys_included(client, world, new_session):
    product = world.product()
    made = _open(client, product)
    assert _key_works(new_session, made["api_key"])
    closed = client.delete(_url(product, made["workspace_id"]), headers=_as(product))
    assert closed.status_code == 202 and closed.json()["status"] == "closed"
    assert not _key_works(new_session, made["api_key"])
    again = client.delete(_url(product, made["workspace_id"]), headers=_as(product))
    assert again.status_code == 202 and again.json()["closed_at"] == closed.json()["closed_at"]
    s = new_session()
    actions = s.execute(text("SELECT action FROM audit_logs WHERE workspace_id = CAST(:o AS uuid)"),
                        {"o": world.owner}).scalars().all()
    s.close()
    assert actions == ["product.workspace_closed"]  # the second close changed nothing


def test_a_product_never_reaches_anothers_workspace(client, world):
    product, other = world.product(), world.product()
    theirs = _open(client, other)
    assert client.delete(_url(product, theirs["workspace_id"]), headers=_as(product)).status_code == 404
    assert client.post(_url(product, theirs["workspace_id"], "/restore"), headers=_as(product)).status_code == 404
    assert client.delete(_url(product, uuid.uuid4()), headers=_as(product)).status_code == 404


def test_a_restore_within_the_grace_opens_it_and_its_keys_again(client, world, new_session):
    product = world.product()
    made = _open(client, product)
    client.delete(_url(product, made["workspace_id"]), headers=_as(product))
    restored = client.post(_url(product, made["workspace_id"], "/restore"), headers=_as(product))
    assert restored.status_code == 200 and restored.json()["status"] == "open" and restored.json()["closed_at"] is None
    assert _key_works(new_session, made["api_key"])
    assert client.post(_url(product, made["workspace_id"], "/restore"), headers=_as(product)).status_code == 409


def test_after_the_grace_a_workspace_cant_be_restored(client, world, new_session):
    product = world.product()
    made = _open(client, product)
    _closed_days_ago(new_session, made["workspace_id"], 8)
    refused = client.post(_url(product, made["workspace_id"], "/restore"), headers=_as(product))
    assert refused.status_code == 410 and "grace" in refused.text


def test_an_admin_closed_workspaces_key_stops_too(client, world, new_session):
    made = _open(client, world.product())
    _closed_days_ago(new_session, made["workspace_id"], 0)
    assert not _key_works(new_session, made["api_key"])


def test_the_daily_job_purges_only_product_workspaces_past_their_grace(client, world, new_session, monkeypatch):
    product = world.product()
    due, recent, kept = (_open(client, product, f"acme-{n}") for n in range(3))
    _closed_days_ago(new_session, due["workspace_id"], 8)
    _closed_days_ago(new_session, recent["workspace_id"], 2)
    purged = []
    monkeypatch.setattr("services.workspace_purge.purge_workspace_sync",
                        lambda workspace_id: purged.append(str(workspace_id)) or SimpleNamespace(skipped_reason=None))
    asyncio.run(product_purge.purge_closed(SimpleNamespace()))
    assert due["workspace_id"] in purged
    assert recent["workspace_id"] not in purged and kept["workspace_id"] not in purged


def test_a_held_purge_fails_the_job_after_the_rest_are_purged(client, world, new_session, monkeypatch):
    product = world.product()
    held, done = _open(client, product, "acme-held"), _open(client, product, "acme-done")
    for made in (held, done):
        _closed_days_ago(new_session, made["workspace_id"], 9)
    tried = []

    def purge(workspace_id):
        tried.append(str(workspace_id))
        return SimpleNamespace(skipped_reason="s3_delete_failed" if str(workspace_id) == held["workspace_id"] else None)

    monkeypatch.setattr("services.workspace_purge.purge_workspace_sync", purge)
    with pytest.raises(RuntimeError, match="held"):
        asyncio.run(product_purge.purge_closed(SimpleNamespace()))
    assert {held["workspace_id"], done["workspace_id"]} <= set(tried)


def test_the_purge_is_a_daily_platform_schedule():
    from core.jobs.registry import CATCH_UP_LATEST, kind_of
    from core.jobs.schedules import PLATFORM, forget

    added = []
    scheduler = SimpleNamespace(add_job=lambda fn, trigger, **kw: added.append((trigger, kw["id"])))
    started = product_purge.schedules_the_closed_purge(lambda self: "started")
    try:
        assert started(SimpleNamespace(_scheduler=scheduler)) == "started"
    finally:
        forget(product_purge.PURGE_KIND, PLATFORM)
    (trigger, timer), (when, _catch_up) = added
    assert timer == f"schedule:{product_purge.PURGE_KIND}:{PLATFORM}" and when == "date"
    assert "hour='2'" in str(trigger) and "minute='50'" in str(trigger)
    assert kind_of(product_purge.PURGE_KIND).catch_up == CATCH_UP_LATEST
