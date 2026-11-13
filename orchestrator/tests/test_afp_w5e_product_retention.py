"""PRD-274 Wave 5e (decision 28): a product's default retention reaches its workspaces' new end users.

``PUT /api/memory/end-users/{id}`` for a new end user of a product's workspace, with no ``retention_days``, records
the product's ``default_retention_days``; an explicit value, ``null`` included, is kept as sent. An end user who
already exists never takes the default (their memories are kept as they consented), and a workspace that is no
product's, or a product with no default, records none, as before. A consent recorded again after an erase is a
new end user, so it takes the default.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from api import end_user_memory as eum
from core.models.end_users import EndUser
from core.models.products import Product
from services.products import mint_key


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    for table in (Product.__table__, EndUser.__table__):
        table.create(eng, checkfirst=True)  # a test database built before these
    with eng.begin() as conn:
        conn.execute(text("ALTER TABLE products ADD COLUMN IF NOT EXISTS reuse_existing_key BOOLEAN NOT NULL "
                          "DEFAULT true"))
        conn.execute(text("ALTER TABLE end_users ADD COLUMN IF NOT EXISTS erasing_at TIMESTAMPTZ"))
    yield eng
    eng.dispose()


@pytest.fixture
def world(new_session):
    """An owner, its product (default 30 days), the product's workspace, and a plain workspace; swept after."""
    owner, customer, plain = (str(uuid.uuid4()) for _ in range(3))
    key = mint_key()
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:o AS uuid), 'w5e-owner'), "
                   "(CAST(:p AS uuid), 'w5e-plain')"), {"o": owner, "p": plain})
    product = Product(slug=f"w5e-{owner[:8]}", name="Support desk", owner_workspace_id=uuid.UUID(owner),
                      key_hash=key.key_hash, key_prefix=key.key_prefix, default_retention_days=30)
    s.add(product)
    s.commit()
    s.execute(text("INSERT INTO workspaces (id, name, settings) VALUES (CAST(:c AS uuid), 'w5e-customer', "
                   "CAST(:s AS jsonb))"), {"c": customer, "s": f'{{"product_id": "{product.id}"}}'})
    s.commit()
    s.close()
    yield SimpleNamespace(owner=owner, customer=customer, plain=plain, product=str(product.id))
    s = new_session.sweep()
    s.execute(text("DELETE FROM end_users WHERE workspace_id = ANY(CAST(:ids AS uuid[]))"), {"ids": [customer, plain]})
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ids AS uuid[]))"), {"ids": [customer, plain]})
    s.execute(text("DELETE FROM products WHERE id = CAST(:p AS uuid)"), {"p": str(product.id)})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:o AS uuid)"), {"o": owner})
    s.commit()
    s.close()


def _put(new_session, workspace_id, end_user_id, **fields):
    ctx = SimpleNamespace(workspace_id=uuid.UUID(workspace_id), api_key_id=None, auth_type="clerk", user=None)
    s = new_session()
    try:
        return eum.record_consent(eum.EndUserConsent(consent=True, **fields), end_user_id=end_user_id, ctx=ctx,
                                  db=s)["retention_days"]
    finally:
        s.close()


def test_a_new_end_user_of_a_products_workspace_takes_its_default(new_session, world):
    assert _put(new_session, world.customer, "shopper-1") == 30


@pytest.mark.parametrize("sent", [7, None])
def test_what_the_product_sends_is_kept_null_included(new_session, world, sent):
    assert _put(new_session, world.customer, "shopper-2", retention_days=sent) == sent


def test_an_end_user_who_exists_never_takes_the_default(new_session, world):
    assert _put(new_session, world.customer, "shopper-3", retention_days=None) is None
    assert _put(new_session, world.customer, "shopper-3") is None  # an update sets what it sends, as before
    assert _put(new_session, world.customer, "shopper-3", retention_days=14) == 14


def test_any_other_workspace_or_a_product_with_no_default_records_none(new_session, world):
    assert _put(new_session, world.plain, "visitor-1") is None
    s = new_session()
    s.execute(text("UPDATE products SET default_retention_days = NULL WHERE id = CAST(:p AS uuid)"),
              {"p": world.product})
    s.commit()
    s.close()
    assert _put(new_session, world.customer, "shopper-4") is None


def test_a_consent_recorded_again_after_an_erase_is_new_and_takes_the_default(new_session, world):
    assert _put(new_session, world.customer, "shopper-5", retention_days=None) is None
    s = new_session()
    s.execute(text("DELETE FROM end_users WHERE workspace_id = CAST(:w AS uuid) AND end_user_id = 'shopper-5'"),
              {"w": world.customer})  # the erase's last step deletes the record
    s.commit()
    s.close()
    assert _put(new_session, world.customer, "shopper-5") == 30
