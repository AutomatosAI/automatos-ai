"""PRD-251D Wave 3, US-D301 — the wave's one migration: ``social_posts.look``.

Pins:

* **One revision.** ``prd251d_wave3`` chains onto ``credit_ledger_sub_tenant_20261113``, the head
  it was written on. ``chat_turn_keys_20261116`` was written on the same head and merged the same
  morning, so ``merge_heads_20261117`` joins the two back into one head; the head pins
  (test_prd209_alembic_single_head, test_prd236_w1_routes) follow the merge.
* **The migration builds the model's schema:** from where PRD-251C leaves a database, the
  reflected ``social_posts`` is the model's, facet by facet, with ``look`` a nullable JSON column.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no duplicate column, from a
  fresh database and from the migration path.
* **The downgrade** drops exactly the column and keeps every post; an upgrade after it works
  again, and a look written to the column reads back as it was written.
"""
from __future__ import annotations

import re
import sys
import uuid
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

from core.models.socials import SocialPost  # noqa: E402
from tests.test_prd251bw2_migration import _model_engine  # noqa: E402
from tests.test_prd251w2_campaigns import (  # noqa: E402
    VERSIONS,
    WAVE1B,
    WAVE2,
    WAVE2B,
    WAVE2C,
    WAVE3D,
    _assert_same_schema,
    _load,
    _run,
    _schema,
    _wave1_engine,
)

POSTS = ("social_posts",)
PARENT = "credit_ledger_sub_tenant_20261113"
LOOK = {"mode": "dark", "logo": False}


def _wave2c_engine():
    """Where PRD-251, PRD-251B and PRD-251C leave ``social_posts``, before this wave."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
        _run(conn, WAVE1B, "upgrade")
        _run(conn, WAVE2B, "upgrade")
        _run(conn, WAVE2C, "upgrade")
    return engine


def _insert_post(conn, **values) -> uuid.UUID:
    post_id = uuid.uuid4()
    conn.execute(sa.insert(SocialPost.__table__).values(
        id=post_id, workspace_id=uuid.uuid4(), created_by="user-1", title="Dark card", content_hash="0" * 64, **values,
    ))
    return post_id


def test_the_one_wave_revision_chains_onto_the_head_it_was_written_on():
    mod = _load(WAVE3D)
    assert (mod.revision, mod.down_revision) == ("prd251d_wave3", PARENT)
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(rf"^down_revision\s*=\s*['\"]{PARENT}['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    # Two revisions were written on that head and merged the same morning (17 Nov). Neither is
    # re-chained, which a database stamped at both would refuse: a merge revision joins them.
    assert chained == ["chat_turn_keys_20261116.py", "prd251d_wave3.py"]
    merge = _load(VERSIONS / "merge_heads_20261117.py")
    assert set(merge.down_revision) == {"prd251d_wave3", "chat_turn_keys_20261116"}


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, engine = _model_engine(), _wave2c_engine()
    try:
        assert "look" not in _schema(engine, POSTS)["social_posts"]["columns"]  # non-vacuous: the wave adds it
        with engine.begin() as conn:
            _run(conn, WAVE3D, "upgrade")
        migrated = _schema(engine, POSTS)
        _assert_same_schema(migrated, _schema(model_engine, POSTS), "the migration path through prd251d_wave3")
        _type, nullable, _default = migrated["social_posts"]["columns"]["look"]
        assert nullable is True
    finally:
        model_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("start", ("fresh", "migration_path"))
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave2c_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE3D, "upgrade")
            _run(conn, WAVE3D, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(model_engine, POSTS), f"create_all first, from {start}")
    finally:
        model_engine.dispose()
        engine.dispose()


def test_the_downgrade_drops_the_column_keeps_every_post_and_the_upgrade_works_again():
    engine = _wave2c_engine()
    try:
        before = _schema(engine, POSTS)
        with engine.begin() as conn:
            _run(conn, WAVE3D, "upgrade")
            post_id = _insert_post(conn, look=LOOK)
            looks = SocialPost.__table__.c.look
            assert conn.execute(sa.select(looks).where(SocialPost.__table__.c.id == post_id)).scalar_one() == LOOK
            _run(conn, WAVE3D, "downgrade")
        _assert_same_schema(_schema(engine, POSTS), before, "after the downgrade")
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT title FROM social_posts")).scalars().all() == ["Dark card"]
        with engine.begin() as conn:
            _run(conn, WAVE3D, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(_model_engine(), POSTS), "upgraded again")
    finally:
        engine.dispose()
