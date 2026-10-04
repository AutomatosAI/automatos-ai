"""PRD-251C Wave 2, US-C201 — the wave's one migration: ``social_posts.batch_key``.

Pins:

* **One revision.** ``prd251c_wave2`` chains onto ``prd251b_wave3`` and is the only revision
  chained there; the head pins (test_prd209_alembic_single_head, test_prd236_w1_routes)
  follow it.
* **The migration builds the model's schema:** from where PRD-251B leaves a database, the
  reflected ``social_posts`` is the model's, facet by facet, with ``batch_key`` indexed
  with the plan.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no duplicate column
  or index, from a fresh database and from the migration path.
* **The downgrade** drops exactly what the upgrade added and keeps every post; an upgrade
  after it works again.
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
    _assert_same_schema,
    _load,
    _run,
    _schema,
    _wave1_engine,
)

POSTS = ("social_posts",)


def _wave2b_engine():
    """Where PRD-251 and PRD-251B leave a database, before this wave."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
        _run(conn, WAVE1B, "upgrade")
        _run(conn, WAVE2B, "upgrade")
    return engine


def test_the_one_wave_revision_chains_onto_prd251b_wave3():
    mod = _load(WAVE2C)
    assert (mod.revision, mod.down_revision) == ("prd251c_wave2", "prd251b_wave3")
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(r"^down_revision\s*=\s*['\"]prd251b_wave3['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["prd251c_wave2.py"]


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, engine = _model_engine(), _wave2b_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE2C, "upgrade")
        migrated = _schema(engine, POSTS)
        _assert_same_schema(migrated, _schema(model_engine, POSTS), "the migration path through prd251c_wave2")
        posts = migrated["social_posts"]
        assert "batch_key" in set(posts["columns"])
        assert ("ix_social_posts_campaign_batch", ("campaign_id", "batch_key"), False) in posts["indexes"]
    finally:
        model_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("start", ("fresh", "migration_path"))
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave2b_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE2C, "upgrade")
            _run(conn, WAVE2C, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(model_engine, POSTS), f"create_all first, from {start}")
    finally:
        model_engine.dispose()
        engine.dispose()


def test_the_downgrade_drops_what_it_added_keeps_every_post_and_the_upgrade_works_again():
    engine = _wave2b_engine()
    post_id = uuid.uuid4()
    try:
        before = _schema(engine, POSTS)
        with engine.begin() as conn:
            _run(conn, WAVE2C, "upgrade")
            conn.execute(sa.insert(SocialPost.__table__).values(
                id=post_id, workspace_id=uuid.uuid4(), created_by="user-1", title="Week post", content_hash="0" * 64,
                batch_key="2026-W42",
            ))
            _run(conn, WAVE2C, "downgrade")
        _assert_same_schema(_schema(engine, POSTS), before, "after the downgrade")
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT title FROM social_posts")).scalars().all() == ["Week post"]
        with engine.begin() as conn:
            _run(conn, WAVE2C, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(_model_engine(), POSTS), "upgraded again")
    finally:
        engine.dispose()
