"""PRD-251C Wave 4, US-C401 — the wave's one migration: ``social_post_stats`` and
``social_voice_examples``.

Pins:

* **One revision.** ``prd251c_wave4`` chains onto ``prd251c_wave2`` and is the only revision
  chained there; the head pins (test_prd209_alembic_single_head, test_prd236_w1_routes)
  follow it.
* **The migration builds the model's schema:** from where Wave 2 leaves a database, the two
  reflected tables are the models', facet by facet: one stats row per target and reading,
  the keys and their ON DELETE, the indexes.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no duplicate table or
  index, from a fresh database and from the migration path.
* **The downgrade** drops exactly what the upgrade added and keeps every post; an upgrade
  after it works again.
"""
from __future__ import annotations

import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402

from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget, SocialVoiceExample  # noqa: E402
from tests.test_prd251bw2_migration import _model_engine as _wave2b_model_engine  # noqa: E402
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

WAVE4C = VERSIONS / "prd251c_wave4.py"
NEW = ("social_post_stats", "social_voice_examples")
BEFORE = ("social_posts", "social_post_targets")


def _model_engine():
    """What create_all builds from the models, the wave's two tables included."""
    engine = _wave2b_model_engine()
    SocialPost.metadata.create_all(engine, tables=[SocialPostTarget.__table__, SocialPostStat.__table__, SocialVoiceExample.__table__])
    return engine


def _wave2c_engine():
    """Where PRD-251B and PRD-251C Wave 2 leave a database, before this wave."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
        _run(conn, WAVE1B, "upgrade")
        _run(conn, WAVE2B, "upgrade")
        _run(conn, WAVE2C, "upgrade")
    return engine


def test_the_one_wave_revision_chains_onto_prd251c_wave2():
    mod = _load(WAVE4C)
    assert (mod.revision, mod.down_revision) == ("prd251c_wave4", "prd251c_wave2")
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(r"^down_revision\s*=\s*['\"]prd251c_wave2['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["prd251c_wave4.py"]


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, engine = _model_engine(), _wave2c_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE4C, "upgrade")
        migrated = _schema(engine, NEW)
        _assert_same_schema(migrated, _schema(model_engine, NEW), "the migration path through prd251c_wave4")
        stats, examples = migrated["social_post_stats"], migrated["social_voice_examples"]
        assert {"target_id", "reading", "read_at", "numbers", "source_action"} <= set(stats["columns"])
        assert (("target_id",), "social_post_targets", ("id",), "CASCADE") in stats["fks"]
        assert (("post_id",), "social_posts", ("id",), "SET NULL") in examples["fks"]
        assert {"draft", "approved", "created_at"} <= set(examples["columns"])
    finally:
        model_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("start", ("fresh", "migration_path"))
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave2c_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE4C, "upgrade")
            _run(conn, WAVE4C, "upgrade")
        _assert_same_schema(_schema(engine, NEW), _schema(model_engine, NEW), f"create_all first, from {start}")
    finally:
        model_engine.dispose()
        engine.dispose()


def test_the_downgrade_drops_what_it_added_keeps_every_post_and_the_upgrade_works_again():
    engine = _wave2c_engine()
    try:
        before = _schema(engine, BEFORE)
        with engine.begin() as conn:
            _run(conn, WAVE4C, "upgrade")
            conn.execute(sa.insert(SocialPost.__table__).values(
                id=uuid.uuid4(), workspace_id=uuid.uuid4(), created_by="user-1", title="Went out", content_hash="0" * 64,
            ))
            conn.execute(sa.insert(SocialVoiceExample.__table__).values(
                id=uuid.uuid4(), workspace_id=uuid.uuid4(), draft="Auto's.", approved="Ours.", created_at=datetime.now(timezone.utc),
            ))
            _run(conn, WAVE4C, "downgrade")
        assert not any(sa.inspect(engine).has_table(table) for table in NEW)
        _assert_same_schema(_schema(engine, BEFORE), before, "after the downgrade")
        with engine.connect() as conn:
            assert conn.execute(sa.text("SELECT title FROM social_posts")).scalars().all() == ["Went out"]
        with engine.begin() as conn:
            _run(conn, WAVE4C, "upgrade")
        _assert_same_schema(_schema(engine, NEW), _schema(_model_engine(), NEW), "upgraded again")
    finally:
        engine.dispose()
