"""PRD-251B Wave 2, US-B201 — the wave's one migration: plans, slot keys, music and the content bank.

Pins:

* **One revision.** ``prd251b_wave2`` chains onto ``prd251b_wave1`` and is the only
  revision chained there; the head pins (test_prd209_alembic_single_head,
  test_prd236_w1_routes) follow it. Its CHECK texts are the model's.
* **The migration builds the model's schema.** On SQLite, alembic ``Operations`` run the
  Socials migrations for real and the reflected ``social_campaigns``, ``social_posts``
  and ``social_topics`` are compared with the models', facet by facet.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no duplicate
  column, CHECK, index or table; from a fresh database and from the migration path.
* **The rules hold:** a bogus kind, status, late policy or topic origin is refused; two
  posts of one plan cannot share a slot key, while posts without one never collide.
* **The downgrade** drops exactly what the upgrade added; an upgrade after it works again.
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
from sqlalchemy.exc import IntegrityError  # noqa: E402

from core.models.socials import (  # noqa: E402
    SOCIAL_CAMPAIGN_KINDS,
    SOCIAL_LATE_POLICIES,
    SOCIAL_PLAN_STATUSES,
    SOCIAL_TOPIC_ORIGINS,
    SocialCampaign,
    SocialPost,
    SocialTopic,
)
from tests.test_prd251w2_campaigns import (  # noqa: E402
    MODEL_TABLES,
    VERSIONS,
    WAVE1B,
    WAVE2,
    WAVE2B,
    _assert_same_schema,
    _load,
    _run,
    _schema,
    _sqlite_engine,
    _wave1_engine,
)

TABLES = ("social_campaigns", "social_posts", "social_topics")
BEFORE_TABLES = ("social_campaigns", "social_posts")


def _model_engine():
    """What create_all builds from the models, the content bank included."""
    engine = _sqlite_engine()
    SocialPost.metadata.create_all(engine, tables=MODEL_TABLES + [SocialTopic.__table__])
    return engine


def _wave1b_engine():
    """Where PRD-251 Waves 0-2 and PRD-251B Wave 1 leave a database, before this wave."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
        _run(conn, WAVE1B, "upgrade")
    return engine


def _in_list(column, values):
    return f"{column} IN ({', '.join(repr(v) for v in values)})"


def test_the_one_wave_revision_chains_onto_wave_1():
    mod = _load(WAVE2B)
    assert (mod.revision, mod.down_revision) == ("prd251b_wave2", "prd251b_wave1")
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(r"^down_revision\s*=\s*['\"]prd251b_wave1['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["prd251b_wave2.py"]
    assert dict(mod.CAMPAIGN_CHECKS) == {
        "ck_social_campaigns_kind": _in_list("kind", SOCIAL_CAMPAIGN_KINDS),
        "ck_social_campaigns_status": _in_list("status", SOCIAL_PLAN_STATUSES),
        "ck_social_campaigns_late_policy": _in_list("late_policy", SOCIAL_LATE_POLICIES),
    }
    assert mod.TOPIC_ORIGIN_CHECK[1] == _in_list("origin", SOCIAL_TOPIC_ORIGINS)


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, engine = _model_engine(), _wave1b_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE2B, "upgrade")
        migrated, model = _schema(engine, TABLES), _schema(model_engine, TABLES)
        _assert_same_schema(migrated, model, "the migration path through prd251b_wave2")
        campaigns, posts, topics = migrated["social_campaigns"], migrated["social_posts"], migrated["social_topics"]
        assert {"kind", "status", "cadence", "late_policy", "slot_overrides"} <= set(campaigns["columns"])
        assert {"slot_key", "music"} <= set(posts["columns"])
        assert ("uq_social_posts_campaign_slot_key", ("campaign_id", "slot_key"), True) in posts["indexes"]
        assert {"title", "facts", "formats", "pinned_on", "used_post_id", "origin"} <= set(topics["columns"])
        assert (("campaign_id",), "social_campaigns", ("id",), "CASCADE") in topics["fks"]
        assert (("used_post_id",), "social_posts", ("id",), "SET NULL") in topics["fks"]
    finally:
        model_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("start", ("fresh", "migration_path"))
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave1b_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE2B, "upgrade")
            _run(conn, WAVE2B, "upgrade")
        _assert_same_schema(_schema(engine, TABLES), _schema(model_engine, TABLES), f"create_all first, from {start}")
        checks = [name for name, _sql in _schema(engine, TABLES)["social_campaigns"]["checks"]]
        assert all(checks.count(name) == 1 for name in checks)
    finally:
        model_engine.dispose()
        engine.dispose()


def _plan(conn, **values):
    row = {"id": uuid.uuid4(), "workspace_id": uuid.uuid4(), "name": "Countdown", "created_by": "user-1", "kind": "plan"}
    row.update(values)
    conn.execute(sa.insert(SocialCampaign.__table__).values(**row))
    return row["id"]


def _post(conn, campaign_id, slot_key):
    conn.execute(sa.insert(SocialPost.__table__).values(
        id=uuid.uuid4(), workspace_id=uuid.uuid4(), created_by="user-1", title="Day 1", content_hash="0" * 64,
        campaign_id=campaign_id, slot_key=slot_key,
    ))


def test_the_rules_hold_and_a_slot_holds_one_post():
    engine = _wave1b_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE2B, "upgrade")
            plan = _plan(conn)
            _post(conn, plan, "r1|2026-10-14|07:00")
            _post(conn, plan, None)
            _post(conn, plan, None)  # posts without a slot never collide
            kind, status, late = conn.execute(sa.text("SELECT kind, status, late_policy FROM social_campaigns")).one()
            assert (kind, status, late) == ("plan", "active", "skip")
        for bad in (
            lambda conn: _post(conn, plan, "r1|2026-10-14|07:00"),
            lambda conn: _plan(conn, kind="bogus"),
            lambda conn: _plan(conn, status="asleep"),
            lambda conn: _plan(conn, late_policy="never"),
            lambda conn: conn.execute(sa.insert(SocialTopic.__table__).values(
                id=uuid.uuid4(), workspace_id=uuid.uuid4(), campaign_id=plan, title="T", created_by="u", origin="bot")),
        ):
            with pytest.raises(IntegrityError), engine.begin() as conn:
                bad(conn)
    finally:
        engine.dispose()


def test_the_downgrade_drops_what_it_added_and_the_upgrade_works_again():
    engine = _wave1b_engine()
    try:
        before = _schema(engine, BEFORE_TABLES)
        with engine.begin() as conn:
            _run(conn, WAVE2B, "upgrade")
            _plan(conn)
            _run(conn, WAVE2B, "downgrade")
        assert not sa.inspect(engine).has_table("social_topics")
        _assert_same_schema(_schema(engine, BEFORE_TABLES), before, "after the downgrade")
        with engine.begin() as conn:
            _run(conn, WAVE2B, "upgrade")
        _assert_same_schema(_schema(engine, TABLES), _schema(_model_engine(), TABLES), "upgraded again")
    finally:
        engine.dispose()
