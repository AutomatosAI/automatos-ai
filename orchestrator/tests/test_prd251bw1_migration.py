"""PRD-251B Wave 1, US-B101 — the wave's one migration: ``planned_for``, ``length_seconds``, the text format.

Pins:

* **One revision.** ``prd251b_wave1`` chains onto ``prd252_ticket_numbers`` (the merge
  revision #859 made for the two heads #852 left) and is the only revision chained
  there; the head pins (test_prd209_alembic_single_head, test_prd236_w1_routes) follow it.
* **The migration builds the model's schema.** On SQLite, alembic ``Operations`` run
  the Socials migrations for real (Waves 0, 1 and 2, then this one) and the reflected
  ``social_posts`` is compared with the model's, facet by facet: the two columns, the
  widened format CHECK, the length CHECK and the planned_for index.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no duplicate
  column, CHECK or index, and the schema is the model's; from a fresh database and
  from the migration path.
* **The CHECKs hold:** a ``text`` post inserts, a bogus format and a zero length are
  refused by the database.
* **The downgrade** restores the old CHECK and drops the columns; an upgrade after it
  works again.
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

from core.models.socials import POST_LENGTH_SECONDS_CHECK, SOCIAL_POST_FORMATS, SocialPost  # noqa: E402
from tests.test_prd251w2_campaigns import (  # noqa: E402
    VERSIONS,
    WAVE2,
    _assert_same_schema,
    _load,
    _model_engine,
    _run,
    _schema,
    _wave1_engine,
)

WAVE1B = VERSIONS / "prd251b_wave1.py"
# The models also carry PRD-251B Wave 2's post columns: a comparison with them runs it after.
WAVE2B = VERSIONS / "prd251b_wave2.py"
BASE_HEAD = "prd252_ticket_numbers"
POSTS = ("social_posts",)


def _wave2_engine():
    """Where Waves 0, 1 and 2 leave a database, before this wave."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
    return engine


def _insert(conn, **values):
    row = {
        "id": uuid.uuid4(), "workspace_id": uuid.uuid4(), "created_by": "user-1",
        "title": "Launch", "content_hash": "0" * 64,
    }
    row.update(values)
    conn.execute(sa.insert(SocialPost.__table__).values(**row))


# ---------------------------------------------------------------------------
# One revision
# ---------------------------------------------------------------------------


def test_the_one_wave_revision_chains_onto_the_single_head():
    mod = _load(WAVE1B)
    assert (mod.revision, mod.down_revision) == ("prd251b_wave1", BASE_HEAD)
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(rf"^down_revision\s*=\s*['\"]{BASE_HEAD}['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    assert chained == ["prd251b_wave1.py"]
    # The migration writes the model's CHECK texts.
    assert mod.POST_LENGTH_CHECK == POST_LENGTH_SECONDS_CHECK
    assert "text" in SOCIAL_POST_FORMATS and mod.POST_FORMAT_CHECK.endswith("'text')")


# ---------------------------------------------------------------------------
# SQLite: the migration builds the model's schema; create_all first; the CHECKs; the downgrade
# ---------------------------------------------------------------------------


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, engine = _model_engine(), _wave2_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE1B, "upgrade")
            _run(conn, WAVE2B, "upgrade")
        migrated, model = _schema(engine, POSTS), _schema(model_engine, POSTS)
        _assert_same_schema(migrated, model, "the migration path through prd251b_wave1")
        posts = migrated["social_posts"]
        assert {"planned_for", "length_seconds"} <= set(posts["columns"])
        assert ("ck_social_posts_length_seconds", POST_LENGTH_SECONDS_CHECK) in posts["checks"]
        assert any(name == "ck_social_posts_format" and "'text'" in sql for name, sql in posts["checks"])
        assert ("ix_social_posts_workspace_planned_for", ("workspace_id", "planned_for"), False) in posts["indexes"]
    finally:
        model_engine.dispose()
        engine.dispose()


@pytest.mark.parametrize("start", ("fresh", "migration_path"))
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave2_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE1B, "upgrade")
            _run(conn, WAVE1B, "upgrade")
            _run(conn, WAVE2B, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(model_engine, POSTS), f"create_all first, from {start}")
        checks = [name for name, _sql in _schema(engine, POSTS)["social_posts"]["checks"]]
        assert checks.count("ck_social_posts_format") == 1 and checks.count("ck_social_posts_length_seconds") == 1
    finally:
        model_engine.dispose()
        engine.dispose()


def test_a_text_post_inserts_and_a_bogus_format_or_a_zero_length_is_refused():
    engine = _wave2_engine()
    try:
        with engine.begin() as conn:
            _run(conn, WAVE1B, "upgrade")
            _insert(conn, format="text")
            _insert(conn, format="video", length_seconds=15)
        for bad in ({"format": "bogus"}, {"format": "video", "length_seconds": 0}):
            with pytest.raises(IntegrityError), engine.begin() as conn:
                _insert(conn, **bad)
    finally:
        engine.dispose()


def test_the_downgrade_restores_the_old_check_and_drops_the_columns_and_the_upgrade_works_again():
    engine = _wave2_engine()
    try:
        before = _schema(engine, POSTS)
        with engine.begin() as conn:
            _run(conn, WAVE1B, "upgrade")
            _insert(conn, format="text")
            _run(conn, WAVE1B, "downgrade")
        _assert_same_schema(_schema(engine, POSTS), before, "after the downgrade")
        with engine.connect() as conn:
            formats = conn.execute(sa.select(SocialPost.__table__.c.format)).scalars().all()
        assert formats == [None]  # the text post keeps its row, without a format the old CHECK refuses
        with engine.begin() as conn:
            _run(conn, WAVE1B, "upgrade")
            _run(conn, WAVE2B, "upgrade")
        _assert_same_schema(_schema(engine, POSTS), _schema(_model_engine(), POSTS), "upgraded again")
    finally:
        engine.dispose()
