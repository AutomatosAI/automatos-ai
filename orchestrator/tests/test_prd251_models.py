"""PRD-251 S0.2 (D2) — social_posts and social_post_targets: the model and the migration, held together.

Two writers build these tables: the model layer (``create_all`` — the fresh
path and the unit tests) and the migrations (existing databases): the
``prd251_socials`` migration, plus the columns Wave 1's one migration adds
(``prd251_wave1``: ``social_posts.voice``, US-111, and ``social_posts.footage``,
US-114). A table that differs between
them is how a column goes missing on one path, so the migrations are RUN here
(alembic ``Operations`` on SQLite) and their schema is compared with the
model's, table by table: columns (type, nullability, server default), primary
key, the unique ``idempotency_key``, check constraints, foreign keys and indexes.

Also pinned:

* both tables build under SQLite via ``create_all`` (the JSON variant, not
  bare JSONB), and a duplicate ``idempotency_key`` is rejected;
* the migration round-trips (upgrade → downgrade → upgrade) on SQLite, and on
  real Postgres inside a rolled-back transaction (``@integration``: JSONB
  columns, the unique key, the seeded master switch) — skipped cleanly when no
  Postgres is reachable, run by ``test.yml``;
* exactly one new revision chained onto ``kb_multimodal_tables``, and no
  ``social_campaigns`` table anywhere (D2: Wave 2, only if series approval ships).
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from alembic.operations import Operations  # noqa: E402
from alembic.runtime.migration import MigrationContext  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from core.models.socials import (  # noqa: E402
    SOCIAL_POST_FORMATS,
    SOCIAL_POST_STATUSES,
    SOCIAL_TARGET_POST_KINDS,
    SOCIAL_TARGET_STATUSES,
    SocialPost,
    SocialPostTarget,
)
from core.models.system_settings import SystemSetting  # noqa: E402

MIGRATION = _ORCH / "alembic" / "versions" / "prd251_socials.py"
WAVE1_MIGRATION = _ORCH / "alembic" / "versions" / "prd251_wave1.py"
MODELS = _ORCH / "core" / "models" / "socials.py"
TABLES = ("social_posts", "social_post_targets")


def _load_migration(path=MIGRATION, name="prd251_socials_migration"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sqlite_engine():
    return sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )


def _model_engine():
    engine = _sqlite_engine()
    SocialPost.metadata.create_all(
        engine, tables=[SocialPost.__table__, SocialPostTarget.__table__]
    )
    return engine


def _run_migration(conn, *steps):
    """Run the migration's upgrade/downgrade on ``conn`` through a real alembic context."""
    mod = _load_migration()
    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        for step in steps:
            getattr(mod, step)()


def _migration_engine():
    engine = _sqlite_engine()
    SystemSetting.__table__.create(bind=engine)  # the seed's table
    with engine.begin() as conn:
        _run_migration(conn, "upgrade")
        # Wave 1's one migration adds social_posts.voice (US-111) and .footage
        # (US-114); its other steps are Postgres DDL on other tables, so only the
        # column steps run here.
        wave1 = _load_migration(WAVE1_MIGRATION, "prd251_wave1_migration_models")
        with Operations.context(MigrationContext.configure(conn)):
            wave1.add_post_voice_column()
            wave1.add_post_footage_column()
    return engine


def _norm(sql) -> str:
    return re.sub(r"\s+", " ", str(sql or "")).strip()


def _schema(engine) -> dict:
    """Everything a writer declares about the two tables, reflected."""
    insp = sa.inspect(engine)
    out = {}
    for table in TABLES:
        out[table] = {
            "columns": {
                c["name"]: (str(c["type"]), c["nullable"], _norm(c.get("default")))
                for c in insp.get_columns(table)
            },
            "pk": tuple(insp.get_pk_constraint(table)["constrained_columns"]),
            "unique": {
                (u["name"], tuple(u["column_names"])) for u in insp.get_unique_constraints(table)
            },
            "checks": {(c["name"], _norm(c["sqltext"])) for c in insp.get_check_constraints(table)},
            "fks": {
                (
                    tuple(f["constrained_columns"]),
                    f["referred_table"],
                    tuple(f["referred_columns"]),
                    (f.get("options") or {}).get("ondelete"),
                )
                for f in insp.get_foreign_keys(table)
            },
            "indexes": {
                (i["name"], tuple(i["column_names"]), bool(i["unique"]))
                for i in insp.get_indexes(table)
            },
        }
    return out


# ---------------------------------------------------------------------------
# D2 — the shape
# ---------------------------------------------------------------------------


def test_d2_value_sets():
    assert SOCIAL_POST_STATUSES == (
        "draft", "rendering", "needs_approval", "changes_requested", "approved", "scheduled",
        "publishing", "published", "partially_published", "failed", "missed", "archived",
    )
    assert SOCIAL_POST_FORMATS == ("video", "image", "carousel", "fact_card", "infographic")
    assert SOCIAL_TARGET_POST_KINDS == ("text", "image", "carousel", "video", "reel", "short", "story")
    assert SOCIAL_TARGET_STATUSES == ("pending", "uploading", "published", "failed")


def test_social_posts_carries_every_d2_column():
    columns = set(SocialPost.__table__.columns.keys())
    assert columns == {
        "id", "workspace_id", "created_by", "campaign_id", "title", "brief", "copy", "format",
        "template_id", "variables", "sources", "media", "status", "content_hash",
        "approved_hash", "approved_by", "approved_at", "override_unsourced", "review_log",
        "scheduled_for", "timezone", "created_at", "updated_at",
        # Wave 1 (S1.5, D11): the voice a render speaks the script with.
        "voice",
        # Wave 1 (S1.8, D12): the footage a post asks its template's slots for.
        "footage",
    }


def test_social_post_targets_carries_every_d2_column():
    columns = set(SocialPostTarget.__table__.columns.keys())
    assert columns == {
        "id", "post_id", "toolkit", "post_kind", "action_plan", "idempotency_key", "status",
        "attempts", "remote_id", "permalink", "error", "published_at",
    }


@pytest.mark.parametrize(
    "table, column",
    [
        ("social_posts", "copy"),
        ("social_posts", "variables"),
        ("social_posts", "sources"),
        ("social_posts", "media"),
        ("social_posts", "voice"),
        ("social_posts", "footage"),
        ("social_posts", "review_log"),
        ("social_post_targets", "action_plan"),
    ],
)
def test_json_columns_are_jsonb_on_postgres_and_json_elsewhere(table, column):
    from sqlalchemy.dialects import postgresql, sqlite

    col_type = SocialPost.metadata.tables[table].c[column].type
    assert isinstance(col_type.dialect_impl(postgresql.dialect()), postgresql.JSONB)
    assert not isinstance(col_type.dialect_impl(sqlite.dialect()), postgresql.JSONB)


def test_the_post_to_target_foreign_key_cascades():
    fk = next(iter(SocialPostTarget.__table__.c.post_id.foreign_keys))
    assert fk.target_fullname == "social_posts.id"
    assert fk.ondelete == "CASCADE"


# ---------------------------------------------------------------------------
# SQLite: create_all builds both tables; the unique key holds
# ---------------------------------------------------------------------------


def test_both_tables_build_under_sqlite_via_create_all():
    engine = _model_engine()
    try:
        assert set(TABLES) <= set(sa.inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_a_duplicate_idempotency_key_is_rejected():
    engine = _model_engine()
    session = sessionmaker(bind=engine)()
    try:
        post = SocialPost(
            workspace_id=uuid.uuid4(), created_by="user-1", title="Launch", content_hash="0" * 64
        )
        session.add(post)
        session.flush()
        assert post.status == "draft" and post.copy == {} and post.review_log == []

        session.add(
            SocialPostTarget(post_id=post.id, toolkit="linkedin", post_kind="text", idempotency_key="key-1")
        )
        session.flush()
        session.add(
            SocialPostTarget(post_id=post.id, toolkit="twitter", post_kind="text", idempotency_key="key-1")
        )
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.close()
        engine.dispose()


def test_a_status_outside_d2_is_rejected():
    engine = _model_engine()
    session = sessionmaker(bind=engine)()
    try:
        session.add(
            SocialPost(
                workspace_id=uuid.uuid4(), created_by="user-1", title="Bad",
                content_hash="0" * 64, status="posted",
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# The migration and the model are the same schema
# ---------------------------------------------------------------------------


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, migration_engine = _model_engine(), _migration_engine()
    try:
        model, migration = _schema(model_engine), _schema(migration_engine)
    finally:
        model_engine.dispose()
        migration_engine.dispose()

    for table in TABLES:
        for facet in ("columns", "pk", "unique", "checks", "fks", "indexes"):
            assert migration[table][facet] == model[table][facet], (
                f"{table}.{facet} drifted between the migrations (prd251_socials, prd251_wave1) and "
                f"core/models/socials.py:\n migration={migration[table][facet]}\n model={model[table][facet]}"
            )

    # Non-vacuous: the reflected facets actually carry the declared constraints.
    targets = model["social_post_targets"]
    assert ("uq_social_post_targets_idempotency_key", ("idempotency_key",)) in targets["unique"]
    assert {name for name, _sql in targets["checks"]} == {
        "ck_social_post_targets_post_kind", "ck_social_post_targets_status",
    }
    assert any(fk[:3] == (("post_id",), "social_posts", ("id",)) for fk in targets["fks"])
    posts = model["social_posts"]
    assert {name for name, _sql in posts["checks"]} == {"ck_social_posts_status", "ck_social_posts_format"}
    assert any(fk[:3] == (("workspace_id",), "workspaces", ("id",)) for fk in posts["fks"])
    assert ("ix_social_posts_workspace_status", ("workspace_id", "status"), False) in posts["indexes"]
    assert (
        "ix_social_posts_workspace_scheduled_for", ("workspace_id", "scheduled_for"), False
    ) in posts["indexes"]


def test_the_migration_round_trips_and_the_downgrade_drops_exactly_what_it_created():
    engine = _sqlite_engine()
    SystemSetting.__table__.create(bind=engine)
    table = SystemSetting.__table__
    try:
        with engine.begin() as conn:
            conn.execute(table.insert().values(category="voice", key="live_enabled", value="false", created_by="prd207"))
            before = set(sa.inspect(conn).get_table_names())

            _run_migration(conn, "upgrade")
            assert set(sa.inspect(conn).get_table_names()) == before | set(TABLES)
            seeded = conn.execute(sa.text("SELECT category, key FROM system_settings ORDER BY category")).fetchall()
            assert [tuple(r) for r in seeded] == [
                ("composio", "denied_actions"), ("socials", "enabled"), ("voice", "live_enabled"),
            ]

            _run_migration(conn, "downgrade")
            assert set(sa.inspect(conn).get_table_names()) == before
            left = conn.execute(sa.text("SELECT category, key FROM system_settings")).fetchall()
            assert [tuple(r) for r in left] == [("voice", "live_enabled")]

            _run_migration(conn, "upgrade")
            assert set(TABLES) <= set(sa.inspect(conn).get_table_names())
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# One revision, no social_campaigns
# ---------------------------------------------------------------------------


def test_one_revision_chained_onto_kb_multimodal_tables():
    mod = _load_migration()
    assert (mod.revision, mod.down_revision) == ("prd251_socials", "kb_multimodal_tables")
    versions = _ORCH / "alembic" / "versions"
    chained = [
        p.name for p in versions.glob("*.py")
        if re.search(r"^down_revision\s*=\s*['\"]kb_multimodal_tables['\"]", p.read_text(encoding="utf-8"), re.M)
    ]
    # One Socials revision. In the merged tree (refresh 2) F049's llm_usage_agent_name
    # chains onto kb_multimodal_tables too, and f049_prd251_merge_heads joins the two.
    assert sorted(chained) == ["llm_usage_agent_name.py", "prd251_socials.py"]


def test_no_social_campaigns_table_or_model():
    assert "social_campaigns" not in SocialPost.metadata.tables
    for path in (MIGRATION, MODELS):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"create_table\(\s*['\"]social_campaigns", source)
        assert "__tablename__ = \"social_campaigns\"" not in source


# ---------------------------------------------------------------------------
# @integration — the round trip on real Postgres (rolled back)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM system_settings LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres round trip needs the test database: {exc}")
    yield engine
    engine.dispose()


def _jsonb_columns(conn, table) -> set:
    rows = conn.execute(
        sa.text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :t AND udt_name = 'jsonb'"
        ),
        {"t": table},
    ).fetchall()
    return {r[0] for r in rows}


@pytest.mark.integration
def test_the_migration_round_trips_on_postgres(pg_engine):
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            insp = sa.inspect(conn)
            if all(insp.has_table(t) for t in TABLES):
                _run_migration(conn, "downgrade")  # the create_all-built tables go first
            assert not any(sa.inspect(conn).has_table(t) for t in TABLES)

            _run_migration(conn, "upgrade")
            assert all(sa.inspect(conn).has_table(t) for t in TABLES)
            assert _jsonb_columns(conn, "social_posts") == {
                "copy", "variables", "sources", "media", "review_log",
            }
            assert _jsonb_columns(conn, "social_post_targets") == {"action_plan"}
            # pg_constraint.confdeltype 'c' = ON DELETE CASCADE.
            delete_rules = conn.execute(
                sa.text(
                    "SELECT conrelid::regclass::text, confrelid::regclass::text, confdeltype "
                    "FROM pg_constraint WHERE contype = 'f' "
                    "AND conrelid::regclass::text IN ('social_posts', 'social_post_targets')"
                )
            ).fetchall()
            assert {tuple(r) for r in delete_rules} == {
                ("social_posts", "workspaces", "c"),
                ("social_post_targets", "social_posts", "c"),
            }
            seeded = conn.execute(
                sa.text(
                    "SELECT value_type FROM system_settings "
                    "WHERE category = 'socials' AND key = 'enabled'"
                )
            ).fetchall()
            assert [tuple(r) for r in seeded] == [("boolean",)]

            ws_id = conn.execute(sa.text("SELECT id FROM workspaces LIMIT 1")).scalar()
            if ws_id is not None:
                post_id = str(uuid.uuid4())
                conn.execute(
                    sa.text(
                        "INSERT INTO social_posts (id, workspace_id, created_by, title, copy, "
                        "variables, sources, media, content_hash, review_log) VALUES "
                        "(CAST(:id AS uuid), CAST(:ws AS uuid), 'u', 't', '{}', '{}', '{}', '{}', :h, '[]')"
                    ),
                    {"id": post_id, "ws": str(ws_id), "h": "0" * 64},
                )
                insert_target = sa.text(
                    "INSERT INTO social_post_targets (id, post_id, toolkit, post_kind, "
                    "action_plan, idempotency_key) VALUES "
                    "(CAST(:id AS uuid), CAST(:post AS uuid), 'linkedin', 'text', '{}', 'dup-key')"
                )
                conn.execute(insert_target, {"id": str(uuid.uuid4()), "post": post_id})
                nested = conn.begin_nested()
                with pytest.raises(IntegrityError):
                    conn.execute(insert_target, {"id": str(uuid.uuid4()), "post": post_id})
                nested.rollback()

            _run_migration(conn, "downgrade")
            assert not any(sa.inspect(conn).has_table(t) for t in TABLES)
            assert conn.execute(
                sa.text("SELECT count(*) FROM system_settings WHERE created_by = 'prd251'")
            ).scalar() == 0
        finally:
            trans.rollback()
