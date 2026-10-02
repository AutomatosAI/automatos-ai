"""PRD-251 Wave 2, US-201 (S2.4a, D2) — ``social_campaigns`` and the campaign link: the wave's one migration.

Pins:

* **The shape (D2).** ``social_campaigns`` carries ``id``, ``workspace_id`` (ON
  DELETE CASCADE), ``name``, ``approval_mode`` (``per_post`` | ``series``, a named
  CHECK, ``per_post`` by default), ``approved_hash_set`` (a JSON list, JSONB on
  Postgres), ``approved_by``/``approved_at``, ``created_by``,
  ``created_at``/``updated_at`` and the (workspace_id, created_at) index.
  ``social_posts.campaign_id`` references it ON DELETE SET NULL, with its own index.
* **One revision.** ``prd251_wave2`` is the only revision chained onto the base head
  ``prd251w1_merge_heads`` and the only one that builds ``social_campaigns``; the
  head pins (test_prd209_alembic_single_head, test_prd236_w1_routes) follow it.
* **The migration builds the model's schema.** On SQLite, alembic ``Operations`` run
  the migrations for real (Waves 0 and 1, then this one) and the reflected schema is
  compared with the model's, table by table and facet by facet. The downgrade drops
  the key, the index and the table; the posts stay.
* **Create_all first (the 89d89c250 rule), then the upgrade twice:** no
  DuplicateTable, no second key or index, and the schema is the model's. Three
  starting points: create_all built everything (a fresh database); create_all built
  only the new table (a Wave 1 database whose backend loaded the new models before
  the migration ran, since create_all never alters a table that exists); nothing
  built (the migration path). On SQLite, and on real Postgres (``@integration``,
  rolled back; in CI a database that does not answer fails these tests instead of
  skipping them).
* **Deleting a campaign keeps its posts,** with ``campaign_id`` NULL: on SQLite with
  foreign keys enforced (both writers' schema), and on Postgres.
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
from sqlalchemy.dialects import postgresql, sqlite  # noqa: E402
from sqlalchemy.exc import IntegrityError  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the keys name)
from core.models.socials import (  # noqa: E402
    SOCIAL_CAMPAIGN_APPROVAL_MODES,
    SocialCampaign,
    SocialPost,
    SocialPostTarget,
)
from core.models.system_settings import SystemSetting  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"
WAVE0 = VERSIONS / "prd251_socials.py"
WAVE1 = VERSIONS / "prd251_wave1.py"
WAVE2 = VERSIONS / "prd251_wave2.py"
# PRD-251B Wave 1 (US-B101): the columns, CHECK and index it adds to social_posts are
# in the model now, so the migration path runs it after Wave 2.
WAVE1B = VERSIONS / "prd251b_wave1.py"
BASE_HEAD = "prd251w1_merge_heads"
CAMPAIGN_FK = "social_posts_campaign_id_fkey"
CAMPAIGN_INDEX = "ix_social_posts_campaign_id"
TABLES = ("social_campaigns", "social_posts", "social_post_targets")
MODEL_TABLES = [SocialCampaign.__table__, SocialPost.__table__, SocialPostTarget.__table__]
FACETS = ("columns", "pk", "unique", "checks", "fks", "indexes")
STARTS = ("fresh", "backend_first", "migration_path")
CAMPAIGN_COLUMNS = {
    "id", "workspace_id", "name", "approval_mode", "approved_hash_set", "approved_by",
    "approved_at", "created_by", "created_at", "updated_at",
}
# The workspaces table the keys name, as a stand-in: the real model is Postgres-first.
WORKSPACES_STANDIN = "CREATE TABLE workspaces (id CHAR(32) PRIMARY KEY)"
# Where Wave 1 left a database: no key, no post index, no campaigns table.
WAVE1_STATE_DDL = (
    f"ALTER TABLE social_posts DROP CONSTRAINT IF EXISTS {CAMPAIGN_FK}",
    f"DROP INDEX IF EXISTS {CAMPAIGN_INDEX}",
    "DROP TABLE IF EXISTS social_campaigns",
)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"{path.stem}_prd251w2_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(conn, path: Path, *steps: str) -> None:
    """Run a revision's steps on ``conn`` through a real alembic context."""
    mod = _load(path)
    with Operations.context(MigrationContext.configure(conn)):
        for step in steps:
            getattr(mod, step)()


def _sqlite_engine():
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with engine.begin() as conn:
        conn.exec_driver_sql(WORKSPACES_STANDIN)
    return engine


def _model_engine():
    """What create_all builds from the models: a fresh database."""
    engine = _sqlite_engine()
    SocialPost.metadata.create_all(engine, tables=MODEL_TABLES)
    return engine


def _wave1_engine():
    """Where Waves 0 and 1 leave a database: their migrations' Socials tables, no campaigns."""
    engine = _sqlite_engine()
    SystemSetting.__table__.create(bind=engine)  # the Wave 0 seed's table
    with engine.begin() as conn:
        _run(conn, WAVE0, "upgrade")
        # Wave 1's column steps; its other steps are Postgres DDL on other tables.
        _run(conn, WAVE1, "add_post_voice_column", "add_post_footage_column")
    return engine


def _migrated_engine():
    """The migration path: Waves 0 and 1, this wave's upgrade, then PRD-251B Wave 1's."""
    engine = _wave1_engine()
    with engine.begin() as conn:
        _run(conn, WAVE2, "upgrade")
        _run(conn, WAVE1B, "upgrade")
    return engine


def _norm(sql) -> str:
    return re.sub(r"\s+", " ", str(sql or "")).strip()


def _facets(insp, table: str) -> dict:
    """Everything a writer declares about one table, reflected."""
    return {
        "columns": {
            c["name"]: (str(c["type"]), c["nullable"], _norm(c.get("default")))
            for c in insp.get_columns(table)
        },
        "pk": tuple(insp.get_pk_constraint(table)["constrained_columns"]),
        "unique": {(u["name"], tuple(u["column_names"])) for u in insp.get_unique_constraints(table)},
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
            (i["name"], tuple(i["column_names"]), bool(i["unique"])) for i in insp.get_indexes(table)
        },
    }


def _schema(engine, tables=TABLES) -> dict:
    insp = sa.inspect(engine)
    return {table: _facets(insp, table) for table in tables}


def _assert_same_schema(actual: dict, expected: dict, label: str) -> None:
    for table in expected:
        for facet in FACETS:
            assert actual[table][facet] == expected[table][facet], (
                f"{table}.{facet} drifted ({label}) from core/models/socials.py:\n"
                f" got={actual[table][facet]}\n model={expected[table][facet]}"
            )


def _campaign_keys(engine) -> list:
    """Every key from social_posts to social_campaigns, by name: a list, so a second one shows."""
    return [
        fk["name"]
        for fk in sa.inspect(engine).get_foreign_keys("social_posts")
        if fk["referred_table"] == "social_campaigns"
    ]


# ---------------------------------------------------------------------------
# D2 — the shape
# ---------------------------------------------------------------------------


def test_social_campaigns_carries_every_d2_column():
    columns = SocialCampaign.__table__.columns
    assert set(columns.keys()) == CAMPAIGN_COLUMNS
    assert SOCIAL_CAMPAIGN_APPROVAL_MODES == ("per_post", "series")
    assert {name for name, column in columns.items() if column.nullable} == {"approved_by", "approved_at"}
    assert (columns["name"].type.length, columns["approval_mode"].type.length) == (200, 16)
    assert columns["approval_mode"].server_default.arg == "per_post"
    hash_set = columns["approved_hash_set"].type
    assert isinstance(hash_set.dialect_impl(postgresql.dialect()), postgresql.JSONB)
    assert not isinstance(hash_set.dialect_impl(sqlite.dialect()), postgresql.JSONB)
    (check,) = [c for c in SocialCampaign.__table__.constraints if isinstance(c, sa.CheckConstraint)]
    assert (check.name, str(check.sqltext)) == (
        "ck_social_campaigns_approval_mode", "approval_mode IN ('per_post', 'series')",
    )
    (index,) = SocialCampaign.__table__.indexes
    assert (index.name, [c.name for c in index.columns]) == (
        "ix_social_campaigns_workspace_created", ["workspace_id", "created_at"],
    )
    (workspace_fk,) = columns["workspace_id"].foreign_keys
    assert (workspace_fk.target_fullname, workspace_fk.ondelete) == ("workspaces.id", "CASCADE")


def test_campaign_id_references_social_campaigns_on_delete_set_null_with_its_index():
    column = SocialPost.__table__.c.campaign_id
    assert column.nullable
    (fk,) = column.foreign_keys
    assert (fk.target_fullname, fk.ondelete, fk.constraint.name) == (
        "social_campaigns.id", "SET NULL", CAMPAIGN_FK,
    )
    indexes = {i.name: [c.name for c in i.columns] for i in SocialPost.__table__.indexes}
    assert indexes[CAMPAIGN_INDEX] == ["campaign_id"]


def test_a_campaign_defaults_to_per_post_with_no_hashes_and_refuses_another_mode():
    engine = _model_engine()
    session = sessionmaker(bind=engine)()
    try:
        campaign = SocialCampaign(workspace_id=uuid.uuid4(), name="Web Summit", created_by="user-1")
        session.add(campaign)
        session.flush()
        row = campaign.to_dict()
        assert set(row) == CAMPAIGN_COLUMNS
        assert (row["id"], row["name"], row["created_by"]) == (str(campaign.id), "Web Summit", "user-1")
        assert (row["approval_mode"], row["approved_hash_set"]) == ("per_post", [])
        assert (row["approved_by"], row["approved_at"]) == (None, None)
        assert row["created_at"] and row["updated_at"]

        session.add(
            SocialCampaign(
                workspace_id=uuid.uuid4(), name="Bad", created_by="user-1", approval_mode="bulk"
            )
        )
        with pytest.raises(IntegrityError):
            session.flush()
    finally:
        session.rollback()
        session.close()
        engine.dispose()


# ---------------------------------------------------------------------------
# One revision
# ---------------------------------------------------------------------------


def test_the_one_wave_revision_chains_onto_the_base_head_and_alone_builds_the_table():
    mod = _load(WAVE2)
    assert (mod.revision, mod.down_revision) == ("prd251_wave2", BASE_HEAD)
    chained = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(rf"^down_revision\s*=\s*['\"]{BASE_HEAD}['\"]", p.read_text(encoding="utf-8"), re.M)
    )
    # Wave 2 is one PRD-251 revision on that base. document_chunks_ingestion_columns
    # branched from the same base on main; prd252_ticket_numbers merges the two
    # heads, and test_prd209 checks there is one head.
    assert [name for name in chained if name.startswith("prd251")] == ["prd251_wave2.py"]
    creators = sorted(
        p.name
        for p in VERSIONS.glob("*.py")
        if re.search(r"create_table\(\s*['\"]social_campaigns['\"]", p.read_text(encoding="utf-8"))
    )
    assert creators == ["prd251_wave2.py"]
    # The migration writes the model's CHECK, key name and index name.
    (check,) = [c for c in SocialCampaign.__table__.constraints if isinstance(c, sa.CheckConstraint)]
    assert mod.CAMPAIGN_APPROVAL_MODE_CHECK == str(check.sqltext)
    assert (mod.POST_CAMPAIGN_FK, mod.POST_CAMPAIGN_INDEX) == (CAMPAIGN_FK, CAMPAIGN_INDEX)


# ---------------------------------------------------------------------------
# SQLite: the migration builds the model's schema; create_all first; the downgrade
# ---------------------------------------------------------------------------


def test_the_migration_builds_exactly_the_model_schema():
    model_engine, migrated = _model_engine(), _migrated_engine()
    try:
        model, migration = _schema(model_engine), _schema(migrated)
    finally:
        model_engine.dispose()
        migrated.dispose()
    _assert_same_schema(migration, model, "the migrations prd251_socials, prd251_wave1, prd251_wave2")

    # Non-vacuous: the reflected facets carry what D2 declares.
    campaigns, posts = model["social_campaigns"], model["social_posts"]
    assert {name for name, _sql in campaigns["checks"]} == {"ck_social_campaigns_approval_mode"}
    assert (("workspace_id",), "workspaces", ("id",), "CASCADE") in campaigns["fks"]
    assert (
        "ix_social_campaigns_workspace_created", ("workspace_id", "created_at"), False
    ) in campaigns["indexes"]
    assert set(campaigns["columns"]) == CAMPAIGN_COLUMNS
    assert (("campaign_id",), "social_campaigns", ("id",), "SET NULL") in posts["fks"]
    assert (CAMPAIGN_INDEX, ("campaign_id",), False) in posts["indexes"]


@pytest.mark.parametrize("start", STARTS)
def test_create_all_first_then_the_upgrade_twice_leaves_the_model_schema(start):
    model_engine = _model_engine()
    engine = _model_engine() if start == "fresh" else _wave1_engine()
    try:
        if start == "backend_first":
            SocialPost.metadata.create_all(engine, tables=MODEL_TABLES)
            # create_all never alters a table that exists: it built the new table only.
            assert sa.inspect(engine).has_table("social_campaigns") and not _campaign_keys(engine)
        with engine.begin() as conn:
            _run(conn, WAVE2, "upgrade")
            _run(conn, WAVE2, "upgrade")
            _run(conn, WAVE1B, "upgrade")  # the model carries PRD-251B Wave 1's columns too
        _assert_same_schema(_schema(engine), _schema(model_engine), f"create_all first, from {start}")
        assert _campaign_keys(engine) == [CAMPAIGN_FK]
    finally:
        model_engine.dispose()
        engine.dispose()


def test_the_downgrade_drops_the_key_the_index_and_the_table_and_the_posts_stay():
    engine = _wave1_engine()
    posts = SocialPost.__table__
    post_id = uuid.uuid4()
    try:
        wave1 = _schema(engine, ("social_posts", "social_post_targets"))
        with engine.begin() as conn:
            _run(conn, WAVE2, "upgrade")
            conn.execute(
                sa.insert(posts).values(
                    id=post_id, workspace_id=uuid.uuid4(), created_by="user-1",
                    title="Launch", content_hash="0" * 64,
                )
            )
            _run(conn, WAVE2, "downgrade")
        assert not sa.inspect(engine).has_table("social_campaigns")
        _assert_same_schema(
            _schema(engine, ("social_posts", "social_post_targets")), wave1, "after the downgrade"
        )
        with engine.connect() as conn:
            kept = conn.execute(sa.select(posts.c.title).where(posts.c.id == post_id)).scalar_one()
        assert kept == "Launch"

        with engine.begin() as conn:
            _run(conn, WAVE2, "upgrade")
        assert _campaign_keys(engine) == [CAMPAIGN_FK]
    finally:
        engine.dispose()


def _enforce_foreign_keys(conn) -> None:
    conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    assert conn.exec_driver_sql("PRAGMA foreign_keys").scalar() == 1


@pytest.mark.parametrize("build", ["model", "migration"])
def test_deleting_a_campaign_leaves_its_posts_with_campaign_id_null(build):
    engine = _model_engine() if build == "model" else _migrated_engine()
    campaigns, posts = SocialCampaign.__table__, SocialPost.__table__
    ws, campaign_id, post_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    post = {"workspace_id": ws, "created_by": "user-1", "title": "Launch", "content_hash": "0" * 64}
    try:
        with engine.connect() as conn:
            _enforce_foreign_keys(conn)
            conn.exec_driver_sql("INSERT INTO workspaces (id) VALUES (?)", (ws.hex,))
            conn.execute(
                sa.insert(campaigns).values(
                    id=campaign_id, workspace_id=ws, name="Web Summit", created_by="user-1"
                )
            )
            conn.execute(sa.insert(posts).values(id=post_id, campaign_id=campaign_id, **post))

            conn.execute(sa.delete(campaigns).where(campaigns.c.id == campaign_id))
            left = conn.execute(
                sa.select(posts.c.id, posts.c.campaign_id).where(posts.c.id == post_id)
            ).one()
            assert (left.id, left.campaign_id) == (post_id, None)

            # Non-vacuous: the key is enforced, so a campaign that does not exist is refused.
            with pytest.raises(IntegrityError):
                conn.execute(sa.insert(posts).values(id=uuid.uuid4(), campaign_id=uuid.uuid4(), **post))
            conn.rollback()
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# @integration — real Postgres, rolled back
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM workspaces LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM social_posts LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("CI") == "true":
            pytest.fail(f"CI runs these checks against its Postgres service, which did not answer: {exc}")
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield engine
    engine.dispose()


def _declared(table: sa.Table) -> dict:
    """The model's own declaration of ``table``, in Postgres terms."""
    pg = postgresql.dialect()
    return {
        "columns": {c.name: (c.type.compile(dialect=pg), c.nullable) for c in table.columns},
        "checks": {c.name for c in table.constraints if isinstance(c, sa.CheckConstraint)},
        "fks": {
            (
                tuple(fk.column_keys),
                fk.referred_table.name,
                tuple(element.column.name for element in fk.elements),
                fk.ondelete,
            )
            for fk in table.foreign_key_constraints
        },
        "indexes": {(i.name, tuple(c.name for c in i.columns), bool(i.unique)) for i in table.indexes},
    }


def _reflected(conn, name: str) -> dict:
    """What Postgres holds for ``name``, in the same terms."""
    insp, pg = sa.inspect(conn), postgresql.dialect()
    return {
        "columns": {c["name"]: (c["type"].compile(dialect=pg), c["nullable"]) for c in insp.get_columns(name)},
        "checks": {c["name"] for c in insp.get_check_constraints(name)},
        "fks": {
            (
                tuple(f["constrained_columns"]),
                f["referred_table"],
                tuple(f["referred_columns"]),
                (f.get("options") or {}).get("ondelete"),
            )
            for f in insp.get_foreign_keys(name)
        },
        "indexes": {
            (i["name"], tuple(i["column_names"]), bool(i["unique"])) for i in insp.get_indexes(name)
        },
    }


def _campaign_key_rows(conn) -> list:
    """Every key on social_posts.campaign_id, with its ON DELETE rule ('n' is SET NULL)."""
    rows = conn.execute(
        sa.text(
            "SELECT con.conname, con.confdeltype FROM pg_constraint con "
            "JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ANY (con.conkey) "
            "WHERE con.contype = 'f' AND con.conrelid = 'social_posts'::regclass "
            "AND a.attname = 'campaign_id'"
        )
    ).fetchall()
    return [tuple(r) for r in rows]


def _indexes_on_campaign_id(conn) -> list:
    """Every index on social_posts over campaign_id alone, by name."""
    rows = conn.execute(
        sa.text(
            "SELECT c.relname FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0] "
            "WHERE i.indrelid = 'social_posts'::regclass AND i.indnatts = 1 "
            "AND a.attname = 'campaign_id'"
        )
    ).fetchall()
    return [r[0] for r in rows]


def _start_from(conn, start: str) -> None:
    """Put the database where a starting point stands, inside the test's transaction."""
    SocialPost.metadata.create_all(bind=conn, tables=MODEL_TABLES)  # a fresh database's create_all
    if start == "fresh":
        return
    for ddl in WAVE1_STATE_DDL:
        conn.execute(sa.text(ddl))
    if start == "backend_first":
        SocialPost.metadata.create_all(bind=conn, tables=MODEL_TABLES)
        # create_all never alters a table that exists: it built the new table only.
        assert _campaign_key_rows(conn) == [] and _indexes_on_campaign_id(conn) == []


@pytest.mark.integration
@pytest.mark.parametrize("start", STARTS)
def test_create_all_first_then_the_upgrade_twice_on_postgres(pg_engine, start):
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            _start_from(conn, start)
            _run(conn, WAVE2, "upgrade")
            _run(conn, WAVE2, "upgrade")

            for table in (SocialCampaign.__table__, SocialPost.__table__):
                assert _reflected(conn, table.name) == _declared(table), f"{table.name}, from {start}"
            assert _campaign_key_rows(conn) == [(CAMPAIGN_FK, "n")]
            assert _indexes_on_campaign_id(conn) == [CAMPAIGN_INDEX]
        finally:
            trans.rollback()


def _insert_campaign(conn, ws_id: str) -> str:
    """A campaign row with no approval_mode given: the column's default applies."""
    campaign_id = str(uuid.uuid4())
    conn.execute(
        sa.text(
            "INSERT INTO social_campaigns (id, workspace_id, name, approved_hash_set, created_by) "
            "VALUES (CAST(:id AS uuid), CAST(:ws AS uuid), 'Web Summit', '[]', 'user-1')"
        ),
        {"id": campaign_id, "ws": ws_id},
    )
    return campaign_id


@pytest.mark.integration
@pytest.mark.parametrize("start", ["fresh", "migration_path"])
def test_deleting_a_campaign_leaves_its_posts_unlinked_on_postgres(pg_engine, start):
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            _start_from(conn, start)
            _run(conn, WAVE2, "upgrade")
            ws_id = str(uuid.uuid4())
            conn.execute(
                sa.text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :name)"),
                {"id": ws_id, "name": f"prd251w2-{ws_id[:8]}"},
            )
            campaign_id = _insert_campaign(conn, ws_id)
            mode = conn.execute(
                sa.text("SELECT approval_mode FROM social_campaigns WHERE id = CAST(:id AS uuid)"),
                {"id": campaign_id},
            ).scalar_one()
            assert mode == "per_post"
            # The CHECK holds whichever writer built the table.
            nested = conn.begin_nested()
            with pytest.raises(IntegrityError):
                conn.execute(
                    sa.text(
                        "UPDATE social_campaigns SET approval_mode = 'bulk' WHERE id = CAST(:id AS uuid)"
                    ),
                    {"id": campaign_id},
                )
            nested.rollback()

            post_id = str(uuid.uuid4())
            conn.execute(
                sa.text(
                    "INSERT INTO social_posts (id, workspace_id, created_by, title, copy, variables, "
                    "sources, media, content_hash, review_log, campaign_id) VALUES "
                    "(CAST(:id AS uuid), CAST(:ws AS uuid), 'u', 't', '{}', '{}', '{}', '{}', :h, '[]', "
                    "CAST(:campaign AS uuid))"
                ),
                {"id": post_id, "ws": ws_id, "h": "0" * 64, "campaign": campaign_id},
            )
            conn.execute(
                sa.text("DELETE FROM social_campaigns WHERE id = CAST(:id AS uuid)"), {"id": campaign_id}
            )
            left = conn.execute(
                sa.text("SELECT campaign_id FROM social_posts WHERE id = CAST(:id AS uuid)"), {"id": post_id}
            ).fetchall()
            assert [tuple(r) for r in left] == [(None,)]
        finally:
            trans.rollback()


@pytest.mark.integration
def test_the_downgrade_on_postgres_drops_the_key_the_index_and_the_table(pg_engine):
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            # create_all built the key and the index; the downgrade finds them by name.
            _start_from(conn, "fresh")
            _run(conn, WAVE2, "downgrade")
            assert conn.execute(sa.text("SELECT to_regclass('social_campaigns')")).scalar() is None
            assert _campaign_key_rows(conn) == [] and _indexes_on_campaign_id(conn) == []
            assert "campaign_id" in {c["name"] for c in sa.inspect(conn).get_columns("social_posts")}

            _run(conn, WAVE2, "upgrade")
            assert _campaign_key_rows(conn) == [(CAMPAIGN_FK, "n")]
            assert _indexes_on_campaign_id(conn) == [CAMPAIGN_INDEX]
        finally:
            trans.rollback()
