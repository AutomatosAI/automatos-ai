"""PRD-251 Wave 1, US-110 (D16) — the media capability registry: an allowlist per toolkit.

``modules/socials/capabilities.py`` says which actions of a workspace's connected
Composio toolkits Socials may call, and what for. Pinned on the real read paths:
the allowlist and the Wave 0 deny list through ``read_system_setting`` (seeded
by the two migrations), the connected toolkits through
``EntityManager.get_connected_apps``, and the cached action schemas in
``composio_actions_cache`` (SQLite copies of the tables, filled with
cached-schema fixtures):

* with fal_ai connected, the registry offers its allowlisted actions, each with
  its cached schema, and nothing else; a toolkit whose allowlisted slugs the
  cache does not hold shows as unavailable;
* a denied slug is never offered, even when a super-admin allowlists it, and a
  deny list that cannot be read or parsed offers nothing;
* an unknown toolkit offers nothing; the allowlist is data — adding a toolkit to
  the setting offers it with no code change, and no toolkit or slug is a string
  constant in the registry's code;
* fail closed: no row, a value that is not the shape, or a read that cannot
  complete offers nothing, and the last two say why and log at ERROR;
* the seed: in the wave's one migration, insert-if-absent, never overwriting an
  edit, removed by the downgrade only where the migration made it; on Postgres
  (``@integration``, rolled back), ``create_all`` first and then the upgrade twice
  leaves exactly one row (the 89d89c250 lesson).
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import core.composio.deny_list as deny_list  # noqa: E402
import core.database.database as database_mod  # noqa: E402
from core.database.database import Base  # noqa: E402
from core.models.composio import ComposioConnection, ComposioEntity  # noqa: E402
from core.models.composio_cache import ComposioActionCache  # noqa: E402
from core.models.system_settings import SettingCategory, SystemSetting  # noqa: E402
from modules.socials import capabilities  # noqa: E402
from modules.socials.capabilities import (  # noqa: E402
    BALANCE,
    CAPABILITIES,
    ESTIMATE,
    GENERATE_IMAGE,
    GENERATE_VIDEO,
    KEY_MEDIA_ACTIONS,
    READ_FAILED_PROBLEM,
    STATUS,
    TTS,
    UPLOAD,
    VOICES,
    media_capabilities,
    parse_media_actions,
)
from tests.helpers_unreadable_settings import UNREADABLE_MODES, settings_unreadable  # noqa: E402

VERSIONS = _ORCH / "alembic" / "versions"
WAVE0_MIGRATION = VERSIONS / "prd251_socials.py"
WAVE1_MIGRATION = VERSIONS / "prd251_wave1.py"
REGISTRY_SOURCE = _ORCH / "modules" / "socials" / "capabilities.py"

WS = uuid.UUID("00000000-0000-0000-0000-00000000a110")
OTHER_WS = uuid.UUID("00000000-0000-0000-0000-00000000a111")
BILLING = "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE"
MEDIA_TOOLKITS = {"fal_ai", "kieai", "higgsfield_mcp", "fish_audio", "elevenlabs"}

# What US-110 promises for fal.ai (slugs verified on docs.composio.dev 2026-09-23).
FAL_OFFERED = {
    "FAL_AI_SUBMIT_ASYNC_JOB": {GENERATE_VIDEO, GENERATE_IMAGE},
    "FAL_AI_QUEUE_GET_STATUS": {STATUS},
    "FAL_AI_GET_QUEUE_REQUEST_RESULT": {STATUS},
    "FAL_AI_UPLOAD_FILE": {UPLOAD},
    "FAL_AI_ESTIMATE_PRICING": {ESTIMATE},
}
# The cached-schema fixtures: what composio_actions_cache holds per app, as the
# metadata sync writes it (app_name upper case, action_name the slug). Each app
# carries actions the allowlist does not name.
FAL_SUBMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "model_id": {"type": "string", "description": "The fal model to run"},
        "input": {"type": "object", "description": "The model's own input"},
        "webhook_url": {"type": "string"},
    },
    "required": ["model_id", "input"],
}
NOT_ALLOWLISTED = {
    "FAL_AI": ["FAL_AI_CANCEL_QUEUE_REQUEST", "FAL_AI_DELETE_REQUEST_PAYLOADS"],
    "KIEAI": ["KIEAI_GENERATE_SUNO_MUSIC"],
    "FISH_AUDIO": ["FISH_AUDIO_CREATE_VOICE_MODEL"],
    "ELEVENLABS": ["ELEVENLABS_DELETE_VOICE_BY_ID", "ELEVENLABS_TEXT_TO_SPEECH_STREAM"],
    "GITHUB": ["GITHUB_CREATE_ISSUE"],
    "RUNWAY": ["RUNWAY_GENERATE_VIDEO"],
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE0 = _load(WAVE0_MIGRATION, "prd251_socials_migration_media")
WAVE1 = _load(WAVE1_MIGRATION, "prd251_wave1_migration_media")
SEED = WAVE1.SOCIALS_MEDIA_ACTIONS_SEED
DENIED_SEED = tuple(WAVE0.COMPOSIO_DENIED_ACTIONS_SEED)


def _seed_slugs(toolkit: str) -> set:
    return {slug for slugs in SEED[toolkit].values() for slug in slugs}


def _cached_fixtures() -> dict:
    """app_name → the slugs its cached schemas hold: every seeded slug, the
    money-moving Higgsfield actions of the deny list, and the extras above."""
    cached = {toolkit.upper(): sorted(_seed_slugs(toolkit)) for toolkit in SEED}
    cached["HIGGSFIELD_MCP"] += list(DENIED_SEED)
    for app, slugs in NOT_ALLOWLISTED.items():
        cached[app] = cached.get(app, []) + slugs
    return cached


# ---------------------------------------------------------------------------
# The harness: SQLite copies of the tables, the settings seeded by the migrations
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, (JSONB, ARRAY)):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table: sa.Table, metadata: sa.MetaData) -> sa.Table:
    """A column-for-column copy SQLite can build (JSONB → JSON, UUID → CHAR(32))."""
    columns = [sa.Column(col.name, _portable(col.type), primary_key=col.primary_key) for col in table.columns]
    return sa.Table(table.name, metadata, *columns)


_TABLES = sa.MetaData()
for _table in (ComposioEntity.__table__, ComposioConnection.__table__, ComposioActionCache.__table__):
    _sqlite_copy(_table, _TABLES)


@pytest.fixture
def env(monkeypatch):
    """read_system_setting → SessionLocal → an in-memory database holding the
    seeded allowlist and deny list; the registry's own session on the same one."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SystemSetting.__table__.create(bind=engine)
    with engine.begin() as conn:
        WAVE0._seed_settings(conn, WAVE0._composio_settings_seed())
        WAVE1.seed_settings(conn, WAVE1.settings_seed())
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    deny_list.reset_cache()
    session = sessionmaker(bind=engine)()
    try:
        yield SimpleNamespace(engine=engine, session=session)
    finally:
        session.close()
        deny_list.reset_cache()
        engine.dispose()


def _cache(env, app: str, *slugs: str) -> None:
    for slug in slugs:
        env.session.add(
            ComposioActionCache(
                app_name=app,
                action_name=slug,
                action_slug=slug.lower().replace("_", "-"),
                display_name=slug.replace("_", " ").title(),
                parameters=FAL_SUBMIT_SCHEMA if slug == "FAL_AI_SUBMIT_ASYNC_JOB" else {"type": "object"},
            )
        )
    env.session.commit()


def _cache_all(env) -> None:
    for app, slugs in _cached_fixtures().items():
        _cache(env, app, *slugs)


def _connect(env, *apps: str, workspace=WS, status="active") -> None:
    entity = env.session.query(ComposioEntity).filter(ComposioEntity.workspace_id == workspace).first()
    if entity is None:
        entity = ComposioEntity(workspace_id=workspace, composio_entity_id=str(workspace))
        env.session.add(entity)
        env.session.flush()
    for app in apps:
        env.session.add(
            ComposioConnection(entity_id=entity.id, app_name=app, status=status, connection_id=f"ca_{app.lower()}")
        )
    env.session.commit()


def _setting(env, category: str, key: str, value) -> None:
    table = SystemSetting.__table__
    with env.engine.begin() as conn:
        if value is None:
            conn.execute(table.delete().where(table.c.category == category, table.c.key == key))
        else:
            conn.execute(table.update().where(table.c.category == category, table.c.key == key).values(value=value))
    deny_list.reset_cache()


def _set_allowlist(env, value) -> None:
    _setting(env, "socials", KEY_MEDIA_ACTIONS, value if value is None or isinstance(value, str) else json.dumps(value))


def _offered(caps) -> dict:
    return {toolkit: set(actions) for toolkit, actions in caps.offered.items()}


# ---------------------------------------------------------------------------
# AC1 — with fal_ai connected, its allowlisted actions and nothing else
# ---------------------------------------------------------------------------


def test_with_fal_ai_connected_the_registry_offers_its_allowlisted_actions_and_nothing_else(env):
    _cache_all(env)
    _connect(env, "FAL_AI")

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {"fal_ai": set(FAL_OFFERED)}
    assert set(FAL_OFFERED) == _seed_slugs("fal_ai")
    for slug, expected in FAL_OFFERED.items():
        action = caps.action("fal_ai", slug)
        assert (action.toolkit, action.slug, set(action.capabilities)) == ("fal_ai", slug, expected)
    # Cached for fal, but not on the allowlist: never offered.
    for slug in NOT_ALLOWLISTED["FAL_AI"]:
        assert caps.action("fal_ai", slug) is None
    # The recipe builds its call from the cached schema.
    assert dict(caps.action("fal_ai", "FAL_AI_SUBMIT_ASYNC_JOB").parameters) == FAL_SUBMIT_SCHEMA
    assert caps.action("FAL_AI", "fal_ai_submit_async_job").slug == "FAL_AI_SUBMIT_ASYNC_JOB"

    assert caps.toolkits(GENERATE_VIDEO) == caps.toolkits(GENERATE_IMAGE) == ("fal_ai",)
    assert caps.toolkits(ESTIMATE) == caps.toolkits(UPLOAD) == ("fal_ai",)
    assert caps.toolkits(TTS) == caps.toolkits(VOICES) == caps.toolkits(BALANCE) == ()
    assert [a.slug for a in caps.actions("fal_ai", STATUS)] == [
        "FAL_AI_GET_QUEUE_REQUEST_RESULT",
        "FAL_AI_QUEUE_GET_STATUS",
    ]
    assert {a.slug for a in caps.actions("fal_ai")} == set(FAL_OFFERED)
    # The composer's connect links: allowlisted toolkits not yet connected.
    assert caps.connectable(TTS) == ("elevenlabs", "fish_audio")
    assert caps.connectable(GENERATE_VIDEO) == ("higgsfield_mcp", "kieai")
    assert caps.connectable(ESTIMATE) == ()
    assert (caps.problem, dict(caps.withheld)) == (None, {})


def test_connections_are_per_workspace(env):
    _cache_all(env)
    _connect(env, "FAL_AI", workspace=OTHER_WS)

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {}
    assert caps.connected == frozenset()
    assert set(caps.connectable(GENERATE_VIDEO)) == {"fal_ai", "kieai", "higgsfield_mcp"}


def test_every_seeded_toolkit_offers_exactly_its_allowlisted_actions(env):
    _cache_all(env)
    _connect(env, *(toolkit.upper() for toolkit in SEED), "GITHUB", "RUNWAY")

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {toolkit: _seed_slugs(toolkit) for toolkit in SEED}
    assert set(caps.toolkits(TTS)) == {"fish_audio", "elevenlabs"}
    assert set(caps.toolkits(GENERATE_VIDEO)) == {"fal_ai", "kieai", "higgsfield_mcp"}
    assert set(caps.toolkits(BALANCE)) == {"kieai", "higgsfield_mcp", "fish_audio"}
    assert caps.connectable(TTS) == ()
    assert {"github", "runway"} <= caps.connected


def test_an_allowlisted_slug_the_cached_schemas_do_not_hold_is_not_offered(env):
    _cache(env, "ELEVENLABS", "ELEVENLABS_TEXT_TO_SPEECH")  # the voices list never synced
    _connect(env, "ELEVENLABS", "FISH_AUDIO")  # fish_audio: nothing cached at all

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {"elevenlabs": {"ELEVENLABS_TEXT_TO_SPEECH"}}
    assert caps.toolkits(VOICES) == ()
    # Fish Audio is connected but unavailable: no connect link either.
    assert "fish_audio" in caps.connected and "fish_audio" not in caps.offered
    assert caps.connectable(TTS) == ()


# ---------------------------------------------------------------------------
# AC2 — the Wave 0 deny list always wins
# ---------------------------------------------------------------------------


def test_the_seeded_allowlist_names_no_denied_slug():
    allowlisted = {slug for toolkit in SEED for slug in _seed_slugs(toolkit)}
    assert allowlisted.isdisjoint(DENIED_SEED)
    assert BILLING in DENIED_SEED


def test_a_denied_slug_is_never_offered_even_when_allowlisted(env):
    _cache_all(env)
    _connect(env, "HIGGSFIELD_MCP")
    edited = json.loads(json.dumps(SEED))
    edited["higgsfield_mcp"]["balance"].append(BILLING)
    edited["higgsfield_mcp"]["generate_video"].extend(DENIED_SEED)
    _set_allowlist(env, edited)

    caps = media_capabilities(env.session, WS)

    assert caps.action("higgsfield_mcp", BILLING) is None
    assert _offered(caps) == {"higgsfield_mcp": _seed_slugs("higgsfield_mcp")}
    assert set(caps.withheld) == set(DENIED_SEED)
    assert caps.withheld[BILLING].startswith(deny_list.BLOCKED_PREFIX)
    assert BILLING in caps.withheld[BILLING]


def test_a_seeded_slug_the_super_admin_denies_is_withheld_on_the_next_read(env):
    _cache_all(env)
    _connect(env, "HIGGSFIELD_MCP")
    _setting(env, "composio", "denied_actions", json.dumps(["HIGGSFIELD_MCP_GENERATE_VIDEO"]))

    caps = media_capabilities(env.session, WS)

    assert caps.action("higgsfield_mcp", "HIGGSFIELD_MCP_GENERATE_VIDEO") is None
    assert "HIGGSFIELD_MCP_GENERATE_VIDEO" in caps.withheld
    assert caps.toolkits(GENERATE_VIDEO) == ()
    assert caps.toolkits(GENERATE_IMAGE) == ("higgsfield_mcp",)


def test_a_deny_list_that_is_not_a_list_offers_nothing(env):
    _cache_all(env)
    _connect(env, "FAL_AI", "FISH_AUDIO")
    _setting(env, "composio", "denied_actions", "not json")

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {}
    assert set(caps.withheld) == _seed_slugs("fal_ai") | _seed_slugs("fish_audio")
    assert all(reason == deny_list.BLOCKED_PREFIX + deny_list.UNREADABLE_REASON for reason in caps.withheld.values())


def test_a_deny_list_that_cannot_be_read_offers_nothing(env, monkeypatch):
    _cache_all(env)
    _connect(env, "FAL_AI")

    def read_fails():
        raise sa.exc.OperationalError("SELECT system_settings.value", {}, Exception("connection dropped"))

    monkeypatch.setattr(deny_list, "_read_denied_actions", read_fails)
    deny_list.reset_cache()

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {}
    assert set(caps.withheld) == _seed_slugs("fal_ai")
    assert all(reason == deny_list.BLOCKED_PREFIX + deny_list.READ_FAILED_REASON for reason in caps.withheld.values())


# ---------------------------------------------------------------------------
# AC3 — an unknown toolkit offers nothing; the allowlist is data
# ---------------------------------------------------------------------------


def test_an_unknown_toolkit_offers_nothing(env):
    _cache_all(env)
    _connect(env, "RUNWAY", "GITHUB")

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {}
    assert caps.connected == frozenset({"runway", "github"})
    assert set(caps.allowlisted) == MEDIA_TOOLKITS
    assert "runway" not in caps.connectable(GENERATE_VIDEO)


def test_a_toolkit_added_to_the_setting_is_offered_with_no_code_change(env):
    _cache_all(env)
    _connect(env, "RUNWAY", "FAL_AI")
    edited = json.loads(json.dumps(SEED))
    # Written the way a person might type it: the registry matches case-insensitively.
    edited["Runway"] = {"generate_video": ["runway_generate_video"]}
    del edited["fal_ai"]
    _set_allowlist(env, edited)

    caps = media_capabilities(env.session, WS)

    assert _offered(caps) == {"runway": {"RUNWAY_GENERATE_VIDEO"}}
    assert caps.toolkits(GENERATE_VIDEO) == ("runway",)
    assert "fal_ai" not in caps.allowlisted


def test_no_toolkit_or_slug_is_a_string_constant_in_the_registry_code():
    tree = ast.parse(REGISTRY_SOURCE.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    strings = [
        node.value.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]
    names = {toolkit.lower() for toolkit in SEED} | {
        slug.lower() for toolkit in SEED for slug in _seed_slugs(toolkit)
    } | {slug.lower() for slug in DENIED_SEED}
    assert strings, "the AST walk found no strings at all"
    assert [s for s in strings if any(name in s for name in names)] == []
    # The allowlist is read from the settings plane, not imported from the seed.
    imported = {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any("alembic" in name or "prd251" in name for name in imported), imported
    assert "core.llm.manager" in imported


# ---------------------------------------------------------------------------
# Fail closed: no row, a value that is not the shape, a read that cannot complete
# ---------------------------------------------------------------------------


def test_no_row_offers_nothing(env):
    _cache_all(env)
    _connect(env, "FAL_AI")
    _set_allowlist(env, None)

    caps = media_capabilities(env.session, WS)

    assert (_offered(caps), dict(caps.allowlisted), caps.problem) == ({}, {}, None)


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        '"fal_ai"',
        '{"fal_ai": ["FAL_AI_UPLOAD_FILE"]}',
        '{"fal_ai": {"teleport": ["FAL_AI_UPLOAD_FILE"]}}',
        '{"fal_ai": {"upload": "FAL_AI_UPLOAD_FILE"}}',
        '{"fal_ai": {"upload": [""]}}',
        '{"fal_ai": {"upload": [7]}}',
        '{" ": {"upload": ["FAL_AI_UPLOAD_FILE"]}}',
    ],
)
def test_a_value_that_is_not_the_shape_offers_nothing_says_why_and_logs_at_error(env, monkeypatch, raw):
    _cache_all(env)
    _connect(env, "FAL_AI")
    _set_allowlist(env, raw)
    logger = MagicMock(name="logger")
    monkeypatch.setattr(capabilities, "logger", logger)

    caps = media_capabilities(env.session, WS)

    assert (_offered(caps), dict(caps.allowlisted)) == ({}, {})
    assert caps.problem.startswith("The Socials media allowlist (system setting socials.media_actions) is not usable (")
    logger.error.assert_called_once()
    assert "is not usable" in logger.error.call_args.args[0]


@pytest.mark.parametrize("mode", UNREADABLE_MODES)
def test_a_read_that_cannot_complete_offers_nothing_and_logs_at_error(env, monkeypatch, mode):
    _cache_all(env)
    _connect(env, "FAL_AI")
    logger = MagicMock(name="logger")
    monkeypatch.setattr(capabilities, "logger", logger)

    with settings_unreadable(mode, monkeypatch):
        caps = media_capabilities(env.session, WS)

    assert (_offered(caps), dict(caps.allowlisted)) == ({}, {})
    assert caps.problem == READ_FAILED_PROBLEM
    assert caps.connected == frozenset({"fal_ai"})
    logger.error.assert_called_once()
    assert "could not be read" in logger.error.call_args.args[0]
    assert logger.error.call_args.kwargs["exc_info"] is True


def test_parse_media_actions_normalises_and_merges():
    assert parse_media_actions(None) == parse_media_actions("  ") == {}
    parsed = parse_media_actions(
        json.dumps({"Fal_AI ": {"generate_video": [" fal_ai_submit_async_job"], "generate_image": ["FAL_AI_SUBMIT_ASYNC_JOB"]}})
    )
    assert parsed == {"fal_ai": {"FAL_AI_SUBMIT_ASYNC_JOB": frozenset({GENERATE_VIDEO, GENERATE_IMAGE})}}


def test_an_unknown_capability_is_a_programming_error(env):
    caps = media_capabilities(env.session, WS)
    for ask in (caps.toolkits, caps.connectable, lambda name: caps.actions("fal_ai", name)):
        with pytest.raises(ValueError, match="unknown media capability"):
            ask("teleport")


# ---------------------------------------------------------------------------
# The seed: the wave's one migration
# ---------------------------------------------------------------------------


def test_the_seed_lives_in_the_one_wave_migration_and_the_registry_reads_it():
    # The same revision seeds the Socials post gate's list too (US-118, D14b).
    rows = {row["key"]: row for row in WAVE1.settings_seed()}
    assert set(rows) == {KEY_MEDIA_ACTIONS, "post_actions"}
    row = rows[KEY_MEDIA_ACTIONS]
    assert (row["category"], row["key"], row["value_type"]) == (
        SettingCategory.SOCIALS.value,
        KEY_MEDIA_ACTIONS,
        "json",
    )
    assert row["default_value"] == row["value"]
    parsed = parse_media_actions(row["value"])
    assert set(parsed) == MEDIA_TOOLKITS
    assert {capability for toolkit in SEED for capability in SEED[toolkit]} <= set(CAPABILITIES)
    # The slugs of the story's notes (docs.composio.dev, 2026-09-23/25).
    assert parsed["kieai"] == {
        "KIEAI_GENERATE_VEO_VIDEO": frozenset({GENERATE_VIDEO}),
        "KIEAI_GET_VEO_VIDEO_DETAILS": frozenset({STATUS}),
        "KIEAI_GENERATE_FLUX_KONTEXT_IMAGE": frozenset({GENERATE_IMAGE}),
        "KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS": frozenset({STATUS}),
        "KIEAI_GET_ACCOUNT_CREDITS": frozenset({BALANCE}),
    }
    assert parsed["higgsfield_mcp"] == {
        "HIGGSFIELD_MCP_GENERATE_VIDEO": frozenset({GENERATE_VIDEO}),
        "HIGGSFIELD_MCP_GENERATE_IMAGE": frozenset({GENERATE_IMAGE}),
        "HIGGSFIELD_MCP_JOBS_WAIT": frozenset({STATUS}),
        "HIGGSFIELD_MCP_JOB_STATUS": frozenset({STATUS}),
        "HIGGSFIELD_MCP_MEDIA_UPLOAD": frozenset({UPLOAD}),
        "HIGGSFIELD_MCP_BALANCE": frozenset({BALANCE}),
    }
    assert parsed["fish_audio"] == {
        "FISH_AUDIO_SYNTHESIZE_SPEECH": frozenset({TTS}),
        "FISH_AUDIO_LIST_VOICE_MODELS": frozenset({VOICES}),
        "FISH_AUDIO_GET_ACCOUNT_BALANCE": frozenset({BALANCE}),
    }
    assert parsed["elevenlabs"] == {
        "ELEVENLABS_TEXT_TO_SPEECH": frozenset({TTS}),
        "ELEVENLABS_GET_VOICES_LIST": frozenset({VOICES}),
    }
    assert {slug: set(caps) for slug, caps in parsed["fal_ai"].items()} == FAL_OFFERED


def _settings_engine():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SystemSetting.__table__.create(bind=engine)
    return engine


def _rows(conn):
    table = SystemSetting.__table__
    return conn.execute(
        table.select().where(table.c.category == "socials").order_by(table.c.key, table.c.created_by)
    ).fetchall()


def _media_row(conn):
    (row,) = [r for r in _rows(conn) if r.key == KEY_MEDIA_ACTIONS]
    return row


def test_the_seed_is_insert_if_absent_and_never_overwrites_an_edit():
    engine = _settings_engine()
    try:
        with engine.begin() as conn:
            WAVE1.seed_settings(conn, WAVE1.settings_seed())
            WAVE1.seed_settings(conn, WAVE1.settings_seed())  # a re-run adds nothing
            # The allowlist, and the Socials post gate's list (US-118) beside it.
            assert [(r.key, r.created_by) for r in _rows(conn)] == [
                (KEY_MEDIA_ACTIONS, "prd251_wave1"),
                ("post_actions", "prd251_wave1"),
            ]
            row = _media_row(conn)
            assert json.loads(row.value) == SEED

            table = SystemSetting.__table__
            conn.execute(table.update().where(table.c.id == row.id).values(value='{"runway": {}}'))
            WAVE1.seed_settings(conn, WAVE1.settings_seed())
            assert _media_row(conn).value == '{"runway": {}}'
    finally:
        engine.dispose()


def test_the_downgrade_deletes_only_the_row_the_migration_seeded():
    engine = _settings_engine()
    table = SystemSetting.__table__
    try:
        with engine.begin() as conn:
            conn.execute(table.insert().values(category="socials", key="enabled", value="true", created_by="prd251"))
            WAVE1.seed_settings(conn, WAVE1.settings_seed())
            WAVE1.unseed_settings(conn, WAVE1.settings_seed())
            assert [(r.key, r.created_by) for r in _rows(conn)] == [("enabled", "prd251")]

            # A row a person made is never the migration's to delete.
            conn.execute(table.insert().values(category="socials", key=KEY_MEDIA_ACTIONS, value="{}", created_by="admin"))
            WAVE1.unseed_settings(conn, WAVE1.settings_seed())
            assert [(r.key, r.created_by) for r in _rows(conn)] == [("enabled", "prd251"), (KEY_MEDIA_ACTIONS, "admin")]
    finally:
        engine.dispose()


def test_upgrade_and_downgrade_run_the_seed_steps():
    source = WAVE1_MIGRATION.read_text(encoding="utf-8")
    upgrade = ast.get_source_segment(source, _function(source, "upgrade"))
    downgrade = ast.get_source_segment(source, _function(source, "downgrade"))
    assert "seed_settings(op.get_bind(), settings_seed())" in upgrade
    assert "unseed_settings(op.get_bind(), settings_seed())" in downgrade


def _function(source: str, name: str):
    return next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _run_migration(conn, *steps):
    """Run the revision's upgrade/downgrade on ``conn`` through a real alembic context."""
    mod = _load(WAVE1_MIGRATION, "prd251_wave1_migration_media_pg")
    ctx = MigrationContext.configure(conn)
    with Operations.context(ctx):
        for step in steps:
            getattr(mod, step)()


@pytest.fixture(scope="module")
def pg_engine():
    from core.database.database import get_database_url

    try:
        engine = sa.create_engine(get_database_url(), pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(sa.text("SELECT 1 FROM system_settings LIMIT 1"))
            conn.execute(sa.text("SELECT 1 FROM document_templates LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the Postgres checks need the test database: {exc}")
    yield engine
    engine.dispose()


def _media_rows(conn):
    return conn.execute(
        sa.text(
            "SELECT value, default_value, value_type, created_by FROM system_settings "
            "WHERE category = 'socials' AND key = 'media_actions'"
        )
    ).fetchall()


@pytest.mark.integration
def test_create_all_first_then_the_upgrade_twice_seeds_the_allowlist_once(pg_engine):
    """A backend that already loaded the new models runs create_all before the migration."""
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            Base.metadata.create_all(bind=conn)
            _run_migration(conn, "upgrade")
            _run_migration(conn, "upgrade")

            (row,) = _media_rows(conn)
            assert row.value_type == "json"
            assert set(parse_media_actions(row.default_value)) == MEDIA_TOOLKITS

            _run_migration(conn, "downgrade")
            assert [r for r in _media_rows(conn) if r.created_by == "prd251_wave1"] == []
            _run_migration(conn, "upgrade")
            assert len(_media_rows(conn)) == 1
        finally:
            trans.rollback()
