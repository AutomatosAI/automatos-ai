"""PRD-251 Wave 1, US-118 (S3.5, D14b) — one way out: agents cannot post directly when Socials is on.

``core/composio/post_gate.py`` refuses an agent's direct Composio call to a social
PUBLISH action in a workspace with Socials on, before any Composio call, and
points it at the draft tool. Pinned on the real read paths (``read_system_setting``
→ ``system_settings`` seeded by both migrations, and the gate's own read of the
workspace's Socials switch), with the Composio SDK, the LinkedIn direct API and a
Playbook step's spine mocked:

* the list is DATA: ``socials.post_actions``, seeded in the wave's one migration
  (insert-if-absent, never overwriting an edit, removed by the downgrade only
  where the migration made it; on Postgres ``create_all`` first, then the upgrade
  twice, leaves one row); no slug is a constant in the gate;
* in a Socials-on workspace INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH is refused with
  the message and no Composio call is made: the agent executor, both agent tool
  paths, the LinkedIn image workaround, a name validation resolved onto a listed
  action, and a Playbook step;
* a read (LINKEDIN_GET_MY_INFO) and an upload (TWITTER_UPLOAD_MEDIA) still run;
  a Socials-off workspace (either switch) posts exactly as before;
* fail closed: a malformed list, or one that cannot be read with nothing cached,
  refuses in a Socials-on workspace (a failed refresh keeps the cached list); a
  Socials switch that cannot be read refuses a listed action; the Wave 0 deny
  list is checked first and always wins;
* the way through: only ``PLATFORM_PUBLISHER`` passes, nothing passes it yet, and
  the deny list still applies to it;
* the gate follows the deny list at both agent sites, before anything executes,
  and every other Composio execution site is classified as no agent posting path.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import threading
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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
from sqlalchemy import exc as sa_exc  # noqa: E402
from sqlalchemy.dialects.postgresql import ARRAY, JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import config as config_module  # noqa: E402
import core.models  # noqa: E402,F401  (registers every mapper)
import core.composio.deny_list as deny_list  # noqa: E402
import core.composio.post_gate as post_gate  # noqa: E402
import core.database.database as database_mod  # noqa: E402
import core.llm.manager as llm_manager  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
from core.composio.client import ComposioClient  # noqa: E402
from core.composio.tool_executor import ComposioToolExecutor  # noqa: E402
from core.database.database import Base  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from tests.helpers_unreadable_settings import UNREADABLE_MODES, settings_unreadable  # noqa: E402
# The Wave 0 deny list's harness: one Playbook LLM step's boundaries, and its
# inventory of every Composio execution site in the source.
from tests.test_prd251_composio_deny import (  # noqa: E402
    _execution_sites,
    _function,
    _Playbook,
    _source_files,
)

VERSIONS = _ORCH / "alembic" / "versions"
WAVE1_MIGRATION = VERSIONS / "prd251_wave1.py"

# Hex with letters, so SQLite keeps every UUID column as text.
WS_ON = uuid.UUID("00000000-0000-0000-0000-0000000118a1")
WS_OFF = uuid.UUID("00000000-0000-0000-0000-0000000118a2")
WS_GONE = uuid.UUID("00000000-0000-0000-0000-0000000118a3")  # no such workspace
CREATED = datetime(2026, 9, 1, 9, 0)

PUBLISH = "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH"
LINKEDIN_POST = "LINKEDIN_CREATE_LINKED_IN_POST"
READ = "LINKEDIN_GET_MY_INFO"
UPLOAD = "TWITTER_UPLOAD_MEDIA"
BILLING = "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE"
IMAGES = {"text": "Launch day", "images": ["/workspace/launch.png"]}
BLOCKED = "This action is blocked in Automatos: "
# The story's words (PRD S3.5, US-118).
REFUSAL = (
    "Socials is on for this workspace: draft the post with platform_create_social_post; "
    "a person approves it in the Socials tab and the platform publishes it."
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WAVE0 = _load(VERSIONS / "prd251_socials.py", "prd251_socials_migration_post_gate")
WAVE1 = _load(WAVE1_MIGRATION, "prd251_wave1_migration_post_gate")
POST_SEED = list(WAVE1.SOCIALS_POST_ACTIONS_SEED)
DENIED_SEED = tuple(WAVE0.COMPOSIO_DENIED_ACTIONS_SEED)


# ---------------------------------------------------------------------------
# The harness: the settings both migrations seed, and two workspaces, on SQLite
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
WORKSPACES = _sqlite_copy(Workspace.__table__, _TABLES)


@pytest.fixture
def env(monkeypatch):
    """read_system_setting and the gate's workspace read → SessionLocal → one
    in-memory database: the Wave 0 deny list, the wave's settings (the post gate's
    list among them), the Socials master switch ON, a Socials-on and a Socials-off
    workspace."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    _TABLES.create_all(engine)
    SystemSetting.__table__.create(bind=engine)
    with engine.begin() as conn:
        WAVE0._seed_settings(conn, WAVE0._composio_settings_seed())
        WAVE1.seed_settings(conn, WAVE1.settings_seed())
        conn.execute(SystemSetting.__table__.insert().values(
            category="socials", key="enabled", value="true", value_type="boolean", created_by="prd251",
        ))
    factory = sessionmaker(bind=engine)
    session = factory()
    for ws_id, settings in ((WS_ON, {"socials": {"enabled": True}}), (WS_OFF, {})):
        session.add(Workspace(
            id=ws_id, name=f"ws-{ws_id.hex[-2:]}", plan="basic", plan_limits={},
            settings=settings, onboarding={}, created_at=CREATED, updated_at=CREATED,
        ))
    session.commit()
    session.close()
    monkeypatch.setattr(database_mod, "SessionLocal", factory)
    deny_list.reset_cache()
    post_gate.reset_cache()
    try:
        yield SimpleNamespace(engine=engine)
    finally:
        deny_list.reset_cache()
        post_gate.reset_cache()
        engine.dispose()


def _set(engine, category, key, value):
    table = SystemSetting.__table__
    with engine.begin() as conn:
        conn.execute(
            table.update().where(table.c.category == category, table.c.key == key)
            .values(value=value if isinstance(value, str) else json.dumps(value))
        )


def _unreadable_read(error, category="socials", key="post_actions"):
    """``read_system_setting`` that fails as ``error`` for one row and reads the rest."""
    real = llm_manager.read_system_setting

    def read(cat, k):
        if (cat, k) == (category, key):
            raise error
        return real(cat, k)

    return read


READ_FAILURES = [
    sa_exc.TimeoutError("QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00"),
    sa_exc.OperationalError("SELECT system_settings.value", {}, Exception("server closed the connection unexpectedly")),
    sa_exc.ProgrammingError("SELECT system_settings.value", {}, Exception('relation "system_settings" does not exist')),
]


class _Agent:
    """ComposioToolExecutor as the agent paths call it, with everything past the
    gates stubbed: the workspace's Composio entity, file uploads, entity
    extraction, the LinkedIn image workaround and the SDK (behind the real
    ComposioClient, which checks the deny list again)."""

    def __init__(self, monkeypatch):
        import core.composio.linkedin_image_workaround as lw

        self.monkeypatch = monkeypatch
        self.sdk = MagicMock(name="ComposioSDK")
        self.sdk.tools.execute.return_value = {"successful": True, "data": {"ok": True}}
        client = ComposioClient(api_key="test-key")
        client._composio = self.sdk
        self.db = MagicMock(name="db")
        self.executor = ComposioToolExecutor(db=self.db, client=client)
        self.entity = MagicMock(name="get_entity_for_workspace", return_value={"composio_entity_id": "entity-1"})
        self.uploads = AsyncMock(name="_resolve_file_uploads", side_effect=lambda action, params, workspace_id: (params, []))
        self.linkedin = AsyncMock(
            name="execute_linkedin_image_post", return_value={"success": True, "data": {"id": "urn:li:share:1"}, "error": None},
        )
        monkeypatch.setattr(self.executor, "get_entity_for_workspace", self.entity)
        monkeypatch.setattr(self.executor, "_resolve_file_uploads", self.uploads)
        monkeypatch.setattr(self.executor, "_extract_and_store_entities", MagicMock(name="_extract_and_store_entities"))
        monkeypatch.setattr(lw, "execute_linkedin_image_post", self.linkedin)

    def validates(self, cached_name, app):
        """The agent's validation passes: the app assigned and connected, and the
        name asked for mapped in the cache onto ``cached_name``."""
        import core.composio.entity_manager as entity_manager_mod

        self.monkeypatch.setattr(entity_manager_mod.EntityManager, "get_connected_apps", lambda _self, ws: [app])
        self.db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(
            action_name=cached_name, app_name=app,
        )
        self.monkeypatch.setattr(self.executor, "validate_feature_access", lambda *args, **kwargs: True)

    async def execute(self, action, workspace_id, params=None, **kwargs):
        return await self.executor.execute(
            action=action, params=dict(params or {}), agent_id=7, workspace_id=workspace_id, **kwargs,
        )

    def tools(self):
        """What exec_composio receives from the unified executor."""
        return SimpleNamespace(composio_executor=self.executor, db=MagicMock(name="unified-db"))

    def assert_nothing_ran(self):
        self.entity.assert_not_called()
        self.uploads.assert_not_called()
        self.linkedin.assert_not_called()
        self.sdk.tools.execute.assert_not_called()


def _assert_refused(result, action, refusal=REFUSAL):
    assert result["success"] is False
    assert result["error"] == refusal
    assert result["error_type"] == "socials_post_gate"
    assert result["use_tool"] == "platform_create_social_post"
    assert result["action"] == action


class _Step(_Playbook):
    """One Playbook LLM step in a given workspace."""

    async def run(self, workspace_id=WS_ON):
        from api import recipe_executor

        return await recipe_executor._execute_step(
            db=MagicMock(name="db"),
            agent=SimpleNamespace(id=7, name="Poster"),
            clean_prompt="Post the launch",
            workspace_id=workspace_id,
            max_iterations=3,
        )


# ---------------------------------------------------------------------------
# The list is data, seeded in the wave's one migration
# ---------------------------------------------------------------------------


def test_the_list_is_seeded_in_the_one_wave_migration():
    assert WAVE1.revision == "prd251_wave1"
    rows = {row["key"]: row for row in WAVE1.settings_seed()}
    row = rows["post_actions"]
    assert (row["category"], row["value_type"]) == ("socials", "json")
    assert row["default_value"] == row["value"]
    assert json.loads(row["value"]) == POST_SEED
    assert post_gate.parse_post_actions(row["value"]) == frozenset(POST_SEED)
    assert "platform_create_social_post" in row["description"]

    # The publish actions the old publisher skills call (the story's notes).
    assert {PUBLISH, "TWITTER_CREATION_OF_A_POST", LINKEDIN_POST, "LINKEDIN_CREATE_POST"} <= set(POST_SEED)
    # Reads, uploads and containers are not posting actions; the Wave 0 deny list is its own list.
    not_posting = {
        READ, UPLOAD, "TWITTER_INITIALIZE_MEDIA_UPLOAD", "TWITTER_APPEND_MEDIA_UPLOAD", "TWITTER_UPLOAD_LARGE_MEDIA",
        "LINKEDIN_INITIALIZE_IMAGE_UPLOAD", "LINKEDIN_REGISTER_IMAGE_UPLOAD", "INSTAGRAM_POST_IG_USER_MEDIA",
        "INSTAGRAM_CREATE_CAROUSEL_CONTAINER", "INSTAGRAM_CREATE_MEDIA_CONTAINER", "TIKTOK_FETCH_PUBLISH_STATUS",
    }
    assert not_posting.isdisjoint(POST_SEED)
    assert set(DENIED_SEED).isdisjoint(POST_SEED)
    # TikTok and YouTube join when composio_actions_cache confirms their slugs (the story's notes).
    assert not any(slug.startswith(("TIKTOK_", "YOUTUBE_")) for slug in POST_SEED)
    assert len(set(POST_SEED)) == len(POST_SEED)
    assert all(re.fullmatch(r"[A-Z0-9_]+", slug) for slug in POST_SEED)


def _settings_engine():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SystemSetting.__table__.create(bind=engine)
    return engine


def _post_rows(conn):
    table = SystemSetting.__table__
    return conn.execute(
        table.select().where(table.c.category == "socials", table.c.key == "post_actions").order_by(table.c.created_by)
    ).fetchall()


def test_the_seed_is_insert_if_absent_and_the_downgrade_removes_only_its_own_row():
    engine = _settings_engine()
    table = SystemSetting.__table__
    try:
        with engine.begin() as conn:
            WAVE1.seed_settings(conn, WAVE1.settings_seed())
            WAVE1.seed_settings(conn, WAVE1.settings_seed())  # a re-run adds nothing
            (row,) = _post_rows(conn)
            assert row.created_by == "prd251_wave1" and json.loads(row.value) == POST_SEED

            conn.execute(table.update().where(table.c.id == row.id).values(value='["X_POST"]'))
            WAVE1.seed_settings(conn, WAVE1.settings_seed())
            assert [r.value for r in _post_rows(conn)] == ['["X_POST"]']  # a super-admin's edit is kept

            WAVE1.unseed_settings(conn, WAVE1.settings_seed())
            assert _post_rows(conn) == []
            conn.execute(table.insert().values(category="socials", key="post_actions", value="[]", created_by="admin"))
            WAVE1.unseed_settings(conn, WAVE1.settings_seed())
            assert [r.created_by for r in _post_rows(conn)] == ["admin"]  # never the migration's to delete
    finally:
        engine.dispose()


def test_the_list_is_read_in_one_place_and_no_slug_is_a_constant_in_the_gate():
    readers = sorted(
        path.relative_to(_ORCH).as_posix() for path in _source_files()
        if '"post_actions"' in path.read_text(encoding="utf-8")
    )
    assert readers == ["alembic/versions/prd251_wave1.py", "core/composio/post_gate.py"]
    gate = (_ORCH / "core" / "composio" / "post_gate.py").read_text(encoding="utf-8")
    assert [slug for slug in POST_SEED if slug in gate] == []


@pytest.mark.parametrize("raw, expected", [
    (None, frozenset()),
    ("", frozenset()),
    ("   ", frozenset()),
    (MagicMock(), frozenset()),  # not text — the column only ever holds text
    ('["a_b", " C_D ", ""]', frozenset({"A_B", "C_D"})),
])
def test_parse_post_actions(raw, expected):
    assert post_gate.parse_post_actions(raw) == expected


@pytest.mark.parametrize("raw", ["not json", '{"slug": "X"}', "[1, 2]", '"INSTAGRAM"'])
def test_parse_post_actions_refuses_anything_but_a_list_of_slugs(raw):
    with pytest.raises(ValueError):
        post_gate.parse_post_actions(raw)


# ---------------------------------------------------------------------------
# AC1/AC2: refused with the message, before any Composio call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_agents_instagram_publish_is_refused_in_a_socials_workspace(env, monkeypatch):
    agent = _Agent(monkeypatch)

    result = await agent.execute(PUBLISH, WS_ON, {"creation_id": "17890"})

    assert post_gate.SOCIALS_ON_REFUSAL == REFUSAL
    _assert_refused(result, PUBLISH)
    agent.db.query.assert_not_called()  # refused before access validation
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("slug", [
    PUBLISH, "INSTAGRAM_CREATE_POST", "TWITTER_CREATION_OF_A_POST", "LINKEDIN_CREATE_POST",
    "LINKEDIN_CREATE_VIDEO_POST", "LINKEDIN_CREATE_ARTICLE_OR_URL_SHARE",
])
async def test_both_agent_tool_paths_are_refused_whatever_the_case(env, monkeypatch, slug):
    from modules.tools.execution import exec_composio

    agent = _Agent(monkeypatch)
    per_action = await exec_composio.execute_composio_tool(
        agent.tools(), SimpleNamespace(name=f"composio_{slug}", metadata={"action": slug}), {},
        agent_id=7, workspace_id=WS_ON,
    )
    meta_tool = await exec_composio.execute_composio_execute(
        agent.tools(), "composio_execute", {"action": slug.lower(), "params": {"text": "hi"}},
        agent_id=7, workspace_id=WS_ON,
    )

    for result in (per_action, meta_tool):
        _assert_refused(result, slug)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
async def test_the_linkedin_image_workaround_is_refused_before_it_runs(env, monkeypatch):
    agent = _Agent(monkeypatch)

    result = await agent.execute(LINKEDIN_POST, WS_ON, IMAGES)

    _assert_refused(result, LINKEDIN_POST)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("asked, cached_name", [
    ("create_linked_in_post", LINKEDIN_POST),           # unprefixed
    ("linkedin-create-linked-in-post", LINKEDIN_POST),  # slug form
    ("post to linkedin", "CREATE LINKED IN POST"),      # display name, rebuilt with the app prefix
])
async def test_the_gate_checks_the_action_validation_resolved(env, monkeypatch, asked, cached_name):
    """The name asked for is not listed; the action it resolves to is, and it never
    reaches the LinkedIn direct API (which bypasses the Composio client)."""
    agent = _Agent(monkeypatch)
    agent.validates(cached_name, "LINKEDIN")
    assert await post_gate.post_action_refusal(asked, WS_ON) is None

    result = await agent.execute(asked, WS_ON, IMAGES, app_name="LINKEDIN")

    _assert_refused(result, LINKEDIN_POST)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, tool_args", [(PUBLISH, {"creation_id": "17890"}), (LINKEDIN_POST, IMAGES)])
async def test_a_playbook_step_is_refused_before_dedup_the_linkedin_workaround_uploads_or_the_spine(
    env, monkeypatch, tool_name, tool_args,
):
    step = _Step(monkeypatch, tool_name, tool_args)

    result = await step.run(WS_ON)

    assert result["status"] == "success"
    (call,) = result["execution"]["tool_calls"]
    assert call["action"] == tool_name
    assert call["result"] == f"Error executing {tool_name}: {REFUSAL}"
    (message,) = [m for m in result["execution"]["messages"] if m.get("role") == "tool"]
    assert REFUSAL in message["content"]
    step.spine.execute_and_format.assert_not_called()
    step.resolve_uploads.assert_not_called()
    step.linkedin.assert_not_called()
    step.get_client.assert_not_called()


# ---------------------------------------------------------------------------
# AC3: reads and uploads still run; a Socials-off workspace posts as before
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("slug, app", [(READ, "LINKEDIN"), (UPLOAD, "TWITTER")])
async def test_a_read_and_an_upload_still_run_in_a_socials_workspace(env, monkeypatch, slug, app):
    from modules.tools.execution import exec_composio

    agent = _Agent(monkeypatch)
    agent.validates(slug, app)

    result = await exec_composio.execute_composio_execute(
        agent.tools(), "composio_execute", {"action": slug, "params": {"media": "/workspace/a.png"}},
        agent_id=7, workspace_id=WS_ON,
    )

    assert result["success"] is True, result
    agent.sdk.tools.execute.assert_called_once()
    assert agent.sdk.tools.execute.call_args.kwargs["slug"] == slug


@pytest.mark.asyncio
async def test_a_socials_off_workspace_posts_exactly_as_before(env, monkeypatch):
    from modules.tools.execution import exec_composio

    agent = _Agent(monkeypatch)
    agent.validates(PUBLISH, "INSTAGRAM")
    published = await exec_composio.execute_composio_execute(
        agent.tools(), "composio_execute", {"action": PUBLISH, "params": {"creation_id": "17890"}},
        agent_id=7, workspace_id=WS_OFF,
    )
    assert published["success"] is True, published
    assert agent.sdk.tools.execute.call_args.kwargs["slug"] == PUBLISH

    agent.validates(LINKEDIN_POST, "LINKEDIN")
    linkedin = await agent.execute(LINKEDIN_POST, WS_OFF, IMAGES)
    assert linkedin["success"] is True, linkedin
    agent.linkedin.assert_called_once()  # the image workaround, exactly as before


@pytest.mark.asyncio
async def test_a_socials_off_playbook_step_reaches_the_linkedin_workaround(env, monkeypatch):
    step = _Step(monkeypatch, LINKEDIN_POST, IMAGES)
    step.linkedin.return_value = {"success": True, "data": {"id": "urn:li:share:1"}, "error": None}

    result = await step.run(WS_OFF)

    (call,) = result["execution"]["tool_calls"]
    assert "urn:li:share:1" in call["result"]
    step.linkedin.assert_called_once()


@pytest.mark.asyncio
async def test_socials_off_is_either_switch_off(env, monkeypatch):
    assert await post_gate.post_action_refusal(PUBLISH, WS_OFF) is None  # the workspace switch
    assert await post_gate.post_action_refusal(PUBLISH, WS_GONE) is None  # no such workspace
    _set(env.engine, "socials", "enabled", "false")  # the platform master switch
    agent = _Agent(monkeypatch)

    result = await agent.execute(PUBLISH, WS_ON, {"creation_id": "17890"}, skip_validation=True)

    assert result["success"] is True, result
    agent.sdk.tools.execute.assert_called_once()


# ---------------------------------------------------------------------------
# AC4: fail closed for a Socials-on workspace; the Wave 0 deny list wins
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["not json", '{"slug": "X"}', "[1, 2]", '"INSTAGRAM"'])
async def test_a_malformed_list_refuses_every_call_in_a_socials_workspace_and_none_elsewhere(env, monkeypatch, raw):
    logger = MagicMock(name="logger")
    monkeypatch.setattr(post_gate, "logger", logger)
    _set(env.engine, "socials", "post_actions", raw)
    agent = _Agent(monkeypatch)

    for slug in (PUBLISH, READ):
        refused = await agent.execute(slug, WS_ON)
        assert refused["error_type"] == "socials_post_gate"
        assert refused["error"].startswith("Socials is on for this workspace and its list of posting actions")
        assert "is not a JSON list of action slugs" in refused["error"]
    agent.assert_nothing_ran()
    assert "unreadable" in logger.error.call_args.args[0]

    assert await post_gate.post_action_refusal(PUBLISH, WS_OFF) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", READ_FAILURES, ids=["pool exhausted", "connection dropped", "table missing"])
async def test_a_list_that_cannot_be_read_refuses_in_a_socials_workspace_and_the_deny_list_still_wins(
    env, monkeypatch, error,
):
    logger = MagicMock(name="logger")
    monkeypatch.setattr(post_gate, "logger", logger)
    monkeypatch.setattr(llm_manager, "read_system_setting", _unreadable_read(error))
    agent = _Agent(monkeypatch)

    for slug in (PUBLISH, READ):
        refused = await agent.execute(slug, WS_ON)
        assert refused["error_type"] == "socials_post_gate"
        assert "could not be read, so no Composio action runs here" in refused["error"]
    agent.assert_nothing_ran()
    assert "could not be read and nothing is cached" in logger.error.call_args.args[0]
    assert logger.error.call_args.kwargs["exc_info"] is True

    assert await post_gate.post_action_refusal(PUBLISH, WS_OFF) is None  # Socials off: as before

    denied = await agent.execute(BILLING, WS_ON)  # the deny list, read as usual, is first
    assert denied["error_type"] == "action_denied"
    assert denied["error"] == BLOCKED + deny_list.DENIED_REASON.format(slug=BILLING)


@pytest.mark.asyncio
async def test_a_failed_refresh_keeps_the_cached_list(env, monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(post_gate, "_now", lambda: now[0])
    monkeypatch.setattr(post_gate, "_start_background_refresh", post_gate._refresh)
    logger = MagicMock(name="logger")
    monkeypatch.setattr(post_gate, "logger", logger)
    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) == REFUSAL

    monkeypatch.setattr(llm_manager, "read_system_setting", _unreadable_read(READ_FAILURES[1]))
    now[0] += config_module.config.SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS + 1

    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) == REFUSAL
    assert await post_gate.post_action_refusal(READ, WS_ON) is None
    assert any("could not be refreshed" in c.args[0] for c in logger.warning.call_args_list)
    logger.error.assert_not_called()


@pytest.mark.asyncio
async def test_a_socials_switch_that_cannot_be_read_refuses_a_listed_action(env, monkeypatch):
    real = socials_settings.read_system_setting

    def master_unreadable(category, key):
        if (category, key) == ("socials", "enabled"):
            raise READ_FAILURES[1]
        return real(category, key)

    monkeypatch.setattr(socials_settings, "read_system_setting", master_unreadable)
    agent = _Agent(monkeypatch)
    unread = post_gate.SWITCH_UNREAD_REFUSAL.format(slug=PUBLISH)
    assert unread.startswith("Automatos could not read whether Socials is on for this workspace")

    for workspace in (WS_ON, WS_OFF):  # "off" cannot be told apart from "could not read"
        _assert_refused(await agent.execute(PUBLISH, workspace), PUBLISH, unread)
    agent.assert_nothing_ran()
    assert await post_gate.post_action_refusal(READ, WS_ON) is None  # not listed: no switch is read

    monkeypatch.setattr(socials_settings, "read_system_setting", real)
    WORKSPACES.drop(env.engine)  # the workspace switch cannot be read
    assert await post_gate.post_action_refusal(PUBLISH, WS_OFF) == unread


@pytest.mark.asyncio
async def test_the_deny_list_wins_over_the_gate(env, monkeypatch):
    _set(env.engine, "composio", "denied_actions", [*DENIED_SEED, PUBLISH])
    agent = _Agent(monkeypatch)

    result = await agent.execute(PUBLISH, WS_ON)

    assert result["error_type"] == "action_denied"
    assert result["error"] == BLOCKED + deny_list.DENIED_REASON.format(slug=PUBLISH)
    agent.assert_nothing_ran()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", UNREADABLE_MODES)
async def test_with_every_setting_unreadable_the_deny_list_refuses_first(monkeypatch, mode):
    with settings_unreadable(mode, monkeypatch):
        agent = _Agent(monkeypatch)
        result = await agent.execute(PUBLISH, WS_ON)

    assert result["error_type"] == "action_denied"
    assert result["error"] == BLOCKED + deny_list.READ_FAILED_REASON
    agent.assert_nothing_ran()


@pytest.mark.parametrize("mode", UNREADABLE_MODES)
def test_the_strict_master_switch_raises_where_the_lenient_one_is_off(monkeypatch, mode):
    with settings_unreadable(mode, monkeypatch):
        with pytest.raises(sa_exc.SQLAlchemyError):
            socials_settings.socials_master_switch()
        assert socials_settings.socials_master_enabled() is False


# ---------------------------------------------------------------------------
# The way through: the platform publisher only, and nothing passes it yet
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_the_platform_publisher_passes_a_listed_action(env, monkeypatch):
    agent = _Agent(monkeypatch)
    for impostor in (True, "platform_publisher", object(), post_gate._WayThrough("the platform publisher (PRD-251 Wave 3)")):
        _assert_refused(await agent.execute(PUBLISH, WS_ON, way_through=impostor), PUBLISH)
    agent.assert_nothing_ran()

    published = await agent.execute(PUBLISH, WS_ON, skip_validation=True, way_through=post_gate.PLATFORM_PUBLISHER)
    assert published["success"] is True, published
    assert agent.sdk.tools.execute.call_args.kwargs["slug"] == PUBLISH

    denied = await agent.execute(BILLING, WS_ON, skip_validation=True, way_through=post_gate.PLATFORM_PUBLISHER)
    assert denied["error_type"] == "action_denied"  # the deny list applies to the publisher too
    agent.sdk.tools.execute.assert_called_once()


def test_nothing_passes_the_way_through_yet():
    """The platform publisher (Wave 3) will. Until then the one use is the executor
    handing its own parameter to the gate."""
    offenders = []
    for path in _source_files():
        source = path.read_text(encoding="utf-8")
        if "way_through" not in source:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                forwarded = isinstance(keyword.value, ast.Name) and keyword.value.id == "way_through"
                if keyword.arg == "way_through" and not forwarded:
                    offenders.append((path.relative_to(_ORCH).as_posix(), node.lineno))
    assert offenders == []
    holders = [p.relative_to(_ORCH).as_posix() for p in _source_files() if "PLATFORM_PUBLISHER" in p.read_text(encoding="utf-8")]
    assert holders == ["core/composio/post_gate.py"]


# ---------------------------------------------------------------------------
# Structure: after the deny list, before anything executes, at both agent sites
# ---------------------------------------------------------------------------

GATE_SITES = [
    ("core/composio/tool_executor.py", "ComposioToolExecutor.execute"),
    ("api/recipe_executor.py", "_execute_step"),
]
# Every other Composio execution site the Wave 0 inventory finds, and why an agent
# never posts through it. A new site fails the test below until it is classified:
# gated like the two above, or listed here with its reason.
NOT_AGENT_POSTING_PATHS = {
    ("core/composio/client.py", "ComposioClient.execute_action"):
        "the SDK chokepoint; the agent executor gates before it calls it",
    ("api/composio.py", "test_linkedin_upload_init"): "an admin's LinkedIn upload smoke test; posts nothing",
    ("api/shopify.py", "_product_sync_impl"): "Shopify product sync",
    ("api/shopify.py", "_orders_sync_impl"): "Shopify orders sync",
    ("core/credentials/integration_bridges/shopify.py", "shopify_access_token"): "Shopify credential probe",
    ("modules/rag/services/cloud_file_downloader.py", "CloudFileDownloader._execute_via_rest_api"): "cloud file download",
    ("modules/rag/services/cloud_file_downloader.py", "CloudFileDownloader._download_via_sdk"): "cloud file download",
    ("services/composio_api_service.py", "ComposioAPIService.execute_action"): "v2 REST service; no importers",
    ("api/workspace_github.py", "list_github_repos"): "lists the workspace's GitHub repos",
    ("modules/tools/composio_tool_router.py", "ComposioToolRouter.execute_tool"):
        "the Tool Router meta-tool path; UnifiedToolExecutor never dispatches to it",
    ("modules/tools/services/composio_tool_service.py", "ComposioToolService.execute_action"): "no callers",
}


def test_every_composio_execution_site_is_classified_for_the_post_gate():
    assert set(_execution_sites()) == set(GATE_SITES) | set(NOT_AGENT_POSTING_PATHS)


@pytest.mark.parametrize("site", GATE_SITES)
def test_the_gate_follows_each_deny_check_before_anything_executes(site):
    checks = sorted(
        (node.lineno, node.func.id) for node in ast.walk(_function(*site))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {"composio_action_denial_async", "post_action_refusal"}
    )
    names = [name for _line, name in checks]
    assert names and names == ["composio_action_denial_async", "post_action_refusal"] * (len(names) // 2), checks
    first_execution = min(line for _kind, line in _execution_sites()[site])
    assert checks[-1][0] < first_execution, (checks, first_execution)


# ---------------------------------------------------------------------------
# The cache: an edit applies within one TTL, with no restart
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_edit_applies_within_one_ttl_with_no_restart(env, monkeypatch):
    now = [1_000.0]
    monkeypatch.setattr(post_gate, "_now", lambda: now[0])
    monkeypatch.setattr(post_gate, "_start_background_refresh", post_gate._refresh)
    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) == REFUSAL

    _set(env.engine, "socials", "post_actions", [slug for slug in POST_SEED if slug != PUBLISH])
    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) == REFUSAL  # the cached list, within the TTL

    now[0] += config_module.config.SOCIALS_POST_ACTIONS_CACHE_TTL_SECONDS + 1
    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) is None  # same process: the edit applies
    assert await post_gate.post_action_refusal(LINKEDIN_POST, WS_ON) == REFUSAL


@pytest.mark.asyncio
async def test_the_gate_never_reads_on_the_event_loop_thread(env, monkeypatch):
    """F105: the list is read once, in a worker thread, then answered from memory;
    the switches are read only for a listed action, in a worker thread too."""
    loop_thread = threading.get_ident()
    threads = {"list": [], "switches": []}
    real_read, real_switches = post_gate._read_post_actions, post_gate._socials_on

    def read():
        threads["list"].append(threading.get_ident())
        return real_read()

    def switches(workspace_id):
        threads["switches"].append(threading.get_ident())
        return real_switches(workspace_id)

    monkeypatch.setattr(post_gate, "_read_post_actions", read)
    monkeypatch.setattr(post_gate, "_socials_on", switches)

    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) == REFUSAL
    assert await post_gate.post_action_refusal(READ, WS_ON) is None

    assert len(threads["list"]) == 1 and len(threads["switches"]) == 1
    assert loop_thread not in threads["list"] + threads["switches"]


@pytest.mark.asyncio
async def test_no_row_lists_nothing(env):
    table = SystemSetting.__table__
    with env.engine.begin() as conn:
        conn.execute(table.delete().where(table.c.category == "socials", table.c.key == "post_actions"))

    assert await post_gate.post_action_refusal(PUBLISH, WS_ON) is None


# ---------------------------------------------------------------------------
# Postgres: create_all first, then the upgrade twice (the 89d89c250 lesson)
# ---------------------------------------------------------------------------


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


def _run_migration(conn, *steps):
    """Run the revision's upgrade/downgrade on ``conn`` through a real alembic context."""
    mod = _load(WAVE1_MIGRATION, "prd251_wave1_migration_post_gate_pg")
    with Operations.context(MigrationContext.configure(conn)):
        for step in steps:
            getattr(mod, step)()


def _pg_post_rows(conn):
    return conn.execute(
        sa.text(
            "SELECT value, default_value, value_type, created_by FROM system_settings "
            "WHERE category = 'socials' AND key = 'post_actions'"
        )
    ).fetchall()


@pytest.mark.integration
def test_create_all_first_then_the_upgrade_twice_seeds_the_list_once(pg_engine):
    with pg_engine.connect() as conn:
        trans = conn.begin()
        try:
            conn.execute(sa.text("SET LOCAL lock_timeout = '5s'"))
            Base.metadata.create_all(bind=conn)
            _run_migration(conn, "upgrade")
            _run_migration(conn, "upgrade")

            (row,) = _pg_post_rows(conn)
            assert (row.value_type, row.created_by) == ("json", "prd251_wave1")
            assert json.loads(row.value) == POST_SEED == json.loads(row.default_value)

            _run_migration(conn, "downgrade")
            assert [r for r in _pg_post_rows(conn) if r.created_by == "prd251_wave1"] == []
            _run_migration(conn, "upgrade")
            assert len(_pg_post_rows(conn)) == 1
        finally:
            trans.rollback()
