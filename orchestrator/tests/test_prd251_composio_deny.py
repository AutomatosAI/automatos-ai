"""PRD-251 S0.6 (D16) — the Composio deny list, platform-wide.

Composio's Higgsfield toolkit carries HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE
("charges real money") and nothing in Automatos stopped an agent calling it:
no per-action deny list, no billing keyword in the capability classifier, and
the policy plane ships ``off``. Pinned here, with the policy plane OFF:

* the list is DATA — the ``composio.denied_actions`` system setting, seeded by
  the one PRD-251 migration with the eight D16 slugs and visible to the
  super-admin in Settings → System Settings; no slug is a code constant;
* the real read path (``get_system_setting`` → ``system_settings`` on SQLite):
  removing a slug unblocks it on the next call, no restart; matching is
  case-insensitive; no row denies nothing; a malformed value fails CLOSED;
* every Composio execution entry point refuses a denied slug with "This action
  is blocked in Automatos: <reason>" and the (mocked) Composio SDK / HTTP call is
  never made — the Composio client, the agent executor and the three agent tool
  paths, the tool router and tool service, a Playbook step, the LinkedIn smoke
  route, both Shopify bulk syncs, the Shopify credential probe, the cloud-file
  REST download and the v2 API service; the agent executor also checks the
  action its validation resolved the name to, not only the name asked for;
* an AST inventory of every execution site in the orchestrator (SDK
  ``tools.execute``, the ``execute_action`` wrapper, Composio REST URLs, the
  LinkedIn direct API) proves each one calls the ONE helper before executing,
  or goes through ``ComposioClient.execute_action``, which does. A new site
  fails this test until it is classified.
"""
from __future__ import annotations

import ast
import functools
import json
import os
import re
import sys
import uuid
import warnings
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

import importlib.util  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import config as config_module  # noqa: E402
import core.composio.client as client_mod  # noqa: E402
import core.composio.deny_list as deny_list  # noqa: E402
import core.database.database as database_mod  # noqa: E402
from core.composio.client import ComposioClient  # noqa: E402
from core.composio.tool_executor import ComposioToolExecutor  # noqa: E402
from core.models.system_settings import SettingCategory, SystemSetting  # noqa: E402

BILLING = "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE"
LINKEDIN_POST = "LINKEDIN_CREATE_LINKED_IN_POST"
SHOPIFY_BULK = "SHOPIFY_BULK_QUERY_OPERATION"
SHOPIFY_PROBE = "SHOPIFY_GET_SHOP_DETAILS"
D16 = [
    "HIGGSFIELD_MCP_CONFIRM_BILLING_PURCHASE",
    "HIGGSFIELD_MCP_CANCEL_TRIAL_AUTO_RENEWAL",
    "HIGGSFIELD_MCP_CONFIRM_TRIAL_CANCEL",
    "HIGGSFIELD_MCP_CREATE_WEBSITE",
    "HIGGSFIELD_MCP_DEPLOY_WEBSITE",
    "HIGGSFIELD_MCP_PUBLISH_WEBSITE",
    "HIGGSFIELD_MCP_PARTICIPATE_IN_CONTEST",
    "HIGGSFIELD_MCP_APPS_INVOKE",
]
MIGRATION = _ORCH / "alembic" / "versions" / "prd251_socials.py"
BLOCKED = "This action is blocked in Automatos: "


# ---------------------------------------------------------------------------
# Fixtures: the real read path on SQLite, seeded by the migration; policy off
# ---------------------------------------------------------------------------


def _load_migration():
    spec = importlib.util.spec_from_file_location("prd251_socials_migration_deny", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings_engine():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    SystemSetting.__table__.create(bind=engine)
    return engine


@pytest.fixture(autouse=True)
def policy_plane_off(monkeypatch):
    """Whatever else holds, the deny list works with the policy plane OFF."""
    monkeypatch.setattr(config_module.config, "POLICY_PLANE_MODE", "off", raising=False)
    monkeypatch.setattr(config_module.config, "POLICY_PLANE_ENABLED", False, raising=False)


@pytest.fixture
def settings_db(monkeypatch):
    """get_system_setting → SessionLocal → an in-memory system_settings table,
    seeded exactly as the migration seeds it."""
    engine = _settings_engine()
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    mod = _load_migration()
    with engine.begin() as conn:
        mod._seed_settings(conn, mod._composio_settings_seed())
    yield engine
    engine.dispose()


def _set_denied(engine, value):
    raw = value if isinstance(value, str) else json.dumps(value)
    table = SystemSetting.__table__
    with engine.begin() as conn:
        conn.execute(
            table.update()
            .where(table.c.category == "composio", table.c.key == "denied_actions")
            .values(value=raw)
        )


def _sdk():
    sdk = MagicMock(name="ComposioSDK")
    sdk.tools.execute.return_value = {"successful": True, "data": {"ok": True}}
    return sdk


def _client(sdk):
    client = ComposioClient(api_key="test-key")
    client._composio = sdk
    return client


def _assert_blocked(message, slug=BILLING):
    assert message.startswith(BLOCKED), message
    assert slug in message


# ---------------------------------------------------------------------------
# The list is data: the seed, the category, visible to the super-admin
# ---------------------------------------------------------------------------


def test_the_seed_holds_the_eight_d16_slugs_in_the_one_migration():
    mod = _load_migration()
    (row,) = mod._composio_settings_seed()
    assert (row["category"], row["key"], row["value_type"]) == ("composio", "denied_actions", "json")
    assert json.loads(row["value"]) == D16
    assert row["default_value"] == row["value"]

    engine = _settings_engine()
    try:
        with engine.begin() as conn:
            mod._seed_settings(conn, mod._composio_settings_seed())
            mod._seed_settings(conn, mod._composio_settings_seed())  # a re-run adds nothing
            rows = conn.execute(
                SystemSetting.__table__.select().where(SystemSetting.__table__.c.category == "composio")
            ).fetchall()
    finally:
        engine.dispose()
    assert len(rows) == 1
    assert json.loads(rows[0].value) == D16 and rows[0].created_by == "prd251"


def test_the_seed_never_overwrites_the_super_admins_list():
    mod = _load_migration()
    engine = _settings_engine()
    try:
        with engine.begin() as conn:
            conn.execute(SystemSetting.__table__.insert().values(
                category="composio", key="denied_actions", value='["X_Y"]', value_type="json", created_by="admin",
            ))
            mod._seed_settings(conn, mod._composio_settings_seed())
            rows = conn.execute(SystemSetting.__table__.select()).fetchall()
    finally:
        engine.dispose()
    assert [(r.value, r.created_by) for r in rows] == [('["X_Y"]', "admin")]


def test_the_list_is_a_system_setting_never_a_code_constant():
    assert SettingCategory.COMPOSIO.value == "composio"
    assert deny_list.KEY_DENIED_ACTIONS == "denied_actions"
    # No D16 slug is a constant anywhere the runtime reads (the migration seeds it).
    offenders = []
    for path in _source_files():
        if path.relative_to(_ORCH).as_posix().startswith("alembic/"):
            continue
        text = path.read_text(encoding="utf-8")
        if any(slug in text for slug in D16):
            offenders.append(path.relative_to(_ORCH).as_posix())
    assert offenders == []


def test_the_check_never_depends_on_the_policy_plane_or_the_classifier():
    tree = ast.parse((_ORCH / "core" / "composio" / "deny_list.py").read_text(encoding="utf-8"))
    imported = {
        (node.module or "") if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("policy" in name or "capabilities" in name for name in imported), imported
    assert {"core.llm.manager", "core.models.system_settings"} <= imported


def test_the_seeded_list_is_visible_to_the_super_admin_in_system_settings(settings_db):
    import api.system_settings as system_settings_api
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    session_factory = sessionmaker(bind=settings_db)

    def _db():
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(system_settings_api.router)
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = lambda: RequestContext(
        workspace_id=uuid.uuid4(),
        user=UserContext(id="root", clerk_user_id="clerk_root", system_role="super_admin"),
        auth_type="clerk",
    )
    response = TestClient(app).get("/api/system-settings/by-category")

    assert response.status_code == 200, response.text
    composio = next(cat for cat in response.json() if cat["category"] == "composio")
    (setting,) = composio["settings"]
    assert setting["key"] == "denied_actions" and setting["value_type"] == "json"
    assert json.loads(setting["value"]) == D16


# ---------------------------------------------------------------------------
# The helper: the real read path, no restart, case-insensitive, fail-closed
# ---------------------------------------------------------------------------


def test_removing_the_slug_from_the_setting_unblocks_it_with_no_restart(settings_db):
    sdk = _sdk()
    client = _client(sdk)

    blocked = client.execute_action(BILLING, {}, "entity-1")
    assert blocked["success"] is False and blocked["error_type"] == "action_denied"
    _assert_blocked(blocked["error"])
    sdk.tools.execute.assert_not_called()

    _set_denied(settings_db, [slug for slug in D16 if slug != BILLING])

    allowed = client.execute_action(BILLING, {}, "entity-1")  # same process, same client
    assert allowed["success"] is True
    sdk.tools.execute.assert_called_once()
    assert sdk.tools.execute.call_args.kwargs["slug"] == BILLING


@pytest.mark.parametrize("stored, asked", [
    ([BILLING.lower()], BILLING),
    ([BILLING], BILLING.lower()),
    ([f"  {BILLING.title()} "], BILLING),
])
def test_slug_matching_is_case_insensitive(settings_db, stored, asked):
    _set_denied(settings_db, stored)
    _assert_blocked(deny_list.composio_action_denial(asked))


def test_an_unlisted_action_runs(settings_db):
    assert deny_list.composio_action_denial("SLACK_SEND_MESSAGE") is None


def test_no_row_denies_nothing(monkeypatch):
    engine = _settings_engine()
    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    try:
        assert deny_list.composio_action_denial(BILLING) is None
    finally:
        engine.dispose()


@pytest.mark.parametrize("raw", ["not json", '{"slug": "X"}', '[1, 2]', '"HIGGSFIELD"'])
def test_a_malformed_list_fails_closed_for_every_action(settings_db, raw):
    _set_denied(settings_db, raw)
    denial = deny_list.composio_action_denial("SLACK_SEND_MESSAGE")
    assert denial == BLOCKED + deny_list.UNREADABLE_REASON


def test_an_empty_list_denies_nothing(settings_db):
    _set_denied(settings_db, [])
    assert deny_list.composio_action_denial(BILLING) is None


@pytest.mark.parametrize("raw, expected", [
    (None, frozenset()),
    ("", frozenset()),
    ("   ", frozenset()),
    (MagicMock(), frozenset()),  # not text — the column only ever holds text
    ('["a_b", " C_D ", ""]', frozenset({"A_B", "C_D"})),
])
def test_parse_denied_actions(raw, expected):
    assert deny_list.parse_denied_actions(raw) == expected


def test_the_refusal_names_the_slug_and_the_reason(settings_db):
    denial = deny_list.composio_action_denial(BILLING)
    assert denial == BLOCKED + deny_list.DENIED_REASON.format(slug=BILLING)
    assert "stay with a person" in denial


# ---------------------------------------------------------------------------
# Every entry point refuses before any network call (policy plane OFF)
# ---------------------------------------------------------------------------


def test_the_composio_client_refuses_before_the_sdk(settings_db):
    sdk = _sdk()
    result = _client(sdk).execute_action(BILLING.lower(), {"amount": 100}, "entity-1")
    assert result == {"success": False, "data": None, "error": result["error"], "error_type": "action_denied"}
    _assert_blocked(result["error"])
    sdk.tools.execute.assert_not_called()


@pytest.mark.asyncio
async def test_the_agent_executor_refuses_before_validation_uploads_or_the_sdk(settings_db):
    sdk = _sdk()
    db = MagicMock(name="db")
    executor = ComposioToolExecutor(db=db, client=_client(sdk))

    result = await executor.execute(
        action=BILLING.lower(), params={"amount": 100}, agent_id=1, workspace_id=uuid.uuid4(),
    )

    assert result["success"] is False and result["error_type"] == "action_denied"
    assert result["action"] == BILLING
    _assert_blocked(result["error"])
    db.query.assert_not_called()  # no access validation ran
    sdk.tools.execute.assert_not_called()


@pytest.mark.asyncio
async def test_the_agent_executor_refuses_a_denied_linkedin_post_before_the_workaround(settings_db, monkeypatch):
    import core.composio.linkedin_image_workaround as lw

    _set_denied(settings_db, [LINKEDIN_POST])
    direct = AsyncMock(name="execute_linkedin_image_post")
    monkeypatch.setattr(lw, "execute_linkedin_image_post", direct)
    sdk = _sdk()
    executor = ComposioToolExecutor(db=MagicMock(), client=_client(sdk))

    result = await executor.execute(
        action=LINKEDIN_POST, params={"text": "hi", "images": ["/workspace/a.png"]},
        agent_id=1, workspace_id=uuid.uuid4(),
    )

    _assert_blocked(result["error"], LINKEDIN_POST)
    direct.assert_not_called()
    sdk.tools.execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("asked, cached_name", [
    ("create_linked_in_post", LINKEDIN_POST),           # unprefixed
    ("linkedin-create-linked-in-post", LINKEDIN_POST),  # slug form
    ("post to linkedin", "CREATE LINKED IN POST"),      # display name, rebuilt with the app prefix
])
async def test_the_agent_executor_checks_the_action_validation_resolved(settings_db, monkeypatch, asked, cached_name):
    """Validation can resolve the name asked for onto another action. The list
    holds the action that runs, so a denied LinkedIn post never reaches the
    direct API (which bypasses the checked client) under another name."""
    import core.composio.entity_manager as entity_manager_mod
    import core.composio.linkedin_image_workaround as lw

    _set_denied(settings_db, [LINKEDIN_POST])
    assert deny_list.composio_action_denial(asked) is None  # the name asked for is not listed

    direct = AsyncMock(name="execute_linkedin_image_post")
    monkeypatch.setattr(lw, "execute_linkedin_image_post", direct)
    monkeypatch.setattr(entity_manager_mod.EntityManager, "get_connected_apps", lambda self, ws: ["LINKEDIN"])
    db = MagicMock(name="db")
    # Every .first(): the agent's LINKEDIN assignment, then the cache row the name resolves to.
    db.query.return_value.filter.return_value.first.return_value = SimpleNamespace(action_name=cached_name)
    sdk = _sdk()
    executor = ComposioToolExecutor(db=db, client=_client(sdk))
    entity = MagicMock(return_value={"composio_entity_id": "entity-1"})
    uploads = AsyncMock(side_effect=lambda action, params, workspace_id: (params, []))
    monkeypatch.setattr(executor, "validate_feature_access", lambda *args, **kwargs: True)
    monkeypatch.setattr(executor, "get_entity_for_workspace", entity)
    monkeypatch.setattr(executor, "_resolve_file_uploads", uploads)

    result = await executor.execute(
        action=asked, params={"text": "hi", "images": ["/workspace/a.png"]},
        agent_id=1, workspace_id=uuid.uuid4(), app_name="LINKEDIN",
    )

    assert result["success"] is False and result["error_type"] == "action_denied"
    assert result["action"] == LINKEDIN_POST
    _assert_blocked(result["error"], LINKEDIN_POST)
    entity.assert_not_called()
    uploads.assert_not_called()
    direct.assert_not_called()
    sdk.tools.execute.assert_not_called()


def _agent_executor(sdk):
    return SimpleNamespace(composio_executor=ComposioToolExecutor(db=MagicMock(), client=_client(sdk)), db=MagicMock())


@pytest.mark.asyncio
async def test_the_agent_tool_paths_refuse(settings_db):
    from modules.tools.execution import exec_composio

    sdk = _sdk()
    executor = _agent_executor(sdk)
    ws = uuid.uuid4()

    per_action = await exec_composio.execute_composio_tool(
        executor, SimpleNamespace(name=f"composio_{BILLING}", metadata={"action": BILLING}),
        {}, agent_id=1, workspace_id=ws,
    )
    meta_tool = await exec_composio.execute_composio_execute(
        executor, "composio_execute", {"action": BILLING.lower(), "params": {}}, agent_id=1, workspace_id=ws,
    )

    for result in (per_action, meta_tool):
        assert result["success"] is False
        _assert_blocked(result["error"])
    sdk.tools.execute.assert_not_called()


def test_the_tool_router_and_tool_service_paths_refuse(settings_db, monkeypatch):
    from modules.tools.composio_tool_router import ComposioToolRouter
    import modules.tools.services.composio_tool_service as tool_service_mod

    sdk = _sdk()
    client = _client(sdk)
    monkeypatch.setattr(client_mod, "get_composio_client", lambda: client)
    monkeypatch.setattr(tool_service_mod, "get_composio_client", lambda: client)

    routed = ComposioToolRouter(db_session=MagicMock(), workspace_id=uuid.uuid4(), assigned_apps=[]).execute_tool(
        BILLING, {}, entity_id="entity-1",
    )
    serviced = tool_service_mod.ComposioToolService(MagicMock()).execute_action(BILLING, {}, "entity-1")

    for result in (routed, serviced):
        assert result["success"] is False
        _assert_blocked(result["error"])
    sdk.tools.execute.assert_not_called()


class _Playbook:
    """The boundaries of one Playbook LLM step (api/recipe_executor._execute_step)."""

    def __init__(self, monkeypatch, tool_name, tool_args):
        import core.composio.client as composio_client
        import core.composio.linkedin_image_workaround as lw
        import core.composio.tool_executor as tool_executor
        import modules.agents.factory.agent_factory as agent_factory
        import modules.context as context_mod
        import modules.tools.services.composio_tool_service as tool_service
        import modules.tools.tool_router as tool_router
        import services.cli_ticket_lane as cli_lane

        self.responses = [
            SimpleNamespace(
                tool_calls=[{"id": "tc-1", "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}],
                content="", usage=None,
            ),
            SimpleNamespace(tool_calls=None, content="done", usage=None),
        ]
        responses = self.responses

        class _LLM:
            async def generate_response(self, messages, tools):
                return responses.pop(0)

        class _Factory:
            def __init__(self, db_session):
                pass

            async def activate_agent(self, agent_id):
                return SimpleNamespace(llm_manager=_LLM())

        class _Context:
            def __init__(self, db):
                pass

            async def build_context(self, **kwargs):
                return SimpleNamespace(system_prompt="system", tools=[])

        class _ToolService:
            def __init__(self, db):
                pass

            def get_tools_for_step(self, **kwargs):
                return SimpleNamespace(
                    tools=[{"type": "function", "function": {"name": tool_name}}],
                    app_names=[tool_name.split("_", 1)[0]],
                    action_set={tool_name},
                    entity_id="entity-1",
                    strategy="sdk_search",
                    search_ms=1,
                )

        self.spine = MagicMock(name="tool_router")
        self.spine.execute_and_format = AsyncMock(name="execute_and_format")
        self.resolve_uploads = AsyncMock(name="resolve_file_uploads")
        self.get_client = MagicMock(name="get_composio_client")
        self.linkedin = AsyncMock(name="execute_linkedin_image_post")

        monkeypatch.setattr(cli_lane, "is_cli_agent", lambda db, agent_id: False)
        monkeypatch.setattr(agent_factory, "AgentFactory", _Factory)
        monkeypatch.setattr(context_mod, "ContextService", _Context)
        monkeypatch.setattr(tool_service, "ComposioToolService", _ToolService)
        monkeypatch.setattr(tool_router, "get_tool_router", lambda: self.spine)
        monkeypatch.setattr(tool_executor, "resolve_file_uploads", self.resolve_uploads)
        monkeypatch.setattr(composio_client, "get_composio_client", self.get_client)
        monkeypatch.setattr(lw, "execute_linkedin_image_post", self.linkedin)

    async def run(self):
        from api import recipe_executor

        return await recipe_executor._execute_step(
            db=MagicMock(name="db"),
            agent=SimpleNamespace(id=7, name="Poster"),
            clean_prompt="Post the launch",
            workspace_id=uuid.uuid4(),
            max_iterations=3,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name, tool_args, denied", [
    (BILLING, {"amount": 100}, None),
    (LINKEDIN_POST, {"text": "hi", "images": ["/workspace/a.png"]}, [LINKEDIN_POST]),
])
async def test_a_playbook_step_refuses_before_dedup_linkedin_uploads_or_the_spine(
    settings_db, monkeypatch, tool_name, tool_args, denied,
):
    if denied:
        _set_denied(settings_db, denied)
    playbook = _Playbook(monkeypatch, tool_name, tool_args)

    result = await playbook.run()

    assert result["status"] == "success"
    (call,) = result["execution"]["tool_calls"]
    assert call["action"] == tool_name
    assert call["result"].startswith(f"Error executing {tool_name}: {BLOCKED}")
    tool_messages = [m for m in result["execution"]["messages"] if m.get("role") == "tool"]
    assert BLOCKED in tool_messages[0]["content"]
    playbook.spine.execute_and_format.assert_not_called()
    playbook.resolve_uploads.assert_not_called()
    playbook.linkedin.assert_not_called()
    playbook.get_client.assert_not_called()


@pytest.mark.asyncio
async def test_the_linkedin_smoke_route_refuses_before_the_credential_or_linkedin(settings_db, monkeypatch):
    import api.composio as composio_api
    import core.composio.linkedin_image_workaround as lw

    _set_denied(settings_db, [LINKEDIN_POST])
    load = MagicMock(name="_load_linkedin_credentials")
    token = AsyncMock(name="_get_access_token")
    monkeypatch.setattr(lw, "_load_linkedin_credentials", load)
    monkeypatch.setattr(lw, "_get_access_token", token)

    with pytest.raises(HTTPException) as refused:
        await composio_api.test_linkedin_upload_init(ctx=SimpleNamespace(workspace_id=uuid.uuid4()))

    assert refused.value.status_code == 403
    _assert_blocked(refused.value.detail, LINKEDIN_POST)
    load.assert_not_called()
    token.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("sync", ["products", "orders"])
async def test_the_shopify_bulk_syncs_refuse_before_the_workspace_entity_or_bulk_query(settings_db, monkeypatch, sync):
    import api.shopify as shopify_api

    _set_denied(settings_db, [SHOPIFY_BULK])
    get_client = MagicMock(name="get_composio_client")
    monkeypatch.setattr(client_mod, "get_composio_client", get_client)
    db = MagicMock(name="db")

    with pytest.raises(HTTPException) as refused:
        if sync == "products":
            await shopify_api._product_sync_impl("ws-1", db)
        else:
            await shopify_api._orders_sync_impl("ws-1", 90, 2, db)

    assert refused.value.status_code == 403
    _assert_blocked(refused.value.detail, SHOPIFY_BULK)
    db.query.assert_not_called()
    get_client.assert_not_called()


@pytest.mark.parametrize("denied, probed", [([SHOPIFY_PROBE], False), (D16, True)])
def test_the_shopify_credential_probe_is_skipped_when_denied(settings_db, monkeypatch, denied, probed):
    import core.credentials.integration_bridges.shopify as bridge
    from core.credentials.integration_bridges.base import BridgeContext

    _set_denied(settings_db, denied)
    sdk = _sdk()
    sdk.connected_accounts.list.return_value = SimpleNamespace(items=[])
    sdk.connected_accounts.initiate.return_value = SimpleNamespace(id="conn-1", status="ACTIVE")
    monkeypatch.setattr(bridge, "get_composio_client", lambda: SimpleNamespace(composio=sdk))
    monkeypatch.setattr(bridge, "_resolve_entity_id", lambda workspace_id: "entity-1")
    monkeypatch.setattr(bridge, "_persist_connection", MagicMock(name="_persist_connection"))

    result = bridge.shopify_access_token(BridgeContext(
        workspace_id=uuid.uuid4(), credential_id=1, credential_type_name="shopifyAccessTokenApi",
        decrypted_data={"shopSubdomain": "acme", "accessToken": "shpat_test", "appSecretKey": "s"},
    ))

    assert result.status == "connected"
    if probed:
        sdk.tools.execute.assert_called_once()
        assert sdk.tools.execute.call_args.args[0] == SHOPIFY_PROBE
    else:
        sdk.tools.execute.assert_not_called()


@pytest.mark.asyncio
async def test_the_cloud_file_rest_download_refuses_before_any_request(settings_db, monkeypatch):
    import modules.rag.services.cloud_file_downloader as downloader_mod

    http = MagicMock(name="httpx.AsyncClient")
    monkeypatch.setattr(downloader_mod.httpx, "AsyncClient", http)
    downloader = downloader_mod.CloudFileDownloader(MagicMock(name="db"))
    downloader._get_api_key = MagicMock(name="_get_api_key")
    downloader._get_entity_id = MagicMock(name="_get_entity_id")

    with pytest.raises(RuntimeError) as refused:
        await downloader._execute_via_rest_api(BILLING, "higgsfield_mcp", {}, uuid.uuid4())

    _assert_blocked(str(refused.value))
    http.assert_not_called()
    downloader._get_api_key.assert_not_called()


@pytest.mark.asyncio
async def test_the_v2_api_service_refuses_before_the_http_call(settings_db):
    from services.composio_api_service import ComposioAPIService, ComposioConfig

    service = ComposioAPIService(ComposioConfig(api_key="k"))
    service._client = MagicMock(name="httpx")
    service._client.post = AsyncMock(name="post")

    result = await service.execute_action("entity-1", BILLING, {})

    assert result["success"] is False and result["error_type"] == "action_denied"
    _assert_blocked(result["error"])
    service._client.post.assert_not_called()


# ---------------------------------------------------------------------------
# The inventory: every execution site found in the source calls the ONE helper
# ---------------------------------------------------------------------------

HELPER = "composio_action_denial"
_REST_EXECUTE = re.compile(r"/tools/execute/|/actions/\{[^}]*\}/execute")
# Every detection below needs one of these tokens in the file's text, so a file
# without any of them cannot hold a site — only candidates are parsed (a full
# parse of all ~900 files overran CI's 60 s per-test timeout).
_CANDIDATE = re.compile(r"tools\s*\.\s*execute|execute_action|execute_linkedin_image_post|_initialize_image_upload|/execute")
_LINKEDIN_DIRECT = {"execute_linkedin_image_post", "_initialize_image_upload"}
_LINKEDIN_MODULE = "core/composio/linkedin_image_workaround.py"

# Sites that execute (SDK tools.execute, a Composio REST URL, the LinkedIn direct
# API) and so call the helper themselves, before executing.
DIRECT = {
    ("core/composio/client.py", "ComposioClient.execute_action"),
    ("core/composio/tool_executor.py", "ComposioToolExecutor.execute"),
    ("api/recipe_executor.py", "_execute_step"),
    ("api/composio.py", "test_linkedin_upload_init"),
    ("api/shopify.py", "_product_sync_impl"),
    ("api/shopify.py", "_orders_sync_impl"),
    ("core/credentials/integration_bridges/shopify.py", "shopify_access_token"),
    ("modules/rag/services/cloud_file_downloader.py", "CloudFileDownloader._execute_via_rest_api"),
    ("services/composio_api_service.py", "ComposioAPIService.execute_action"),
}
# Sites that run an action only through ComposioClient.execute_action (DIRECT).
VIA_CLIENT = {
    ("api/workspace_github.py", "list_github_repos"),
    ("modules/tools/composio_tool_router.py", "ComposioToolRouter.execute_tool"),
    ("modules/tools/services/composio_tool_service.py", "ComposioToolService.execute_action"),
    ("modules/rag/services/cloud_file_downloader.py", "CloudFileDownloader._download_via_sdk"),
}


@functools.lru_cache(maxsize=1)
def _source_files():
    """Every orchestrator source file (no tests, no vendored or hidden trees)."""
    files = []
    for path in sorted(_ORCH.rglob("*.py")):
        parts = path.relative_to(_ORCH).parts
        if parts[0] == "tests" or "tests" in parts[:-1]:
            continue
        if any(part.startswith(".") or part in ("node_modules", "site-packages") for part in parts):
            continue
        files.append(path)
    return tuple(files)


def _fstring_text(node: ast.JoinedStr) -> str:
    """An f-string's literal parts with each placeholder as ``{}`` — no
    ast.get_source_segment, which re-splits the whole file on every call."""
    return "".join(
        part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
        for part in node.values
    )


@functools.lru_cache(maxsize=1)
def _execution_sites():
    """{(file, qualified function): [(kind, line)]} for every Composio execution."""
    sites = {}
    for path in _source_files():
        rel = path.relative_to(_ORCH).as_posix()
        if rel.startswith("alembic/"):
            continue
        source = path.read_text(encoding="utf-8")
        if not _CANDIDATE.search(source):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)

        def visit(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    visit(child, scope + [child.name])
                    continue
                kind = None
                if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == "execute" and isinstance(child.func.value, ast.Attribute)
                        and child.func.value.attr == "tools"):
                    kind = "sdk"
                elif isinstance(child, ast.Attribute) and child.attr == "execute_action":
                    kind = "wrapper"
                elif isinstance(child, ast.Call) and rel != _LINKEDIN_MODULE:
                    func = child.func
                    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                    if name in _LINKEDIN_DIRECT:
                        kind = "linkedin"
                elif isinstance(child, ast.JoinedStr) and _REST_EXECUTE.search(_fstring_text(child)):
                    kind = "rest"
                if kind:
                    sites.setdefault((rel, ".".join(scope) or "<module>"), []).append((kind, child.lineno))
                visit(child, scope)

        visit(tree, [])
    return sites


def _function(rel, qualname):
    source = (_ORCH / rel).read_text(encoding="utf-8")
    node = ast.parse(source)
    for part in qualname.split("."):
        node = next(
            child for child in ast.iter_child_nodes(node)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and child.name == part
        )
    return node


def test_every_execution_site_is_classified():
    """A new Composio execution site fails here until it calls the helper (or the checked client)."""
    assert set(_execution_sites()) == DIRECT | VIA_CLIENT


@pytest.mark.parametrize("site", sorted(DIRECT))
def test_each_direct_site_calls_the_one_helper_before_it_executes(site):
    executions = _execution_sites()[site]
    checks = [
        node.lineno for node in ast.walk(_function(*site))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == HELPER
    ]
    assert checks, f"{site} never calls {HELPER}"
    first_execution = min(line for _kind, line in executions)
    assert min(checks) < first_execution, (site, checks, executions)


@pytest.mark.parametrize("site", sorted(VIA_CLIENT))
def test_each_via_client_site_goes_through_the_checked_client(site):
    assert {kind for kind, _line in _execution_sites()[site]} == {"wrapper"}


def test_the_check_is_one_helper_not_a_copy():
    holders = [
        path.relative_to(_ORCH).as_posix() for path in _source_files()
        if "This action is blocked in Automatos" in path.read_text(encoding="utf-8")
    ]
    assert holders == ["core/composio/deny_list.py"]
    readers = [
        path.relative_to(_ORCH).as_posix() for path in _source_files()
        if "denied_actions" in path.read_text(encoding="utf-8")
    ]
    assert sorted(readers) == ["alembic/versions/prd251_socials.py", "core/composio/deny_list.py"]
