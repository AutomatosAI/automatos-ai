"""PRD-233 S2 — tool-routing degrade seam: honest without Composio.

One predicate (``core.composio.client.composio_available``: key configured via
config AND SDK importable), ONE discovery exclusion point
(``tool_router._offerable_candidates``), one explicit refusal shape
(``error_code == "integrations_unavailable"``) at the execution entries, and a
boot bootstrap that syncs the catalogue on the first keyed boot and re-binds
the seeded marketplace agents.

Locked here:
* key configured ⇒ Composio tools offered + executed exactly as today;
* key absent ⇒ (a) excluded from discovery, (b) native tools still route,
  (c) a direct Composio call returns ``integrations_unavailable`` — never a
  silent success, never a fail-open re-route, (d) the F018 destructive gate is
  untouched;
* bootstrap: empty cache + key ⇒ sync scheduled; populated ⇒ no sync; no key
  ⇒ nothing; the rebind is idempotent and never touches user-created items.

Everything but the last class is pure (no DB, no network). The real-DB rebind
test follows the test.yml Postgres pattern and skips when none is reachable.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import json  # noqa: E402
import logging  # noqa: E402
import uuid  # noqa: E402
from typing import Any, Dict, List  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

import core.composio.bootstrap as bs  # noqa: E402
import core.composio.client as cc  # noqa: E402
import modules.tools.tool_router as tr  # noqa: E402
from config import config  # noqa: E402
from modules.tools.registry.tool_registry import ToolCategory, ToolSpec  # noqa: E402

DESTRUCTIVE_INTENT = "delete every message in #general"
BENIGN_INTENT = "send hello to #general"


# ---------------------------------------------------------------------------
# Fixtures: flip the ONE predicate
# ---------------------------------------------------------------------------


class _FakeSDK:
    """Stands in for ``composio.Composio`` — the import succeeded."""


@pytest.fixture
def composio_on(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", "test-key-not-a-secret")
    monkeypatch.setattr(cc, "_get_composio", lambda: _FakeSDK)
    cc.reset_composio_availability()
    yield
    cc.reset_composio_availability()


@pytest.fixture
def composio_off(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", None)
    cc.reset_composio_availability()
    yield
    cc.reset_composio_availability()


# ---------------------------------------------------------------------------
# 1. The predicate
# ---------------------------------------------------------------------------


def test_available_with_key_and_sdk(composio_on):
    assert cc.composio_available() is True
    assert cc.composio_unavailable_reason() is None


def test_unavailable_without_key(composio_off):
    assert cc.composio_available() is False
    assert cc.composio_unavailable_reason() == cc.COMPOSIO_UNAVAILABLE_NO_KEY
    assert "COMPOSIO_API_KEY" in cc.composio_unavailable_reason()


def test_unavailable_when_sdk_missing(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", "test-key-not-a-secret")

    def _no_sdk():
        raise ImportError("composio not installed")

    monkeypatch.setattr(cc, "_get_composio", _no_sdk)
    cc.reset_composio_availability()
    try:
        assert cc.composio_available() is False
        assert cc.composio_unavailable_reason() == cc.COMPOSIO_UNAVAILABLE_NO_SDK
        # The honest reason must survive tool_router's fatal-dependency
        # classifier (it keys on "composio-openai").
        assert tr._is_fatal_dependency_error(cc.COMPOSIO_UNAVAILABLE_NO_SDK) is False
    finally:
        cc.reset_composio_availability()


def test_probe_is_cached_per_process_and_resettable(monkeypatch):
    monkeypatch.setattr(config, "COMPOSIO_API_KEY", None)
    cc.reset_composio_availability()
    try:
        assert cc.composio_available() is False
        # Flipping config WITHOUT a reset keeps the cached verdict.
        monkeypatch.setattr(config, "COMPOSIO_API_KEY", "now-set")
        monkeypatch.setattr(cc, "_get_composio", lambda: _FakeSDK)
        assert cc.composio_available() is False
        cc.reset_composio_availability()
        assert cc.composio_available() is True
    finally:
        cc.reset_composio_availability()


# ---------------------------------------------------------------------------
# 2. Discovery — the ONE exclusion point
# ---------------------------------------------------------------------------


def _spec(name: str, integration: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        category=ToolCategory.API_TOOLS,
        description=f"{name} description",
        executor_class="Fake",
        executor_method="run",
        metadata={"integration_type": integration} if integration else {},
    )


NATIVE = "search_knowledge"
COMPOSIO_TOOLS = ("composio_execute", "composio_SLACK_SEND_MESSAGE", "sdk_named_composio_tool")
_CANDIDATES = [
    _spec(NATIVE),
    _spec("composio_execute", "composio"),
    _spec("composio_SLACK_SEND_MESSAGE", "composio"),
    # metadata-only marker (no composio_ prefix) — the seam must catch it too
    _spec("sdk_named_composio_tool", "composio"),
]


def _dispatcher_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "platform_execute",
            "description": "dispatcher",
            "parameters": {"type": "object", "properties": {"action": {"enum": ["platform_list_agents"]}}},
        },
    }


def _discover() -> List[str]:
    fake_tool_registry = MagicMock()
    fake_tool_registry.get_all_tools.return_value = list(_CANDIDATES)
    action_registry = MagicMock()
    action_registry.to_dispatcher_schema.return_value = _dispatcher_schema()
    action_registry.get_all.return_value = []
    action_registry.to_first_class_schemas.return_value = []
    with patch.object(tr, "registry_get_tool_registry", return_value=fake_tool_registry), \
            patch.object(tr, "SessionLocal", return_value=MagicMock()), \
            patch("modules.tools.discovery.get_action_registry", return_value=action_registry):
        tools = tr.get_tools_for_agent(agent_id=None, workspace_id=None)
    return [t["function"]["name"] for t in tools]


def test_key_configured_offers_composio_tools_as_today(composio_on):
    names = _discover()
    for name in COMPOSIO_TOOLS:
        assert name in names, f"{name} must be offered when a key is configured"
    assert NATIVE in names
    assert "platform_execute" in names


def test_key_absent_excludes_every_composio_tool_from_discovery(composio_off):
    names = _discover()
    for name in COMPOSIO_TOOLS:
        assert name not in names, f"{name} must NOT be offered without a key"
    # (b) native platform tools + the dispatcher are untouched
    assert NATIVE in names
    assert "platform_execute" in names


def test_exclusion_is_a_passthrough_when_available(composio_on):
    candidates = list(_CANDIDATES)
    assert tr._offerable_candidates(candidates, "trace") is candidates


def test_exclusion_logs_once_at_info_with_the_reason(composio_off, monkeypatch, caplog):
    monkeypatch.setattr(tr, "_composio_exclusion_logged", False)
    with caplog.at_level(logging.DEBUG, logger=tr.logger.name):
        tr._offerable_candidates(list(_CANDIDATES), "t1")
        tr._offerable_candidates(list(_CANDIDATES), "t2")
    info = [r for r in caplog.records if r.levelno == logging.INFO and "not offered" in r.getMessage()]
    assert len(info) == 1
    assert cc.COMPOSIO_UNAVAILABLE_NO_KEY in info[0].getMessage()
    assert "3 Composio tool(s)" in info[0].getMessage()


@pytest.mark.asyncio
async def test_filtered_composio_actions_empty_with_reason(composio_off):
    result = await tr.get_filtered_composio_actions("send a message", ["SLACK"], db_session=MagicMock())
    assert result["success"] is False
    assert result["actions"] == []
    assert result["error_code"] == tr.INTEGRATIONS_UNAVAILABLE
    assert result["error"] == cc.COMPOSIO_UNAVAILABLE_NO_KEY


@pytest.mark.asyncio
async def test_filtered_composio_actions_unchanged_when_available(composio_on, monkeypatch):
    # Today's first branch (filter module absent) — byte-for-byte the old shape.
    monkeypatch.setattr(tr, "CAPABILITY_FILTER_AVAILABLE", False)
    result = await tr.get_filtered_composio_actions("send a message", ["SLACK"], db_session=MagicMock())
    assert result == {
        "success": False,
        "error": "Capability filter not available",
        "actions": [],
        "capabilities": [],
    }


# ---------------------------------------------------------------------------
# 3. Execution seam
# ---------------------------------------------------------------------------


class _FakeExecutor:
    def __init__(self) -> None:
        self.calls: List[str] = []

    async def execute_tool(self, tool_name, tool_args, agent_id, **kwargs):
        self.calls.append(tool_name)
        return {"success": True, "data": {"ok": True}, "tool": tool_name}


def _executor_patches(executor: _FakeExecutor):
    session = MagicMock()
    return (
        patch.object(tr, "_get_executor_for_request", return_value=executor),
        # execute_tool imports SessionLocal locally from the database module.
        patch("core.database.database.SessionLocal", return_value=session),
    )


@pytest.mark.asyncio
async def test_native_tools_still_route_without_composio(composio_off):
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool(NATIVE, {"query": "x"}, agent_id=1, workspace_id=uuid.uuid4())
    assert result["success"] is True
    assert executor.calls == [NATIVE]


@pytest.mark.asyncio
async def test_direct_composio_call_refused_explicitly_without_composio(composio_off):
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool(
            "composio_execute",
            {"action": "SLACK_SEND_MESSAGE", "params": {"channel": "#general", "text": "hi"}},
            agent_id=1,
            workspace_id=uuid.uuid4(),
        )
    assert result["success"] is False
    assert result["error_code"] == tr.INTEGRATIONS_UNAVAILABLE
    assert result["error_type"] == tr.INTEGRATIONS_UNAVAILABLE
    assert cc.COMPOSIO_UNAVAILABLE_NO_KEY in result["error"]
    assert "COMPOSIO_API_KEY" in result["error"]
    # never a fail-open pass-through: nothing reached the executor
    assert executor.calls == []


@pytest.mark.asyncio
async def test_per_action_composio_tool_refused_too(composio_off):
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool("composio_SLACK_SEND_MESSAGE", {"channel": "#g"}, agent_id=1)
    assert result["error_code"] == tr.INTEGRATIONS_UNAVAILABLE
    assert executor.calls == []


@pytest.mark.asyncio
async def test_composio_call_passes_through_when_available(composio_on):
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool("composio_execute", {"action": "SLACK_SEND_MESSAGE"}, agent_id=1)
    assert result["success"] is True
    assert executor.calls == ["composio_execute"]
    assert "error_code" not in result


@pytest.mark.asyncio
async def test_router_envelope_carries_integrations_unavailable(composio_off):
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        envelope = await tr.ToolRouter().execute_and_format(
            "composio_execute",
            {"action": "SLACK_DELETE_MESSAGE", "params": {"ts": "1"}},
            agent_id=1,
            workspace_id=uuid.uuid4(),
            original_intent=DESTRUCTIVE_INTENT,  # would hit F018 if it ever got that far
        )
    assert envelope["success"] is False
    assert envelope["error_type"] == tr.INTEGRATIONS_UNAVAILABLE
    assert envelope["raw_result"]["error_code"] == tr.INTEGRATIONS_UNAVAILABLE
    assert envelope["fatal_error"] is False  # honest message, not the generic "restart"
    assert "Integrations are disabled" in envelope["llm_context"]
    assert executor.calls == []


# ---------------------------------------------------------------------------
# 4. F018 fail-closed gate — unchanged in BOTH availability states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["composio_on", "composio_off"])
def test_f018_gate_unchanged(state, request, monkeypatch):
    request.getfixturevalue(state)
    monkeypatch.setattr(tr, "CAPABILITY_FILTER_AVAILABLE", False)
    monkeypatch.setattr(config, "COMPOSIO_DESTRUCTIVE_FAIL_CLOSED", True)

    eligible, reason = tr.validate_action_for_intent("SLACK_DELETE_MESSAGE", DESTRUCTIVE_INTENT)
    assert eligible is False
    assert "confirmation required" in reason

    eligible, reason = tr.validate_action_for_intent("SLACK_SEND_MESSAGE", BENIGN_INTENT)
    assert eligible is True

    # explicit authorisation still opens the gate, exactly as before
    eligible, _ = tr.validate_action_for_intent("SLACK_DELETE_MESSAGE", DESTRUCTIVE_INTENT, allow_destructive=True)
    assert eligible is True


@pytest.mark.asyncio
async def test_validated_execution_blocked_by_f018_when_available(composio_on, monkeypatch):
    monkeypatch.setattr(tr, "CAPABILITY_FILTER_AVAILABLE", False)
    monkeypatch.setattr(config, "COMPOSIO_DESTRUCTIVE_FAIL_CLOSED", True)
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool_with_validation(
            "composio_execute", {"action": "SLACK_DELETE_MESSAGE"}, DESTRUCTIVE_INTENT, agent_id=1
        )
    assert result["error_type"] == "validation_blocked"
    assert result["blocked_by"] == "capability_validation"
    assert executor.calls == []


@pytest.mark.asyncio
async def test_validated_execution_refused_before_f018_when_unavailable(composio_off, monkeypatch):
    monkeypatch.setattr(tr, "CAPABILITY_FILTER_AVAILABLE", False)
    executor = _FakeExecutor()
    p1, p2 = _executor_patches(executor)
    with p1, p2:
        result = await tr.execute_tool_with_validation(
            "composio_execute", {"action": "SLACK_DELETE_MESSAGE"}, DESTRUCTIVE_INTENT, agent_id=1
        )
    # honest: "no integrations" — not "confirm this destructive action"
    assert result["error_code"] == tr.INTEGRATIONS_UNAVAILABLE
    assert executor.calls == []


# ---------------------------------------------------------------------------
# 4b. GET /api/tools/integrations/status — the UI reads the SAME predicate
# ---------------------------------------------------------------------------


def _status_session(apps_cached: int, last_sync: str | None, job_status: str | None) -> MagicMock:
    session = MagicMock()
    session.query.return_value.scalar.return_value = apps_cached
    stats_row = MagicMock(stat_value={"timestamp": last_sync}) if last_sync else None
    session.query.return_value.filter.return_value.first.return_value = stats_row
    job = MagicMock(status=job_status) if job_status else None
    session.query.return_value.order_by.return_value.first.return_value = job
    return session


@pytest.mark.asyncio
async def test_status_endpoint_reports_disabled_state(composio_off):
    from api.tools import integrations_status

    out = await integrations_status(ctx=MagicMock(), db=_status_session(0, None, None))
    assert out.available is False
    assert out.reason == cc.COMPOSIO_UNAVAILABLE_NO_KEY
    assert out.key_configured is False
    assert out.apps_cached == 0
    assert out.last_sync is None
    assert out.sync_status is None


@pytest.mark.asyncio
async def test_status_endpoint_reports_available_state(composio_on):
    from api.tools import integrations_status

    out = await integrations_status(
        ctx=MagicMock(), db=_status_session(880, "2026-08-29T00:00:00Z", "completed")
    )
    assert out.available is True and out.reason is None
    assert out.key_configured is True
    assert out.apps_cached == 880
    assert out.last_sync == "2026-08-29T00:00:00Z"
    assert out.sync_status == "completed"


# ---------------------------------------------------------------------------
# 5. Boot bootstrap (pure — session + service mocked)
# ---------------------------------------------------------------------------


def _boot_session(apps_cached: int) -> MagicMock:
    session = MagicMock()
    session.query.return_value.scalar.return_value = apps_cached  # catalog_app_count
    session.query.return_value.filter.return_value.first.return_value = None  # no sync in flight
    return session


class _Runner:
    def __init__(self) -> None:
        self.scheduled: List[Any] = []

    def __call__(self, target) -> None:
        self.scheduled.append(target)


def test_bootstrap_schedules_sync_when_key_and_empty_cache(composio_on, monkeypatch):
    session = _boot_session(0)
    runner = _Runner()
    rebind = MagicMock(return_value=0)
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr(bs, "rebind_seeded_agents", rebind)
    result = bs.ensure_catalog_on_boot(start_background=runner)
    assert result["status"] == "scheduled"
    assert runner.scheduled == [bs.run_catalog_sync_and_rebind]
    rebind.assert_not_called()  # the rebind runs AFTER the sync, on the thread
    session.close.assert_called_once()


def test_bootstrap_no_sync_when_cache_populated(composio_on, monkeypatch):
    session = _boot_session(880)
    runner = _Runner()
    rebind = MagicMock(return_value=0)
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr(bs, "rebind_seeded_agents", rebind)
    result = bs.ensure_catalog_on_boot(start_background=runner)
    assert result == {"status": "ready", "apps_cached": 880, "rebound": 0}
    assert runner.scheduled == []
    rebind.assert_called_once_with(session)


def test_bootstrap_touches_nothing_without_key(composio_off, monkeypatch):
    factory = MagicMock()
    runner = _Runner()
    monkeypatch.setattr(bs, "SessionLocal", factory)
    result = bs.ensure_catalog_on_boot(start_background=runner)
    assert result == {"status": "unavailable", "reason": cc.COMPOSIO_UNAVAILABLE_NO_KEY}
    factory.assert_not_called()
    assert runner.scheduled == []


def test_bootstrap_probe_failure_never_raises(composio_on, monkeypatch):
    session = MagicMock()
    session.query.side_effect = RuntimeError("relation does not exist")
    recorded = MagicMock()
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr(bs, "record_error", recorded)
    result = bs.ensure_catalog_on_boot(start_background=_Runner())
    assert result["status"] == "error"
    recorded.assert_called_once()
    session.close.assert_called_once()


def test_background_sync_runs_service_then_rebind(monkeypatch):
    session = MagicMock()
    service = MagicMock()
    service.run_full_sync.return_value = {"apps_synced": 5, "actions_synced": 50, "errors_count": 0}
    service_cls = MagicMock(return_value=service)
    rebind = MagicMock(return_value=31)
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr(bs, "rebind_seeded_agents", rebind)
    monkeypatch.setattr("services.metadata_sync_service.MetadataSyncService", service_cls)
    result = bs.run_catalog_sync_and_rebind()
    service_cls.assert_called_once_with(session)
    service.run_full_sync.assert_called_once_with()
    rebind.assert_called_once_with(session)
    assert result["status"] == "completed" and result["rebound"] == 31
    session.close.assert_called_once()


def test_background_sync_failure_is_recorded_not_raised(monkeypatch):
    session = MagicMock()
    service = MagicMock()
    service.run_full_sync.side_effect = RuntimeError("composio 401")
    recorded = MagicMock()
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr("services.metadata_sync_service.MetadataSyncService", MagicMock(return_value=service))
    monkeypatch.setattr(bs, "record_error", recorded)
    result = bs.run_catalog_sync_and_rebind()
    assert result["status"] == "failed" and "composio 401" in result["error"]
    recorded.assert_called_once()
    session.rollback.assert_called_once()
    session.close.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Rebind — pure resolution
# ---------------------------------------------------------------------------


_CATALOG = {"SLACK": (11, "https://logo/slack.png"), "GMAIL": (12, None)}


def test_rebound_metadata_resolves_intended_names():
    before = {"tools": [], "tool_names": ["SLACK", "GMAIL", "NOTINCACHE"], "tool_icons": [], "skills": ["x"]}
    after = bs.rebound_metadata(before, _CATALOG)
    assert after == {"tools": [11, 12], "tool_names": ["SLACK", "GMAIL", "NOTINCACHE"],
                     "tool_icons": ["https://logo/slack.png", ""], "skills": ["x"]}
    # input never mutated
    assert before["tools"] == [] and before["tool_icons"] == []


def test_rebound_metadata_is_none_when_already_bound():
    bound = {"tools": [11, 12], "tool_names": ["slack", "gmail"], "tool_icons": ["https://logo/slack.png", ""]}
    assert bs.rebound_metadata(bound, _CATALOG) is None


def test_rebound_metadata_is_none_when_nothing_resolvable():
    unresolved = {"tools": [], "tool_names": ["NOTINCACHE"], "tool_icons": []}
    assert bs.rebound_metadata(unresolved, _CATALOG) is None


# ---------------------------------------------------------------------------
# 7. Rebind — real DB, idempotent, seeded rows only
# ---------------------------------------------------------------------------

PROBE_APP = "PRD233TESTAPP"
PROBE_LOGO = "https://logo/prd233.png"
PROBE_NAMES = [PROBE_APP, "PRD233MISSING"]


@pytest.fixture(scope="module")
def engine():
    """Real Postgres engine; skip the whole class cleanly when none is up."""
    from sqlalchemy import create_engine, text
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
            # marketplace_items has no ORM model (alembic-only), so a create_all
            # test DB may lack it — skip honestly rather than error at INSERT.
            present = c.execute(
                text("SELECT to_regclass('marketplace_items') IS NOT NULL "
                     "AND to_regclass('composio_apps_cache') IS NOT NULL")
            ).scalar()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"rebind suite needs a reachable Postgres: {exc}")
    if not present:
        pytest.skip("rebind suite needs marketplace_items + composio_apps_cache (alembic schema)")
    yield eng
    eng.dispose()


def _insert_item(session, creator: str, name: str) -> int:
    from sqlalchemy import text

    metadata = {"tools": [], "tool_names": PROBE_NAMES, "tool_icons": [], "agent_type": "custom"}
    return session.execute(
        text(
            "INSERT INTO marketplace_items (type, name, description, creator_name, category, tags, "
            "install_count, is_featured, is_approved, version, metadata, created_at, updated_at) "
            "VALUES ('agent', :name, 'prd233 probe', :creator, 'Test', CAST('[]' AS jsonb), "
            "0, false, false, '1.0.0', CAST(:metadata AS jsonb), NOW(), NOW()) RETURNING id"
        ),
        {"name": name, "creator": creator, "metadata": json.dumps(metadata)},
    ).scalar()


@pytest.fixture
def probe_rows(engine, new_session):
    """One throwaway cache app + one seeded item + one user item; all deleted on teardown."""
    from sqlalchemy import text

    s = new_session()
    s.execute(text("DELETE FROM composio_apps_cache WHERE app_name = :n"), {"n": PROBE_APP})
    s.execute(
        text(
            "INSERT INTO composio_apps_cache (app_name, app_slug, display_name, logo_url, categories, auth_schemes, "
            "action_count, trigger_count, status) VALUES (:n, :slug, :d, :logo, CAST('[]' AS jsonb), "
            "CAST('[]' AS jsonb), 0, 0, 'ACTIVE') RETURNING id"
        ),
        {"n": PROBE_APP, "slug": PROBE_APP.lower(), "d": "PRD233 Probe App", "logo": PROBE_LOGO},
    )
    seeded_id = _insert_item(s, bs.SEEDED_CREATOR_NAME, "PRD233 Rebind Probe (seeded)")
    user_id = _insert_item(s, "Some User", "PRD233 Rebind Probe (user)")
    s.commit()
    yield seeded_id, user_id
    t = new_session.sweep()
    t.execute(text("DELETE FROM marketplace_items WHERE id IN (:a, :b)"), {"a": seeded_id, "b": user_id})
    t.execute(text("DELETE FROM composio_apps_cache WHERE app_name = :n"), {"n": PROBE_APP})
    t.commit()
    t.close()


def _metadata(session, item_id: int) -> Dict[str, Any]:
    from sqlalchemy import text

    return session.execute(text("SELECT metadata FROM marketplace_items WHERE id = :id"), {"id": item_id}).scalar()


@pytest.mark.integration
def test_rebind_is_idempotent_and_touches_only_seeded_rows(probe_rows, new_session):
    from sqlalchemy import text

    seeded_id, user_id = probe_rows
    cache_id = None
    s = new_session()
    cache_id = s.execute(
        text("SELECT id FROM composio_apps_cache WHERE app_name = :n"), {"n": PROBE_APP}
    ).scalar()

    # first pass binds the seeded row (only the app that exists in the cache)
    assert bs.rebind_seeded_agents(s) == 1
    seeded = _metadata(s, seeded_id)
    assert seeded["tools"] == [cache_id]
    assert seeded["tool_icons"] == [PROBE_LOGO]
    assert seeded["tool_names"] == PROBE_NAMES  # intended names preserved
    assert seeded["agent_type"] == "custom"       # unrelated metadata untouched

    # the user-published item with the same shape is never touched
    user = _metadata(s, user_id)
    assert user["tools"] == [] and user["tool_icons"] == []

    # second pass: same rows, zero writes
    assert bs.rebind_seeded_agents(s) == 0
    assert _metadata(s, seeded_id) == seeded
    assert _metadata(s, user_id) == user
    s.close()
