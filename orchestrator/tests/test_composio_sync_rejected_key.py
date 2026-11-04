"""A rejected Composio key fails the catalogue sync, and says so (4 Nov 2026).

Seen on a self-hosted install: with an invalid ``COMPOSIO_KEY`` Composio answers
every call ``401 APIKey_InvalidAPIKey``, yet the sync job ended ``completed`` with
0 apps, 0 actions and 0 errors, the boot log said "finished: apps=0", and the Tools
page listed nothing and gave no reason, so the operator kept pressing Sync.

Locked here:
* the SDK's 401 on the toolkits list fails the job: ``failed``, one error, the key
  message in ``error_details`` and in the result the Sync button reads;
* the bulk tools fetch raises on 401/403 too; other callers of
  ``get_available_apps`` still get ``[]``;
* the boot sync logs and records the failure, never "finished";
* the status endpoint carries the reason for the Tools page, and only that one;
* a Sync while a sync runs returns that job instead of starting another.

Pure: the Composio SDK handle and the DB session are faked; no network, no DB.
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

import logging  # noqa: E402
from datetime import datetime  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402
from composio_client import AuthenticationError  # noqa: E402

import core.composio.bootstrap as bs  # noqa: E402
import core.composio.client as cc  # noqa: E402
from core.composio.key_rejection import ComposioAuthError  # noqa: E402
from core.models.composio_cache import ComposioSyncJob  # noqa: E402

KEY_MESSAGE = "Composio rejected the API key. Check COMPOSIO_KEY."


def _rejected_key_error() -> AuthenticationError:
    """What the Composio SDK (composio_client) raises when Composio answers 401."""
    body = {"error": {"message": "Invalid API key", "slug": "APIKey_InvalidAPIKey", "status": 401}}
    request = httpx.Request("GET", "https://backend.composio.dev/api/v3/toolkits")
    response = httpx.Response(401, request=request, json=body)
    return AuthenticationError(f"Error code: 401 - {body}", response=response, body=body)


def _client(toolkits_list_effect) -> cc.ComposioClient:
    """A ComposioClient whose SDK handle's ``toolkits.list`` raises ``toolkits_list_effect``."""
    client = cc.ComposioClient(api_key="test-key-not-a-secret")
    client._composio = SimpleNamespace(toolkits=SimpleNamespace(list=MagicMock(side_effect=toolkits_list_effect)))
    return client


def _sync_session(in_flight: ComposioSyncJob | None = None) -> MagicMock:
    """A session for the sync route: ``in_flight`` is what the in-flight query finds."""
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = in_flight
    return session


@pytest.fixture
def rejecting_client(monkeypatch) -> cc.ComposioClient:
    client = _client(_rejected_key_error())
    bulk = MagicMock(return_value=[])
    monkeypatch.setattr(client, "get_all_actions_bulk", bulk)
    monkeypatch.setattr("services.metadata_sync_service.get_composio_client", lambda: client)
    return client


# ---------------------------------------------------------------------------
# 1. The client
# ---------------------------------------------------------------------------


def test_a_401_on_the_toolkits_list_raises_composio_auth_error():
    with pytest.raises(ComposioAuthError) as excinfo:
        _client(_rejected_key_error()).fetch_available_apps()

    assert str(excinfo.value) == KEY_MESSAGE
    assert excinfo.value.status_code == 401
    assert isinstance(excinfo.value.__cause__, AuthenticationError)


def test_other_callers_still_get_an_empty_list():
    """``/api/composio/apps`` and the rest keep the never-raise contract."""
    assert _client(_rejected_key_error()).get_available_apps() == []


@pytest.mark.parametrize("status_code", [401, 403])
def test_the_bulk_tools_fetch_raises_on_a_rejected_key(monkeypatch, status_code):
    page = SimpleNamespace(status_code=status_code, text='{"error": "APIKey_InvalidAPIKey"}', json=lambda: {})
    monkeypatch.setattr("requests.get", MagicMock(return_value=page))

    with pytest.raises(ComposioAuthError):
        cc.ComposioClient(api_key="test-key-not-a-secret").get_all_actions_bulk()


def test_the_bulk_tools_fetch_keeps_stopping_quietly_on_other_errors(monkeypatch):
    page = SimpleNamespace(status_code=500, text="upstream error", json=lambda: {})
    monkeypatch.setattr("requests.get", MagicMock(return_value=page))

    assert cc.ComposioClient(api_key="test-key-not-a-secret").get_all_actions_bulk() == []


# ---------------------------------------------------------------------------
# 2. The sync job
# ---------------------------------------------------------------------------


def test_a_rejected_key_fails_the_sync_job_with_the_key_message(rejecting_client):
    from services.metadata_sync_service import MetadataSyncService

    db = MagicMock()
    result = MetadataSyncService(db).run_full_sync()

    job = db.add.call_args.args[0]
    assert isinstance(job, ComposioSyncJob)
    assert job.status == "failed"
    assert job.errors_count == 1
    assert job.error_details == {"errors": [KEY_MESSAGE]}
    assert job.apps_synced == 0 and job.actions_synced == 0
    assert result["status"] == "failed"
    assert result["errors_count"] == 1
    assert result["error"] == KEY_MESSAGE
    rejecting_client.get_all_actions_bulk.assert_not_called()
    db.rollback.assert_called_once()


def test_a_failed_toolkits_list_no_longer_completes_an_empty_sync(monkeypatch):
    """Any failure to list the apps fails the job; only the key message is shown to people."""
    from services.metadata_sync_service import MetadataSyncService

    client = _client(RuntimeError("connection reset by peer"))
    monkeypatch.setattr("services.metadata_sync_service.get_composio_client", lambda: client)
    db = MagicMock()

    result = MetadataSyncService(db).run_full_sync()

    job = db.add.call_args.args[0]
    assert job.status == "failed"
    assert job.error_details == {"errors": ["connection reset by peer"]}
    assert result["status"] == "failed"
    assert result["error"] is None


# ---------------------------------------------------------------------------
# 3. The boot sync
# ---------------------------------------------------------------------------


def test_boot_sync_logs_and_records_a_rejected_key_as_a_failure(rejecting_client, monkeypatch, caplog):
    session = MagicMock()
    recorded = MagicMock()
    monkeypatch.setattr(bs, "SessionLocal", lambda: session)
    monkeypatch.setattr(bs, "rebind_seeded_agents", MagicMock(return_value=0))
    monkeypatch.setattr(bs, "record_error", recorded)
    caplog.set_level(logging.INFO, logger=bs.logger.name)

    result = bs.run_catalog_sync_and_rebind()

    assert result["status"] == "failed"
    assert result["error"] == KEY_MESSAGE
    assert result["sync"]["status"] == "failed"
    recorded.assert_called_once()
    assert recorded.call_args.kwargs["subsystem"] == "startup"
    assert recorded.call_args.kwargs["operation"] == "composio_catalog_sync"
    assert str(recorded.call_args.kwargs["error"]) == KEY_MESSAGE
    boot_lines = [r.getMessage() for r in caplog.records if r.name == bs.logger.name]
    assert any("Composio catalogue sync failed" in line and KEY_MESSAGE in line for line in boot_lines)
    assert not any("finished" in line for line in boot_lines)
    session.close.assert_called_once()


def test_boot_sync_still_reports_a_completed_sync_as_finished(monkeypatch, caplog):
    service = MagicMock()
    service.run_full_sync.return_value = {"status": "completed", "apps_synced": 5, "actions_synced": 50,
                                          "errors_count": 0, "error": None}
    monkeypatch.setattr(bs, "SessionLocal", MagicMock)
    monkeypatch.setattr(bs, "rebind_seeded_agents", MagicMock(return_value=2))
    monkeypatch.setattr("services.metadata_sync_service.MetadataSyncService", MagicMock(return_value=service))
    recorded = MagicMock()
    monkeypatch.setattr(bs, "record_error", recorded)
    caplog.set_level(logging.INFO, logger=bs.logger.name)

    result = bs.run_catalog_sync_and_rebind()

    assert result["status"] == "completed" and result["rebound"] == 2
    recorded.assert_not_called()
    assert any("finished: apps=5" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. POST /api/tools/sync and GET /api/tools/integrations/status
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_route_returns_the_running_job_instead_of_starting_another(monkeypatch):
    from api.tools import sync

    running = ComposioSyncJob(id=41, job_type="full_sync", status="running", started_at=datetime(2026, 11, 4, 9, 0))
    monkeypatch.setattr("api.tools.composio_available", lambda: True)
    db = _sync_session(in_flight=running)

    with patch("api.tools.MetadataSyncService") as service:
        out = sync(sync_type="full", ctx=MagicMock(), db=db)

    service.assert_not_called()
    db.query.assert_called_once_with(ComposioSyncJob)
    assert out["already_running"] is True
    assert out["job_id"] == 41
    assert out["status"] == "running"
    assert out["started_at"] == "2026-11-04T09:00:00"


@pytest.mark.asyncio
async def test_sync_route_hands_the_button_the_key_message(rejecting_client, monkeypatch):
    from api.tools import sync

    monkeypatch.setattr("api.tools.composio_available", lambda: True)

    out = sync(sync_type="full", ctx=MagicMock(), db=_sync_session(in_flight=None))

    assert out["status"] == "failed"
    assert out["error"] == KEY_MESSAGE
    assert "already_running" not in out


def _status_session(job: ComposioSyncJob) -> MagicMock:
    session = MagicMock()
    session.query.return_value.scalar.return_value = 0  # empty catalogue
    session.query.return_value.filter.return_value.first.return_value = None  # never synced
    session.query.return_value.order_by.return_value.first.return_value = job
    return session


@pytest.mark.asyncio
async def test_status_endpoint_carries_the_rejected_key_reason(monkeypatch):
    from api.tools import integrations_status

    monkeypatch.setattr("api.tools.composio_available", lambda: True)
    monkeypatch.setattr("api.tools.composio_unavailable_reason", lambda: None)
    failed = ComposioSyncJob(status="failed", errors_count=1, error_details={"errors": [KEY_MESSAGE]})

    out = await integrations_status(ctx=MagicMock(), db=_status_session(failed))

    assert out.sync_status == "failed"
    assert out.sync_error == KEY_MESSAGE


@pytest.mark.asyncio
async def test_status_endpoint_never_shows_raw_errors(monkeypatch):
    from api.tools import integrations_status

    monkeypatch.setattr("api.tools.composio_available", lambda: True)
    monkeypatch.setattr("api.tools.composio_unavailable_reason", lambda: None)
    failed = ComposioSyncJob(status="failed", errors_count=1,
                             error_details={"errors": ["(psycopg2.OperationalError) server closed the connection"]})

    out = await integrations_status(ctx=MagicMock(), db=_status_session(failed))

    assert out.sync_status == "failed"
    assert out.sync_error is None
