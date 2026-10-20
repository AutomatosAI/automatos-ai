"""PRD-266 US-016b: a new content report's summary to the moderation inbox.

On SaaS with an https ``MODERATION_WEBHOOK_URL``, each NEW report is posted once,
after the response, through the existing generic webhook driver. A local install
posts nothing even when the setting is set (D15); an unset, http or malformed
value posts nothing and is never logged. The summary holds the report id, the
workspace id, the target, the reason, an excerpt and the note, each cut to 200
characters, and nothing of the reporter or the workspace's name. A failed post
leaves the 201 and is logged at ERROR without the URL or the driver's error text.

DB-free: the workspace role, the service and the driver (or httpx under the real
driver) are faked.
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

import api.content_reports as reports_api  # noqa: E402
import channels.drivers.webhook as webhook_mod  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
from channels.drivers import get_driver  # noqa: E402
from channels.drivers.base import SendResult  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from services.content_reports import moderation, service  # noqa: E402
from services.content_reports.targets import clip_excerpt  # noqa: E402

WS_ID = uuid.uuid4()
ROUTE = "/api/v1/reports"
MESSAGE_ID = str(uuid.uuid4())
BODY = {"target_type": "message", "target_id": MESSAGE_ID, "reason": "harmful", "note": "Not OK."}
# K19: an example.test host, never a Slack-shaped webhook URL.
INBOX_HOST = "moderation.example.test"
INBOX_URL = f"https://{INBOX_HOST}/inbox/T0KEN-in-the-path"
REPORTER_ID = "reporter-7731"
REPORTER_CLERK = "user_reporter_clerk_7731"
REPORTER_EMAIL = "reporter-7731@reports.example.test"
WORKSPACE_NAME = "Quokka Moderation Test Workspace"


class _NoRowsDB:
    """A session the route never reads: the service is faked."""

    def query(self, *_args, **_kwargs):
        raise AssertionError("the route reads nothing itself")


def _reporter_ctx():
    return RequestContext(
        workspace_id=WS_ID,
        user=UserContext(id=REPORTER_ID, clerk_user_id=REPORTER_CLERK, email=REPORTER_EMAIL,
                         system_role="user"),
        auth_type="clerk",
    )


def _filed(*, report_id=41, duplicate=False, excerpt="An Auto reply.", note="Not OK."):
    return service.FiledReport(
        report_id=report_id, duplicate=duplicate, workspace_id=str(WS_ID), target_type="message",
        target_id=MESSAGE_ID, reason="harmful", note=note, excerpt=excerpt,
    )


def _client():
    app = FastAPI()
    app.include_router(reports_api.router)
    app.dependency_overrides[get_request_context_hybrid] = _reporter_ctx
    app.dependency_overrides[get_db] = _NoRowsDB
    return TestClient(app)


def _line(text, label):
    """The value of the summary line ``label: …``."""
    (value,) = [line[len(label) + 2:] for line in text.splitlines() if line.startswith(f"{label}: ")]
    return value


def _no_secret_in(records):
    for record in records:
        logged = " ".join(filter(None, (record.getMessage(), record.exc_text, str(record.args))))
        assert INBOX_HOST not in logged and "T0KEN" not in logged, logged


@pytest.fixture(autouse=True)
def _warn_once_fresh():
    moderation._warn_unusable_setting.cache_clear()
    yield
    moderation._warn_unusable_setting.cache_clear()


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MODERATION_WEBHOOK_URL", INBOX_URL)


@pytest.fixture
def filed(monkeypatch):
    """The service, faked: answers ``state["result"]``; any role may report."""
    state = {"result": _filed()}

    async def _file_report(db, ctx, **kwargs):
        return state["result"]

    monkeypatch.setattr(reports_api, "file_report", _file_report)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "viewer")
    return state


@pytest.fixture
def driver(monkeypatch):
    """The webhook driver, faked: records each send; answers ``result`` or raises ``error``."""
    state = SimpleNamespace(platforms=[], sends=[], result=SendResult(ok=True, latency_ms=3), error=None)

    class _Driver:
        async def send(self, **kwargs):
            state.sends.append(kwargs)
            if state.error is not None:
                raise state.error
            return state.result

    def _get_driver(platform):
        state.platforms.append(platform)
        return _Driver

    monkeypatch.setattr(moderation, "get_driver", _get_driver)
    return state


# ---------------------------------------------------------------------------
# Who posts: SaaS with an https URL, nobody else
# ---------------------------------------------------------------------------


def test_saas_with_an_https_url_posts_exactly_once_with_the_summary(saas, filed, driver):
    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    assert driver.platforms == ["webhook"]
    assert driver.sends == [{
        "workspace_id": str(WS_ID), "config": {"webhook_url": INBOX_URL}, "target": None,
        "text": moderation.moderation_summary(_filed()),
    }]


def test_saas_with_the_setting_unset_posts_nothing(monkeypatch, filed, driver, caplog):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MODERATION_WEBHOOK_URL", "")
    caplog.set_level(logging.DEBUG)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    assert driver.platforms == [] and driver.sends == []
    assert not [r for r in caplog.records
                if r.name == moderation.logger.name and r.levelno >= logging.WARNING]


def test_a_local_install_posts_nothing_even_when_the_setting_is_set(monkeypatch, filed, driver):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "MODERATION_WEBHOOK_URL", INBOX_URL)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    assert moderation.moderation_url() is None
    assert asyncio.run(moderation.post_to_moderation("text", report_id=1, workspace_id=str(WS_ID))) is False
    assert driver.platforms == [] and driver.sends == []


@pytest.mark.parametrize("value", [
    f"http://{INBOX_HOST}/inbox/T0KEN-in-the-path",
    f"{INBOX_HOST}/inbox/T0KEN-in-the-path",
    f"ftp://{INBOX_HOST}/T0KEN",
    "https:///T0KEN-no-host",
    "https://",
    f"https://[{INBOX_HOST}/T0KEN",
    "T0KEN garbage",
])
def test_an_unusable_setting_posts_nothing_and_its_value_is_never_logged(monkeypatch, driver, caplog, value):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MODERATION_WEBHOOK_URL", value)
    caplog.set_level(logging.DEBUG)

    posted = asyncio.run(moderation.post_to_moderation("text", report_id=7, workspace_id=str(WS_ID)))

    assert posted is False
    assert driver.platforms == [] and driver.sends == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "MODERATION_WEBHOOK_URL" in warnings[0].getMessage()
    _no_secret_in(caplog.records)


def test_an_unusable_setting_is_warned_about_once(monkeypatch, driver, caplog):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MODERATION_WEBHOOK_URL", f"http://{INBOX_HOST}/inbox")
    caplog.set_level(logging.WARNING)

    for report_id in (1, 2, 3):
        asyncio.run(moderation.post_to_moderation("text", report_id=report_id, workspace_id=str(WS_ID)))

    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


def test_a_duplicate_report_posts_nothing(saas, filed, driver):
    filed["result"] = _filed(duplicate=True)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 200
    assert resp.json() == {"report_id": 41, "status": "already_received"}
    assert driver.platforms == [] and driver.sends == []


def test_a_report_that_fails_posts_nothing(saas, filed, driver, monkeypatch):
    async def _not_found(db, ctx, **kwargs):
        raise service.ReportTargetNotFound()

    monkeypatch.setattr(reports_api, "file_report", _not_found)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 404
    assert driver.sends == []


# ---------------------------------------------------------------------------
# What is posted: the summary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("excerpt", [
    pytest.param(lambda message: clip_excerpt(message), id="as-the-service-files-it"),
    pytest.param(lambda message: message, id="uncut"),
])
def test_the_posted_excerpt_and_note_are_at_most_200_characters(excerpt):
    message_5000 = "x" * 5000

    summary = moderation.moderation_summary(_filed(excerpt=excerpt(message_5000), note="n" * 500))

    assert _line(summary, "Excerpt") == "x" * 200
    assert _line(summary, "Note") == "n" * 200


def test_escaping_never_takes_the_excerpt_or_the_note_past_200_characters():
    hostile = "<&>" * 1667

    summary = moderation.moderation_summary(_filed(excerpt=hostile, note=hostile[:500]))

    for label in ("Excerpt", "Note"):
        posted = _line(summary, label)
        assert 0 < len(posted) <= 200
        assert posted.replace("&lt;", "").replace("&amp;", "").replace("&gt;", "") == ""


def test_the_summary_names_the_report_the_workspace_the_target_and_the_reason():
    summary = moderation.moderation_summary(_filed(note=None))

    assert summary.splitlines()[0] == "Automatos content report 41"
    assert _line(summary, "Workspace") == str(WS_ID)
    assert _line(summary, "Target") == f"message {MESSAGE_ID}"
    assert _line(summary, "Reason") == "harmful"
    assert _line(summary, "Excerpt") == "An Auto reply."
    assert _line(summary, "Note") == "(none)"


def test_the_text_carries_no_reporter_and_no_workspace_name(saas, filed, driver):
    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    (sent,) = driver.sends
    for personal in (REPORTER_ID, REPORTER_CLERK, REPORTER_EMAIL, "reporter", WORKSPACE_NAME):
        assert personal not in sent["text"]
    fields = set(service.FiledReport.__dataclass_fields__)
    assert not {f for f in fields if any(k in f for k in ("user", "reporter", "email", "name"))}


def test_a_note_cannot_ping_a_channel_carry_a_link_or_forge_a_line():
    note = "<!channel> see <https://phish.example.test|the doc>\nReason: other & more"

    summary = moderation.moderation_summary(_filed(note=note))

    assert "<" not in summary and ">" not in summary
    assert [line for line in summary.splitlines() if line.startswith("Reason:")] == ["Reason: harmful"]
    assert _line(summary, "Note").startswith("&lt;!channel&gt; see &lt;https://phish.example.test|the doc&gt;")
    assert _line(summary, "Note").endswith("Reason: other &amp; more")


# ---------------------------------------------------------------------------
# Failure: the report stands, the URL is never logged
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("result, status", [
    (SendResult(ok=False, latency_ms=5, error=f"network error: ConnectError for {INBOX_URL}",
                retryable=True), "none"),
    (SendResult(ok=False, latency_ms=5, error="webhook returned HTTP 503", retryable=True), "503"),
])
def test_a_driver_failure_leaves_the_201_and_never_logs_the_url(saas, filed, driver, caplog, result, status):
    driver.result = result
    caplog.set_level(logging.DEBUG)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    assert resp.json() == {"report_id": 41, "status": "received"}
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "report 41" in errors[0].getMessage() and f"HTTP status {status}" in errors[0].getMessage()
    assert "retryable True" in errors[0].getMessage()
    _no_secret_in(caplog.records)


def test_a_driver_that_raises_is_logged_by_class_only(saas, filed, driver, caplog):
    driver.error = httpx.ConnectError(f"cannot reach {INBOX_URL}")
    caplog.set_level(logging.DEBUG)

    resp = _client().post(ROUTE, json=BODY)

    assert resp.status_code == 201
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "ConnectError" in errors[0].getMessage()
    assert errors[0].exc_info is None
    _no_secret_in(caplog.records)


# ---------------------------------------------------------------------------
# When and how: after the response, through the existing webhook driver
# ---------------------------------------------------------------------------


def test_the_route_answers_before_the_post_runs(saas, filed, driver):
    async def _answer_then_run_tasks():
        tasks = BackgroundTasks()
        answer = await reports_api.report_content(
            reports_api.ReportRequest(**BODY), Response(), tasks, ctx=_reporter_ctx(), db=_NoRowsDB(),
        )
        sent_before_tasks = list(driver.sends)
        queued = list(tasks.tasks)
        await tasks()
        return answer, sent_before_tasks, queued

    answer, sent_before_tasks, queued = asyncio.run(_answer_then_run_tasks())

    assert answer == {"report_id": 41, "status": "received"}
    assert sent_before_tasks == []
    (task,) = queued
    assert task.func is moderation.post_to_moderation
    assert task.args == (moderation.moderation_summary(_filed()),)
    assert task.kwargs == {"report_id": 41, "workspace_id": str(WS_ID)}
    assert len(driver.sends) == 1


def test_the_post_goes_through_the_existing_webhook_driver(saas, monkeypatch):
    posts = []

    class _AsyncClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, json, headers):
            posts.append((url, json, self.timeout))
            return SimpleNamespace(status_code=200)

    monkeypatch.setattr(webhook_mod.httpx, "AsyncClient", _AsyncClient)
    summary = moderation.moderation_summary(_filed())

    posted = asyncio.run(moderation.post_to_moderation(summary, report_id=41, workspace_id=str(WS_ID)))

    assert posted is True
    assert get_driver(moderation.WEBHOOK_PLATFORM) is webhook_mod.WebhookDriver
    assert posts == [(INBOX_URL, {"text": summary, "message": summary, "ai_generated": True}, 15.0)]


def test_the_real_drivers_network_error_text_never_reaches_a_log(saas, caplog):
    class _UnreachableClient:
        def __init__(self, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, json, headers):
            raise httpx.ConnectError(f"cannot reach {url}")

    caplog.set_level(logging.DEBUG)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(webhook_mod.httpx, "AsyncClient", _UnreachableClient)
        posted = asyncio.run(moderation.post_to_moderation("text", report_id=9, workspace_id=str(WS_ID)))

    assert posted is False
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "report 9" in errors[0].getMessage() and "HTTP status none" in errors[0].getMessage()
    _no_secret_in(caplog.records)
