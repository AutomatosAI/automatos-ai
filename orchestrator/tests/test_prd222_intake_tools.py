"""PRD-222 W1S8/US-008 — the business-intake tools (scan + status).

Contract tests (3-file registration + schema truth) and handler tests against
fakes — no DB, no live Firecrawl. ``start_business_scan`` is exercised with a fake
Firecrawl client and a captured ``launch_guarded`` so the "real pipeline is
started, scoped to the caller's workspace" claim is proven without running it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import api.wizard as wiz
from config import config
from modules.tools.discovery.action_registry import ActionRegistry
from modules.tools.discovery.actions_intake import register_intake_actions
from modules.tools.discovery.handlers_intake import (
    _summarize_profile,
    get_intake_status,
    scan_business_site,
)

SCAN = "platform_scan_business_site"
STATUS = "platform_get_intake_status"


def _run(coro):
    return asyncio.run(coro)


def _defs():
    reg = ActionRegistry()
    register_intake_actions(reg)
    return reg._actions


# --------------------------------------------------------------------------- #
# Contract — 3-file registration + schema truth (AC1)
# --------------------------------------------------------------------------- #


def test_both_tools_registered():
    d = _defs()
    assert d[SCAN].name == SCAN
    assert d[STATUS].name == STATUS


def test_required_matches_handler_truth():
    d = _defs()
    # Each handler hard-fails without its one param → it belongs in required[].
    assert d[SCAN].parameters["required"] == ["domain"]
    assert d[STATUS].parameters["required"] == ["profile_id"]


def test_registered_via_full_registry_init():
    reg = ActionRegistry()  # runs register_all_actions
    assert reg.get(SCAN) is not None
    assert reg.get(STATUS) is not None


def test_registered_in_executor_handler_map():
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(MagicMock(), uuid4())
    assert SCAN in executor._handlers
    assert STATUS in executor._handlers


# --------------------------------------------------------------------------- #
# scan_business_site — honest degrade (AC2) + real pipeline start (AC1)
# --------------------------------------------------------------------------- #


def test_scan_missing_domain_errors():
    res = _run(scan_business_site(MagicMock(), uuid4(), {}))
    assert res["success"] is False and "domain" in res["error"]


def test_scan_unconfigured_firecrawl_returns_honest_result(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", None)
    res = _run(scan_business_site(MagicMock(), uuid4(), {"domain": "acme.com"}))
    # No 503 through the tool — an honest {configured:false}, success envelope.
    assert res["success"] is True
    assert res["data"]["configured"] is False
    assert "doc upload" in res["data"]["alternatives"]


def test_scan_configured_delegates_to_start_business_scan(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", "fc-fake-key-000")

    async def _stub(db, workspace_id, domain, **kw):
        return {"profile_id": "p-1", "started": True, "archetype": "ecommerce", "selected_count": 3}

    monkeypatch.setattr(wiz, "start_business_scan", _stub)
    res = _run(scan_business_site(MagicMock(), uuid4(), {"domain": "https://ACME.com/x"}))
    assert res["success"] is True
    assert res["data"] == {"profile_id": "p-1", "started": True, "archetype": "ecommerce", "selected_count": 3}


def test_scan_start_never_raises_through_the_tool(monkeypatch):
    monkeypatch.setattr(config, "FIRECRAWL_API_KEY", "fc-fake-key-000")

    async def _boom(db, workspace_id, domain, **kw):
        raise RuntimeError("firecrawl exploded")

    monkeypatch.setattr(wiz, "start_business_scan", _boom)
    db = MagicMock()
    res = _run(scan_business_site(db, uuid4(), {"domain": "acme.com"}))
    assert res["success"] is False and "exploded" in res["error"]


# --------------------------------------------------------------------------- #
# start_business_scan — creates a workspace-scoped profile + launches the pipeline
# (reuses the wizard's own primitives; Firecrawl + launch mocked). AC1.
# --------------------------------------------------------------------------- #


class _FakeFirecrawl:
    async def map(self, domain):
        return [f"https://{domain}/", f"https://{domain}/pricing", f"https://{domain}/about"]


class _FakeDB:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid4()


def test_start_business_scan_scopes_profile_and_launches_pipeline(monkeypatch):
    ws_id = uuid4()
    captured = {}

    def _capture_launch(coro, **kwargs):
        captured["kwargs"] = kwargs
        coro.close()  # never actually run the pipeline in a unit test

    monkeypatch.setattr(wiz, "_firecrawl_client", lambda: _FakeFirecrawl())
    monkeypatch.setattr(wiz, "launch_guarded", _capture_launch)
    monkeypatch.setattr(config, "FIRECRAWL_MAX_PAGES_PER_SCAN", 20)

    db = _FakeDB()
    res = _run(wiz.start_business_scan(db, ws_id, "https://ACME.com/path", goals=["book appts"]))

    assert res["started"] is True and res["profile_id"]
    # The profile was created scoped to the CALLER's workspace, domain normalized.
    profile = db.added[0]
    assert profile.workspace_id == ws_id
    assert profile.domain == "acme.com"
    assert profile.status == "scraping"
    # The real background pipeline was launched for this workspace + profile.
    assert captured["kwargs"]["workspace_id"] == str(ws_id)
    assert captured["kwargs"]["extra"]["profile_id"] == res["profile_id"]


# --------------------------------------------------------------------------- #
# get_intake_status — workspace-scoped read; cross-workspace refused (AC4)
# --------------------------------------------------------------------------- #


def _profile(**over):
    base = dict(
        id=uuid4(), status="scraping", domain="acme.com", archetype="ecommerce",
        company_name="Acme", raw_map_urls=["a", "b", "c"], selected_urls=["a"],
        quality_findings={},
    )
    base.update(over)
    return SimpleNamespace(**base)


def _db_first(result):
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = result
    return db


def test_status_missing_profile_id_errors():
    res = _run(get_intake_status(MagicMock(), uuid4(), {}))
    assert res["success"] is False and "profile_id" in res["error"]


def test_status_invalid_profile_id_errors():
    res = _run(get_intake_status(MagicMock(), uuid4(), {"profile_id": "not-a-uuid"}))
    assert res["success"] is False and "invalid" in res["error"]


def test_status_returns_stage_and_summary_for_owned_profile():
    prof = _profile()
    res = _run(get_intake_status(_db_first(prof), uuid4(), {"profile_id": str(prof.id)}))
    assert res["success"] is True
    data = res["data"]
    assert data["stage"] == "scraping"
    assert data["domain"] == "acme.com"
    assert data["pages_found"] == 3 and data["pages_selected"] == 1


def test_status_cross_workspace_profile_is_refused():
    # The handler filters by workspace_id; a profile in another workspace never
    # matches → not-found, never leaked/confirmed.
    res = _run(get_intake_status(_db_first(None), uuid4(), {"profile_id": str(uuid4())}))
    assert res["success"] is False and "not found" in res["error"]


def test_summarize_profile_dumps_no_scraped_content():
    # Summary is stage + shape counts only — never the scraped page bodies.
    data = _summarize_profile(_profile(raw_map_urls=["u"] * 40))
    assert data["pages_found"] == 40
    assert set(data) == {
        "profile_id", "stage", "domain", "archetype",
        "company_name", "pages_found", "pages_selected", "quality_findings",
    }
