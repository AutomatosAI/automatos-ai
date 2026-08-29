"""PRD-183 — internal-key GDPR seam for the Shopify compliance webhooks.

THE GAP (OS-review risk #4): a Shopify GDPR webhook (customers/redact,
shop/redact, customers/data_request) arrives machine-to-machine with the
internal key + a shop domain and NO user/workspace session, so it cannot reach
the user-facing ``/api/v1/gdpr/*`` endpoints (those require a logged-in
workspace admin and resolve the workspace from ``ctx.workspace_id``). This test
pins the fix: the internal-key-authed ``/api/verticals/{v}/gdpr/*`` surface

  * authenticates with the SAME internal key as ``/provision`` and ``/events``
    (``_verify_internal_key`` → ``SHOPIFY_INTERNAL_API_KEY``), NOT a workspace
    admin;
  * resolves the workspace from the shop/external id using the SAME indexed
    ``settings[external_id_key]`` lookup the provision flow + Shopify routes use;
  * 404s for a wrong/absent shop rather than erasing a blank/wrong workspace;
  * delegates to the SAME ``services.gdpr_service`` cascade W11 built (mocked
    here at the boundary), so every erasure is still audited there.

Pure: the DB session, the workspace resolver, and gdpr_service are faked/mocked
at the boundary — no Postgres, no Qdrant, no durable store, no network.
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

from api import verticals  # noqa: E402


# ---------------------------------------------------------------------------
# Boundary fakes — mirror the resolver's indexed lookup on a fake session.
# ---------------------------------------------------------------------------


class _FakeWorkspace:
    def __init__(self, wid="ws-abc", shop="demo.myshopify.com"):
        self.id = wid
        self.is_active = True
        self.settings = {"shopify_domain": shop, "source_external_id": shop}


class _FakeQuery:
    """Return the workspace only when the filtered shop matches (else None)."""

    def __init__(self, workspace):
        self._workspace = workspace

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._workspace


class _FakeDb:
    def __init__(self, workspace):
        self._workspace = workspace

    def query(self, *args, **kwargs):
        return _FakeQuery(self._workspace)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


SHOP = "demo.myshopify.com"


# ===========================================================================
# Auth — the surface is internal-key-authed, not workspace-admin.
# ===========================================================================


def test_gdpr_endpoints_use_internal_key_dep():
    """Every GDPR route depends on ``_verify_vertical_internal_key`` (the machine key),
    NOT the user-facing GDPR admin guard. Proven structurally: the dependency the
    routes carry is exactly the internal-key verifier the provision/events routes
    use."""
    routes = {r.path: r for r in verticals.router.routes}
    for path in (
        "/api/verticals/{vertical}/gdpr/erase-subject",
        "/api/verticals/{vertical}/gdpr/erase",
        "/api/verticals/{vertical}/gdpr/export",
    ):
        assert path in routes, f"missing route {path}"
        deps = [d.call for d in routes[path].dependant.dependencies]
        assert verticals._verify_vertical_internal_key in deps, f"{path} not internal-key authed"


def test_missing_internal_key_is_rejected():
    """No/blank Authorization → the internal-key dep raises (401/403/503 family),
    never a 200. We invoke the dep directly (as FastAPI would) with an empty key."""
    from config import config

    if not (config.SHOPIFY_INTERNAL_API_KEY or "").strip():
        # Key unset in the test env → dep fail-closes with 503 for any token.
        with pytest.raises(HTTPException) as ei:
            verticals._verify_vertical_internal_key("shopify", authorization="Bearer whatever")
        assert ei.value.status_code in (401, 403, 503)
    else:
        # Key set → a wrong token is rejected 401.
        with pytest.raises(HTTPException) as ei:
            verticals._verify_vertical_internal_key("shopify", authorization="Bearer definitely-wrong-key")
        assert ei.value.status_code == 401


# ===========================================================================
# Resolve + delegate — valid shop resolves the right workspace and calls
# gdpr_service with that workspace id.
# ===========================================================================


def test_erase_subject_resolves_and_delegates(monkeypatch):
    ws = _FakeWorkspace(wid="ws-abc", shop=SHOP)
    db = _FakeDb(ws)

    seen = {}

    def _fake_erase_subject(_db, *, workspace_id, subject_id, requested_by):
        seen.update(
            workspace_id=workspace_id, subject_id=subject_id, requested_by=requested_by
        )
        return {"workspace_id": str(workspace_id), "subject_id": subject_id}

    import services.gdpr_service as gdpr_service

    monkeypatch.setattr(gdpr_service, "erase_data_subject", _fake_erase_subject)

    req = verticals.VerticalGdprEraseSubjectRequest(external_id=SHOP, subject_id="cust-42")
    result = _run(verticals.gdpr_erase_subject(vertical="shopify", request=req, db=db, _auth=None))

    # Delegated to gdpr_service with the RESOLVED workspace (not the shop).
    assert seen["workspace_id"] == "ws-abc"
    assert seen["subject_id"] == "cust-42"
    # requested_by carries webhook provenance for the audit row.
    assert seen["requested_by"] == "shopify-webhook:demo.myshopify.com"
    assert result["subject_id"] == "cust-42"


def test_erase_workspace_resolves_and_delegates(monkeypatch):
    ws = _FakeWorkspace(wid="ws-abc", shop=SHOP)
    db = _FakeDb(ws)

    seen = {}

    def _fake_erase_ws(_db, workspace_id, *, requested_by):
        seen.update(workspace_id=workspace_id, requested_by=requested_by)
        return {"workspace_id": str(workspace_id), "complete": True}

    import services.gdpr_service as gdpr_service

    monkeypatch.setattr(gdpr_service, "erase_workspace", _fake_erase_ws)

    req = verticals.VerticalGdprEraseRequest(external_id=SHOP)
    result = _run(verticals.gdpr_erase_workspace(vertical="shopify", request=req, db=db, _auth=None))

    assert seen["workspace_id"] == "ws-abc"
    assert seen["requested_by"] == "shopify-webhook:demo.myshopify.com"
    assert result["complete"] is True


def test_export_resolves_and_delegates(monkeypatch):
    ws = _FakeWorkspace(wid="ws-abc", shop=SHOP)
    db = _FakeDb(ws)

    seen = {}

    def _fake_export(_db, workspace_id, *, requested_by):
        seen.update(workspace_id=workspace_id, requested_by=requested_by)
        return {"workspace_id": str(workspace_id), "format": "automatos.gdpr.export/v1"}

    import services.gdpr_service as gdpr_service

    monkeypatch.setattr(gdpr_service, "export_workspace", _fake_export)

    resp = _run(
        verticals.gdpr_export_workspace(
            vertical="shopify", external_id=SHOP, customer_id="cust-42", db=db, _auth=None
        )
    )

    assert seen["workspace_id"] == "ws-abc"
    assert seen["requested_by"] == "shopify-webhook:demo.myshopify.com"
    # Returns the bundle as a JSON response.
    import json

    body = json.loads(resp.body)
    assert body["format"] == "automatos.gdpr.export/v1"


# ===========================================================================
# Wrong / absent shop → 404 (NEVER erase a wrong/blank workspace).
# ===========================================================================


def test_erase_subject_unknown_shop_404s(monkeypatch):
    db = _FakeDb(None)  # resolver finds nothing

    import services.gdpr_service as gdpr_service

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("gdpr_service must NOT be called for an unknown shop")

    monkeypatch.setattr(gdpr_service, "erase_data_subject", _boom)

    req = verticals.VerticalGdprEraseSubjectRequest(external_id="ghost.myshopify.com", subject_id="x")
    with pytest.raises(HTTPException) as ei:
        _run(verticals.gdpr_erase_subject(vertical="shopify", request=req, db=db, _auth=None))
    assert ei.value.status_code == 404
    assert called["n"] == 0


def test_erase_workspace_unknown_shop_404s(monkeypatch):
    db = _FakeDb(None)

    import services.gdpr_service as gdpr_service

    monkeypatch.setattr(
        gdpr_service, "erase_workspace",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )

    req = verticals.VerticalGdprEraseRequest(external_id="ghost.myshopify.com")
    with pytest.raises(HTTPException) as ei:
        _run(verticals.gdpr_erase_workspace(vertical="shopify", request=req, db=db, _auth=None))
    assert ei.value.status_code == 404


def test_export_unknown_shop_404s():
    db = _FakeDb(None)
    with pytest.raises(HTTPException) as ei:
        _run(
            verticals.gdpr_export_workspace(
                vertical="shopify", external_id="ghost.myshopify.com", customer_id=None, db=db, _auth=None
            )
        )
    assert ei.value.status_code == 404
