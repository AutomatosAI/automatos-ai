"""#1442 follow-up: an incomplete erasure is a 503 its caller retries, and the retry finds the
workspace and finishes the job.

The erase routes answered 200 with ``"complete": false``, so Shopify never asked again; and the
``shop/redact`` webhook looked up only active workspaces. Shopify sends ``shop/redact`` 48 hours
after an uninstall, which deactivates the workspace, so the lookup 404'd; a retry after a partial
erase (which closes the workspace first) would have 404'd too. Now the webhook erases every
workspace of the shop that is no longer active (a reinstalled shop's live one is kept), and an
incomplete erasure answers 503. The real erasure on the real database; the memory stores stubbed.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

SHOP = "acme-billing.myshopify.com"


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def erasure(monkeypatch):
    """The real erasure with the memory stores stubbed (``fail`` makes durable memory raise once)."""
    from config import config
    from services import gdpr_service, workspace_purge

    fail = {"durable": 0}

    def _durable(ws, subject_id=None):
        if fail["durable"]:
            fail["durable"] -= 1
            raise RuntimeError("qdrant unreachable")
        return 2

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 1)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", _durable)
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)
    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    return fail


def _shop_workspace(new_session, *, active: bool) -> str:
    from sqlalchemy import text

    ws = str(uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, is_active, settings) VALUES (CAST(:id AS uuid), 'Acme Billing', "
                   ":active, CAST(:settings AS jsonb))"),
              {"id": ws, "active": active, "settings": json.dumps({"shopify_domain": SHOP, "source_external_id": SHOP})})
    s.commit()
    s.close()
    return ws


def _exists(new_session, ws) -> bool:
    from sqlalchemy import text

    s = new_session()
    try:
        return s.execute(text("SELECT 1 FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws}).first() is not None
    finally:
        s.close()


def _drop(new_session, *ids):
    from sqlalchemy import text

    s = new_session.sweep()
    for ws in ids:
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _redact(new_session):
    from api import verticals

    s = new_session()
    try:
        return verticals.gdpr_erase_workspace(vertical="shopify", request=SimpleNamespace(external_id=SHOP),
                                              db=s, _auth=None)
    finally:
        s.close()


def test_an_uninstalled_shops_workspace_is_found_and_erased(new_session, erasure):
    ws = _shop_workspace(new_session, active=False)  # deactivated on uninstall
    try:
        answer = _redact(new_session)
        assert answer["complete"] is True and not _exists(new_session, ws)
    finally:
        _drop(new_session, ws)


def test_an_incomplete_erasure_is_a_503_and_the_retry_finishes_it(new_session, erasure):
    ws = _shop_workspace(new_session, active=False)
    erasure["durable"] = 1  # the first attempt cannot reach durable memory
    try:
        with pytest.raises(HTTPException) as refused:
            _redact(new_session)
        assert refused.value.status_code == 503
        assert refused.value.detail["complete"] is False
        assert refused.value.detail["failed"] == ["durable_memory", "sql"]  # the purge waits for the memories
        assert _exists(new_session, ws)  # so the retry can find the workspace, closed
        answer = _redact(new_session)  # Shopify asks again
        assert answer["complete"] is True and not _exists(new_session, ws)
    finally:
        _drop(new_session, ws)


def test_a_reinstalled_shops_live_workspace_is_kept(new_session, erasure):
    old = _shop_workspace(new_session, active=False)
    live = _shop_workspace(new_session, active=True)
    try:
        assert _redact(new_session)["complete"] is True
        assert not _exists(new_session, old) and _exists(new_session, live)
    finally:
        _drop(new_session, old, live)


def test_several_gone_workspaces_are_all_erased_and_answered_together(new_session, erasure):
    first, second = _shop_workspace(new_session, active=False), _shop_workspace(new_session, active=False)
    try:
        answer = _redact(new_session)
        assert answer["complete"] is True and len(answer["workspaces"]) == 2
        assert not _exists(new_session, first) and not _exists(new_session, second)
    finally:
        _drop(new_session, first, second)


def test_the_owners_own_erasure_answers_503_when_incomplete(monkeypatch):
    from api import gdpr
    from services import gdpr_service

    monkeypatch.setattr(gdpr_service, "erase_workspace",
                        lambda db, ws, requested_by: {"complete": False, "failed": ["sql"]})
    ctx = SimpleNamespace(workspace_id="ws-1", user=None)
    monkeypatch.setattr(gdpr, "_actor_ref", lambda ctx: "user:1")
    with pytest.raises(HTTPException) as refused:
        gdpr.erase_my_workspace(body={"confirm_workspace_id": "ws-1"}, ctx=ctx, db=None)
    assert refused.value.status_code == 503 and refused.value.detail["failed"] == ["sql"]


def test_an_unknown_shop_is_still_a_404(new_session, erasure):
    with pytest.raises(HTTPException) as refused:
        _redact(new_session)
    assert refused.value.status_code == 404


def test_an_answer_never_carries_an_exceptions_text_or_a_sign_in_id():
    from api.gdpr import erasure_answer

    stopped = {"workspace_id": "ws-1", "complete": False, "failed": ["sql"], "derived": {},
               "sql": {"skipped_reason": "exception: connection to 10.0.0.5 refused (password=…)",
                       "clerk_user_id": "user_2abc", "rows_deleted": {"agents": 3}}}
    with pytest.raises(HTTPException) as refused:
        erasure_answer([stopped])
    detail = refused.value.detail
    assert detail["sql"]["skipped_reason"] == "failed" and "clerk_user_id" not in detail["sql"]
    assert "10.0.0.5" not in str(detail) and detail["sql"]["rows_deleted"] == {"agents": 3}
    kept = erasure_answer([{**stopped, "complete": True, "failed": [],
                            "sql": {"skipped_reason": None, "workspace_row_deleted": True}}])
    assert kept["sql"]["skipped_reason"] is None and kept["sql"]["workspace_row_deleted"] is True


def test_a_purge_that_raises_records_no_exception_text(new_session, monkeypatch):
    from services import gdpr_service, workspace_purge

    def _boom(*args, **kwargs):
        raise RuntimeError("connection to 10.0.0.5 refused")

    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    # The memories erase cleanly, so the purge reaches the SQL step that raises.
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(workspace_purge, "_discover_scoped_tables", _boom)
    ws = _shop_workspace(new_session, active=False)
    try:
        from sqlalchemy import text

        s = new_session()
        s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:id AS uuid)"), {"id": ws})
        s.commit()
        s.close()
        result = workspace_purge.purge_workspace_sync(ws)
        assert result.skipped_reason == "exception" and "10.0.0.5" not in str(result.to_dict())
    finally:
        _drop(new_session, ws)


def test_a_workspace_that_could_not_be_closed_keeps_its_memories_for_the_retry(monkeypatch):
    """CodeRabbit on #1443: erasing a live workspace's memories left it half erased once the
    purge refused it. When the close fails, no store is touched and the erasure is incomplete."""
    from services import gdpr_service

    touched = []

    def _close_fails(ws):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(gdpr_service, "_close_workspace", _close_fails)
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: touched.append("field"))
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: touched.append("durable"))
    monkeypatch.setattr(gdpr_service, "_purge_sql", lambda ws: touched.append("sql") or {})
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)
    result = gdpr_service.erase_workspace(None, "ws-1", requested_by="user:1")
    assert touched == [] and result["complete"] is False
    assert result["failed"][:3] == ["close", "field_memory", "durable_memory"]


def test_a_failed_s3_delete_holds_the_purge_and_a_rerun_finishes_it(new_session, monkeypatch):
    """CodeRabbit on #1443: deleting the row after a failed S3 delete left those objects where no
    retry could find them. The purge stops after S3, before the memories, the sign-in and the rows."""
    from services import workspace_purge

    attempts = {"s3_fails": 1}
    memories = []

    def _s3(workspace_id, result):
        if attempts["s3_fails"]:
            attempts["s3_fails"] -= 1
            result.s3_errors = 2

    monkeypatch.setattr(workspace_purge, "_purge_workspace_s3", _s3)
    monkeypatch.setattr(workspace_purge, "_erase_workspace_memories", lambda ws, result: memories.append(ws) or True)
    ws = _shop_workspace(new_session, active=False)
    try:
        from sqlalchemy import text

        s = new_session()
        s.execute(text("UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:id AS uuid)"), {"id": ws})
        s.commit()
        s.close()
        held = workspace_purge.purge_workspace_sync(ws)
        assert held.skipped_reason == workspace_purge.S3_DELETE_FAILED and not held.workspace_row_deleted
        assert memories == [] and _exists(new_session, ws)
        rerun = workspace_purge.purge_workspace_sync(ws)
        assert rerun.workspace_row_deleted and not _exists(new_session, ws) and len(memories) == 1
    finally:
        _drop(new_session, ws)


def test_the_answer_passes_the_s3_hold_on_as_itself():
    from api.gdpr import public_erasure
    from services.workspace_purge import S3_DELETE_FAILED

    answer = public_erasure({"complete": False, "sql": {"skipped_reason": S3_DELETE_FAILED, "s3_errors": 2}})
    assert answer["sql"]["skipped_reason"] == S3_DELETE_FAILED
