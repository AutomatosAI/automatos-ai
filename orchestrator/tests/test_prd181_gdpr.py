"""PRD-181 S3 + S4 — GDPR data export + erasure-with-derived-data-cascade.

Risk #4 (UK right-to-erasure): a delete that leaves the subject in field memory /
vectors / durable memory is NOT a GDPR delete. These tests prove the cascade reaches every
store, and that every erasure is audited.

Everything external is mocked at the boundary:
  - Postgres  → the reused ``workspace_purge`` result is faked.
  - Qdrant    → the field adapter's ``erase_workspace`` is a stub returning a count.
  - durable memory → the unified-memory ``erase_workspace_memories`` is a stub.
So the test needs no live infra; it verifies the *orchestration* and the audit.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


class _FakeSession:
    def __init__(self, rows: Dict[str, List[dict]] | None = None) -> None:
        self._rows = rows or {}
        self.added: List[Any] = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.committed = False


# ===========================================================================
# S3 — export
# ===========================================================================

def test_gdpr_export_bundles_sql_and_derived(monkeypatch):
    """Export returns a portable JSON bundle covering primary (SQL) + derived
    (Qdrant field, durable memory) stores for a workspace."""
    from services import gdpr_service

    ws = uuid4()

    monkeypatch.setattr(
        gdpr_service, "_export_sql_tables",
        lambda db, workspace_id: {"agents": [{"id": 1}], "board_tasks": [{"id": 9}]},
    )
    monkeypatch.setattr(
        gdpr_service, "_export_field_memory",
        lambda workspace_id, subject_id=None: [{"key": "fact", "value": "x"}],
    )
    monkeypatch.setattr(
        gdpr_service, "_export_durable_memory",
        lambda workspace_id, subject_id=None: [{"id": "m1", "text": "durable"}],
    )

    bundle = gdpr_service.export_workspace(_FakeSession(), ws)

    assert bundle["workspace_id"] == str(ws)
    assert bundle["format"] == "automatos.gdpr.export/v1"
    assert bundle["sql"]["agents"] == [{"id": 1}]
    assert bundle["derived"]["field_memory"] == [{"key": "fact", "value": "x"}]
    assert bundle["derived"]["durable_memory"] == [{"id": "m1", "text": "durable"}]
    # portable = JSON-serialisable
    import json

    json.dumps(bundle)


def test_gdpr_export_is_audited(monkeypatch):
    from services import gdpr_service

    calls: List[str] = []
    monkeypatch.setattr(gdpr_service, "_export_sql_tables", lambda db, ws: {})
    monkeypatch.setattr(gdpr_service, "_export_field_memory", lambda ws, subject_id=None: [])
    monkeypatch.setattr(gdpr_service, "_export_durable_memory", lambda ws, subject_id=None: [])
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: calls.append(action))

    gdpr_service.export_workspace(_FakeSession(), uuid4())
    assert "gdpr:export" in calls


# ===========================================================================
# S4 — erasure cascade
# ===========================================================================

def test_gdpr_erasure_cascade(monkeypatch):
    """After erasure the workspace has zero SQL rows AND zero Qdrant field
    vectors AND zero durable memories — the derived-data cascade runs."""
    from services import gdpr_service

    ws = uuid4()
    ran: Dict[str, Any] = {}

    # SQL: reuse workspace_purge — fake a successful purge result.
    def _fake_purge(workspace_id):
        ran["sql"] = str(workspace_id)
        return {"rows_deleted": {"agents": 3, "board_tasks": 1}, "workspace_row_deleted": True}

    monkeypatch.setattr(gdpr_service, "_purge_sql", _fake_purge)

    def _fake_field(workspace_id, subject_id=None):
        ran["field"] = (str(workspace_id), subject_id)
        return 12  # points deleted

    def _fake_durable(workspace_id, subject_id=None):
        ran["durable"] = (str(workspace_id), subject_id)
        return 5  # memories deleted

    monkeypatch.setattr(gdpr_service, "_erase_field_memory", _fake_field)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", _fake_durable)

    result = gdpr_service.erase_workspace(_FakeSession(), ws, requested_by="user:9")

    # every store was reached
    assert ran["sql"] == str(ws)
    assert ran["field"][0] == str(ws)
    assert ran["durable"][0] == str(ws)
    # the result reports zeros-after (counts of what was removed)
    assert result["sql"]["workspace_row_deleted"] is True
    assert result["derived"]["field_memory_deleted"] == 12
    assert result["derived"]["durable_memory_deleted"] == 5
    assert result["complete"] is True


def test_erasure_is_audited(monkeypatch):
    """Every erasure writes an AuditLog governance row (§S4)."""
    from services import gdpr_service

    calls: List[Dict[str, Any]] = []
    monkeypatch.setattr(gdpr_service, "_purge_sql", lambda ws: {"rows_deleted": {}, "workspace_row_deleted": True})
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: calls.append({"action": action, **kw}))

    gdpr_service.erase_workspace(_FakeSession(), uuid4(), requested_by="user:9")
    assert any(c["action"] == "gdpr:erasure" for c in calls), "erasure must be audited"


def test_erase_data_subject_entrypoint_exists(monkeypatch):
    """The single subject-level entrypoint the future Shopify customers/redact
    webhook calls exists and cascades where the data-subject tag is present."""
    from services import gdpr_service

    ws = uuid4()
    seen: Dict[str, Any] = {}
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: seen.setdefault("field", subject_id) or 0)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: seen.setdefault("durable", subject_id) or 0)
    monkeypatch.setattr(gdpr_service, "_erase_subject_sql", lambda db, ws, subject_id: {"deleted": 0})
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)

    result = gdpr_service.erase_data_subject(
        _FakeSession(), workspace_id=ws, subject_id="cust_123", requested_by="webhook:shopify"
    )
    assert seen["field"] == "cust_123"
    assert seen["durable"] == "cust_123"
    # gaps are surfaced, not hidden
    assert "gaps" in result


def test_erasure_reports_gaps(monkeypatch):
    """Stores that lack a data-subject tag are reported as documented gaps on a
    subject-level erasure (never silently skipped)."""
    from services import gdpr_service

    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_erase_subject_sql", lambda db, ws, subject_id: {"deleted": 0})
    monkeypatch.setattr(gdpr_service, "_audit_gdpr", lambda db, ws, action, **kw: None)

    result = gdpr_service.erase_data_subject(
        _FakeSession(), workspace_id=uuid4(), subject_id="cust_123", requested_by="webhook:shopify"
    )
    gaps = result["gaps"]
    assert isinstance(gaps, list) and len(gaps) >= 1
    # each gap names the store and why subject-granularity is unavailable
    assert all("store" in g and "reason" in g for g in gaps)
    stores = {g["store"] for g in gaps}
    # PRD-196 S6: field_memory + durable_memory now do a real subject filter-
    # delete, so they are NO LONGER gaps — only SQL remains a structural gap.
    assert "sql" in stores
    assert "field_memory" not in stores and "durable_memory" not in stores
    # pre-tag rows in the (now-tagged) stores are reported as untagged history,
    # never claimed erased.
    caveat = result["untagged_history"]
    assert set(caveat["stores"]) == {"field_memory", "durable_memory"}
