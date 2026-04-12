"""Unit tests for DeliverableService (PRD-129).

These tests use a MagicMock SQLAlchemy session — they assert the SQL that
the service issues and the shape of the dict envelopes it returns. A proper
integration test against a real Postgres lives in the API test suite.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4

import pytest

from services.deliverable_service import (
    DeliverableService,
    EXTENSION_TO_ARTIFACT,
    _infer_artifact_type,
    _humanize_basename,
    _slugify,
)


WORKSPACE_ID = uuid4()


# ---------------------------------------------------------------------------
# Helpers: build fake DB result rows
# ---------------------------------------------------------------------------

def _make_row(**overrides):
    defaults = {
        "id": uuid4(),
        "workspace_id": WORKSPACE_ID,
        "source_type": "chat",
        "source_id": "task-1",
        "agent_id": 42,
        "agent_name": "Scout",
        "artifact_type": "report",
        "title": "Weekly Report",
        "summary": "Summary text",
        "storage_type": "workspace",
        "file_path": "reports/scout/weekly.md",
        "file_name": "weekly.md",
        "file_type": "md",
        "file_size_bytes": 1024,
        "preview_url": None,
        "preview_type": None,
        "extra": {},
        "status": "ready",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    row = MagicMock()
    for k, v in defaults.items():
        setattr(row, k, v)
    return row


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    def test_slugify_basic(self):
        assert _slugify("Hello World") == "hello-world"
        assert _slugify("  Lots___of   spaces  ") == "lots-of-spaces"
        assert _slugify("a--b") == "a-b"

    def test_slugify_strips_punctuation(self):
        assert _slugify("Quarterly Report (Q3) — v2!") == "quarterly-report-q3-v2"

    def test_infer_artifact_type_known_extensions(self):
        assert _infer_artifact_type("file.png") == "image"
        assert _infer_artifact_type("FOO.JPG") == "image"
        assert _infer_artifact_type("report.md") == "report"
        assert _infer_artifact_type("analysis.pdf") == "document"
        assert _infer_artifact_type("deck.pptx") == "slide"
        assert _infer_artifact_type("data.csv") == "spreadsheet"
        assert _infer_artifact_type("main.py") == "code"
        assert _infer_artifact_type("bundle.zip") == "archive"
        assert _infer_artifact_type("clip.mp4") == "video"
        assert _infer_artifact_type("voice.mp3") == "audio"

    def test_infer_artifact_type_unknown_defaults_document(self):
        assert _infer_artifact_type("mystery.xyz") == "document"
        assert _infer_artifact_type("") == "document"
        assert _infer_artifact_type("no_extension") == "document"

    def test_humanize_basename(self):
        assert _humanize_basename("reports/weekly-sales-report.md") == "Weekly Sales Report"
        assert _humanize_basename("foo_bar_baz.txt") == "Foo Bar Baz"

    def test_extension_map_covers_all_categories(self):
        types = set(EXTENSION_TO_ARTIFACT.values())
        assert {
            "image", "report", "document", "slide",
            "spreadsheet", "code", "archive", "audio", "video",
        } <= types


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_new_row_returns_success(self):
        db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, i: (uuid4(), True)[i]
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(
            file_path="reports/scout/weekly.md",
            title="Weekly Report",
            source_type="heartbeat",
            source_id="hb-1",
            agent_id=42,
            agent_name="Scout",
            summary="Summary",
            file_size_bytes=2048,
        )

        assert out["success"] is True
        assert out["artifact_type"] == "report"
        assert out["title"] == "Weekly Report"
        assert out["created"] is True
        db.execute.assert_called_once()
        db.commit.assert_called_once()

        # Bound params passed correctly
        _, params = db.execute.call_args[0]
        assert params["workspace_id"] == str(WORKSPACE_ID)
        assert params["file_path"] == "reports/scout/weekly.md"
        assert params["source_type"] == "heartbeat"
        assert params["file_size_bytes"] == 2048
        assert params["artifact_type"] == "report"

    def test_register_idempotent_update(self):
        """Re-registering the same path returns created=False."""
        db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, i: (uuid4(), False)[i]
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="reports/scout/weekly.md")

        assert out["success"] is True
        assert out["created"] is False

    def test_register_infers_artifact_type_when_omitted(self):
        db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, i: (uuid4(), True)[i]
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="charts/q3.png")
        assert out["artifact_type"] == "image"

        _, params = db.execute.call_args[0]
        assert params["artifact_type"] == "image"
        assert params["file_name"] == "q3.png"
        assert params["file_type"] == "png"

    def test_register_requires_file_path(self):
        svc = DeliverableService(MagicMock(), WORKSPACE_ID)
        out = svc.register(file_path="")
        assert out["success"] is False
        assert "file_path" in out["error"]

    def test_register_rolls_back_on_error(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("boom")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="reports/x.md")
        assert out["success"] is False
        db.rollback.assert_called_once()

    def test_register_does_not_call_workspace_client(self):
        """register() must never hit WorkspaceClient — size is passed in."""
        db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, i: (uuid4(), True)[i]
        db.execute.return_value.fetchone.return_value = row

        with patch("services.deliverable_service.WorkspaceClient") as wc_mock:
            svc = DeliverableService(db, WORKSPACE_ID)
            svc.register(file_path="x.md", file_size_bytes=100)
            wc_mock.assert_not_called()


# ---------------------------------------------------------------------------
# list_deliverables()
# ---------------------------------------------------------------------------

class TestListDeliverables:
    def _setup_db(self, rows, total):
        db = MagicMock()
        count_result = MagicMock()
        count_result.scalar.return_value = total
        list_result = MagicMock()
        list_result.fetchall.return_value = rows
        db.execute.side_effect = [count_result, list_result]
        return db

    def test_list_returns_rows_and_total(self):
        rows = [_make_row(), _make_row()]
        db = self._setup_db(rows, total=2)
        svc = DeliverableService(db, WORKSPACE_ID)

        out = svc.list_deliverables()

        assert out["success"] is True
        assert out["total"] == 2
        assert len(out["deliverables"]) == 2
        assert out["limit"] == 24
        assert out["offset"] == 0
        assert out["deliverables"][0]["title"] == "Weekly Report"

    def test_list_applies_filters(self):
        db = self._setup_db([], total=0)
        svc = DeliverableService(db, WORKSPACE_ID)

        svc.list_deliverables(
            artifact_type="image",
            source_type="chat",
            agent_id=42,
            search="weekly",
            date_from="2026-01-01",
            date_to="2026-12-31",
            limit=10,
            offset=5,
        )

        # Both calls (count + list) should have filter params populated
        count_call_params = db.execute.call_args_list[0][0][1]
        assert count_call_params["artifact_type"] == "image"
        assert count_call_params["source_type"] == "chat"
        assert count_call_params["agent_id"] == 42
        assert count_call_params["search"] == "%weekly%"
        assert count_call_params["date_from"] == "2026-01-01"
        assert count_call_params["date_to"] == "2026-12-31"

        list_call_params = db.execute.call_args_list[1][0][1]
        assert list_call_params["limit"] == 10
        assert list_call_params["offset"] == 5

    def test_list_clamps_limit(self):
        db = self._setup_db([], total=0)
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.list_deliverables(limit=999)
        assert out["limit"] == 100

    def test_list_handles_exception(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("db down")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.list_deliverables()
        assert out["success"] is False
        assert out["deliverables"] == []
        assert out["total"] == 0


# ---------------------------------------------------------------------------
# get_deliverable()
# ---------------------------------------------------------------------------

class TestGetDeliverable:
    @pytest.mark.asyncio
    async def test_get_returns_not_found_when_missing(self):
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = None
        svc = DeliverableService(db, WORKSPACE_ID)

        out = await svc.get_deliverable("00000000-0000-0000-0000-000000000000")
        assert out["success"] is False
        assert "not found" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_get_without_content(self):
        row = _make_row()
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = row
        svc = DeliverableService(db, WORKSPACE_ID)

        out = await svc.get_deliverable(str(row.id), include_content=False)
        assert out["success"] is True
        assert "content" not in out["deliverable"]

    @pytest.mark.asyncio
    async def test_get_with_content_reads_file(self):
        row = _make_row(artifact_type="report", file_path="reports/x.md")
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = row

        with patch("services.deliverable_service.WorkspaceClient") as wc_mock:
            ws = MagicMock()
            ws.read_file = AsyncMock(return_value={"success": True, "content": "# Hi"})
            wc_mock.return_value = ws

            svc = DeliverableService(db, WORKSPACE_ID)
            out = await svc.get_deliverable(str(row.id), include_content=True)

        assert out["success"] is True
        assert out["deliverable"]["content"] == "# Hi"
        ws.read_file.assert_awaited_once_with("reports/x.md")

    @pytest.mark.asyncio
    async def test_get_image_with_content_returns_url_not_bytes(self):
        row = _make_row(
            artifact_type="image",
            file_path="charts/q3.png",
            preview_url="https://cdn/q3.png",
        )
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = row

        with patch("services.deliverable_service.WorkspaceClient") as wc_mock:
            svc = DeliverableService(db, WORKSPACE_ID)
            out = await svc.get_deliverable(str(row.id), include_content=True)
            wc_mock.assert_not_called()  # never reads image bytes inline

        assert out["success"] is True
        assert out["deliverable"]["content"] is None
        assert out["deliverable"]["content_url"] == "https://cdn/q3.png"

    @pytest.mark.asyncio
    async def test_get_read_file_failure_sets_content_error(self):
        row = _make_row(artifact_type="report", file_path="reports/x.md")
        db = MagicMock()
        db.execute.return_value.fetchone.return_value = row

        with patch("services.deliverable_service.WorkspaceClient") as wc_mock:
            ws = MagicMock()
            ws.read_file = AsyncMock(return_value={"success": False, "error": "boom"})
            wc_mock.return_value = ws

            svc = DeliverableService(db, WORKSPACE_ID)
            out = await svc.get_deliverable(str(row.id), include_content=True)

        assert out["deliverable"]["content"] is None
        assert out["deliverable"]["content_error"] == "boom"


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_get_stats_success(self):
        db = MagicMock()
        total_result = MagicMock()
        total_result.scalar.return_value = 7
        by_type_result = MagicMock()
        by_type_result.fetchall.return_value = [
            MagicMock(artifact_type="report", cnt=4),
            MagicMock(artifact_type="image", cnt=3),
        ]
        by_agent_result = MagicMock()
        ag = MagicMock(agent_id=42, cnt=5)
        ag.agent_name = "Scout"
        by_agent_result.fetchall.return_value = [ag]
        db.execute.side_effect = [total_result, by_type_result, by_agent_result]

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.get_stats()

        assert out["success"] is True
        assert out["total"] == 7
        assert out["by_type"] == {"report": 4, "image": 3}
        assert out["by_agent"] == [{"agent_id": 42, "agent_name": "Scout", "count": 5}]

    def test_get_stats_handles_exception(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("oops")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.get_stats()
        assert out["success"] is False
        assert out["total"] == 0
        assert out["by_type"] == {}


# ---------------------------------------------------------------------------
# soft_delete()
# ---------------------------------------------------------------------------

class TestSoftDelete:
    def test_soft_delete_success(self):
        db = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = (uuid4(),)
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("some-id")

        assert out["success"] is True
        db.commit.assert_called_once()

    def test_soft_delete_not_found(self):
        db = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchone.return_value = None
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("missing-id")
        assert out["success"] is False
        assert "not found" in out["error"].lower()

    def test_soft_delete_rolls_back_on_error(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("fail")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("some-id")
        assert out["success"] is False
        db.rollback.assert_called_once()
