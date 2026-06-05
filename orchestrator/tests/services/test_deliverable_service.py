"""Unit tests for DeliverableService (PRD-133b corrected).

These tests use a MagicMock SQLAlchemy session — they assert the SQL that
the service issues and the shape of the dict envelopes it returns. A proper
integration test against a real Postgres lives in the API test suite.

PRD-133b changes exercised here:
  * ``register()`` now refuses ``artifact_type in {'blog_post','report'}``.
  * Reads (list/get/stats) go through ``v_workspace_outputs``.
  * ``soft_delete()`` resolves artifact_type first, then targets the right
    source table (blog_posts / agent_reports / deliverables).
  * ``apply_retention()`` branches heartbeat → agent_reports, else → deliverables.
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
    _workspace_file_url,
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
            file_path="outputs/scout/chart.png",
            title="Weekly Chart",
            source_type="task",
            source_id="task-1",
            agent_id=42,
            agent_name="Scout",
            summary="Summary",
            file_size_bytes=2048,
        )

        assert out["success"] is True
        assert out["artifact_type"] == "image"
        assert out["title"] == "Weekly Chart"
        assert out["created"] is True
        db.execute.assert_called_once()
        db.commit.assert_called_once()

        # Bound params passed correctly
        _, params = db.execute.call_args[0]
        assert params["workspace_id"] == str(WORKSPACE_ID)
        assert params["file_path"] == "outputs/scout/chart.png"
        assert params["source_type"] == "task"
        assert params["file_size_bytes"] == 2048
        assert params["artifact_type"] == "image"

    def test_register_idempotent_update(self):
        """Re-registering the same path returns created=False."""
        db = MagicMock()
        row = MagicMock()
        row.__getitem__ = lambda self, i: (uuid4(), False)[i]
        result_mock = MagicMock()
        result_mock.fetchone.return_value = row
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="outputs/scout/chart.png")

        assert out["success"] is True
        assert out["created"] is False

    def test_register_refuses_blog_post(self):
        """PRD-133b: blog posts are owned by BlogService; never writable here."""
        db = MagicMock()
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="posts/intro.md", artifact_type="blog_post")
        assert out["success"] is False
        assert "blog_post" in out["error"]
        db.execute.assert_not_called()

    def test_register_refuses_report_by_inference(self):
        """PRD-133b: .md files infer 'report' — must be rejected."""
        db = MagicMock()
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.register(file_path="reports/scout/weekly.md")
        assert out["success"] is False
        assert "report" in out["error"]
        db.execute.assert_not_called()

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
        # Non-report/blog path so we reach the INSERT, which then explodes.
        out = svc.register(file_path="outputs/chart.png")
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
            svc.register(file_path="outputs/chart.png", file_size_bytes=100)
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
        # content_url is recomputed fresh from the workspace path (not echoed
        # from the row's stored preview_url) — older rows had stale preview_url
        # pointing at /files/content (JSON), so the service always rebuilds the
        # binary /files/raw URL for images.
        assert out["deliverable"]["content_url"] == _workspace_file_url(
            WORKSPACE_ID, "charts/q3.png"
        )

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
# soft_delete() — two-step: resolve artifact_type via view, UPDATE source table
# ---------------------------------------------------------------------------

class TestSoftDelete:
    @staticmethod
    def _setup(db, artifact_type: str | None):
        """Wire db.execute to return a row with .artifact_type on first call
        (SELECT via view), then swallow subsequent UPDATE."""
        select_result = MagicMock()
        if artifact_type is None:
            select_result.fetchone.return_value = None
        else:
            lookup_row = MagicMock()
            lookup_row.artifact_type = artifact_type
            select_result.fetchone.return_value = lookup_row
        update_result = MagicMock()
        db.execute.side_effect = [select_result, update_result]

    def _extract_update_sql(self, db):
        """Second execute call is the UPDATE — return its SQL text."""
        update_call = db.execute.call_args_list[1]
        return str(update_call[0][0])

    def test_soft_delete_report_targets_agent_reports(self):
        db = MagicMock()
        self._setup(db, artifact_type="report")

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("some-id")

        assert out["success"] is True
        db.commit.assert_called_once()
        assert "UPDATE agent_reports" in self._extract_update_sql(db)

    def test_soft_delete_blog_post_targets_blog_posts(self):
        db = MagicMock()
        self._setup(db, artifact_type="blog_post")

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("blog-id")

        assert out["success"] is True
        assert "UPDATE blog_posts" in self._extract_update_sql(db)

    def test_soft_delete_ad_hoc_targets_deliverables(self):
        db = MagicMock()
        self._setup(db, artifact_type="image")

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("img-id")

        assert out["success"] is True
        assert "UPDATE deliverables" in self._extract_update_sql(db)

    def test_soft_delete_not_found(self):
        db = MagicMock()
        self._setup(db, artifact_type=None)

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("missing-id")
        assert out["success"] is False
        assert "not found" in out["error"].lower()
        # No UPDATE issued
        assert db.execute.call_count == 1

    def test_soft_delete_rolls_back_on_error(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("fail")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.soft_delete("some-id")
        assert out["success"] is False
        db.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# apply_retention() — heartbeat → agent_reports, else → deliverables
# ---------------------------------------------------------------------------

class TestApplyRetention:
    def test_heartbeat_targets_agent_reports(self):
        db = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [MagicMock(id=uuid4()) for _ in range(3)]
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.apply_retention(source_type="heartbeat", keep_per_agent=50)

        assert out["success"] is True
        assert out["pruned"] == 3
        sql = str(db.execute.call_args[0][0])
        assert "agent_reports" in sql
        assert "heartbeat_result_id IS NOT NULL" in sql
        db.commit.assert_called_once()

    def test_non_heartbeat_targets_deliverables(self):
        db = MagicMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        db.execute.return_value = result_mock

        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.apply_retention(source_type="task", keep_per_agent=10)

        assert out["success"] is True
        assert out["pruned"] == 0
        sql = str(db.execute.call_args[0][0])
        assert "FROM deliverables" in sql
        params = db.execute.call_args[0][1]
        assert params["source_type"] == "task"
        assert params["keep"] == 10

    def test_apply_retention_rolls_back_on_error(self):
        db = MagicMock()
        db.execute.side_effect = RuntimeError("boom")
        svc = DeliverableService(db, WORKSPACE_ID)
        out = svc.apply_retention(source_type="heartbeat")
        assert out["success"] is False
        db.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# get_deliverable() — blog_post inline branch (PRD-133b)
# ---------------------------------------------------------------------------

class TestGetBlogPostInline:
    @pytest.mark.asyncio
    async def test_blog_post_returns_content_from_blog_posts_table(self):
        """PRD-133b: blog_post content is read inline from blog_posts.content,
        never from workspace filesystem."""
        blog_id = uuid4()
        view_row = _make_row(
            id=blog_id,
            artifact_type="blog_post",
            file_path="posts/intro.md",
            preview_url="/api/workspaces/x/files/content?path=posts/intro.md",
        )
        blog_row = MagicMock()
        blog_row.__getitem__ = lambda self, i: ["# Inline content"][i]

        view_result = MagicMock()
        view_result.fetchone.return_value = view_row
        blog_result = MagicMock()
        blog_result.fetchone.return_value = blog_row

        db = MagicMock()
        db.execute.side_effect = [view_result, blog_result]

        with patch("services.deliverable_service.WorkspaceClient") as wc_mock:
            svc = DeliverableService(db, WORKSPACE_ID)
            out = await svc.get_deliverable(str(blog_id), include_content=True)
            wc_mock.assert_not_called()  # never touches filesystem for blogs

        assert out["success"] is True
        assert out["deliverable"]["content"] == "# Inline content"
        assert out["deliverable"]["content_truncated"] is False
