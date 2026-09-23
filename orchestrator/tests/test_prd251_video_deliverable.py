"""PRD-251 S0.4b — video is a Deliverable.

Agents already produce MP4s and ``deliverable_service`` already maps .mp4/.mov/
.avi/.mkv/.webm to ``video``, but ``video`` was missing from
``AGENT_REGISTERABLE_ARTIFACT_TYPES``: both agent registration paths
(``workspace_write_file`` auto-register and a Claude Code session's files)
skipped every video. Pinned here:

* an MP4 registers as an agent Deliverable of type ``video`` — through
  ``DeliverableService.register`` itself and through both agent paths;
* every video extension is served as raw bytes by the ONE workspace file route
  the preview already uses for binary Deliverables (``/files/raw``), so the
  player gets bytes, never the JSON of ``/files/content``;
* ``get_deliverable(include_content=True)`` streams a video by URL like an
  image — it never reads the bytes inline (which would fail and replace the
  player with "Unable to load content").
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import quote
from uuid import UUID, uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import services.deliverable_service as deliverable_service  # noqa: E402
from services.deliverable_service import (  # noqa: E402
    AGENT_REGISTERABLE_ARTIFACT_TYPES,
    EXTENSION_TO_ARTIFACT,
    STREAMED_ARTIFACT_TYPES,
    DeliverableService,
    _infer_artifact_type,
    _workspace_file_url,
)

WS = "00000000-0000-0000-0000-0000000025a1"
VIDEO_EXTENSIONS = sorted(ext for ext, kind in EXTENSION_TO_ARTIFACT.items() if kind == "video")


def _raw_url(path: str) -> str:
    return f"/api/workspaces/{WS}/files/raw?path={quote(path)}"


class _RecordingService:
    """Stands in for DeliverableService on the agent paths: records register()."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    def register(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"success": True, "deliverable_id": "d-video-1", "created": True,
                "artifact_type": kwargs.get("artifact_type")}


@pytest.fixture
def recording_service(monkeypatch):
    _RecordingService.calls = []
    monkeypatch.setattr(deliverable_service, "DeliverableService", _RecordingService)
    return _RecordingService


# ── the type ─────────────────────────────────────────────────────────────────

def test_an_mp4_is_a_video_and_video_is_agent_registrable():
    assert _infer_artifact_type("renders/launch.mp4") == "video"
    assert "video" in AGENT_REGISTERABLE_ARTIFACT_TYPES
    assert {"mp4", "mov", "webm"} <= {ext.lstrip(".") for ext in VIDEO_EXTENSIONS}
    for ext in VIDEO_EXTENSIONS:
        assert _infer_artifact_type(f"clips/cut{ext}") in AGENT_REGISTERABLE_ARTIFACT_TYPES


def test_archives_and_audio_stay_out():
    assert "archive" not in AGENT_REGISTERABLE_ARTIFACT_TYPES
    assert "audio" not in AGENT_REGISTERABLE_ARTIFACT_TYPES


@pytest.mark.parametrize("ext", VIDEO_EXTENSIONS)
def test_every_video_extension_is_served_as_raw_bytes(ext):
    path = f"renders/launch{ext}"
    assert _workspace_file_url(WS, path) == _raw_url(path)
    upper = f"renders/LAUNCH{ext.upper()}"
    assert "/files/raw?" in _workspace_file_url(WS, upper)


def test_text_files_still_use_the_content_route():
    assert "/files/content?" in _workspace_file_url(WS, "notes/plan.md")
    assert "/files/content?" in _workspace_file_url(WS, "src/app.py")


# ── register() ───────────────────────────────────────────────────────────────

def test_registering_an_mp4_as_an_agent_video_deliverable_succeeds():
    db = MagicMock()
    new_id = uuid4()
    row = MagicMock()
    row.__getitem__ = lambda self, i: (new_id, True)[i]
    db.execute.return_value.fetchone.return_value = row

    out = DeliverableService(db, WS).register(
        file_path="renders/launch.mp4",
        source_type="chat",
        agent_id=7,
        agent_name="Studio",
        artifact_type="video",
        file_size_bytes=4_200_000,
    )

    assert out == {"success": True, "deliverable_id": str(new_id), "created": True,
                   "artifact_type": "video", "title": "Launch"}
    params = db.execute.call_args.args[1]
    assert params["artifact_type"] == "video"
    assert params["file_type"] == "mp4"
    assert params["preview_url"] == _raw_url("renders/launch.mp4")
    db.commit.assert_called_once()


def test_registering_without_a_type_infers_video_from_the_extension():
    db = MagicMock()
    row = MagicMock()
    row.__getitem__ = lambda self, i: (uuid4(), True)[i]
    db.execute.return_value.fetchone.return_value = row

    out = DeliverableService(db, WS).register(file_path="renders/teaser.mov", agent_id=7)

    assert out["success"] is True
    assert out["artifact_type"] == "video"
    assert db.execute.call_args.args[1]["preview_url"].startswith(f"/api/workspaces/{WS}/files/raw?")


# ── the two agent registration paths ─────────────────────────────────────────

def _load_exec_workspace():
    """Load exec_workspace by path — its own imports are stdlib-only, and the
    modules.tools package would pull the whole tool stack."""
    target = _ORCH / "modules" / "tools" / "execution" / "exec_workspace.py"
    spec = importlib.util.spec_from_file_location("prd251_exec_workspace_under_test", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_agent_writing_an_mp4_registers_a_video_deliverable(recording_service, monkeypatch):
    import core.database.database as database_mod

    monkeypatch.setattr(database_mod, "SessionLocal", lambda: contextlib.nullcontext(MagicMock()))
    exec_workspace = _load_exec_workspace()

    exec_workspace._auto_register_deliverable(
        workspace_id=WS,
        file_path="renders/launch.mp4",
        write_result={"success": True, "size": 4_200_000},
        agent_id=None,
        caller_context={"mission_id": "m-1"},
        trace_id="t-1",
    )

    assert len(recording_service.calls) == 1
    call = recording_service.calls[0]
    assert call["artifact_type"] == "video"
    assert call["file_path"] == "renders/launch.mp4"
    assert call["file_size_bytes"] == 4_200_000
    assert (call["source_type"], call["source_id"]) == ("mission", "m-1")


def test_a_claude_code_session_mp4_registers_a_video_deliverable(recording_service, monkeypatch, tmp_path):
    from services import cli_host_service

    monkeypatch.setattr(cli_host_service.config, "WORKSPACE_VOLUME_PATH", str(tmp_path), raising=False)
    video = tmp_path / WS / "renders" / "launch.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    task = SimpleNamespace(workspace_id=UUID(WS), id=68)

    registered = cli_host_service._register_session_deliverables(
        MagicMock(), task, [f"/host/workspaces/{WS}/renders/launch.mp4"],
        agent_id=7, agent_name="Studio", session_id="s-1",
    )

    assert registered == [{"id": "d-video-1", "file_path": "renders/launch.mp4",
                           "title": "launch.mp4", "artifact_type": "video"}]
    call = recording_service.calls[0]
    assert call["artifact_type"] == "video"
    assert call["file_size_bytes"] == video.stat().st_size
    assert (call["source_type"], call["source_id"]) == ("task", "68")


# ── the preview fetch: stream by URL, never inline ───────────────────────────

def _video_row(**overrides):
    from datetime import datetime, timezone

    values = {
        "id": uuid4(), "workspace_id": UUID(WS), "source_type": "chat", "source_id": None,
        "agent_id": 7, "agent_name": "Studio", "artifact_type": "video", "title": "Launch",
        "summary": None, "storage_type": "workspace", "file_path": "renders/launch.mp4",
        "file_name": "launch.mp4", "file_type": "mp4", "file_size_bytes": 4_200_000,
        "preview_url": None, "preview_type": "file", "extra": {}, "status": "ready",
        "created_at": datetime.now(timezone.utc), "updated_at": datetime.now(timezone.utc),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
@pytest.mark.parametrize("file_path", ["renders/launch.mp4", "renders/launch.mov", "renders/launch.mkv"])
async def test_a_video_streams_by_url_and_is_never_read_inline(file_path):
    row = _video_row(file_path=file_path, file_name=file_path.rsplit("/", 1)[-1])
    db = MagicMock()
    db.execute.return_value.fetchone.return_value = row

    with patch("services.deliverable_service.WorkspaceClient") as workspace_client:
        workspace_client.return_value.read_file = AsyncMock(return_value={"success": False, "error": "binary"})
        out = await DeliverableService(db, WS).get_deliverable(str(row.id), include_content=True)
        workspace_client.assert_not_called()

    deliverable = out["deliverable"]
    assert out["success"] is True
    assert deliverable["content"] is None
    assert "content_error" not in deliverable
    assert deliverable["content_url"] == _raw_url(file_path)
    assert deliverable["preview_url"] == deliverable["content_url"]


def test_images_and_videos_are_the_streamed_types():
    assert STREAMED_ARTIFACT_TYPES == frozenset({"image", "video"})
