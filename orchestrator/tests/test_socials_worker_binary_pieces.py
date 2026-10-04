"""A binary file reaches the workspace a piece at a time (3 Oct 2026, the Socials copies).

The worker's ``/files/write`` takes text, under aiohttp's 1 MB body limit. A picture or a
video now goes through the same route as base64 pieces (``executor._write_chunk``):

* pieces are written to ``<path>.part`` (the first over it, the rest after it) and the
  last one moves it to the path, so the Explorer never shows a file half-written;
* both paths stay inside the workspace, a piece that is not base64 is refused, and a
  file may not grow past ``WORKER_MAX_BINARY_WRITE_BYTES``;
* every workspace has ``socials/images`` and ``socials/videos``;
* ``WorkspaceClient.write_binary`` cuts any stream into pieces of ``BINARY_PIECE_BYTES``
  and stops at the first refusal.
"""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core import workspace_client  # noqa: E402
from tests.helpers_workspace_worker import answer, worker_server  # noqa: E402

WS = str(uuid.UUID("00000000-0000-0000-0000-0000000000c1"))
VIDEO = bytes(range(256)) * 40  # 10 KB of every byte value


def _piece(data, append, rename_to=None):
    piece = {"base64": base64.b64encode(data).decode(), "append": append}
    return {**piece, "rename_to": rename_to} if rename_to else piece


async def _write(http, base, path, content):
    async with http.post(f"{base}/workspaces/{WS}/files/write", json={"path": path, "content": content}) as resp:
        return await answer(resp)


def test_pieces_land_whole_at_the_path_and_never_half_written(monkeypatch, tmp_path):
    async def run():
        async with worker_server(monkeypatch, tmp_path) as (base, http):
            part, final = "socials/videos/clip.mp4.part", "socials/videos/clip.mp4"
            assert (await _write(http, base, part, _piece(VIDEO[:4000], False)))[0] == 200
            assert (await _write(http, base, part, _piece(VIDEO[4000:8000], True)))[0] == 200
            assert not (tmp_path / WS / final).exists()  # not there until the last piece
            status, body = await _write(http, base, part, _piece(VIDEO[8000:], True, rename_to=final))
            return status, body

    status, body = asyncio.run(run())
    assert status == 200 and body["path"] == "socials/videos/clip.mp4" and body["size_bytes"] == len(VIDEO)
    assert (tmp_path / WS / "socials/videos/clip.mp4").read_bytes() == VIDEO
    assert not (tmp_path / WS / "socials/videos/clip.mp4.part").exists()
    assert (tmp_path / WS / "socials/images").is_dir()  # both folders come with the workspace


@pytest.mark.parametrize(
    "path, content, refused",
    [
        ("../outside.png", _piece(b"x", False), "outside"),
        ("socials/images/a.png.part", _piece(b"x", False, rename_to="../../escape.png"), "outside"),
        ("socials/images/a.png", {"base64": "not base64!", "append": False}, "Not a base64 piece"),
        ("socials/images/big.png", _piece(b"x" * 101, False), "File too large"),
    ],
)
def test_a_piece_outside_the_workspace_not_base64_or_too_big_is_refused(monkeypatch, tmp_path, path, content, refused):
    monkeypatch.setenv("WORKER_MAX_BINARY_WRITE_BYTES", "100")

    async def run():
        async with worker_server(monkeypatch, tmp_path) as (base, http):
            return await _write(http, base, path, content)

    status, body = asyncio.run(run())
    assert status == 400 and refused.lower() in body["error"].lower()
    assert not (tmp_path / "escape.png").exists() and not (tmp_path / "outside.png").exists()


def test_text_still_writes_as_text(monkeypatch, tmp_path):
    async def run():
        async with worker_server(monkeypatch, tmp_path) as (base, http):
            return await _write(http, base, "reports/note.md", "# Hello")

    assert asyncio.run(run())[0] == 200
    assert (tmp_path / WS / "reports/note.md").read_text() == "# Hello"


class _Recorder:
    """``WorkspaceClient.write_file`` recorded: each piece decoded, the third refused when asked."""

    def __init__(self, refuse_at=None):
        self.calls, self.refuse_at = [], refuse_at

    async def __call__(self, path, content):  # set on the class, it is called unbound: (path, content)
        self.calls.append((path, base64.b64decode(content["base64"]), content["append"], content.get("rename_to")))
        if self.refuse_at == len(self.calls):
            return {"success": False, "error": "disk full"}
        return {"success": True, "path": content.get("rename_to") or path}


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk


def test_the_client_cuts_any_stream_into_pieces_and_renames_on_the_last(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(workspace_client, "BINARY_PIECE_BYTES", 4)
    monkeypatch.setattr(workspace_client.WorkspaceClient, "write_file", recorder)
    client = workspace_client.WorkspaceClient(WS)

    result = asyncio.run(client.write_binary("socials/images/a.png", _stream(b"abc", b"defgh", b"ij")))

    assert result == {"success": True, "path": "socials/images/a.png"}
    part = "socials/images/a.png.part"
    assert recorder.calls == [
        (part, b"abcd", False, None), (part, b"efgh", True, None), (part, b"ij", True, "socials/images/a.png"),
    ]


def test_the_client_stops_at_the_first_refusal(monkeypatch):
    recorder = _Recorder(refuse_at=2)
    monkeypatch.setattr(workspace_client, "BINARY_PIECE_BYTES", 4)
    monkeypatch.setattr(workspace_client.WorkspaceClient, "write_file", recorder)

    result = asyncio.run(workspace_client.WorkspaceClient(WS).write_binary("v.mp4", _stream(b"x" * 13)))

    assert result == {"success": False, "error": "disk full"} and len(recorder.calls) == 2
