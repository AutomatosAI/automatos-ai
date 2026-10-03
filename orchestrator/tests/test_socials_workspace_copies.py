"""Approved posts in the Workspace Explorer (3 Oct 2026): ``modules/socials/workspace_copies.py``.

* A video post's files go to ``socials/videos/``, every other post's to ``socials/images/``,
  named ``<day>-<title>-<post id>-<file name>``: the day it goes out (its schedule, else
  its slot) in its own timezone, else the day it was approved; a file with no stored
  object, or an error, is not planned.
* The copy streams each stored object to the worker; a missing object or a refused write
  is logged and skipped, the rest still copied.
* Planning never fails an approval: storage off or no media plans nothing; a listing that
  fails rolls back and is logged.
* The copy runs in the background from a route on the event loop and from one in the
  threadpool.
* Approving a post starts its copy; a series approval starts one for each post it approved,
  never for a post it left.
"""
from __future__ import annotations

import asyncio
import os
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from modules.socials import workspace_copies  # noqa: E402
from modules.socials.media_urls import MediaFile  # noqa: E402
from tests.test_prd251_api import _approved, _create, _post  # noqa: E402
from tests.test_prd251w2_series_approval import _approve, _campaign, _series_switch, _shown, _waiting  # noqa: E402

api = api_harness.api
POST_ID = uuid.UUID("5153d784-d247-4e99-9df6-c48e7be916ea")
WS = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
NOW = datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)


def _post_row(**over):
    row = {"id": POST_ID, "workspace_id": WS, "title": "Autumn colour week!", "format": "image", "timezone": "Europe/Lisbon",
           "scheduled_for": None, "planned_for": None, "media": {"4:5": [{"name": "image-4x5.png"}]}}
    return SimpleNamespace(**{**row, **over})


def _file(name="image-4x5.png", key="social-media/ws/post/image-4x5.png", error=None):
    return MediaFile(aspect="4:5", deliverable_id="d1", name=name, key=key, error=error)


# ---------------------------------------------------------------------------
# Where a file goes
# ---------------------------------------------------------------------------


def test_a_picture_goes_to_socials_images_named_by_day_title_and_post():
    assert workspace_copies.copy_path(_post_row(), _file(), date(2026, 10, 5)) == \
        "socials/images/2026-10-05-autumn-colour-week-5153d784-image-4x5.png"


def test_a_video_goes_to_socials_videos_and_odd_names_are_made_safe():
    post = _post_row(format="video", title="  ")
    assert workspace_copies.copy_path(post, _file(name="../9x16 clip.mp4"), date(2026, 10, 5)) == \
        "socials/videos/2026-10-05-post-5153d784-9x16-clip.mp4"


def test_the_day_is_when_it_goes_out_in_its_own_timezone_else_today():
    late = datetime(2026, 10, 5, 23, 30, tzinfo=timezone.utc)  # already the 6th in Lisbon (UTC+1)
    assert workspace_copies.copy_plan(_post_row(planned_for=late), [_file()], NOW)[0][1].startswith("socials/images/2026-10-06-")
    assert workspace_copies.copy_plan(_post_row(scheduled_for=late, planned_for=NOW), [_file()], NOW)[0][1].startswith(
        "socials/images/2026-10-06-")
    assert workspace_copies.copy_plan(_post_row(timezone="Mars/Olympus"), [_file()], NOW)[0][1].startswith(
        "socials/images/2026-10-03-")


def test_only_stored_files_are_planned():
    files = [_file(), _file(name="gone.png", key=None, error="its file is not in storage"), _file(name="x.png", error="moved")]
    assert workspace_copies.copy_plan(_post_row(), files, NOW) == [
        ("social-media/ws/post/image-4x5.png", "socials/images/2026-10-03-autumn-colour-week-5153d784-image-4x5.png"),
    ]


# ---------------------------------------------------------------------------
# The copy
# ---------------------------------------------------------------------------


class _Store:
    def __init__(self, objects):
        self.objects = objects

    def open(self, key):
        chunks = self.objects.get(key)
        return None if chunks is None else SimpleNamespace(body=iter(chunks), content_type="image/png", content_length=0)


class _Worker:
    written = {}
    refuse = set()

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def write_binary(self, path, pieces):
        data = b"".join([piece async for piece in pieces])
        if path in self.refuse:
            return {"success": False, "error": "disk full"}
        self.written[(self.workspace_id, path)] = data
        return {"success": True, "path": path}


@pytest.fixture
def worker(monkeypatch):
    _Worker.written, _Worker.refuse = {}, set()
    monkeypatch.setattr(workspace_copies, "WorkspaceClient", _Worker)
    return _Worker


def test_each_file_is_streamed_into_the_workspace_and_a_missing_or_refused_one_is_skipped(worker, monkeypatch, caplog):
    objects = {"k/a.png": [b"PNG", b"-bytes"], "k/b.png": [b"other"], "k/c.png": [b"refused"]}
    monkeypatch.setattr(workspace_copies.media_store, "MediaStore", lambda: _Store(objects))
    worker.refuse = {"socials/images/c.png"}
    plan = [("k/a.png", "socials/images/a.png"), ("k/gone.png", "socials/images/gone.png"),
            ("k/b.png", "socials/images/b.png"), ("k/c.png", "socials/images/c.png")]

    written = asyncio.run(workspace_copies.copy_files(WS, plan))

    assert written == ["socials/images/a.png", "socials/images/b.png"]
    assert worker.written == {(str(WS), "socials/images/a.png"): b"PNG-bytes", (str(WS), "socials/images/b.png"): b"other"}
    assert "k/gone.png is not in storage" in caplog.text and "disk full" in caplog.text


# ---------------------------------------------------------------------------
# Never in the approval's way
# ---------------------------------------------------------------------------


class _Db:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(workspace_copies, "_start", lambda workspace_id, plan: calls.append((workspace_id, list(plan))))
    monkeypatch.setattr(workspace_copies, "is_storage_configured", lambda: True)
    return calls


def test_a_post_with_files_starts_its_copy(started, monkeypatch):
    monkeypatch.setattr(workspace_copies, "resolve_post_media", lambda db, post: [_file()])
    workspace_copies.copy_when_approved(_Db(), _post_row(), now=NOW)
    assert started == [(WS, [("social-media/ws/post/image-4x5.png",
                              "socials/images/2026-10-03-autumn-colour-week-5153d784-image-4x5.png")])]


def test_no_storage_or_no_media_plans_nothing_and_a_failed_listing_is_logged(started, monkeypatch, caplog):
    workspace_copies.copy_when_approved(_Db(), _post_row(media={}), now=NOW)
    monkeypatch.setattr(workspace_copies, "is_storage_configured", lambda: False)
    workspace_copies.copy_when_approved(_Db(), _post_row(), now=NOW)
    assert started == []

    monkeypatch.setattr(workspace_copies, "is_storage_configured", lambda: True)

    def broken(db, post):
        raise RuntimeError("deliverables table is gone")

    monkeypatch.setattr(workspace_copies, "resolve_post_media", broken)
    db = _Db()
    workspace_copies.copy_when_approved(db, _post_row(), now=NOW)  # never raises
    assert started == [] and db.rollbacks == 1
    assert "could not be copied to the workspace" in caplog.text


@pytest.fixture
def launched(monkeypatch):
    calls = []

    def launch(coro, *, subsystem, operation, workspace_id):
        coro.close()  # a coroutine object; the copy itself is tested above
        calls.append((subsystem, operation, workspace_id))

    monkeypatch.setattr(workspace_copies, "launch_guarded", launch)
    return calls


def test_a_broken_storage_check_or_launch_never_fails_the_approval(started, monkeypatch, caplog):
    """CI found it: a storage check outside the guard would have answered the approve with a 500."""
    def broken():
        raise AttributeError("the storage check is gone")

    monkeypatch.setattr(workspace_copies, "is_storage_configured", broken)
    db = _Db()
    workspace_copies.copy_when_approved(db, _post_row(), now=NOW)  # never raises
    assert started == [] and db.rollbacks == 1 and "could not be copied to the workspace" in caplog.text


def test_the_copy_runs_in_the_background_from_the_loop_and_from_the_threadpool(launched):
    plan = [("k", "socials/images/a.png")]

    async def from_a_route_on_the_loop():
        workspace_copies._start(WS, plan)

    async def from_a_route_in_the_threadpool():
        await anyio.to_thread.run_sync(workspace_copies._start, WS, plan)

    asyncio.run(from_a_route_on_the_loop())
    anyio.run(from_a_route_in_the_threadpool)
    assert launched == [("socials", "workspace_copy", WS)] * 2


# ---------------------------------------------------------------------------
# The approvals start it
# ---------------------------------------------------------------------------


@pytest.fixture
def copies(monkeypatch):
    seen = []
    monkeypatch.setattr(workspace_copies, "copy_when_approved", lambda db, post: seen.append(str(post.id)))
    return seen


def test_approving_a_post_starts_its_copy(api, copies):
    post = _approved(api)
    assert copies == [post["id"]]

    other = _create(api, title="Not approved")
    _post(api, other["id"], "submit")
    assert _post(api, other["id"], "approve", {"content_hash": "0" * 64}).status_code == 409
    assert copies == [post["id"]]  # a refused approval copies nothing


def test_a_series_approval_copies_the_posts_it_approved_never_one_it_left(api, copies):
    _series_switch(api, True)
    campaign = _campaign(api)
    first = _waiting(api, campaign["id"], "Three weeks to go")
    second = _waiting(api, campaign["id"], "Two weeks to go")
    changed = {**second, "content_hash": "0" * 64}  # the approver saw an older version: left

    resp = _approve(api, campaign["id"], _shown(first, changed))

    assert resp.status_code == 200, resp.text
    assert [p["id"] for p in resp.json()["approved"]] == [first["id"]]
    assert copies == [first["id"]]
