"""Approved posts in the Workspace Explorer (3 Oct 2026, Gerard: "have a socials directory").

A post's files live in object storage (``media_store``), where publishing reads them, and
are Deliverables there (``storage_type='s3'``), so the Workspace Explorer, which shows the
workspace's own folders, never listed them. When a post is approved, the files it will
publish (each rendered size, or the person's own picture or video) are copied into the
workspace too:

* ``socials/videos/`` for a video post, ``socials/images/`` for every other post;
* each named ``<day>-<title>-<post id>-<file name>``: the day it is planned for (else the
  day it was approved), so a folder sorts by date, and a second approval of the same post
  writes the same names over the first.

Locally the workspace root IS the deliverables folder (#722), so the copies carry no
workspace id: ``~/Development/deliverables/socials/images/...``. The worker creates both
folders with every workspace (``DEFAULT_SUBDIRS``).

A copy is a convenience: it never holds up or undoes an approval. The files are listed in
the approval's request; the copying runs in the background, each file streamed from
storage to the worker (``WorkspaceClient.write_binary``); a file that cannot be copied is
logged and skipped. The copies are not registered again as Deliverables: the stored ones
already are, and a second row would show every post twice.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import anyio

from core.storage import is_storage_configured
from core.utils.background_tasks import launch_guarded
from core.workspace_client import WorkspaceClient
from modules.socials import media_store
from modules.socials.media_urls import MediaFile, resolve_post_media

logger = logging.getLogger(__name__)

IMAGES_FOLDER = "socials/images"
VIDEOS_FOLDER = "socials/videos"
VIDEO_FORMAT = "video"
TITLE_MAX_CHARS = 60
TITLE_FALLBACK = "post"
POST_ID_CHARS = 8
_NOT_A_NAME = re.compile(r"[^a-z0-9]+")
_NOT_A_FILE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

Plan = Sequence[Tuple[str, str]]  # (storage key, workspace path)


def _slug(title: Any) -> str:
    slug = _NOT_A_NAME.sub("-", str(title or "").lower()).strip("-")
    return slug[:TITLE_MAX_CHARS].strip("-") or TITLE_FALLBACK


def _day(post: Any, now: datetime) -> date:
    """The day the post goes out (its schedule, else its slot), else ``now``, in its timezone."""
    try:
        zone = ZoneInfo(post.timezone or "UTC")
    except (ZoneInfoNotFoundError, ValueError):
        zone = ZoneInfo("UTC")
    when = post.scheduled_for or post.planned_for or now
    return when.astimezone(zone).date()


def copy_path(post: Any, media: MediaFile, day: date) -> str:
    """Where ``media`` of ``post`` goes in the workspace (the module docstring)."""
    folder = VIDEOS_FOLDER if post.format == VIDEO_FORMAT else IMAGES_FOLDER
    name = _NOT_A_FILE_NAME.sub("-", (media.name or "").rsplit("/", 1)[-1]).strip("-") or "file"
    return f"{folder}/{day.isoformat()}-{_slug(post.title)}-{str(post.id)[:POST_ID_CHARS]}-{name}"


def copy_plan(post: Any, files: Sequence[MediaFile], now: datetime) -> List[Tuple[str, str]]:
    """``(storage key, workspace path)`` for each file of ``post`` that has a stored object."""
    day = _day(post, now)
    return [(media.key, copy_path(post, media, day)) for media in files if media.key and not media.error]


async def _pieces(body: Iterator[bytes]) -> AsyncIterator[bytes]:
    """A stored object's chunks, each read off the event loop (the storage client blocks)."""
    while True:
        piece = await asyncio.to_thread(next, body, None)
        if piece is None:
            return
        yield piece


async def copy_files(workspace_id: Any, plan: Plan) -> List[str]:
    """Stream each ``(key, path)`` of ``plan`` from storage into the workspace; the paths written."""
    store = media_store.MediaStore()
    client = WorkspaceClient(str(workspace_id))
    written: List[str] = []
    for key, path in plan:
        stored = await asyncio.to_thread(store.open, key)
        if stored is None:
            logger.warning("[Socials] %s is not in storage, so it was not copied to %s", key, path)
            continue
        result = await client.write_binary(path, _pieces(stored.body))
        if result.get("success"):
            written.append(path)
        else:
            logger.warning("[Socials] %s could not be copied into the workspace: %s", path, result.get("error"))
    return written


def _start(workspace_id: Any, plan: Plan) -> None:
    """Launch ``copy_files`` on the event loop, from a route on the loop or one in the threadpool."""
    launch = functools.partial(launch_guarded, subsystem="socials", operation="workspace_copy", workspace_id=workspace_id)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        anyio.from_thread.run_sync(lambda: launch(copy_files(workspace_id, plan)))
        return
    launch(copy_files(workspace_id, plan))


def copy_when_approved(db: Any, post: Any, now: Optional[datetime] = None) -> None:
    """Start copying the approved ``post``'s files into the workspace. Never raises: a copy
    that cannot be planned or started is logged, and the approval stands."""
    try:
        if not post.media or not is_storage_configured():
            return
        plan = copy_plan(post, resolve_post_media(db, post), now or datetime.now(timezone.utc))
        if plan:
            _start(post.workspace_id, plan)
    except Exception:
        db.rollback()
        logger.exception("[Socials] the files of post %s could not be copied to the workspace", post.id)
