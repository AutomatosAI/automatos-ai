"""F105 (4 Nov): a rendered Socials file hands its request's connection back before it streams.

``GET /api/socials/posts/{id}/media/{file}`` loads the caller's post on the request's
``get_db`` session, then streams the file from storage. FastAPI closes that session (the
Socials gate's too) only when the response ends, so a long video held a pooled
connection idle in transaction for the whole download, and the lookup ran on the event
loop. The route is now a plain ``def`` (FastAPI's threadpool) and ends the transaction
once the post is loaded, before storage is opened. The response is unchanged.

DB-free: no database, storage or socket.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from fastapi import HTTPException

from api import socials
from modules.socials import media_store

WS = uuid4()
POST_ID = uuid4()
FILE = "video-9x16.mp4"
BYTES = b"\x00\x00\x00\x18ftypmp42"


class _RequestSession:
    def __init__(self, events):
        self._events = events

    def rollback(self):
        self._events.append("request transaction ended")


def _patch(monkeypatch, events):
    def _load(db, ctx, post_id):
        events.append("post loaded")
        return NS(workspace_id=WS, id=post_id)

    class _Store:
        def open(self, key):
            events.append(f"storage opened {key}")
            return NS(body=iter([BYTES]), content_type="video/mp4", content_length=len(BYTES))

    monkeypatch.setattr(socials, "_load", _load)
    monkeypatch.setattr(media_store, "MediaStore", _Store)


def test_the_request_transaction_ends_before_the_file_streams(monkeypatch):
    events = []
    _patch(monkeypatch, events)

    response = socials.get_social_post_media(
        post_id=POST_ID, file_name=FILE, db=_RequestSession(events), ctx=NS(workspace_id=WS),
    )

    key = media_store.media_key(WS, POST_ID, FILE)
    assert events == ["post loaded", "request transaction ended", f"storage opened {key}"]
    assert response.media_type == "video/mp4"
    assert response.headers["content-length"] == str(len(BYTES))
    assert response.headers["content-disposition"] == f'inline; filename="{FILE}"'
    assert response.headers["cache-control"] == socials.MEDIA_CACHE_CONTROL


def test_a_bad_file_name_is_still_not_found_before_storage_is_touched(monkeypatch):
    events = []
    _patch(monkeypatch, events)

    with pytest.raises(HTTPException) as missing:
        socials.get_social_post_media(post_id=POST_ID, file_name="..mp4", db=_RequestSession(events),
                                      ctx=NS(workspace_id=WS))

    assert missing.value.status_code == 404 and missing.value.detail == socials.MEDIA_NOT_FOUND
    assert events == ["post loaded"]


def test_the_media_route_runs_in_the_threadpool():
    assert not inspect.iscoroutinefunction(socials.get_social_post_media)
