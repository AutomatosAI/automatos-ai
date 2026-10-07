"""7 Oct 2026: a post's media reaches Composio under the canonical tool slug.

Composio's files API (``files.create_presigned_url``) stopped accepting the
hyphenated tool slug the platform built from the action
(``instagram-post-ig-user-media``): 400 "Invalid tool_slug" (code 4802). Every
Instagram and X post with media failed, the 06:00 scheduled post and a manual
publish-now among them. The canonical action slug (``INSTAGRAM_POST_IG_USER_MEDIA``)
is accepted. Both places that stage a file for Composio, the publisher's
``upload_spec`` and the executor's own upload conversion, pass it.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.composio import tool_executor, upload_spec

ACTIONS = ("INSTAGRAM_POST_IG_USER_MEDIA", "TWITTER_UPLOAD_MEDIA", "LINKEDIN_UPLOAD_VIDEO")


@pytest.mark.parametrize("action", ACTIONS)
def test_the_tool_slug_is_the_canonical_action_slug(action):
    assert upload_spec.composio_tool_slug(action) == action
    assert upload_spec.composio_tool_slug(action.lower()) == action
    assert upload_spec.composio_tool_slug(f" {action.lower().replace('_', '-')} ") == action  # the old hyphenated form


@pytest.mark.parametrize("action,toolkit", [("INSTAGRAM_POST_IG_USER_MEDIA", "instagram"), ("TWITTER_UPLOAD_MEDIA", "twitter")])
def test_the_publisher_uploads_a_staged_file_under_the_canonical_slug(monkeypatch, tmp_path, action, toolkit):
    uploaded = []

    class Uploadable:
        @staticmethod
        def from_path(**kwargs):
            uploaded.append(kwargs)
            return SimpleNamespace(model_dump=lambda: {"s3key": kwargs["file"].name})

    monkeypatch.setattr(upload_spec, "_file_uploadable_class", lambda: Uploadable)
    monkeypatch.setattr(upload_spec, "get_composio_client", lambda: SimpleNamespace(composio=SimpleNamespace(client="http")))
    staged = tmp_path / "card.png"
    staged.write_bytes(b"png")

    upload_spec.resolve_upload_spec(action, {"media": staged}, ["media"], toolkit)

    (call,) = uploaded
    assert call["tool"] == action and call["toolkit"] == toolkit


def test_the_executors_own_upload_conversion_uses_the_same_slug():
    source = inspect.getsource(tool_executor._resolve_single_file_standalone)
    assert source.count("tool=upload_spec.composio_tool_slug(action_slug)") == 2  # from_url and from_path
