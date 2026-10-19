"""P256-FIX-SOC-4 (F431-b, 14 Oct): an agent's Socials tools give a post its channels.

Post 7e7c402c, which Auto edited and submitted, ended with no channels: a post's targets were
written only by PUT /posts/{id}/targets and the plan maker, and the create and update tools had no
field for them, so a post Auto made published nowhere. Auto wrote copy.channels.instagram, which is
the post's text for Instagram, not a channel.

Now both tools take ``channels`` (["instagram"]): each connected channel becomes a target with the
kind the post's format posts as there (the plan maker's choice), checked against the workspace's
channels before anything is staged. Create writes the post and its channels in one commit, so its
render sizes for them; update carries them in the same compare-and-set as the rest of the edit. A
channel the workspace has not connected, or one that posts nothing of the format, is refused saying
what it has, and nothing is saved. REST keeps refusing ``channels``: PUT /targets sets them there.
"""
from __future__ import annotations

import pytest

import api.socials as socials_api
import api.socials_targets as socials_targets
from modules.socials import service
from modules.socials import targets as post_targets
from modules.socials.capabilities import ChannelKind, SocialChannel
from tests.test_prd251w1_social_post_tools import WS, _created, _draft_fields, _posts, _row, _template, _tool
from tests.test_prd251w1_social_post_tools import env as _env

env = _env  # the SQLite Socials harness


def _kind(kind):
    return ChannelKind(kind=kind, available=True, reason=None, needs_public_storage=False, steps=())


CHANNELS = [
    SocialChannel(toolkit="instagram", label="Instagram", verified=True, setup_note=None,
                  post_kinds=tuple(_kind(k) for k in ("image", "carousel", "reel", "story"))),
    SocialChannel(toolkit="linkedin", label="LinkedIn", verified=True, setup_note=None,
                  post_kinds=(_kind("text"), _kind("image"))),
]


@pytest.fixture
def channels(env, monkeypatch):
    """The workspace's connected channels; the targets route is never what sets an agent's."""
    read = []

    def _channels(db, workspace_id):
        read.append(workspace_id)
        return CHANNELS

    def _not_this_way(*a, **k):
        raise AssertionError("an agent's channels are written with the post, not by a second commit")

    monkeypatch.setattr(socials_targets, "social_channels", _channels)
    monkeypatch.setattr(socials_targets, "set_post_targets", _not_this_way)
    env.channel_reads = read
    return env


def _targets(row):
    return [(t.toolkit, t.post_kind, t.status) for t in sorted(row.targets, key=lambda t: t.toolkit)]


# ── create ───────────────────────────────────────────────────────────────────────────

def test_a_carousel_made_with_instagram_has_its_channel_from_the_start(channels):
    made = _created(_tool(channels, "platform_create_social_post", **_draft_fields(format="carousel", variables={},
                                                                                     sources={}),
                          channels=["instagram"], render=False))

    row = _row(channels, made["id"])
    assert _targets(row) == [("instagram", "carousel", "pending")]
    assert post_targets.target_set(row) == [{"toolkit": "instagram", "post_kind": "carousel", "options": {}}]
    assert row.content_hash == service.compute_content_hash(row)
    assert [entry["action"] for entry in row.review_log] == ["draft"]          # one write: no edit after it
    assert channels.channel_reads == [WS]                                        # the registry, read once


def test_a_rendered_post_reaches_its_render_with_its_channels(channels, monkeypatch):
    seen = []

    async def _render(db, workspace, post, actor):
        seen.append([(t.toolkit, t.post_kind) for t in post.targets])
        return post.to_dict()

    monkeypatch.setattr(socials_api, "render_post", _render)
    _template(channels)

    result = _tool(channels, "platform_create_social_post", **_draft_fields(), template="Countdown",
                   channels=["Instagram"])                                       # a channel's name, any case

    assert result["success"] is True, result
    assert seen == [[("instagram", "reel")]]                                     # a video posts as a reel there


def test_a_channel_that_is_not_connected_is_refused_and_nothing_is_saved(channels):
    result = _tool(channels, "platform_create_social_post", **_draft_fields(), channels=["tiktok"], render=False)

    assert result["success"] is False
    assert "tiktok is not a channel connected in this workspace" in result["error"]
    assert "the connected channels are instagram, linkedin" in result["error"]
    assert _posts(channels) == 0


def test_a_channel_that_posts_nothing_of_the_format_is_refused(channels):
    result = _tool(channels, "platform_create_social_post", **_draft_fields(format="text", variables={}, sources={}),
                   channels=["instagram"], render=False)

    assert result["success"] is False
    assert result["error"] == "Instagram does not post a text: it posts image, carousel, reel, story"
    assert _posts(channels) == 0


# ── update ───────────────────────────────────────────────────────────────────────────

def _waiting(env):
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), render=False))
    assert _tool(env, "platform_submit_social_post", post_id=post["id"])["success"] is True
    return _row(env, post["id"])


def test_channels_alone_change_a_waiting_post(channels):
    before = _waiting(channels)
    hash_before = before.content_hash

    result = _tool(channels, "platform_update_social_post", post_id=str(before.id), channels=["instagram"])

    assert result["success"] is True, result
    row = _row(channels, before.id)
    assert _targets(row) == [("instagram", "reel", "pending")]                   # a video posts as a reel there
    assert row.status == "needs_approval" and row.content_hash != hash_before
    assert row.content_hash == service.compute_content_hash(row)


def test_an_empty_list_clears_the_channels(channels):
    post = _created(_tool(channels, "platform_create_social_post", **_draft_fields(), channels=["instagram"],
                          render=False))

    assert _tool(channels, "platform_update_social_post", post_id=post["id"], channels=[])["success"] is True
    assert _row(channels, post["id"]).targets == []


def test_an_update_naming_a_channel_that_is_not_connected_changes_nothing(channels):
    before = _waiting(channels)
    hash_before, targets_before = before.content_hash, _targets(before)

    result = _tool(channels, "platform_update_social_post", post_id=str(before.id), channels=["tiktok"],
                   title="A new title")

    assert result["success"] is False and "the connected channels are instagram, linkedin" in result["error"]
    row = _row(channels, before.id)
    assert (row.content_hash, _targets(row), row.title) == (hash_before, targets_before, before.title)


def test_rest_keeps_refusing_channels(channels):
    rest = channels.client.post("/api/socials/posts", json={**_draft_fields(), "channels": ["instagram"]})

    assert rest.status_code == 422 and "channels" in rest.text
