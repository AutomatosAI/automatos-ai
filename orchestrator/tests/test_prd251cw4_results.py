"""PRD-251C Wave 4, US-C402 — reading results.

Pinned:

* **The reads** (Composio's answers mocked): X's public metrics, Instagram's insights (a
  story's own metrics), LinkedIn's reactions, YouTube's statistics (given as text); a number
  the platform does not give is left out, never a zero; TikTok has no read.
* **The job** (the S0.3b API harness, SQLite): a published target is read 1 day and 7 days
  after it went out, once each, kept in ``social_post_stats``; a target that never went out,
  or whose window has passed, is not read; a toolkit the workspace cannot read now
  (not connected) is skipped and nothing is kept, so the next tick tries again; a workspace
  that takes no background reads is skipped; a read the platform refuses is kept with no
  numbers and not tried again.
* **The registry** says why a read cannot run: not connected, deny-listed, or not synced.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio
import pytest
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget  # noqa: E402
from modules.socials import capabilities, result_reads  # noqa: E402
from services import socials_results  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251w3_publisher import FakeExecutor, ok, refused  # noqa: E402

api = api_harness.api
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)
TWEET = {"data": {"id": "1850", "public_metrics": {"like_count": 12, "retweet_count": 3, "reply_count": 2, "quote_count": 0,
                                                   "impression_count": 940}}}


# ── the reads ──────────────────────────────────────────────────────────────


def test_each_channels_numbers_come_from_its_own_answer():
    reads = result_reads.READS
    assert result_reads.numbers_from(reads["twitter"], TWEET) == {"views": 940, "likes": 12, "reposts": 3, "replies": 2, "quotes": 0}
    insights = {"data": [{"name": "reach", "values": [{"value": 410}]}, {"name": "likes", "total_value": {"value": 31}},
                         {"name": "saved", "values": [{"value": 4}]}]}
    assert result_reads.numbers_from(reads["instagram"], insights) == {"reach": 410, "likes": 31, "saves": 4}
    assert result_reads.numbers_from(reads["linkedin"], {"paging": {"total": 18, "count": 1}, "elements": []}) == {"reactions": 18}
    video = {"items": [{"id": "abc", "statistics": {"viewCount": "1203", "likeCount": "88", "commentCount": "5"}}]}
    assert result_reads.numbers_from(reads["youtube"], video) == {"views": 1203, "likes": 88, "comments": 5}
    assert "tiktok" not in reads  # Composio's TikTok toolkit gives no video statistics


def test_a_number_the_platform_does_not_give_is_left_out():
    partial = {"data": {"public_metrics": {"like_count": 5, "impression_count": None, "retweet_count": "n/a"}}}
    assert result_reads.numbers_from(result_reads.READS["twitter"], partial) == {"likes": 5}
    assert result_reads.numbers_from(result_reads.READS["instagram"], {"error": "unsupported metric"}) == {}
    assert result_reads.engagement({"likes": 5, "views": 900, "comments": 2}) == 7  # views are not engagement


def test_the_params_name_the_target_and_a_story_asks_its_own_metrics():
    reads = result_reads.READS
    assert result_reads.params_for(reads["twitter"], "1850", "image") == {"id": "1850", "tweet_fields": ["public_metrics"]}
    assert result_reads.params_for(reads["youtube"], "abc", "short") == {"id": ["abc"], "parts": ["statistics"]}
    story = result_reads.params_for(reads["instagram"], "1790", "story")
    assert story == {"ig_media_id": "1790", "metric": ["views", "reach", "replies", "shares"]}
    assert "saved" in result_reads.params_for(reads["instagram"], "1790", "image")["metric"]


# ── the job ────────────────────────────────────────────────────────────────


@pytest.fixture
def results(api, monkeypatch):
    engine = api.session.get_bind()
    SocialPost.metadata.create_all(engine, tables=[SocialPostStat.__table__])
    factory = sessionmaker(bind=engine)
    readable = {"twitter": None, "instagram": None, "linkedin": None, "youtube": None}
    monkeypatch.setattr(socials_results, "_session", factory)
    monkeypatch.setattr(socials_results, "background_allowed", lambda workspace: workspace is not None)
    monkeypatch.setattr(socials_results, "runnable_actions", lambda db, ws, actions: {t: readable.get(t) for t in actions})
    api.readable, api.factory = readable, factory
    return api


def _went_out(api, toolkit="twitter", *, ago=timedelta(days=1, hours=2), status="published", remote_id="1850"):
    post = SocialPost(id=uuid.uuid4(), workspace_id=WS_A, created_by="u", title="Went out", content_hash="0" * 64, status="published")
    target = SocialPostTarget(
        id=uuid.uuid4(), post_id=post.id, toolkit=toolkit, post_kind="image", action_plan={}, idempotency_key=f"sp:{post.id}:{toolkit}",
        status=status, remote_id=remote_id if status == "published" else None, published_at=NOW - ago if status == "published" else None,
    )
    api.session.add_all([post, target])
    api.session.commit()
    return target


def _run(executor, now=NOW):
    return anyio.run(socials_results.run_reads, now, executor)


def _kept(api):
    api.session.expire_all()
    return [(str(row.target_id), row.reading, row.numbers, row.source_action) for row in api.session.query(SocialPostStat).all()]


def test_a_target_is_read_a_day_after_it_went_out_and_again_at_seven_days_once_each(results):
    target = _went_out(results)
    executor = FakeExecutor({"TWITTER_POST_LOOKUP_BY_POST_ID": [ok(TWEET)]})
    assert _run(executor) == {"read": 1, "refused": 0, "skipped": 0}
    (call,) = executor.calls
    assert (call.action, call.params, call.workspace_id, call.app_name) == (
        "TWITTER_POST_LOOKUP_BY_POST_ID", {"id": "1850", "tweet_fields": ["public_metrics"]}, WS_A, "TWITTER",
    )
    numbers = {"views": 940, "likes": 12, "reposts": 3, "replies": 2, "quotes": 0}
    assert _kept(results) == [(str(target.id), 1, numbers, "TWITTER_POST_LOOKUP_BY_POST_ID")]
    assert _run(executor, NOW + timedelta(hours=1)) == {"read": 0, "refused": 0, "skipped": 0}  # never twice
    assert _run(executor, NOW + timedelta(days=6)) == {"read": 1, "refused": 0, "skipped": 0}  # the 7-day reading
    assert sorted(reading for _, reading, _, _ in _kept(results)) == [1, 7]


def test_a_target_that_never_went_out_or_whose_window_passed_is_not_read(results):
    _went_out(results, status="pending")
    _went_out(results, ago=timedelta(days=3, hours=1))  # its 1-day window (2 days) has passed, 7 days not yet
    _went_out(results, toolkit="tiktok")  # no read for TikTok
    executor = FakeExecutor({})
    assert _run(executor) == {"read": 0, "refused": 0, "skipped": 0}
    assert executor.calls == [] and _kept(results) == []


def test_a_toolkit_not_connected_is_skipped_and_tried_again_at_the_next_tick(results):
    _went_out(results)
    results.readable["twitter"] = "twitter is not connected"
    executor = FakeExecutor({"TWITTER_POST_LOOKUP_BY_POST_ID": [ok(TWEET)]})
    assert _run(executor) == {"read": 0, "refused": 0, "skipped": 1}
    assert executor.calls == [] and _kept(results) == []
    results.readable["twitter"] = None
    assert _run(executor, NOW + timedelta(hours=1))["read"] == 1


def test_a_read_the_platform_refuses_is_kept_with_no_numbers_and_not_tried_again(results):
    target = _went_out(results, toolkit="instagram", remote_id="1790")
    executor = FakeExecutor({"INSTAGRAM_GET_IG_MEDIA_INSIGHTS": [refused("(#100) The media was deleted")]})
    assert _run(executor) == {"read": 0, "refused": 1, "skipped": 0}
    assert _kept(results) == [(str(target.id), 1, {}, "INSTAGRAM_GET_IG_MEDIA_INSIGHTS")]
    assert _run(executor, NOW + timedelta(hours=1))["refused"] == 0


def test_a_workspace_that_takes_no_background_reads_is_skipped(results, monkeypatch):
    _went_out(results)
    monkeypatch.setattr(socials_results, "background_allowed", lambda workspace: False)
    executor = FakeExecutor({"TWITTER_POST_LOOKUP_BY_POST_ID": [ok(TWEET)]})
    assert _run(executor) == {"read": 0, "refused": 0, "skipped": 1} and executor.calls == []


# ── the registry ───────────────────────────────────────────────────────────


def test_the_registry_says_why_a_read_cannot_run(monkeypatch):
    monkeypatch.setattr(capabilities, "_connected_toolkits", lambda db, ws: frozenset({"twitter", "instagram"}))
    monkeypatch.setattr(capabilities, "_cached_actions", lambda db, wanted: {("twitter", "TWITTER_POST_LOOKUP_BY_POST_ID"): object()})
    why = capabilities.runnable_actions(None, WS_A, {
        "twitter": "TWITTER_POST_LOOKUP_BY_POST_ID", "instagram": "INSTAGRAM_GET_IG_MEDIA_INSIGHTS", "youtube": "YOUTUBE_GET_VIDEO_DETAILS_BATCH",
    })
    assert why["twitter"] is None
    assert why["instagram"].startswith("Missing action INSTAGRAM_GET_IG_MEDIA_INSIGHTS")
    assert why["youtube"] == "youtube is not connected"
