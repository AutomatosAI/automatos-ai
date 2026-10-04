"""PRD-251C Wave 4, US-C408 — Posted, and a post's numbers (C7).

On the S0.3b API harness (SQLite). Pinned:

* ``GET /api/socials/posted`` lists the posts that went out, newest first (by their last
  channel), each with its plan, its topic (the bank's topic it was made from, else its brief's
  first line), its receipts and its numbers; a post that has not gone out is not listed, nor
  another workspace's;
* a post's numbers are each channel's latest reading (the 7-day one once taken), summed for
  the numbers given; a post nothing was read for has none;
* the filters: plan, channel (a post that went out there) and format.
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget, SocialTopic  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)

api = api_harness.api
NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def board(bank):  # noqa: F811
    """The bank harness with the results' table."""
    SocialPost.metadata.create_all(bank.session.get_bind(), tables=[SocialPostStat.__table__])
    return bank


def _went_out(api, title, *, ago_hours, channels=("twitter",), fmt="image", plan=None, workspace=WS_A, status="published"):
    post = SocialPost(id=uuid.uuid4(), workspace_id=workspace, created_by="u", title=title, content_hash="0" * 64, status=status,
                      format=fmt, brief=f"{title} brief\nmore", campaign_id=uuid.UUID(plan["id"]) if plan else None)
    api.session.add(post)
    targets = []
    for toolkit in channels:
        went = status in ("published", "partially_published")
        target = SocialPostTarget(
            id=uuid.uuid4(), post_id=post.id, toolkit=toolkit, post_kind="image", action_plan={}, idempotency_key=f"sp:{post.id}:{toolkit}",
            status="published" if went else "pending", remote_id="r-1" if went else None,
            permalink=f"https://{toolkit}.test/{post.id}" if went else None, published_at=NOW - timedelta(hours=ago_hours) if went else None,
        )
        api.session.add(target)
        targets.append(target)
    api.session.commit()
    return post, targets


def _read(api, post, target, reading, numbers):
    api.session.add(SocialPostStat(workspace_id=post.workspace_id, post_id=post.id, target_id=target.id, reading=reading,
                                   read_at=NOW, numbers=numbers, source_action="ACTION"))
    api.session.commit()


def _posted(api, query=""):
    resp = api.client.get(f"/api/socials/posted{query}")
    assert resp.status_code == 200, resp.text
    return resp.json()["posts"]


def test_what_went_out_newest_first_with_its_receipts_numbers_and_topic(board):
    plan = _create_plan(board)
    older, older_targets = _went_out(board, "Older", ago_hours=48, channels=("twitter", "linkedin"), plan=plan)
    newer, _ = _went_out(board, "Newer", ago_hours=2)
    _went_out(board, "Waiting", ago_hours=1, status="needs_approval")
    _went_out(board, "Theirs", ago_hours=1, workspace=WS_B)
    board.session.add(SocialTopic(workspace_id=WS_A, campaign_id=uuid.UUID(plan["id"]), title="The stand", facts=[], formats=[],
                                 used_post_id=older.id, created_by="u"))
    board.session.commit()
    twitter, linkedin = older_targets
    _read(board, older, twitter, 1, {"likes": 3, "views": 100})
    _read(board, older, twitter, 7, {"likes": 9, "views": 400})  # the week's reading wins
    _read(board, older, linkedin, 1, {"reactions": 4})
    posts = _posted(board)
    assert [post["title"] for post in posts] == ["Newer", "Older"]
    first, second = posts
    assert (first["topic"], first["numbers"], first["plan_id"]) == ("Newer brief", None, None)
    assert (second["topic"], second["plan_name"]) == ("The stand", "Countdown to Lisbon")
    assert [receipt["toolkit"] for receipt in second["receipts"]] == ["twitter", "linkedin"]
    assert second["numbers"]["numbers"] == {"likes": 9, "views": 400, "reactions": 4}
    assert second["numbers"]["by_channel"] == {"twitter": {"likes": 9, "views": 400}, "linkedin": {"reactions": 4}}
    assert (second["numbers"]["engagement"], second["numbers"]["reading"]) == (13, 7)


def test_the_filters_plan_channel_and_format(board):
    plan = _create_plan(board)
    _went_out(board, "Plan image on X", ago_hours=3, plan=plan)
    _went_out(board, "Video on LinkedIn", ago_hours=2, channels=("linkedin",), fmt="video")
    _went_out(board, "Image on both", ago_hours=1, channels=("twitter", "linkedin"))
    assert [p["title"] for p in _posted(board, f"?plan_id={plan['id']}")] == ["Plan image on X"]
    assert [p["title"] for p in _posted(board, "?channel=linkedin")] == ["Image on both", "Video on LinkedIn"]
    assert [p["title"] for p in _posted(board, "?format=video")] == ["Video on LinkedIn"]
    assert board.client.get("/api/socials/posted?channel=Not%20a%20toolkit").status_code == 422
