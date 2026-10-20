"""P256-FIX-SOC-4 (F432-b, 14 Oct): "what did I post today and when?" reads what went out.

platform_get_social_history orders a workspace's posts by when they are scheduled, planned or made,
includes the posts planned ahead, and its items carry no receipts, so "what went out today" had no
read. ``posted.posted`` already lists what went out, ordered by when it went out and with each
channel's receipt, but no agent tool read it.

Now ``went_out`` true reads it for the last ``days``: only posts that went out, newest first by when,
each with when, where, the platform's id and the link, and the links survive the chat's cut. Without
it the history is as it was, with no receipts, so platform_get_social_plan (which embeds the
history) keeps its size.
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from core.context_guard import truncate_to_token_budget
from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget, SocialTopic
from modules.tools.discovery import handlers_socials
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.test_prd251w1_social_post_tools import WS, WS_OTHER
from tests.test_prd251w1_social_post_tools import env as _env

env = _env  # the SQLite Socials harness

ACTION = "platform_get_social_history"
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
NOW = datetime.now(timezone.utc)


@pytest.fixture
def week(env):
    """Two posts out today, one three days ago, another workspace's today, and three planned for next week."""
    SocialPost.metadata.create_all(env.session.get_bind(), tables=[SocialTopic.__table__, SocialPostStat.__table__])

    def post(title, *, status, out=None, planned=None, workspace=WS, channel="instagram"):
        row = SocialPost(id=uuid.uuid4(), workspace_id=workspace, created_by="agent:7", title=title, status=status,
                         format="image", copy={"base": f"{title}."}, content_hash="a" * 64, planned_for=planned)
        env.session.add(row)
        env.session.add(SocialPostTarget(
            id=uuid.uuid4(), post_id=row.id, toolkit=channel, post_kind="image", action_plan={},
            idempotency_key=f"sp:{row.id}:{channel}:image", status="published" if out else "pending",
            remote_id=f"remote-{title}" if out else None, published_at=out,
            permalink=f"https://{channel}.test/p/{title.replace(' ', '-')}" if out else None))
        return row

    post("Morning roast", status="published", out=NOW - timedelta(hours=5))
    post("Evening tasting", status="published", out=NOW - timedelta(hours=1), channel="linkedin")
    post("Monday's offer", status="published", out=NOW - timedelta(days=3))
    post("Theirs today", status="published", out=NOW - timedelta(hours=2), workspace=WS_OTHER)
    for n, status in enumerate(("approved", "scheduled", "scheduled"), 1):
        post(f"Next week {n}", status=status, planned=NOW + timedelta(days=6 + n))
    env.session.commit()
    return env


def _history(env, **params):
    return asyncio.run(handlers_socials.get_social_history(env.session, WS, params))


def test_what_went_out_today_newest_first_with_its_receipts(week):
    answer = _history(week, went_out=True, days=1)

    assert answer["success"] is True and answer["count"] == 2
    assert [post["title"] for post in answer["posts"]] == ["Evening tasting", "Morning roast"]
    evening = answer["posts"][0]["receipts"]
    assert [(r["toolkit"], r["remote_id"], r["permalink"]) for r in evening] == [
        ("linkedin", "remote-Evening tasting", "https://linkedin.test/p/Evening-tasting")]
    assert evening[0]["published_at"] and answer["posts"][0]["went_out_at"]


def test_the_links_survive_the_chats_cut(week):
    seen = truncate_to_token_budget(ToolResultFormatter.format_for_llm(_history(week, went_out=True, days=1), ACTION),
                                    CHAT_TOOL_TOKENS)

    assert "https://linkedin.test/p/Evening-tasting" in seen and "https://instagram.test/p/Morning-roast" in seen


def test_a_longer_window_reaches_further_back(week):
    titles = [post["title"] for post in _history(week, went_out=True, days=7)["posts"]]

    assert titles == ["Evening tasting", "Morning roast", "Monday's offer"]


def test_without_went_out_the_history_is_as_it_was(week):
    answer = _history(week)

    assert answer["success"] is True and answer["posts"]
    assert not [post for post in answer["posts"] if "receipts" in post]
    assert {post["title"] for post in answer["posts"]} >= {"Next week 1", "Morning roast"}   # planned ahead too


def test_went_out_must_be_true_or_false(week):
    answer = _history(week, went_out="today")

    assert answer == {"success": False, "error": "went_out must be true or false."}


def test_the_tool_says_how_to_ask_for_it():
    from modules.tools.discovery.action_registry import get_action_registry

    action = get_action_registry().get(ACTION)
    assert action.parameters["properties"]["went_out"]["type"] == "boolean"
    assert "what did I post today and when?" in action.examples
