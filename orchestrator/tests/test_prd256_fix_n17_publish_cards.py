"""P256-FIX-N17-11 (F443, night 16): a publish card shows the post, and says what the click does.

Night 16's blog card #2705 (19:59, chat 5f87dd20) read "card raised: publish a blog post
title='Guji Uraga: our Ethiopian lot'" for a call with ``publish_immediately`` false, which saves
a draft; #2744 the same. The social card #2710 (20:30) read "submit a social post to publish" and
named no post. Neither action had card lines (``card_question.READERS``), and the handler read
the string 'false' as true.

Now a blog card says whether the post goes live and shows its excerpt, category, tags, first line
and length; its verb says "save … as a draft (nothing is published)" when it is one. A social
card names the post this workspace holds under the id, its channels, format, caption and time,
the reviewer's note, and that it publishes only after it is approved in Socials. An id that is not a
post of this workspace gives no lines: the card asks on its act alone. post_id stays off the
card's subject line (F091).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery import owner_only
from modules.tools.discovery.card_question import READERS, platform_question
from modules.tools.discovery.card_question_text import question
from modules.tools.discovery.handlers_blog import goes_live

BLOG = "platform_publish_blog_post"
SOCIAL = "platform_submit_social_post"
TITLE = "Guji Uraga: our Ethiopian lot"
BODY = ("# Guji Uraga: our Ethiopian lot\n\n"
        "Our new Ethiopian lot comes from the Uraga highlands, washed and dried on raised beds.\n\n"
        + "It tastes of bergamot, peach and black tea, with a long, clean finish. " * 12)
# Night 16, line 42 (19:59, #2705): the call the card was raised for.
DRAFT_CALL = {"title": TITLE, "content": BODY, "excerpt": "A washed Ethiopian lot from the Uraga highlands.",
              "category": "Coffees", "tags": "ethiopia, washed, new", "publish_immediately": False}
CAPTION = "Our Guji Uraga is in: bergamot, peach and black tea."
PLANNED = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)


def _card(action, params, db=None, ws=None):
    """(the card's act, its question), as owner_only.platform_ask writes them."""
    act = owner_only._act(db, ws, action, params, f"title={params['title']!r}" if "title" in params else "")
    return act, platform_question(db, ws, action, params, act)


# ── Every owner-only action has card lines and a verb ───────────────────────────────

def test_every_owner_only_action_has_card_lines_and_a_verb():
    owner_only_actions = owner_only.OWNER_ONLY_ACTIONS | owner_only.SESSION_CREATES | owner_only.PLAN_EDITS

    assert sorted(a for a in owner_only_actions if a not in READERS) == []
    assert all(owner_only.VERBS.get(a) for a in owner_only_actions - owner_only.CARD_MOVES)
    assert BLOG in READERS and SOCIAL in READERS


# ── A blog card says whether the post goes live, and shows it ───────────────────────

def test_a_blog_draft_card_says_it_is_a_draft_and_shows_the_post():
    act, asked = _card(BLOG, DRAFT_CALL)

    assert act.startswith("save a blog post as a draft (nothing is published)") and TITLE in act
    assert "Publish a blog post" not in asked and "publish a blog post" not in asked
    assert "- goes live: no, saved as a draft; nothing is published" in asked
    assert "- excerpt: A washed Ethiopian lot from the Uraga highlands." in asked
    assert "- category: Coffees" in asked and "- tags: ethiopia, washed, new" in asked
    assert "- first line: Our new Ethiopian lot comes from the Uraga highlands" in asked   # not the heading
    assert f"- length: {len(BODY.split())} words" in asked


def test_a_blog_publish_card_says_it_goes_live():
    act, asked = _card(BLOG, {**DRAFT_CALL, "publish_immediately": True, "excerpt": None})

    assert act.startswith("publish a blog post") and "draft" not in act
    assert "- goes live: yes, on this workspace's blog" in asked
    assert "- excerpt: none given; the post's opening is used" in asked


def test_a_string_false_never_publishes(db_session, seed_workspace):
    from modules.tools.discovery.handlers_blog import publish_blog_post

    assert goes_live({"publish_immediately": "false"}) is False and goes_live({"publish_immediately": "true"}) is True
    assert goes_live({"publish_immediately": True}) is True and goes_live({}) is False
    assert goes_live({"publish_immediately": 1}) is False
    assert _card(BLOG, {**DRAFT_CALL, "publish_immediately": "false"})[0].startswith("save a blog post as a draft")

    ws = UUID(seed_workspace())
    with patch("core.services.blog_service.BlogService.create_post", new=AsyncMock(
            return_value=NS(id=uuid4(), title=TITLE, slug="guji-uraga", status="draft", file_path="blog/guji.md"))) as made:
        out = asyncio.run(publish_blog_post(db_session, ws, {**DRAFT_CALL, "publish_immediately": "false"}))
    assert out["success"] is True and made.await_args.kwargs["status"] == "draft"


# ── A social card names the post, its channels and caption ──────────────────────────

def _post(db, ws, **fields):
    from core.models.socials import SocialPost, SocialPostTarget

    post = SocialPost(workspace_id=ws, created_by="agent:7", title=fields.get("title", "Guji Uraga launch"),
                      copy={"base": f"{CAPTION}\nRoasted on Tuesday, shipped Wednesday.",
                            "channels": {channel: "" for channel in fields.get("written_for", ("x",))}},
                      format="image", content_hash="h" * 64, planned_for=fields.get("planned_for", PLANNED))
    db.add(post)
    db.flush()
    for toolkit in fields.get("targets", ()):
        db.add(SocialPostTarget(post_id=post.id, toolkit=toolkit, post_kind="image", action_plan={},
                                idempotency_key=f"sp:{post.id}:{toolkit}:image"))
    db.flush()
    db.refresh(post)
    return post


@pytest.fixture
def posts(db_session, seed_workspace):
    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    return NS(db=db_session, ws=ws, other=other, ours=_post(db_session, ws, targets=("linkedin", "instagram")),
              bare=_post(db_session, ws, title="Bare", planned_for=None, written_for=("threads", "bluesky")),
              theirs=_post(db_session, other, title="Someone else's launch"))


def test_a_social_submit_card_names_the_post_its_channels_and_caption(posts):
    params = {"post_id": str(posts.ours.id), "note": "Check the tasting notes against the cupping sheet."}
    act, asked = _card(SOCIAL, params, posts.db, posts.ws)

    assert act == "send a social post for approval (nothing posts yet)"
    assert asked.splitlines() == [
        "Send a social post for approval (nothing posts yet):",
        f"- post: 'Guji Uraga launch' ({posts.ours.id})",
        "- status: draft → needs approval",
        "- channels: instagram image, linkedin image",
        "- format: image",
        f"- caption: {CAPTION}",
        "- rendered files: 0",
        "- planned for: 09:00 on 20 Oct UTC",
        "- note: Check the tasting notes against the cupping sheet.",
        "- the click sends it for approval; it publishes only after a person approves it in the Socials tab",
    ]
    bare = _card(SOCIAL, {"post_id": str(posts.bare.id)}, posts.db, posts.ws)[1]
    # P256-FIX-SOC-4 (F431-b): text written for a channel is not a channel; no targets publish nowhere.
    assert "- channels: none: it would publish nowhere" in bare and "- planned for: (empty)" in bare
    assert "bluesky" not in bare and "threads" not in bare
    assert str(posts.ours.id) not in act                                                  # F091: no id on the subject


def test_a_social_card_never_reads_another_workspaces_post(posts):
    act, asked = _card(SOCIAL, {"post_id": str(posts.theirs.id)}, posts.db, posts.ws)

    assert asked == question(act, []) and "Someone else's launch" not in asked


@pytest.mark.parametrize("post_id", ["#2710", "2710", None, ""])
def test_a_post_id_that_is_not_a_uuid_still_asks(posts, post_id):
    act, asked = _card(SOCIAL, {"post_id": post_id}, posts.db, posts.ws)

    assert asked == question(act, []) == "Send a social post for approval (nothing posts yet)."


# ── Through the executor: the card the owner sees carries the lines ────────────────

def test_the_owners_chat_raises_the_draft_card_with_the_post_on_it(db_session, seed_workspace, monkeypatch):
    from sqlalchemy import text

    import modules.tools.discovery.follows_the_owner as guard
    from core.models.approval_grants import ApprovalGrant
    from core.models.core import Agent
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    db_session.add(auto)
    db_session.flush()
    tag = uuid4().hex[:8]
    owner = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                               {"e": f"n1711-{tag}@harbourline.test", "u": f"n1711-{tag}"}).scalar()
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: None)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True
    chat = {"driving_user_id": str(owner), "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}

    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        reply = asyncio.run(executor.execute(BLOG, {**DRAFT_CALL, "_agent_id": auto.id}, chat))

    assert reply["success"] is False and reply.get("requires_confirmation") is True, reply
    grant = db_session.get(ApprovalGrant, reply["grant_id"])
    assert grant.question_md.startswith("Save a blog post as a draft (nothing is published)")
    assert "- goes live: no, saved as a draft; nothing is published" in grant.question_md
    assert "- first line: Our new Ethiopian lot" in grant.question_md
