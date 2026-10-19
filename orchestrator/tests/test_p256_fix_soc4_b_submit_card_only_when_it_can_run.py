"""P256-FIX-SOC-4 (F431-c, 14 Oct card #2648): a submit card is raised only when its click can send the post.

Auto called platform_submit_social_post in the owner's chat for post 7e7c402c, which a render had
already left waiting for approval. Submit is owner-only (Decision D1), so the card came before
anything read the post; the click re-ran submit, which refused, and the click was given back. The
card named no post and said the click publishes.

Now the post is read before the card: one that cannot be sent for approval is answered in plain
words, with no card and no grant; an id that names no post gets the tool's own words. A card shows
the post, its channels (or that it has none), its caption and that it publishes only after a person
approves it; the click sends it, signed by who clicked. A post that moved between the card and the
click gives the click back (F193). A unique prefix of the id names the post, and the card binds the
call to its whole id.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

SUBMIT = "platform_submit_social_post"
CLICKER = "user:clicker-7"
CAPTION = "Our Guji Uraga is in: bergamot, peach and black tea."


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    """A workspace with Socials on, its owner, Auto's executor, and posts to submit."""
    import modules.socials.notify as notify
    import modules.socials.settings as socials_settings
    import modules.tools.discovery.follows_the_owner as guard
    from core.models.core import Agent
    from core.models.workspaces import Workspace
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    ws = UUID(seed_workspace())
    db_session.get(Workspace, ws).settings = {"socials": {"enabled": True}}
    monkeypatch.setattr(socials_settings, "read_system_setting", lambda category, key: "true")
    told = []
    monkeypatch.setattr(notify, "notify_approval_pending", lambda workspace, post_id, title: told.append(post_id))
    auto = Agent(name="Auto", agent_type="system", description="", status="active", configuration={},
                 workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws), is_system_agent=True)
    db_session.add(auto)
    db_session.flush()
    tag = uuid4().hex[:8]
    owner = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                               {"e": f"soc4-{tag}@harbourline.test", "u": f"soc4-{tag}"}).scalar()
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    monkeypatch.setattr(guard, "owner_turn", lambda db, workspace, ctx: None)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True   # the dial on: an owner-only action still asks
    chat = {"driving_user_id": str(owner), "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}
    return NS(db=db_session, ws=ws, auto=auto.id, executor=executor, chat=chat, told=told)


def _post(desk, *, status="draft", targets=(), title="Guji Uraga launch"):
    from core.models.socials import SocialPost, SocialPostTarget

    post = SocialPost(workspace_id=desk.ws, created_by="agent:7", title=title, status=status, format="carousel",
                      copy={"base": f"{CAPTION}\nRoasted on Tuesday, shipped Wednesday."}, content_hash="h" * 64,
                      media={"4:5": [{"deliverable_id": "d1", "name": "slide-1.png"},
                                     {"deliverable_id": "d2", "name": "slide-2.png"}]})
    desk.db.add(post)
    desk.db.flush()
    for toolkit in targets:
        desk.db.add(SocialPostTarget(post_id=post.id, toolkit=toolkit, post_kind="carousel", action_plan={},
                                     idempotency_key=f"sp:{post.id}:{toolkit}:carousel"))
    desk.db.flush()
    desk.db.refresh(post)
    return post


def _submit(desk, params):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(desk.executor.execute(SUBMIT, {**params, "_agent_id": desk.auto}, desk.chat))


def _grants(desk):
    from core.models.approval_grants import ApprovalGrant

    return desk.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == desk.ws).count()


def _grant(desk, grant_id):
    from core.models.approval_grants import ApprovalGrant

    return desk.db.get(ApprovalGrant, grant_id)


def _click(desk, reply):
    from core.services.approval_grants import grant_grant

    grant = _grant(desk, reply["grant_id"])
    grant_grant(grant, granted_by=CLICKER)
    desk.db.flush()
    return grant


# ── (1) a post already waiting: no card, plain words ─────────────────────────────────

def test_a_post_already_waiting_raises_no_card_and_says_so(desk):
    post = _post(desk, status="needs_approval", targets=("instagram",))
    before = (post.status, post.content_hash, list(post.review_log or []))

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["success"] is False and "requires_confirmation" not in reply, reply
    assert "already waiting for approval in the Socials tab" in reply["error"]
    assert "nothing was sent and no card was raised" in reply["error"]
    assert "publish it nowhere" not in reply["error"]
    assert _grants(desk) == 0
    desk.db.refresh(post)
    assert (post.status, post.content_hash, list(post.review_log or [])) == before


def test_a_waiting_post_with_no_channels_says_it_would_publish_nowhere(desk):
    post = _post(desk, status="needs_approval")

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["success"] is False and "requires_confirmation" not in reply
    assert "It has no channels, so an approval would publish it nowhere" in reply["error"]
    assert _grants(desk) == 0


@pytest.mark.parametrize("status, said", [
    ("rendering", "is rendering"), ("approved", "is already approved"), ("scheduled", "approved and scheduled"),
    ("published", "is already published"), ("failed", "has failed"), ("missed", "is missed"),
])
def test_every_other_status_gets_one_line_and_no_card(desk, status, said):
    post = _post(desk, status=status, targets=("instagram",))

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["success"] is False and "requires_confirmation" not in reply
    assert said in reply["error"] and reply["error"].endswith("Nothing was sent and no card was raised.")
    assert _grants(desk) == 0


# ── (2)(3) a draft: the card shows the post, where it goes, and what the click does ──

def test_a_drafts_card_shows_the_post_its_channels_and_caption(desk):
    post = _post(desk, targets=("instagram",))

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["success"] is False and reply["requires_confirmation"] is True, reply
    asked = _grant(desk, reply["grant_id"]).question_md
    assert "Guji Uraga launch" in asked and str(post.id) in asked
    assert "- channels: instagram carousel" in asked and f"- caption: {CAPTION}" in asked
    assert "- status: draft → needs approval" in asked and "- rendered files: 2" in asked
    assert "publishes only after a person approves it in the Socials tab" in asked
    assert "to publish" not in reply["act"]


def test_a_draft_with_no_channels_says_so_on_its_card(desk):
    post = _post(desk)

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["requires_confirmation"] is True
    assert "- channels: none: it would publish nowhere" in _grant(desk, reply["grant_id"]).question_md


# ── (4) the click sends it, signed by who clicked ────────────────────────────────────

def test_the_click_sends_the_post_for_approval_signed_by_the_clicker(desk):
    post = _post(desk, targets=("instagram",))
    asked = _submit(desk, {"post_id": str(post.id)})
    grant = _click(desk, asked)

    ran = _submit(desk, dict(grant.details["params"]))

    assert ran["success"] is True and ran["approved_via_grant_id"] == grant.id, ran
    desk.db.refresh(post)
    assert post.status == "needs_approval"
    assert (post.review_log[-1]["action"], post.review_log[-1]["by"]) == ("submit", CLICKER)
    assert desk.told == [post.id]


# ── (5) the post moved between the card and the click: the click is given back ──────

def test_a_post_that_moved_before_the_click_is_unchanged_and_the_click_given_back(desk):
    post = _post(desk, targets=("instagram",))
    asked = _submit(desk, {"post_id": str(post.id)})
    grant = _click(desk, asked)
    post.status = "needs_approval"
    desk.db.flush()

    ran = _submit(desk, dict(grant.details["params"]))

    assert ran["success"] is False and "cannot submit a post that is needs approval" in ran["error"]
    desk.db.refresh(post)
    assert post.status == "needs_approval" and not [e for e in post.review_log or [] if e.get("action") == "submit"]
    assert _grant(desk, grant.id).status == "granted"          # F193: given back for its retry


# ── (6) an id that names no post: the tool's own words, no card ──────────────────────

@pytest.mark.parametrize("post_id, said", [
    ("not-a-post", "post_id 'not-a-post' is not a post's id: platform_list_social_posts lists them."),
    (str(uuid4()), "Post not found in this workspace: platform_list_social_posts lists them."),
])
def test_an_id_that_names_no_post_is_refused_before_the_card(desk, post_id, said):
    reply = _submit(desk, {"post_id": post_id})

    assert reply == {**reply, "success": False, "error": said} and "requires_confirmation" not in reply
    assert _grants(desk) == 0


def test_another_workspaces_post_is_never_asked_about(desk, seed_workspace):
    from core.models.socials import SocialPost

    theirs = SocialPost(workspace_id=UUID(seed_workspace()), created_by="u", title="Theirs", content_hash="h" * 64)
    desk.db.add(theirs)
    desk.db.flush()

    reply = _submit(desk, {"post_id": str(theirs.id)})

    assert reply["success"] is False and "Post not found in this workspace" in reply["error"]
    assert _grants(desk) == 0


# ── (h) a unique prefix of the id names the post; the card binds the whole id ────────

def test_a_unique_id_prefix_raises_the_card_on_the_whole_id(desk):
    post = _post(desk, targets=("instagram",))

    reply = _submit(desk, {"post_id": str(post.id)[:8]})

    assert reply["requires_confirmation"] is True, reply
    grant = _grant(desk, reply["grant_id"])
    assert grant.details["params"]["post_id"] == str(post.id) and str(post.id) in grant.question_md


def test_a_socials_off_workspace_raises_no_card(desk):
    from core.models.workspaces import Workspace

    post = _post(desk, targets=("instagram",))
    desk.db.get(Workspace, desk.ws).settings = {"socials": {"enabled": False}}
    desk.db.flush()

    reply = _submit(desk, {"post_id": str(post.id)})

    assert reply["success"] is False and "requires_confirmation" not in reply
    assert _grants(desk) == 0
