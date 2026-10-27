"""P256-FIX-SOC-4 (F432-b, 14 Oct): Auto sees a carousel's receipt through the chat's cut.

platform_get_social_post answered the whole post. The chat writes a platform tool's answer as
indented JSON and keeps its first ~2,000 tokens; a post's channels came after its copy, variables,
media, preview and a history whose every agent edit kept a copy of up to 4,000 characters. On a
carousel the receipts fell past the cut, so Auto could not say where or when it went out.

Now the answer puts who the post is, its channels with their receipts, its times, its approval and
what it still needs first, then the rest, with the last five history entries and none of their copy
snapshots. This fails on the old answer: the receipt's values are cut off.
"""
from __future__ import annotations

import asyncio
import inspect
import uuid
from datetime import datetime, timedelta, timezone

from core.context_guard import truncate_to_token_budget
from core.models.socials import SocialPost, SocialPostTarget
from modules.socials.voice_examples import AUTO_COPY_KEY
from modules.tools.discovery import handlers_socials
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.test_prd251w1_social_post_tools import WS
from tests.test_prd251w1_social_post_tools import env as _env

env = _env  # the SQLite Socials harness

ACTION = "platform_get_social_post"
CHAT_TOOL_TOKENS = inspect.signature(ToolLoopExecutor.__init__).parameters["content_truncate_tokens"].default
WENT_OUT = datetime(2026, 10, 14, 18, 5, tzinfo=timezone.utc)
SNAPSHOT = ("Harvest Club opens Friday: six reasons to join, from the first roast of the season to the "
            "tasting notes our roasters wrote for you. " * 40)[:3000]


def _history():
    """Ten entries: a draft and agent edits that each keep a 3,000-character copy, then the review."""
    at = WENT_OUT - timedelta(days=2)
    entries = [{"at": (at + timedelta(minutes=n)).isoformat(), "by": "agent:7", "action": "edit", "agent": "Director",
                "comment": f"Edited by Director: copy (pass {n}).", "fields": ["copy"], AUTO_COPY_KEY: SNAPSHOT}
               for n in range(7)]
    review = [("submit", "agent:7"), ("approve", "user:owner"), ("schedule", "user:owner")]
    return entries + [{"at": (at + timedelta(hours=n + 1)).isoformat(), "by": by, "action": action, "comment": None}
                      for n, (action, by) in enumerate(review)]


def _carousel(env):
    post = SocialPost(
        id=uuid.uuid4(), workspace_id=WS, created_by="agent:7", title="Harvest Club: six reasons", status="published",
        format="carousel", copy={"base": SNAPSHOT, "channels": {"instagram": SNAPSHOT[:1500]}}, content_hash="a" * 64,
        approved_hash="a" * 64, variables={f"reason_{n}": {"value": f"Reason {n}: " + "x" * 80, "claim": False}
                                           for n in range(1, 16)},
        media={"4:5": [{"deliverable_id": f"d{n}", "name": f"slide-{n}.png", "sha256": "b" * 64, "bytes": 90000,
                        "content_type": "image/png", "width": 1080, "height": 1350} for n in range(1, 7)]},
        voice=None, footage={"slide_1": {"prompt": "a harvest table at dawn", "status": "done", "toolkit": "fal_ai"}},
        approved_by="user:owner", approved_at=WENT_OUT - timedelta(days=1), override_unsourced=True,
        scheduled_for=WENT_OUT, planned_for=WENT_OUT - timedelta(hours=1), timezone="Europe/London",
        review_log=_history(),
    )
    env.session.add(post)
    env.session.add(SocialPostTarget(
        id=uuid.uuid4(), post_id=post.id, toolkit="instagram", post_kind="carousel", action_plan={},
        idempotency_key=f"sp:{post.id}:instagram:carousel", status="published", attempts=1,
        remote_id="17945112233445566", permalink="https://www.instagram.com/p/Cx7Harvest/", published_at=WENT_OUT,
    ))
    env.session.commit()
    env.session.expire_all()
    return env.session.get(SocialPost, post.id)


def _what_chat_sees(answer):
    return truncate_to_token_budget(ToolResultFormatter.format_for_llm(answer, ACTION), CHAT_TOOL_TOKENS)


def test_the_receipt_times_and_approval_survive_the_chats_cut(env):
    post = _carousel(env)
    whole = post.to_dict()
    receipt = whole["targets"][0]

    answer = asyncio.run(handlers_socials.get_social_post(env.session, WS, {"post_id": str(post.id)}))
    seen = _what_chat_sees(answer)

    assert answer["success"] is True
    for value in (receipt["remote_id"], receipt["permalink"], receipt["published_at"], whole["approved_at"],
                  whole["scheduled_for"], whole["planned_for"], whole["timezone"]):
        assert value and value in seen, value
    assert '"override_unsourced": true' in seen


def test_the_answer_keeps_the_post_and_trims_only_its_history(env):
    post = _carousel(env)

    read = asyncio.run(handlers_socials.get_social_post(env.session, WS, {"post_id": str(post.id)}))["post"]

    assert read["status"] == "published" and read["title"] == "Harvest Club: six reasons"
    assert {"voice", "footage", "copy", "variables", "media", "sources"} <= set(read)
    assert not {"preview", "content_hash", "approved_hash", "workspace_id"} & set(read)
    # P256-FIX-SOC-5 (F437): the render's state follows the status, so a cut answer still carries it.
    assert list(read)[:7] == ["id", "status", "render", "title", "format", "template_id", "targets"]
    assert read["targets"] == [{key: post.to_dict()["targets"][0][key] for key in (
        "toolkit", "post_kind", "options", "status", "published_at", "remote_id", "permalink", "error", "notes")}]
    assert read["review_log_total"] == 10 and len(read["review_log"]) == 5
    assert [entry["action"] for entry in read["review_log"]] == ["edit", "edit", "submit", "approve", "schedule"]
    assert not [entry for entry in read["review_log"] if AUTO_COPY_KEY in entry]
    assert read["needs"] == []


def test_the_old_answer_lost_the_receipt(env):
    """What the chat kept of the whole post (the answer before this fix): no receipt."""
    post = _carousel(env)

    seen = _what_chat_sees({"success": True, "post": post.to_dict()})

    assert "17945112233445566" not in seen and "https://www.instagram.com/p/Cx7Harvest/" not in seen
