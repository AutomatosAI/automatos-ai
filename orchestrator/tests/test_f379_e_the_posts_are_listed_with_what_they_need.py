"""F379 (night 11, 7 Oct): Auto lists every post it is asked about, and what each still needs.

Night 11 (B20): asked what was waiting for approval, Auto listed 2 of about 35 posts (each came
back whole, and the chat keeps about 2,000 tokens of an answer) and called a carousel with an empty
caption "good to post". GET /api/socials/posts?limit=5 returned all 66 posts (friction 19). Now a
row is compact and says what the post still needs, the answer counts every match first, a queue
picks the posts waiting for approval or not rendered, and the route honours ``limit``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_social_post_tools as harness  # noqa: E402
from modules.socials import service  # noqa: E402
from modules.tools.discovery import social_post_list as listing  # noqa: E402
from tests.test_prd251w1_social_post_tools import ACTOR, _created, _row, _tool  # noqa: E402

env = harness.env  # the US-116 fixture: SQLite, the real router and executor

ROW_KEYS = {"id", "title", "status", "format", "template_id", "created_at", "scheduled_for", "rendered",
            "has_caption", "needs"}


def _post(status, **fields):
    return {"id": "p", "title": "Harvest Club", "status": status, "format": "carousel", "copy": {}, "media": {},
            "review_log": [], **fields}


def test_a_row_says_what_the_post_still_needs_in_plain_words():
    failed = _post("failed", review_log=[{"action": "render_failed", "comment": "fill in point_1_title before rendering"}])
    waiting = _post("needs_approval", media={"4:5": [{"name": "render-01.png"}]}, copy={"base": "Two coffees, £32."})

    assert listing.what_it_needs(failed) == ["its render failed: fill in point_1_title before rendering",
                                             "a caption: its copy is empty"]
    assert listing.what_it_needs(_post("draft")) == ["a render: nothing is rendered yet", "a caption: its copy is empty"]
    assert listing.what_it_needs(waiting) == ["the owner's approval in the Socials tab"]
    assert listing.what_it_needs(_post("needs_approval", media={"4:5": [{}]})) == [
        "the owner's approval in the Socials tab", "a caption: its copy is empty"]   # never "good to post"
    assert set(listing.post_row(waiting)) == ROW_KEYS and listing.post_row(waiting)["rendered"] is True


def test_a_queue_picks_the_posts_by_what_they_need_and_an_unknown_one_is_refused():
    statuses, keep, problem = listing.queue_of("not_rendered")
    rendered_draft = _post("draft", media={"4:5": [{"name": "render.png"}]})

    assert statuses == ("failed", "draft", "changes_requested") and problem is None
    assert keep(_post("draft")) is True and keep(rendered_draft) is False and keep(_post("failed")) is True
    assert listing.queue_of("pending")[2] == "queue must be one of awaiting_approval, not_rendered, open."


def test_the_answer_counts_every_match_before_the_rows():
    answer = listing.listed([_post("draft"), _post("draft"), _post("needs_approval")], lambda post: True, 2)

    assert (answer["total"], answer["count"], answer["limit"]) == (3, 2, 2)
    assert list(answer)[:4] == ["success", "total", "count", "limit"]


def test_the_tool_lists_the_queue_waiting_for_approval_and_the_drafts_not_rendered(env):
    drafts = [_created(_tool(env, "platform_create_social_post", title=f"Draft {n}", render=False)) for n in (1, 2)]
    waiting = _created(_tool(env, "platform_create_social_post", title="Waiting", render=False))
    assert _tool(env, "platform_submit_social_post", post_id=waiting["id"])["success"] is True
    failed = _row(env, drafts[1]["id"])
    service.start_render(failed, ACTOR)
    service.fail_render(failed, ACTOR, "fill in headline before rendering")
    env.session.commit()

    approval = _tool(env, "platform_list_social_posts", queue="awaiting_approval")
    not_rendered = _tool(env, "platform_list_social_posts", queue="not_rendered", limit=1)
    unknown = _tool(env, "platform_list_social_posts", queue="pending")

    assert approval["total"] == 1 and [p["id"] for p in approval["posts"]] == [waiting["id"]]
    assert approval["posts"][0]["needs"][0] == "the owner's approval in the Socials tab"
    assert set(approval["posts"][0]) == ROW_KEYS
    assert not_rendered["total"] == 2 and not_rendered["count"] == 1
    assert unknown["success"] is False and "queue must be one of" in unknown["error"]


@pytest.mark.parametrize("limit, shown", [(None, 3), (2, 2)])
def test_the_route_keeps_the_newest_limit_and_counts_every_match(env, limit, shown):
    for n in (1, 2, 3):
        _created(_tool(env, "platform_create_social_post", title=f"Post {n}", render=False))

    listed = env.client.get("/api/socials/posts", params={} if limit is None else {"limit": limit})

    assert listed.status_code == 200, listed.text
    assert len(listed.json()["posts"]) == shown and listed.json()["total"] == 3
