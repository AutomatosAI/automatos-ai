"""PRD-251D Wave 2, US-D108 — the agent tools say when an edit left the file older than its words.

A post records what its file was made from, and says ``file_current`` false once an edit changed what
a render reads; the owner cannot approve it until it renders again. An agent learns it from the
tools' answers, so it renders again. Pinned, on the US-116 harness (SQLite, the real router and
executor):

* ``platform_update_social_post`` on a rendered post: a caption change answers as before; a change to
  a field answers ``file_current`` false with a message naming the call that renders it again, and
  the post itself says so;
* ``platform_get_social_post`` lists the new render among what the post still needs;
* the answer helpers add nothing for a post whose file is current, nor for an approved post (a render is
  refused while it holds the approval), and the compact post keeps its keys.
"""
from __future__ import annotations

from tests.test_prd251w1_social_post_tools import (  # its environment first
    _created,
    _draft_fields,
    _row,
    _template,
    _tool,
)
from tests.test_prd251w1_social_post_tools import env as _env

from modules.socials import service  # noqa: E402
from modules.tools.discovery import social_post_list as listing  # noqa: E402
from modules.tools.discovery import social_post_results as results  # noqa: E402

env = _env  # the US-116 fixture: SQLite, the real router and executor

VIDEO = {"9:16": [{"deliverable_id": "d-video", "name": "video-9x16.mp4", "sha256": "e" * 64, "bytes": 10,
                   "content_type": "video/mp4", "duration": 30.0}]}


def _rendered(env, post_id):  # noqa: F811
    """The post rendered from what it holds now, as the render's finish leaves it: it waits for approval."""
    row = _row(env, post_id)
    service.start_render(row, "render")
    service.finish_render(row, "render", VIDEO)
    env.session.commit()
    return row


def test_an_agents_edit_that_leaves_the_file_older_says_to_render_again(env):  # noqa: F811
    template_id = _template(env)
    post = _created(_tool(env, "platform_create_social_post", **_draft_fields(), template=str(template_id), render=False))
    _rendered(env, post["id"])

    caption = _tool(env, "platform_update_social_post", post_id=post["id"], copy={"base": "Two weeks to go."})
    assert caption["success"] is True and caption["message"] == results.CHANGED, caption
    assert results.FILE_CURRENT not in caption
    assert _row(env, post["id"]).to_dict()["file_current"] is True  # the caption is not on the file

    edited = _tool(
        env, "platform_update_social_post", post_id=post["id"],
        variables={"headline": {"value": "Two weeks to go", "claim": True}},
    )
    assert edited["success"] is True, edited
    assert edited[results.FILE_CURRENT] is False
    assert edited["message"] == results.FILE_OLDER.format(id=post["id"])
    assert '"render": true' in edited["message"]
    assert _row(env, post["id"]).to_dict()["file_current"] is False

    read = _tool(env, "platform_get_social_post", post_id=post["id"])
    assert read["success"] is True and read["post"]["file_current"] is False
    assert results.NEEDS_NEW_RENDER in read["post"]["needs"]


def test_the_answers_add_nothing_for_a_post_whose_file_is_current():
    current = {"id": "p1", "title": "Open late", "status": "needs_approval", "format": "image", "file_current": True,
               "review_log": []}
    assert results.saved(current, None, results.CHANGED) == {
        "success": True, "post": results.post_view(current, None), "message": results.CHANGED,
    }
    assert results.NEEDS_NEW_RENDER not in results.read_view(current)["needs"]

    older = {**current, "file_current": False}
    answer = results.saved(older, None, results.CHANGED)
    assert (answer["success"], answer[results.FILE_CURRENT]) == (True, False)
    assert answer["message"] == results.FILE_OLDER.format(id="p1")
    assert answer["post"] == results.post_view(current, None)  # the compact post is unchanged
    assert results.read_view(older)["needs"][-1] == results.NEEDS_NEW_RENDER
    # One rule for the read, a list row and a ticket's note (social_post_list.what_it_needs): the render
    # comes first, so a post the owner cannot approve yet is not said to need the owner's approval.
    assert listing.what_it_needs(current) == [listing.NEEDS_APPROVAL, listing.NEEDS_CAPTION]
    assert listing.what_it_needs(older) == [listing.NEEDS_CAPTION, results.NEEDS_NEW_RENDER]
    assert listing.post_row(older)["needs"] == results.read_view(older)["needs"]

    # An approved post keeps its approved file, and a render is refused while it holds the approval:
    # a change to a render setting there (its voice, its footage) is not a render to ask for.
    approved = {**older, "status": "approved"}
    assert results.saved(approved, None, results.CHANGED)["message"] == results.CHANGED
    assert results.FILE_CURRENT not in results.saved(approved, None, results.CHANGED)
    assert results.NEEDS_NEW_RENDER not in results.read_view(approved)["needs"]
    assert listing.what_it_needs(approved) == []
