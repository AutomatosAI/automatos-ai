"""PRD-251D Wave 2, US-D108 — a file older than its words is never approved.

A post that waited for approval, edited and saved, kept its old file, and Approve accepted that
file with the new words: nothing recorded what a file was made from. Now a render records it
(``core/social_file_current.py``: a hash of what a render reads, ``made_from`` on its
``render_done`` entry), a post says ``file_current`` in its JSON, and an approval refuses a post
whose file is not current, saying to render it again. Pinned:

* a render records what it read, and the post is current;
* an edit after a render to a field, the template, the format or the length makes the file older:
  Approve is refused with the sentence and the post is unchanged; a render makes it current
  again and Approve passes; a render that fails after the edit leaves it older;
* a slot move, a caption change, a channel change, a title change, a brief change and a claim mark
  leave it current, and the post is approved;
* a new picture in a photo spot leaves it older; what a shot cost does not, nor does the editor's
  save that sends the spot's prompt back;
* for a video a new voice, new music or a new length leave it older; an image render reads no
  voice or music;
* a text post, a post with no template, a post never rendered and a post rendered before this
  record existed are current, and so is a history the reader cannot use;
* the approve route answers 409 with the sentence (Approve & post now approves through it); a
  plan's week leaves such a post out as ``render_again`` with the sentence and approves the rest;
  the series core does the same;
* the literal names in ``core`` are ``modules.socials``'s.
"""
from __future__ import annotations

import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402  (its environment first)
from core import social_file_current as file_state  # noqa: E402
from core.models.socials import SOCIAL_POST_FORMATS, SocialPost  # noqa: E402
from core.social_render_state import RENDER_DONE  # noqa: E402
from modules.socials import ai_options, campaigns, service, slot_photos  # noqa: E402
from tests.test_prd251_api import WS_A, _create  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)
from tests.test_prd251cw2_approve_week import _approve as _approve_week  # noqa: E402
from tests.test_prd251cw2_approve_week import _post as _week_post  # noqa: E402
from tests.test_prd251cw2_approve_week import _statuses  # noqa: E402

api = api_harness.api

AUTHOR, REVIEWER = "member-1", "reviewer-1"
TEMPLATE, OTHER_TEMPLATE = uuid.uuid4(), uuid.uuid4()
FRIDAY = {"headline": {"value": "OPEN LATE FRIDAY", "claim": False}}
SATURDAY = {"headline": {"value": "OPEN LATE SATURDAY", "claim": False}}
SQUARE = {"1:1": [{"deliverable_id": "d-square", "name": "image-1x1.png", "sha256": "f" * 64, "bytes": 10,
                   "content_type": "image/png"}]}
VIDEO = {"9:16": [{"deliverable_id": "d-video", "name": "video-9x16.mp4", "sha256": "e" * 64, "bytes": 10,
                   "content_type": "video/mp4", "duration": 15.0}]}
LINKEDIN = {"toolkit": "linkedin", "post_kind": "image", "options": {}}
# The person's own picture in the photo spot, as slot_photos.own_file_record writes it.
OWN_PHOTO = {"prompt": "Photo: your own picture (shop.png)", "status": "done", "toolkit": "upload",
             "deliverable_id": "d-shop", "name": "shop.png", "content_type": "image/png"}
OPTION = {"name": "photo_option_1.png", "deliverable_id": "d-option-1", "sha256": "a" * 64, "bytes": 12,
          "content_type": "image/png", "toolkit": "fal_ai"}


class FakeDb:
    def add(self, row):
        self.row = row


def _draft(**fields):
    """An image post on a template with one field, as a person or an agent drafts it."""
    base = {"title": "Open late", "copy": {"base": "Open late on Friday."}, "format": "image", "template_id": TEMPLATE,
            "variables": FRIDAY}
    return service.create_draft(FakeDb(), workspace_id=WS_A, created_by=AUTHOR, **{**base, **fields})


def _render(post, media=SQUARE):
    """The post rendered from what it holds now: it waits for approval with its file."""
    service.start_render(post, AUTHOR)
    return service.finish_render(post, "render", media)


def _current(post) -> bool:
    return post.to_dict()["file_current"]


# ---------------------------------------------------------------------------
# The record, and what makes a file older than its words
# ---------------------------------------------------------------------------


def test_a_render_records_what_its_file_was_made_from():
    post = _render(_draft())
    done = post.review_log[-1]
    assert done["action"] == service.ACTION_RENDER_DONE
    assert re.fullmatch(r"[0-9a-f]{64}", done[file_state.MADE_FROM])
    assert done[file_state.MADE_FROM] == file_state.render_inputs_hash(post)
    assert post.status == service.NEEDS_APPROVAL and _current(post) is True


@pytest.mark.parametrize("change", [
    {"variables": SATURDAY},
    {"template_id": OTHER_TEMPLATE},
    {"format": "carousel"},
    {"length_seconds": 30},
], ids=["a field", "the template", "the format", "the length"])
def test_an_edit_after_a_render_is_refused_until_a_render_makes_the_file_again(change):
    post = _render(_draft())
    service.update_post(post, AUTHOR, change)
    assert post.status == service.NEEDS_APPROVAL and _current(post) is False

    before = (post.status, post.approved_hash, post.approved_by, list(post.review_log))
    with pytest.raises(service.NotPublishable) as refused:
        service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert str(refused.value) == file_state.FILE_OLDER_THAN_WORDS
    assert "Render it again" in str(refused.value)
    assert (post.status, post.approved_hash, post.approved_by, post.review_log) == before

    _render(post)  # rendered again, from what it holds now
    assert _current(post) is True
    service.approve(post, REVIEWER, content_hash=post.content_hash)
    assert post.status == service.APPROVED and post.approved_hash == post.content_hash


def test_a_render_that_fails_after_the_edit_leaves_the_file_as_old_as_it_was():
    post = _render(_draft())
    service.update_post(post, AUTHOR, {"variables": SATURDAY})
    service.start_render(post, AUTHOR)
    service.fail_render(post, AUTHOR, "The renderer could not be reached.")
    assert post.status == service.FAILED
    assert _current(post) is False  # the file is still the one made from Friday's words
    _render(post)
    assert _current(post) is True


@pytest.mark.parametrize("edit", [
    lambda p: service.set_planned_for(p, datetime.now(timezone.utc) + timedelta(days=2), "Europe/Dublin"),
    lambda p: service.update_post(p, AUTHOR, {"copy": {"base": "Open late on Friday, till ten."}}),
    lambda p: service.update_post(p, AUTHOR, {"targets": [LINKEDIN]}),
    lambda p: service.update_post(p, AUTHOR, {"title": "Open late, again"}),
    lambda p: service.update_post(p, AUTHOR, {"brief": "Say it louder"}),
    lambda p: service.update_post(p, AUTHOR, {"variables": {"headline": {**FRIDAY["headline"], "claim": True}}}),
], ids=["a slot move", "a caption change", "a channel change", "a title change", "a brief change", "a claim mark"])
def test_a_change_that_is_not_on_the_file_leaves_it_current_and_the_post_is_approved(edit):
    post = _render(_draft())
    edit(post)
    assert _current(post) is True
    service.approve(post, REVIEWER, content_hash=post.content_hash, override_unsourced=True)
    assert post.status in (service.APPROVED, service.SCHEDULED)  # a slot still ahead schedules it


def test_a_new_picture_in_a_photo_spot_leaves_the_file_older_and_what_a_shot_cost_does_not():
    post = _draft()
    slot_photos.with_own_file(post, "photo", OWN_PHOTO)
    _render(post)
    assert _current(post) is True

    # The editor's save sends each photo spot's prompt back: the spot keeps what it holds.
    service.update_post(post, AUTHOR, {"footage": {"photo": {"prompt": OWN_PHOTO["prompt"]}}})
    assert post.footage["photo"]["deliverable_id"] == "d-shop" and _current(post) is True
    # What a shot cost and when it was made are not on the file.
    post.footage = {"photo": {**post.footage["photo"], "cost_usd": 0.04, "generated_at": "2026-11-16T09:00:00+00:00"}}
    assert _current(post) is True

    # Options asked for the spot, then one picked: the file shows a picture the post no longer holds.
    ai_options.ask(post, "photo", "a shop front at dusk")
    assert _current(post) is False
    assert ai_options.settle(post, "photo", "a shop front at dusk", [OPTION], None)
    assert ai_options.pick(post, "photo", OPTION["name"])
    assert _current(post) is False
    _render(post)
    assert _current(post) is True


@pytest.mark.parametrize("change", [
    {"voice": {"toolkit": "kokoro", "voice_id": "af_bella"}},
    {"music": {"track": None}},
    {"length_seconds": 30},
], ids=["the voice", "the music", "the length"])
def test_a_videos_voice_music_or_length_changed_after_its_render_leaves_its_file_older(change):
    post = _render(_draft(format="video", length_seconds=15), VIDEO)
    assert _current(post) is True
    service.update_post(post, AUTHOR, change)
    assert _current(post) is False


def test_an_image_render_reads_no_voice_and_no_music():
    post = _render(_draft())
    service.update_post(post, AUTHOR, {"voice": {"toolkit": "kokoro", "voice_id": "af_bella"}, "music": {"track": None}})
    assert _current(post) is True


# ---------------------------------------------------------------------------
# Always current
# ---------------------------------------------------------------------------


def test_a_text_post_and_a_post_with_no_template_are_always_current():
    text = _draft(format="text", template_id=None, variables=None)
    text.review_log = [{"action": RENDER_DONE, file_state.MADE_FROM: "0" * 64}]
    assert _current(text) is True

    own = _draft(template_id=None, media={"original": ["d-own"]})  # the person's own still, cropped per channel
    service.start_render(own, AUTHOR)
    service.finish_render(own, "render", SQUARE, keep=("original",))
    service.update_post(own, AUTHOR, {"variables": SATURDAY})
    assert _current(own) is True
    service.approve(own, REVIEWER, content_hash=own.content_hash)
    assert own.status == service.APPROVED


def test_a_post_never_rendered_and_one_rendered_before_the_record_are_current():
    never = _draft()
    service.submit(never, AUTHOR)
    assert _current(never) is True

    # Rendered before US-D108: its render_done entry has no record. Nothing that exists today is refused.
    before = _render(_draft())
    before.review_log = [{key: value for key, value in entry.items() if key != file_state.MADE_FROM}
                         for entry in before.review_log]
    service.update_post(before, AUTHOR, {"variables": SATURDAY})
    assert _current(before) is True
    service.approve(before, REVIEWER, content_hash=before.content_hash)
    assert before.status == service.APPROVED


def test_a_history_the_reader_cannot_use_reads_as_current_and_never_raises():
    for log in (None, [], ["not an entry", None, 7], [{"action": RENDER_DONE, file_state.MADE_FROM: 7}]):
        post = SimpleNamespace(format="image", template_id=TEMPLATE, review_log=log, variables=None, footage=None)
        assert file_state.file_current(post) is True
    unsaved = SocialPost(title="Not flushed yet", format="image", template_id=TEMPLATE)  # every JSON column None
    assert unsaved.to_dict()["file_current"] is True


def test_the_names_in_core_are_the_lifecycles():
    assert RENDER_DONE == service.ACTION_RENDER_DONE
    assert file_state.FOOTAGE_DONE == service.FOOTAGE_DONE
    assert {file_state.TEXT_FORMAT, file_state.VIDEO_FORMAT} <= set(SOCIAL_POST_FORMATS)


# ---------------------------------------------------------------------------
# Every approve path: the route, a plan's week, a series
# ---------------------------------------------------------------------------


def _rendered_in(session, post_id):
    """The committed post on a template with Friday's words, rendered: it waits for approval with its file."""
    row = session.get(SocialPost, uuid.UUID(str(post_id)))
    row.format, row.template_id, row.variables = "image", TEMPLATE, FRIDAY
    _render(row)
    session.commit()
    return row


def test_the_approve_route_refuses_a_file_older_than_its_words_with_409_and_the_sentence(api):  # noqa: F811
    post = _create(api)
    _rendered_in(api.session, post["id"])
    url = f"/api/socials/posts/{post['id']}"
    assert api.client.get(url).json()["file_current"] is True

    edited = api.client.patch(url, json={"variables": SATURDAY})
    assert edited.status_code == 200, edited.text
    assert (edited.json()["status"], edited.json()["file_current"]) == ("needs_approval", False)
    refused = api.client.post(f"{url}/approve", json={"content_hash": edited.json()["content_hash"]})
    assert refused.status_code == 409 and refused.json()["detail"] == file_state.FILE_OLDER_THAN_WORDS
    assert api.client.get(url).json()["status"] == "needs_approval"

    row = api.session.get(SocialPost, uuid.UUID(post["id"]))
    _render(row)  # rendered again: made from Saturday's words
    api.session.commit()
    again = api.client.get(url).json()
    assert again["file_current"] is True
    approved = api.client.post(f"{url}/approve", json={"content_hash": again["content_hash"]})
    assert approved.status_code == 200 and approved.json()["status"] == "approved", approved.text


def _older(session, post):
    """``post`` (a plan's) as a render of Friday's words left it, then edited to Saturday's."""
    post.format, post.template_id, post.variables = "image", TEMPLATE, FRIDAY
    post.media = {"1:1": ["d-square"]}
    post.review_log = [{"at": "2026-10-18T09:00:00+00:00", "by": "render", "action": RENDER_DONE, "comment": None,
                        file_state.MADE_FROM: file_state.render_inputs_hash(post)}]
    post.variables = SATURDAY
    post.content_hash = service.compute_content_hash(post)
    session.commit()
    return post


def test_a_plans_week_leaves_a_file_older_than_its_words_out_saying_why_and_approves_the_rest(bank):  # noqa: F811
    plan = _create_plan(bank)
    kept = _week_post(bank, plan, 19)
    older = _older(bank.session, _week_post(bank, plan, 20))
    resp = _approve_week(bank, plan, [kept, older])
    assert resp.status_code == 200, resp.text  # never the whole batch
    body = resp.json()
    assert [row["id"] for row in body["approved"]] == [str(kept.id)]
    (left,) = body["left"]
    assert (left["post_id"], left["reason"], left["message"]) == (
        str(older.id), campaigns.LEFT_RENDER_AGAIN, file_state.FILE_OLDER_THAN_WORDS,
    )
    assert _statuses(bank, [kept, older]) == ["scheduled", "needs_approval"]


def test_a_series_leaves_a_file_older_than_its_words_out_saying_why(monkeypatch):
    post = _render(_draft())
    service.update_post(post, AUTHOR, {"variables": SATURDAY})
    campaign_id = uuid.uuid4()
    post.campaign_id = campaign_id
    monkeypatch.setattr(campaigns.service, "get_post", lambda db, workspace_id, post_id: post)
    monkeypatch.setattr(campaigns.post_sources, "unresolved", lambda db, workspace_id, sources: {})
    monkeypatch.setattr(campaigns.template_claims, "names", lambda db, workspace_id, template_id: frozenset())
    item = campaigns.ShownPost(post_id=post.id, content_hash=post.content_hash)

    done, left = campaigns._approve_one(object(), (post.workspace_id, campaign_id), REVIEWER, item, None)

    assert done is None and post.status == service.NEEDS_APPROVAL and post.approved_hash is None
    assert (left["reason"], left["message"]) == (campaigns.LEFT_RENDER_AGAIN, file_state.FILE_OLDER_THAN_WORDS)
