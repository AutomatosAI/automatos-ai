"""F378 (night 11, 7 Oct): a retake starts from the post as it is, and can be undone.

B11: a retake ignored "don't invent tasting notes". B16: a post with no brief and no copy
was retaken from old workspace notes and replaced the good version, with no undo. Friction
13: a draft or a failed post could not be retaken. Pinned, on the US-B111 retake harness:

* a draft and a failed post are retaken; the composer is given the post's current take
  (its copy and its fields' values), and the prompt says to change only what the guidance
  asks, the guidance being a hard rule;
* a post with no brief, no copy and no fields is 422 ``nothing_to_retake``, nothing composed;
* the take a retake replaces is kept in the history, and ``/retake/undo`` restores it (409
  ``nothing_to_undo`` once there is none left), logged;
* the composer's warnings and questions come back with the post (``take``);
* the undo route is in the manifest, and apiClient posts to it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251bw1_retake as retake_tests  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from modules.socials import compose, retakes as post_retakes  # noqa: E402
from tests.test_prd251bw1_retake import API_CLIENT, MANIFEST, TAKE, _retake  # noqa: E402
from tests.test_prd251w1_render_lifecycle import _create, _post, _renderable  # noqa: E402

env = render_harness.env
retakes = retake_tests.retakes
UNDO = "/api/socials/posts/{post_id}/retake/undo"


def _undo(env, post_id):
    return env.client.post(UNDO.format(post_id=post_id))


def _text_post(env):
    return _create(env, format="text", variables={}, copy={"base": "Harbour Blend is back, £12 a bag."})


def test_a_draft_is_retaken_from_its_current_take(retakes):
    post = _renderable(retakes)  # a draft
    resp = _retake(retakes, post["id"], "Don't invent tasting notes")
    assert resp.status_code == 202, resp.text
    assert resp.json()["copy"]["base"] == TAKE["copy"]["base"]
    (extra,) = retakes.retakes.extras
    assert extra["current_take"] == {"copy": {"base": "Three weeks to go."}, "variables": {"headline": "Three weeks to go"}}


def test_a_failed_post_is_retaken(retakes):
    post = _renderable(retakes)
    row = _post(retakes, post["id"])
    row.status = "failed"
    retakes.session.commit()
    assert _retake(retakes, post["id"]).status_code == 202
    assert len(retakes.retakes.bodies) == 1


def test_a_post_with_nothing_to_start_from_is_refused(retakes):
    post = _create(retakes, format="text", variables={}, copy={})
    resp = _retake(retakes, post["id"])
    assert resp.status_code == 422
    assert resp.json()["detail"] == {"code": "nothing_to_retake", "message": post_retakes.NOTHING_TO_RETAKE}
    assert retakes.retakes.bodies == []


def test_the_previous_take_is_kept_and_the_undo_restores_it(retakes):
    post = _text_post(retakes)
    assert _retake(retakes, post["id"], "Shorter").status_code == 202
    row = _post(retakes, post["id"])
    assert row.copy == TAKE["copy"]
    entry = next(e for e in row.review_log if e["action"] == post_retakes.ACTION_RETAKE)
    assert entry["previous"]["copy"] == {"base": "Harbour Blend is back, £12 a bag."} and entry["guidance"] == "Shorter"

    resp = _undo(retakes, post["id"])
    assert resp.status_code == 202, resp.text
    assert resp.json()["copy"] == {"base": "Harbour Blend is back, £12 a bag."}
    row = _post(retakes, post["id"])
    assert row.variables == {} and row.review_log[-1]["action"] == post_retakes.ACTION_RETAKE_UNDONE
    assert row.review_log[-1]["undid"] == entry["at"]

    again = _undo(retakes, post["id"])
    assert again.status_code == 409 and again.json()["detail"]["code"] == "nothing_to_undo"


def test_the_composers_warnings_and_questions_come_back_with_the_post(retakes):
    retakes.retakes.proposal = {**TAKE, "warnings": ["Numbers nobody gave: 46"], "questions": ["Source line"]}
    post = _text_post(retakes)
    resp = _retake(retakes, post["id"])
    assert resp.json()["take"] == {"warnings": ["Numbers nobody gave: 46"], "questions": ["Source line"]}


def test_the_prompt_starts_from_the_current_take_and_holds_the_guidance():
    take = {"copy": {"base": "Harbour Blend is back."}, "variables": {"headline": "Back"}}
    ctx = compose.ComposeContext(brief="Harbour Blend\n\nChanges requested: no tasting notes", format="image",
                                 channels=[], templates=[], candidates=[], current_take=take)
    system, material = compose.build_messages(ctx)
    assert compose.RETAKE_RULES_NOTE in system["content"]
    assert "hard rule" in compose.RETAKE_RULES_NOTE and "change only what" in compose.RETAKE_RULES_NOTE
    assert json.loads(material["content"])["current_take"] == take
    bare = compose.ComposeContext(brief="b", format=None, channels=[], templates=[], candidates=[])
    assert compose.RETAKE_RULES_NOTE not in compose.build_messages(bare)[0]["content"]


def test_the_undo_route_is_wired():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": UNDO} in manifest["routes"]
    client = API_CLIENT.read_text(encoding="utf-8")
    call = re.search(r"async undoSocialPostRetake\(postId: string\)[\s\S]*?\n  \}", client)
    assert call and "`/api/socials/posts/${postId}/retake/undo`" in call.group(0) and "method: 'POST'" in call.group(0)
