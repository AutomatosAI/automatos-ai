"""PRD-251C Wave 4, US-C406 — voice from edits.

On the S0.3b API harness (SQLite). Pinned:

* an agent's draft keeps the copy it wrote in its history entry; a person's edit does not;
* approving a post whose copy a person changed from Auto's draft keeps the pair; approving it
  unchanged, or a post no agent drafted, keeps nothing;
* the workspace keeps its newest ``SOCIALS_VOICE_EXAMPLES``;
* ``GET /api/socials/voice-examples`` lists them newest first; an owner removes one
  (``DELETE``), a viewer may not, another workspace's is a 404; a removed one is never given
  to the composer, which gets the rest with the brand voice.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import anyio

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from config import config  # noqa: E402
from core.models.socials import SocialPost, SocialVoiceExample  # noqa: E402
from modules.socials import compose, service, voice_examples  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _ctx  # noqa: E402

api = api_harness.api
AGENT = "Social Media Director"


def _created(api, base, agent=None):
    fields = {"title": "Lisbon", "format": "text", "copy": {"base": base, "channels": {}}}
    return anyio.run(functools.partial(socials_api.create_post, api.session, WS_A, "member-1", fields, agent=agent))


def _drafted_by_auto(api, base):
    """A post an agent drafted (as the plan maker does), waiting for approval."""
    row = _created(api, base, agent=AGENT)
    service.submit(row, "member-1")
    api.session.commit()
    return row


def _edit_and_approve(api, row, base=None):
    if base is not None:
        api.client.patch(f"/api/socials/posts/{row.id}", json={"copy": {"base": base, "channels": {}}})
    api.session.expire_all()
    current = api.session.get(SocialPost, row.id)
    if current.status != "needs_approval":
        api.client.post(f"/api/socials/posts/{row.id}/submit")
        api.session.expire_all()
        current = api.session.get(SocialPost, row.id)
    resp = api.client.post(f"/api/socials/posts/{row.id}/approve", json={"content_hash": current.content_hash})
    assert resp.status_code == 200, resp.text


def _kept(api):
    api.session.expire_all()
    return [(row.draft, row.approved) for row in voice_examples.examples(api.session, WS_A)]


def test_an_agents_draft_keeps_what_it_wrote_and_a_persons_edit_does_not(api):
    row = _drafted_by_auto(api, "We are at Web Summit. Come and say hello!!!")
    assert voice_examples.auto_draft(row) == "We are at Web Summit. Come and say hello!!!"
    service.update_post(row, "member-1", {"copy": {"base": "Find us at stand B12.", "channels": {}}})
    assert voice_examples.auto_draft(row) == "We are at Web Summit. Come and say hello!!!"  # still Auto's own


def test_a_rewrite_approved_is_kept_and_an_unchanged_approval_is_not(api):
    rewritten = _drafted_by_auto(api, "We are at Web Summit. Come and say hello!!!")
    _edit_and_approve(api, rewritten, "Web Summit, stand B12. Come by.")
    assert _kept(api) == [("We are at Web Summit. Come and say hello!!!", "Web Summit, stand B12. Come by.")]
    unchanged = _drafted_by_auto(api, "Three days to go.")
    _edit_and_approve(api, unchanged)
    _edit_and_approve(api, _created(api, "Mine."), "Mine, edited.")  # no agent drafted it
    assert len(_kept(api)) == 1


def test_the_workspace_keeps_its_newest_examples(api, monkeypatch):
    monkeypatch.setattr(config, "SOCIALS_VOICE_EXAMPLES", 2)
    for n in range(3):
        _edit_and_approve(api, _drafted_by_auto(api, f"Auto's draft {n}."), f"Ours {n}.")
    assert [approved for _, approved in _kept(api)] == ["Ours 2.", "Ours 1."]
    assert api.session.query(SocialVoiceExample).count() == 2


def test_the_owner_lists_and_removes_examples_and_the_composer_never_gets_a_removed_one(api):
    for n in range(2):
        _edit_and_approve(api, _drafted_by_auto(api, f"Auto's draft {n}."), f"Ours {n}.")
    listed = api.client.get("/api/socials/voice-examples").json()["examples"]
    assert [item["approved"] for item in listed] == ["Ours 1.", "Ours 0."]
    api.role = "viewer"
    assert api.client.delete(f"/api/socials/voice-examples/{listed[0]['id']}").status_code == 403
    api.role = "owner"
    api.ctx = _ctx(WS_B)
    assert api.client.delete(f"/api/socials/voice-examples/{listed[0]['id']}").status_code == 404
    api.ctx = _ctx(WS_A)
    assert api.client.delete(f"/api/socials/voice-examples/{listed[0]['id']}").status_code == 204
    given = voice_examples.for_composer(api.session, WS_A)
    assert given == [{"auto_wrote": "Auto's draft 0.", "approved": "Ours 0."}]
    ctx = compose.ComposeContext(brief="Lisbon", format="text", channels=[], templates=[], candidates=[], voice_examples=tuple(given))
    system, user = compose.build_messages(ctx)
    assert compose.VOICE_EXAMPLES_NOTE in system["content"] and '"voice_examples"' in user["content"] and "Ours 1." not in user["content"]
