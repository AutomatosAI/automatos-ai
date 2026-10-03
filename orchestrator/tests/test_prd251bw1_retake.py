"""PRD-251B Wave 1, US-B111 — another take from the Queue: ``POST /api/socials/posts/{id}/retake``.

On the S1.1c render harness (SQLite, media-render faked) with the composer's model call
replaced. Pinned:

* a retake of a post waiting for approval composes again with the post's own choices (the
  composer is asked for the post's template, length and format), writes the new take's copy,
  variables and sources (a new content hash), keeps the planned slot, renders again, and the
  render ends the post in needs_approval;
* the reviewer's guidance reaches the composer's brief as "Changes requested: ...";
* a post in changes_requested with no template is sent for approval again at once;
* outside needs_approval and changes_requested it is 409 and nothing is composed; another
  workspace's post is 404; a viewer is refused (403); a model failure is 502 and nothing
  changes;
* the route is a plain ``def``, in the manifest with its method, and apiClient posts to it.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from fastapi.routing import APIRoute  # noqa: E402

import api.socials as socials_api  # noqa: E402
import api.socials_compose as compose_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from modules.socials import compose  # noqa: E402
from tests.test_prd251w1_render_lifecycle import (  # noqa: E402
    WS_OTHER, FakeStore, Renderer, _create, _ctx, _post, _renderable, _run,
)

env = render_harness.env
ROUTE = "/api/socials/posts/{post_id}/retake"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
API_CLIENT = _ORCH.parent / "frontend" / "lib" / "api-client.ts"
TAKE = {
    "title": "A different title",
    "copy": {"base": "Three weeks: the second take.", "per_channel": {"linkedin": "Three weeks to go, said better."}},
    "format": "video",
    "variables": {"headline": {"value": "The second take", "claim": False}},
    "sources": {},
}


@pytest.fixture
def retakes(env, monkeypatch):
    state = SimpleNamespace(bodies=[], proposal=TAKE, fail=None)

    def context(db, workspace_id, body):
        state.bodies.append(body)
        return SimpleNamespace(brief=body.brief)

    async def propose(context, llm_factory, timeout):
        if state.fail is not None:
            raise state.fail
        return state.proposal

    monkeypatch.setattr(compose_api, "compose_context", context)
    monkeypatch.setattr(compose, "propose", propose)
    monkeypatch.setattr(compose, "llm_factory", lambda workspace_id: None)
    env.retakes = state
    return env


def _waiting(env, **body):
    post = _renderable(env, length_seconds=40, **body)
    submitted = env.client.post(f"/api/socials/posts/{post['id']}/submit")
    assert submitted.status_code == 200, submitted.text
    return submitted.json()


def _retake(env, post_id, guidance=None):
    return env.client.post(f"/api/socials/posts/{post_id}/retake", json={"guidance": guidance} if guidance else None)


def test_a_retake_composes_with_the_posts_choices_keeps_the_slot_and_renders_again(retakes):
    post = _waiting(retakes)
    slot = (datetime.now(timezone.utc) + timedelta(days=3)).replace(microsecond=0)
    assert retakes.client.put(f"/api/socials/posts/{post['id']}/slot", json={"planned_for": slot.isoformat()}).status_code == 200

    resp = _retake(retakes, post["id"])
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "rendering" and body["content_hash"] != post["content_hash"]
    assert body["copy"] == {"base": "Three weeks: the second take.", "channels": {"linkedin": "Three weeks to go, said better."}}
    assert body["variables"] == TAKE["variables"] and body["title"] == post["title"]  # the take's copy, not its title
    assert body["planned_for"].startswith(slot.isoformat()[:16])
    (asked,) = retakes.retakes.bodies
    assert str(asked.template_id) == post["template_id"] and asked.length_seconds == 40 and asked.format == "video"
    assert asked.brief == post["brief"] or asked.brief == post["title"]

    (job,) = retakes.launched
    assert _run(job, Renderer(), FakeStore(), retakes.factory) is True
    assert _post(retakes, post["id"]).status == "needs_approval"


def test_the_guidance_reaches_the_composer(retakes):
    post = _waiting(retakes)
    assert _retake(retakes, post["id"], "Shorter, and lead with the date").status_code == 202
    (asked,) = retakes.retakes.bodies
    assert asked.brief.endswith("\n\nChanges requested: Shorter, and lead with the date")


def test_a_post_without_a_template_is_sent_for_approval_again(retakes):
    post = _create(retakes, format="text", variables={})
    assert retakes.client.post(f"/api/socials/posts/{post['id']}/submit").status_code == 200
    asked = retakes.client.post(f"/api/socials/posts/{post['id']}/request-changes", json={"comment": "Warmer"})
    assert asked.status_code == 200 and asked.json()["status"] == "changes_requested"
    resp = _retake(retakes, post["id"], "Warmer")
    assert resp.status_code == 202, resp.text
    assert resp.json()["status"] == "needs_approval" and retakes.launched == []


def test_outside_the_queue_statuses_it_is_409_and_nothing_is_composed(retakes):
    post = _renderable(retakes)  # a draft
    resp = _retake(retakes, post["id"])
    assert resp.status_code == 409
    assert retakes.retakes.bodies == [] and _post(retakes, post["id"]).content_hash == post["content_hash"]


def test_another_workspaces_post_is_404_and_a_viewer_is_refused(retakes):
    post = _waiting(retakes)
    retakes.role = "viewer"
    assert _retake(retakes, post["id"]).status_code == 403
    retakes.role = "owner"
    retakes.ctx = _ctx(WS_OTHER)
    assert _retake(retakes, post["id"]).status_code == 404
    assert retakes.retakes.bodies == []


def test_a_model_failure_is_502_and_nothing_changes(retakes):
    post = _waiting(retakes)
    retakes.retakes.fail = compose.ComposeFailed("the answer could not be read")
    resp = _retake(retakes, post["id"])
    assert resp.status_code == 502
    row = _post(retakes, post["id"])
    assert row.status == "needs_approval" and row.content_hash == post["content_hash"]


def test_the_route_is_wired():
    (route,) = [r for r in socials_api.router.routes if isinstance(r, APIRoute) and r.path == ROUTE]
    assert route.methods == {"POST"} and route.status_code == 202
    assert not inspect.iscoroutinefunction(route.endpoint), "a route over a sync Session is a plain def (F105)"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "POST", "path": ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
    client = API_CLIENT.read_text(encoding="utf-8")
    call = re.search(r"async retakeSocialPost\(postId: string, guidance\?: string\)[\s\S]*?\{([\s\S]*?)\n  \}", client)
    assert call, "apiClient.retakeSocialPost"
    assert "`/api/socials/posts/${postId}/retake`" in call.group(1) and "method: 'POST'" in call.group(1)
