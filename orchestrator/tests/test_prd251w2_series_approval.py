"""PRD-251 Wave 2, US-210 (S2.4, D6) — campaigns and series approval.

The real Socials router (with ``api/socials_campaigns.py`` included) on the S0.3b
SQLite harness (``tests/test_prd251_api.py``, its ``social_campaigns`` table
included). Pins:

* **The workspace switch.** ``workspace.settings['socials'].series_approval`` is
  off unless it is a real ``True`` (fail-closed); ``PUT
  /api/workspaces/current/socials`` sets it (validated as a boolean) and ``GET
  /api/workspaces/current`` reports it beside ``available``/``enabled``.
* **Series approval approves the shown posts, hash-bound (D6).** Each post of the
  campaign waiting for approval whose current hash is the one the approver was
  shown is approved as a single approval would approve it, and its hash joins
  the campaign's ``approved_hash_set``; the campaign records who and when.
* **Nothing else is approved.** A post added afterwards stays needs_approval; an
  edit after the series approval voids that post's approval like any edit; a
  post whose hash changed between display and approval (also mid-request, by
  another worker: the compare-and-set) is left unapproved and reported; a post
  of the campaign the approver was not shown is reported, not approved.
* **Unsourced claims (D7)** need that post's own override, and a claim whose
  source no longer resolves counts as unsourced, as for a single approval.
* **The switch and the mode.** With the workspace switch off, or the campaign in
  ``per_post`` mode, the route answers 409 and approves nothing.
* **Tenant isolation, the role, and the routes.** Another workspace's campaign or
  post is 404; a viewer cannot approve a series; every route is a plain ``def``,
  behind the gate first, and in the committed route manifest.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import uuid
from pathlib import Path

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
from sqlalchemy.orm import sessionmaker  # noqa: E402

import api.socials as socials_api  # noqa: E402
import api.socials_campaigns as campaigns_api  # noqa: E402
import modules.socials.campaigns as campaigns_mod  # noqa: E402
import modules.socials.service as socials_service  # noqa: E402
import modules.socials.settings as socials_settings  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251_settings as settings_harness  # noqa: E402
from core.models.socials import SocialCampaign, SocialPost  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials.settings import require_socials_enabled  # noqa: E402
from tests.test_prd251_api import WS_A, WS_B, _create, _ctx, _post  # noqa: E402

# The S0.3b routes on SQLite (api); the workspace switch's PUT/GET fakes (master, flag_spy).
api = api_harness.api
master = settings_harness.master
flag_spy = settings_harness.flag_spy

CAMPAIGNS = "/api/socials/campaigns"
MANIFEST = _ORCH / "reports" / "route-manifest.json"
NEW_ROUTES = [
    ("GET", "/api/socials/campaigns"),
    ("POST", "/api/socials/campaigns"),
    ("GET", "/api/socials/campaigns/{campaign_id}"),
    ("PATCH", "/api/socials/campaigns/{campaign_id}"),
    ("POST", "/api/socials/campaigns/{campaign_id}/posts/{post_id}"),
    ("DELETE", "/api/socials/campaigns/{campaign_id}/posts/{post_id}"),
    ("POST", "/api/socials/campaigns/{campaign_id}/approve"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _series_switch(api, on: bool, workspace_id=WS_A):
    workspace = api.session.get(Workspace, workspace_id)
    workspace.settings = {"socials": {"enabled": True, "series_approval": on}}
    api.session.commit()


def _campaign(api, name="Web Summit countdown", mode="series"):
    resp = api.client.post(CAMPAIGNS, json={"name": name, "approval_mode": mode})
    assert resp.status_code == 201, resp.text
    return resp.json()


def _add(api, campaign_id, post_id):
    return api.client.post(f"{CAMPAIGNS}/{campaign_id}/posts/{post_id}")


def _waiting(api, campaign_id, title="Countdown", **body):
    """A post of the campaign, submitted: needs_approval. Its copy is its title
    unless ``body`` says otherwise, so posts of other titles hash apart."""
    post = _create(api, title=title, **{"copy": {"base": title}, **body})
    assert _add(api, campaign_id, post["id"]).status_code == 200
    resp = _post(api, post["id"], "submit")
    assert resp.status_code == 200 and resp.json()["status"] == "needs_approval", resp.text
    return resp.json()


def _shown(*posts, override=False):
    return [
        {"post_id": p["id"], "content_hash": p["content_hash"], "override_unsourced": override} for p in posts
    ]


def _approve(api, campaign_id, shown):
    return api.client.post(f"{CAMPAIGNS}/{campaign_id}/approve", json={"posts": shown})


def _row(api, post_id):
    session = sessionmaker(bind=api.session.get_bind())()
    try:
        return session.get(SocialPost, uuid.UUID(post_id)).to_dict()
    finally:
        session.close()


def _campaign_row(api, campaign_id):
    session = sessionmaker(bind=api.session.get_bind())()
    try:
        return session.get(SocialCampaign, uuid.UUID(campaign_id)).to_dict()
    finally:
        session.close()


def _left(body):
    return {entry["post_id"]: entry for entry in body["left"]}


# ---------------------------------------------------------------------------
# The workspace switch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "settings, expected",
    [
        (None, False),
        ({}, False),
        ({"socials": {"enabled": True}}, False),
        ({"socials": {"enabled": True, "series_approval": "true"}}, False),
        ({"socials": {"enabled": True, "series_approval": 1}}, False),
        ({"socials": "series"}, False),
        ({"socials": {"series_approval": True}}, True),
    ],
)
def test_the_series_switch_is_fail_closed(settings, expected):
    assert socials_settings.parse_workspace_socials(settings).series_approval is expected


def test_a_write_takes_series_approval_only_as_a_boolean():
    assert socials_settings.validate_socials_update({"series_approval": True}) == {"series_approval": True}
    for bad in ("true", 1, None, "on"):
        with pytest.raises(ValueError) as exc:
            socials_settings.validate_socials_update({"series_approval": bad})
        assert "series_approval must be a boolean" in str(exc.value)


def test_put_sets_the_switch_and_get_current_reports_it(monkeypatch, flag_spy, master):
    settings_harness._as_role(monkeypatch, "owner")
    master["value"] = "true"
    ws = settings_harness._workspace({"socials": {"enabled": True, "media_monthly_cap_usd": 25}})
    db = settings_harness._FakeDB(ws)
    client = settings_harness._client(db, settings_harness._member_ctx())

    resp = client.put(settings_harness.SWITCH_ROUTE, json={"socials": {"series_approval": True}})

    assert resp.status_code == 200, resp.text
    assert resp.json()["socials"] == {"available": True, "enabled": True, "series_approval": True}
    assert ws.settings["socials"] == {"enabled": True, "media_monthly_cap_usd": 25, "series_approval": True}
    assert client.get("/api/workspaces/current").json()["socials"]["series_approval"] is True
    bad = client.put(settings_harness.SWITCH_ROUTE, json={"socials": {"series_approval": "yes"}})
    assert bad.status_code == 400 and ws.settings["socials"]["series_approval"] is True


# ---------------------------------------------------------------------------
# Campaigns: create, read, list, rename, posts in and out
# ---------------------------------------------------------------------------


def test_create_list_read_and_patch_a_campaign(api):
    campaign = _campaign(api, mode="per_post")
    assert campaign["approval_mode"] == "per_post" and campaign["posts"] == []
    assert campaign["approved_hash_set"] == [] and campaign["created_by"] == "member-1"
    post = _create(api)
    assert _add(api, campaign["id"], post["id"]).json()["campaign_id"] == campaign["id"]

    listed = api.client.get(CAMPAIGNS).json()
    assert listed["total"] == 1 and listed["campaigns"][0]["post_count"] == 1
    read = api.client.get(f"{CAMPAIGNS}/{campaign['id']}").json()
    assert [p["id"] for p in read["posts"]] == [post["id"]]

    patched = api.client.patch(f"{CAMPAIGNS}/{campaign['id']}", json={"name": "Renamed", "approval_mode": "series"})
    assert patched.status_code == 200 and (patched.json()["name"], patched.json()["approval_mode"]) == ("Renamed", "series")
    assert api.client.patch(f"{CAMPAIGNS}/{campaign['id']}", json={"approval_mode": "all"}).status_code == 422
    assert api.client.post(CAMPAIGNS, json={"name": "   "}).status_code == 422
    assert api.client.post(CAMPAIGNS, json={"name": "X", "approved_hash_set": ["0" * 64]}).status_code == 422

    removed = api.client.delete(f"{CAMPAIGNS}/{campaign['id']}/posts/{post['id']}")
    assert removed.status_code == 200 and removed.json()["campaign_id"] is None
    assert api.client.delete(f"{CAMPAIGNS}/{campaign['id']}/posts/{post['id']}").status_code == 404


def test_joining_a_campaign_changes_neither_the_hash_nor_an_approval(api):
    campaign = _campaign(api)
    post = _create(api)
    submitted = _post(api, post["id"], "submit").json()
    approved = _post(api, post["id"], "approve", {"content_hash": submitted["content_hash"]}).json()

    joined = _add(api, campaign["id"], post["id"]).json()

    assert joined["content_hash"] == approved["content_hash"] and joined["status"] == "approved"
    assert joined["approved_hash"] == approved["approved_hash"]


# ---------------------------------------------------------------------------
# D6 — series approval
# ---------------------------------------------------------------------------


def test_approving_a_series_approves_its_waiting_posts_and_records_the_set(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    first = _waiting(api, campaign["id"], "Three weeks to go")
    second = _waiting(api, campaign["id"], "Two weeks to go")

    resp = _approve(api, campaign["id"], _shown(first, second))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert sorted(p["id"] for p in body["approved"]) == sorted([first["id"], second["id"]])
    assert body["left"] == []
    for post in (first, second):
        row = _row(api, post["id"])
        assert row["status"] == "approved" and row["approved_by"] == "member-1"
        assert row["approved_hash"] == row["content_hash"] == post["content_hash"]
        assert row["review_log"][-1]["action"] == "approve"
    stored = _campaign_row(api, campaign["id"])
    assert stored["approved_hash_set"] == sorted([first["content_hash"], second["content_hash"]])
    assert stored["approved_by"] == "member-1" and stored["approved_at"] is not None
    assert body["campaign"]["approved_hash_set"] == stored["approved_hash_set"]


def test_a_post_added_afterwards_stays_needs_approval(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    first = _waiting(api, campaign["id"], "Three weeks to go")
    assert _approve(api, campaign["id"], _shown(first)).status_code == 200

    later = _waiting(api, campaign["id"], "One week to go")
    # Even a post whose content (so hash) is one the series approved is not covered.
    twin = _waiting(api, campaign["id"], "A twin", copy={"base": "Three weeks to go"})

    approved_set = _campaign_row(api, campaign["id"])["approved_hash_set"]
    assert approved_set == [first["content_hash"]] == [twin["content_hash"]]
    for post in (later, twin):
        row = _row(api, post["id"])
        assert row["status"] == "needs_approval" and row["approved_hash"] is None
        with pytest.raises(socials_service.NotPublishable):
            socials_service.assert_publishable(api.session.get(SocialPost, uuid.UUID(post["id"])))
    assert later["content_hash"] not in approved_set


def test_an_edit_after_series_approval_voids_that_posts_approval(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    edited, untouched = _waiting(api, campaign["id"], "Edited"), _waiting(api, campaign["id"], "Untouched")
    assert _approve(api, campaign["id"], _shown(edited, untouched)).status_code == 200

    resp = api.client.patch(f"/api/socials/posts/{edited['id']}", json={"copy": {"base": "New words."}})

    assert resp.status_code == 200
    row = _row(api, edited["id"])
    assert row["status"] == "needs_approval" and row["approved_hash"] != row["content_hash"]
    assert row["review_log"][-1]["action"] == "approval_voided"
    assert row["content_hash"] not in _campaign_row(api, campaign["id"])["approved_hash_set"]
    assert _row(api, untouched["id"])["status"] == "approved"


def test_with_the_workspace_switch_off_the_series_route_refuses(api):
    _series_switch(api, False)
    campaign = _campaign(api)
    post = _waiting(api, campaign["id"])

    resp = _approve(api, campaign["id"], _shown(post))

    assert resp.status_code == 409
    assert "Series approval is off for this workspace" in resp.json()["detail"]
    assert _row(api, post["id"])["status"] == "needs_approval"
    assert _campaign_row(api, campaign["id"])["approved_hash_set"] == []


def test_a_per_post_campaign_refuses_series_approval(api):
    _series_switch(api, True)
    campaign = _campaign(api, mode="per_post")
    post = _waiting(api, campaign["id"])

    resp = _approve(api, campaign["id"], _shown(post))

    assert resp.status_code == 409 and "approves post by post" in resp.json()["detail"]
    assert _row(api, post["id"])["status"] == "needs_approval"


def test_a_post_whose_hash_changed_since_display_is_left_unapproved_and_reported(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    changed, steady = _waiting(api, campaign["id"], "Changed"), _waiting(api, campaign["id"], "Steady")
    edit = api.client.patch(f"/api/socials/posts/{changed['id']}", json={"copy": {"base": "Edited after display."}})
    current_hash = edit.json()["content_hash"]

    resp = _approve(api, campaign["id"], _shown(changed, steady))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [p["id"] for p in body["approved"]] == [steady["id"]]
    left = _left(body)[changed["id"]]
    assert left["reason"] == "changed" and left["content_hash"] == current_hash
    row = _row(api, changed["id"])
    assert row["status"] == "needs_approval" and row["approved_hash"] is None
    assert _campaign_row(api, campaign["id"])["approved_hash_set"] == [steady["content_hash"]]


def test_an_edit_committed_while_the_series_approval_runs_is_left_unapproved(api, monkeypatch):
    """Another worker commits an edit after the series approval loaded the post and
    before it commits: the compare-and-set leaves the post unapproved."""
    _series_switch(api, True)
    campaign = _campaign(api)
    post = _waiting(api, campaign["id"])
    other_worker = sessionmaker(bind=api.session.get_bind())()
    real_get_post = socials_service.get_post
    edits = []

    def load_then_another_worker_edits(db, workspace_id, post_id):
        loaded = real_get_post(db, workspace_id, post_id)
        if not edits:
            row = other_worker.get(SocialPost, post_id)
            socials_service.update_post(row, "editor-2", {"copy": {"base": "Edited by another worker."}})
            edits.append(row.content_hash)
            other_worker.commit()
        return loaded

    monkeypatch.setattr(socials_service, "get_post", load_then_another_worker_edits)
    try:
        resp = _approve(api, campaign["id"], _shown(post))
    finally:
        other_worker.close()

    assert edits and resp.status_code == 200, resp.text
    assert resp.json()["approved"] == []
    left = _left(resp.json())[post["id"]]
    assert left["reason"] == "changed" and left["content_hash"] == edits[0]
    row = _row(api, post["id"])
    assert row["status"] == "needs_approval" and row["approved_hash"] is None
    assert row["copy"] == {"base": "Edited by another worker."}
    assert _campaign_row(api, campaign["id"])["approved_hash_set"] == []


def test_unsourced_posts_need_the_override(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    claim = {"users": {"value": 1200, "claim": True}}
    unsourced = _waiting(api, campaign["id"], "Users", variables=claim)
    sourced = _waiting(api, campaign["id"], "Plain")

    first = _approve(api, campaign["id"], _shown(unsourced, sourced)).json()

    assert [p["id"] for p in first["approved"]] == [sourced["id"]]
    left = _left(first)[unsourced["id"]]
    assert left["reason"] == "unsourced" and left["claims"] == ["users"]
    assert _row(api, unsourced["id"])["status"] == "needs_approval"

    second = _approve(api, campaign["id"], _shown(unsourced, override=True)).json()

    assert [p["id"] for p in second["approved"]] == [unsourced["id"]]
    row = _row(api, unsourced["id"])
    assert row["status"] == "approved" and row["override_unsourced"] is True
    assert row["review_log"][-1]["overridden_claims"] == ["users"]


def test_a_source_that_no_longer_resolves_counts_as_unsourced(api, monkeypatch):
    _series_switch(api, True)
    campaign = _campaign(api)
    post = _waiting(api, campaign["id"], variables={"users": {"value": 1200, "claim": True}})
    monkeypatch.setattr(
        campaigns_mod.post_sources, "unresolved", lambda db, ws, sources: {"users": "the Deliverable is gone"}
    )

    body = _approve(api, campaign["id"], _shown(post)).json()

    left = _left(body)[post["id"]]
    assert left["reason"] == "unsourced" and left["unresolved"] == {"users": "the Deliverable is gone"}
    assert _row(api, post["id"])["status"] == "needs_approval"


def test_posts_not_shown_or_not_waiting_are_reported_not_approved(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    shown = _waiting(api, campaign["id"], "Shown")
    unseen = _waiting(api, campaign["id"], "Not shown")
    draft = _create(api, title="Draft")
    assert _add(api, campaign["id"], draft["id"]).status_code == 200
    outsider = _create(api, title="Outside the campaign")
    assert _post(api, outsider["id"], "submit").status_code == 200

    body = _approve(api, campaign["id"], _shown(shown, draft, _row(api, outsider["id"]))).json()

    assert [p["id"] for p in body["approved"]] == [shown["id"]]
    left = _left(body)
    assert left[unseen["id"]]["reason"] == "not_shown"
    assert left[draft["id"]]["reason"] == "not_waiting"
    assert left[outsider["id"]]["reason"] == "not_in_campaign"
    assert _row(api, unseen["id"])["status"] == "needs_approval"
    assert _row(api, outsider["id"])["status"] == "needs_approval"


@pytest.mark.parametrize(
    "posts",
    [
        [],
        [{"post_id": str(uuid.uuid4()), "content_hash": "not-a-hash"}],
        [{"post_id": str(uuid.uuid4()), "content_hash": "0" * 64, "status": "approved"}],
    ],
)
def test_a_malformed_series_request_is_422(api, posts):
    _series_switch(api, True)
    campaign = _campaign(api)
    assert _approve(api, campaign["id"], posts).status_code == 422


def test_the_same_post_twice_is_422(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    post = _waiting(api, campaign["id"])
    assert _approve(api, campaign["id"], _shown(post, post)).status_code == 422
    assert _row(api, post["id"])["status"] == "needs_approval"


def test_a_viewer_cannot_approve_a_series(api):
    _series_switch(api, True)
    campaign = _campaign(api)
    post = _waiting(api, campaign["id"])
    api.role = "viewer"

    resp = _approve(api, campaign["id"], _shown(post))

    assert resp.status_code == 403 and "socials:approve" in resp.json()["detail"]
    assert _row(api, post["id"])["status"] == "needs_approval"


# ---------------------------------------------------------------------------
# Tenant isolation
# ---------------------------------------------------------------------------


def test_another_workspaces_campaign_and_post_are_404(api):
    _series_switch(api, True, WS_A)
    _series_switch(api, True, WS_B)
    api.ctx = _ctx(WS_B, "member-b")
    theirs = _campaign(api, name="Theirs")
    their_post = _waiting(api, theirs["id"], "Their post")
    api.ctx = _ctx(WS_A)
    mine = _campaign(api, name="Mine")
    my_post = _create(api, title="Mine")

    base = f"{CAMPAIGNS}/{theirs['id']}"
    assert api.client.get(base).status_code == 404
    assert api.client.patch(base, json={"name": "Taken"}).status_code == 404
    assert _add(api, theirs["id"], my_post["id"]).status_code == 404
    assert api.client.delete(f"{base}/posts/{their_post['id']}").status_code == 404
    assert _approve(api, theirs["id"], _shown(their_post)).status_code == 404
    assert _add(api, mine["id"], their_post["id"]).status_code == 404
    # Their post, shown to my series, is not in my campaign.
    body = _approve(api, mine["id"], _shown(their_post)).json()
    assert body["approved"] == [] and _left(body)[their_post["id"]]["reason"] == "not_in_campaign"

    assert [c["id"] for c in api.client.get(CAMPAIGNS).json()["campaigns"]] == [mine["id"]]
    assert _campaign_row(api, theirs["id"])["name"] == "Theirs"
    assert _row(api, their_post["id"])["status"] == "needs_approval"
    assert _row(api, their_post["id"])["campaign_id"] == theirs["id"]


# ---------------------------------------------------------------------------
# The routes: plain def, gated first, manifested
# ---------------------------------------------------------------------------


def _campaign_routes():
    return [route for route in campaigns_api.router.routes if isinstance(route, APIRoute)]


def test_every_campaign_route_is_a_plain_def():
    routes = _campaign_routes()
    assert sorted((m, "/api/socials" + r.path) for r in routes for m in r.methods) == sorted(NEW_ROUTES)
    for route in routes:
        assert not inspect.iscoroutinefunction(route.endpoint), route.path


def test_the_campaign_routes_are_in_the_socials_router_behind_the_gate():
    served = {
        (method, route.path): route
        for route in socials_api.router.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }
    for key in NEW_ROUTES:
        assert key in served, key
        assert served[key].dependant.dependencies[0].call is require_socials_enabled, key


def test_the_campaign_routes_are_in_the_committed_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifested = {(r["method"], r["path"]) for r in manifest["routes"]}
    assert not [route for route in NEW_ROUTES if route not in manifested]
    assert manifest["route_count"] == len(manifest["routes"])


def test_a_post_whose_status_moves_after_the_check_is_left_and_the_series_goes_on(api, monkeypatch):
    """Another reviewer acts between the series' check and its approval: that post is
    left (not waiting), never a 409 that stops the posts after it."""
    _series_switch(api, True)
    campaign = _campaign(api)
    moved = _waiting(api, campaign["id"], title="Moved")
    after = _waiting(api, campaign["id"], title="After")
    assert _post(api, moved["id"], "request-changes", {"comment": "Tighter"}).status_code == 200
    monkeypatch.setattr(campaigns_mod, "_precheck", lambda post, campaign_id, item: None)

    resp = _approve(api, campaign["id"], _shown(moved, after))

    assert resp.status_code == 200, resp.text
    assert [p["id"] for p in resp.json()["approved"]] == [after["id"]]
    assert _left(resp.json())[moved["id"]]["reason"] == "not_waiting"
    assert _row(api, moved["id"])["status"] == "changes_requested"
