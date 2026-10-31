"""PRD-270 US-270-07a (§5.7, K10) — the public Marketplace reads for agents, Playbooks and packages.

``GET /api/public/marketplace/{type}`` (``api/public_marketplace.py`` over ``services/marketplace_public/``)
on Postgres: the real router runs on a mini FastAPI app with no request-context override installed, so a
200 is an unauthenticated one. The rows are flushed on one session and never committed (the teardown
rolls them back), and every walk reads every page, so rows the suite seeded are held to the same rules.
The per-IP limit runs the real ``RateLimitStore`` over a fake Redis. Proven here:

* only approved ``owner_type='marketplace'`` rows appear: never an unapproved agent or Playbook, a
  workspace's agent or a Playbook of another owner, even approved;
* the allowlist: a walk of every key of every answer, nested ones too, fails on any key outside
  ``fields.py``, and on configuration, model_config, system_prompt, custom_persona_prompt, steps,
  inputs, outputs, execution_config, schedule_config, setup_manifest, workspace_id, owner_id,
  created_by, created_by_user_id, original_creator_id, cloned_from_id, public_id or email in any
  spelling; no value of a column off the list (each holds a marker) reaches the body;
* the publisher is the original creator's name, else ``Automatos`` (no name, a name that is an email,
  no creator, ``Agent.created_by``), and no creator's email appears anywhere in the body;
* an icon a browser reads as a URL (a scheme, or ``//``, ``\\`` read as ``/``, tabs, newlines and leading
  control characters dropped) is answered only when it is ``https`` (P270W1-RVW-2): an agent's or a
  Playbook's ``http:``, ``javascript:``, ``data:`` or scheme-relative icon is ``null``, an ``https`` one
  and an emoji or plain icon name are answered as given;
* required apps: an agent's enabled tool assignments, a Playbook's required tools, a package's
  required connects, as distinct lower-case slugs; a package's members only their type and name (never
  their ref);
* the Socials package is listed as a brand-new workspace would be shown it: the master switch and
  ``SOCIALS_ENABLED_DEFAULT`` both on;
* pages are stable (``limit`` 1 three times is ``limit`` 3 once) and ``more`` says whether one follows;
  a page past the end is empty; limit 1-100 (50 by default), else 422;
* ``Cache-Control: public, max-age=300``, nosniff, a weak ETag and 304, a new ETag when the page changes;
  429 with ``Retry-After`` over the per-IP limit in the route's own bucket; 404 on a local edition
  (before the limit counts) and for an unknown type; a failed read a logged 503; every refusal
  ``no-store``;
* the route is a plain ``def`` with no request-context or permission dependency, in the route manifest
  and mounted through ``RouterSpec``.
"""
from __future__ import annotations

import inspect
import json
import logging
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import api.public_marketplace as public_api
import api.public_rate_limit as rate_limit
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.core import Agent, User, WorkflowTemplate as WorkflowRecipe
from core.models.marketplace_packages import MarketplacePackage
from modules.socials.settings import SOCIALS_PACKAGE_SLUG
from router_manifest import MANIFEST_ROUTERS, RouterSpec
from services.marketplace_public import fields
from services.marketplace_public import packages as public_packages
from tests.test_prd264_content_provenance import _FakeRedis

ROUTE = "/api/public/marketplace/"
ORCH = Path(__file__).resolve().parents[1]
TYPES = ("agents", "playbooks", "packages")
MARK = "PRD270-PRIVATE"
TOP = 10 ** 9  # an install count above any seeded row's, so the test's agents lead the first page
FORBIDDEN = {
    "configuration", "model_config", "system_prompt", "custom_persona_prompt", "steps", "inputs", "outputs",
    "execution_config", "schedule_config", "setup_manifest", "workspace_id", "owner_id", "created_by",
    "created_by_user_id", "original_creator_id", "cloned_from_id", "public_id", "email",
}
PAGE_LIMIT, MAX_PAGES = 100, 100
HTTPS_ICON = "https://cdn.example/x.png"
# A stored icon -> the public answer's: a URL only when it is https, any other text as given.
STORED_ICONS = {
    "http://cdn.example/x.png": None, "javascript:alert(1)": None, "//cdn.example/x.png": None,
    "\\\\cdn.example/x.png": None, "data:image/svg+xml,<svg/>": None, HTTPS_ICON: HTTPS_ICON,
    "🧭": "🧭", "rocket": "rocket",
}


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in ("users", "agents", "workflow_recipes", "marketplace_packages", "agent_tool_assignments"):
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the public reads tests need Postgres with the marketplace tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def shop(engine, new_session, monkeypatch):
    """The router on a mini app over one uncommitted session; the hosted edition, Socials on, a fake Redis."""
    db = new_session()
    redis, socials = _FakeRedis(), {"master": True}
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(config, "SOCIALS_ENABLED_DEFAULT", True)
    monkeypatch.setattr(public_packages, "socials_master_enabled", lambda: socials["master"])
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    app = FastAPI()
    app.include_router(public_api.router)
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield NS(db=db, client=TestClient(app), redis=redis, socials=socials, tag=uuid.uuid4().hex[:10])
    finally:
        new_session.sweep().close()  # rolls the flushed rows back: nothing was committed


def _user(shop, name=None, label="creator"):
    email = f"{label}-{shop.tag}@prd270.test"
    user = User(username=f"prd270-{label}-{shop.tag}", email=email, name=name)
    shop.db.add(user)
    shop.db.flush()
    return user


def _agent(shop, name, approved=True, owner="marketplace", creator=None, installs=TOP, tools=(), icon="🧭"):
    """An agent whose every column off the allowlist holds the marker; ``tools`` are (tool_id, enabled)."""
    workspace_id = _workspace(shop) if owner == "workspace" else None
    agent = Agent(
        name=f"{name} {shop.tag}", description=f"{name} does its job.", agent_type="custom", status="active",
        configuration={"api_key": f"{MARK}-configuration"}, model_config={"model": f"{MARK}-model-config"},
        custom_persona_prompt=f"{MARK}-persona", created_by=f"{MARK}-created-by@prd270.test",
        owner_type=owner, owner_id=f"{MARK}-owner", workspace_id=workspace_id, is_approved=approved,
        original_creator_id=None if creator is None else creator.id, install_count=installs,
        marketplace_category="sales", marketplace_icon=icon, tags=["outbound", "", 7],
    )
    shop.db.add(agent)
    shop.db.flush()
    for tool_id, enabled in tools:
        shop.db.execute(text("INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled) VALUES (:a, :t, :e)"),
                        {"a": agent.id, "t": tool_id, "e": enabled})
    return agent


def _workspace(shop):
    ws = uuid.uuid4()
    shop.db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270 public reads')"),
                    {"id": str(ws)})
    return ws


def _playbook(shop, name, approved=True, owner="marketplace", creator=None, tools=("GMAIL",), icon=None):
    workspace_id = _workspace(shop) if owner == "workspace" else None
    playbook = WorkflowRecipe(
        template_id=f"prd270-{uuid.uuid4().hex[:10]}", name=f"{name} {shop.tag}", description=f"{name}, weekly.",
        workspace_id=workspace_id, owner_type=owner, owner_id=f"{MARK}-owner", is_approved=approved,
        original_creator_id=None if creator is None else creator.id, created_by=f"{MARK}-created-by",
        steps=[{"prompt_template": f"{MARK}-step"}], inputs={f"{MARK}-input": {"type": "string"}},
        outputs={f"{MARK}-output": {}}, execution_config={"mode": f"{MARK}-execution"},
        schedule_config={"cron": f"{MARK}-schedule"}, template_definition={"config": f"{MARK}-definition"},
        learning_data={"note": f"{MARK}-learning"}, required_tools=list(tools), install_count=TOP,
        marketplace_category="operations", marketplace_icon=icon, tags=["weekly"],
    )
    shop.db.add(playbook)
    shop.db.flush()
    return playbook


def _package(shop, slug=None, members=None, connects=("SHOPIFY",)):
    package = MarketplacePackage(
        slug=slug or f"prd270-{shop.tag}", name=f"AAA Starter team {shop.tag}", description="Run the store.",
        vertical_tags=["shopify", ""], matching={"vocabulary": [f"{MARK}-matching"]},
        members=members if members is not None else [{"type": "agent", "ref": "store-ops", "name": "Store Ops"}],
        setup_manifest={
            "questions": [{"id": "q", "prompt": f"{MARK}-question"}],
            "required_connects": [{"app_name": app, "note": f"{MARK}-note"} for app in connects],
            "guide_steps": [{"step": 1, "title": f"{MARK}-guide"}],
        },
        showcase=True,
    )
    shop.db.add(package)
    shop.db.flush()
    return package


def _get(shop, kind, headers=None, **params):
    return shop.client.get(ROUTE + kind, params=params, headers=headers or {})


def _page(shop, kind, **params):
    resp = _get(shop, kind, **params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _pages(shop, kind):
    """Every page of ``kind`` as answered, in order."""
    pages = []
    for number in range(MAX_PAGES):
        pages.append(_page(shop, kind, limit=PAGE_LIMIT, offset=number * PAGE_LIMIT))
        if not pages[-1]["more"]:
            return pages
    raise AssertionError(f"more than {MAX_PAGES} pages of {kind}")


def _items(shop, kind):
    return [item for page in _pages(shop, kind) for item in page["items"]]


def _by_key(shop, kind):
    return {item["key"]: item for item in _items(shop, kind)}


def _keys_anywhere(value):
    """Every key in ``value``, nested ones too."""
    if isinstance(value, dict):
        return set(value).union(*(_keys_anywhere(inner) for inner in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys_anywhere(inner) for inner in value))
    return set()


def _spelling(key):
    return key.replace("_", "").lower()


def _calls(dependant):
    found = []
    for sub in dependant.dependencies:
        found += [sub.call] + _calls(sub)
    return found


# ---------------------------------------------------------------------------
# What is listed
# ---------------------------------------------------------------------------


def test_only_approved_marketplace_agents_and_playbooks_appear(shop):
    listed, unapproved = _agent(shop, "Closer"), _agent(shop, "Draft", approved=False)
    workspace_agent = _agent(shop, "Ours", owner="workspace")
    playbook, draft = _playbook(shop, "Pipeline review"), _playbook(shop, "Draft review", approved=False)
    theirs = _playbook(shop, "Their review", owner="workspace")
    agents, playbooks = _by_key(shop, "agents"), _by_key(shop, "playbooks")
    assert str(listed.id) in agents and str(playbook.id) in playbooks
    assert str(unapproved.id) not in agents and str(workspace_agent.id) not in agents
    assert str(draft.id) not in playbooks and str(theirs.id) not in playbooks
    assert agents[str(listed.id)]["name"] == listed.name and agents[str(listed.id)]["category"] == "sales"
    assert agents[str(listed.id)]["tags"] == ["outbound"] and agents[str(listed.id)]["icon"] == "🧭"


def test_every_key_of_every_answer_is_on_the_allowlist_and_nothing_private_leaves(shop):
    creator = _user(shop, name="Ada Lovelace")
    agent = _agent(shop, "Closer", creator=creator, tools=[("GMAIL", True)])
    _playbook(shop, "Pipeline review", creator=creator)
    _package(shop)
    on_lists = fields.ANSWER_FIELDS.union(*fields.ITEM_FIELDS.values(), *fields.NESTED_FIELDS.values())
    assert not {_spelling(key) for key in on_lists} & {_spelling(key) for key in FORBIDDEN}
    for kind in TYPES:
        for page in _pages(shop, kind):
            seen = _keys_anywhere(page)
            assert seen <= on_lists, f"{kind}: off the allowlist: {seen - on_lists}"
            assert not {_spelling(key) for key in seen} & {_spelling(key) for key in FORBIDDEN}
            assert set(page) == fields.ANSWER_FIELDS
            for item in page["items"]:
                assert set(item) == fields.ITEM_FIELDS[kind], f"{kind}: {set(item) ^ fields.ITEM_FIELDS[kind]}"
                for member in item.get("members", []):
                    assert set(member) == fields.PACKAGE_MEMBER_FIELDS
            body = json.dumps(page, ensure_ascii=False)
            assert MARK not in body and creator.email not in body and str(agent.public_id) not in body


def test_the_publisher_is_a_display_name_never_an_email(shop):
    named, nameless = _user(shop, name="  Ada Lovelace ", label="named"), _user(shop, label="nameless")
    emailed = _user(shop, name="grace@prd270.test", label="emailed")
    agents = {key: _agent(shop, key, creator=user) for key, user in
              (("named", named), ("nameless", nameless), ("emailed", emailed), ("orphan", None))}
    playbook = _playbook(shop, "Weekly numbers", creator=nameless)
    listed = _by_key(shop, "agents")
    assert listed[str(agents["named"].id)]["publisher"] == "Ada Lovelace"
    for key in ("nameless", "emailed", "orphan"):
        assert listed[str(agents[key].id)]["publisher"] == fields.DEFAULT_PUBLISHER == "Automatos"
    assert _by_key(shop, "playbooks")[str(playbook.id)]["publisher"] == "Automatos"
    for kind in TYPES:
        body = json.dumps(_pages(shop, kind))
        assert not any(user.email in body for user in (named, nameless, emailed))
        assert "grace@" not in body and MARK not in body


def test_required_apps_are_distinct_lower_case_slugs_and_members_only_their_type_and_name(shop):
    agent = _agent(shop, "Closer", tools=[("GMAIL", True), ("gmail-extra", True), ("SLACK", False)])
    playbook = _playbook(shop, "Pipeline review", tools=("GMAIL", "gmail", " Slack ", 7, ""))
    members = [
        {"type": "agent", "ref": "store-ops", "name": "Store Ops", "description": f"{MARK}-member"},
        {"type": "playbook", "ref": "weekly-numbers"},
        {"type": "workspace", "ref": "ignored", "name": "Ignored"},
        "not-a-member",
    ]
    package = _package(shop, members=members, connects=("SHOPIFY", "GITHUB", "shopify"))
    assert _by_key(shop, "agents")[str(agent.id)]["requiredApps"] == ["gmail", "gmail-extra"]
    assert _by_key(shop, "playbooks")[str(playbook.id)]["requiredApps"] == ["gmail", "slack"]
    listed = _by_key(shop, "packages")[package.slug]
    assert listed["requiredApps"] == ["shopify", "github"] and listed["publisher"] == "Automatos"
    assert listed["members"] == [{"type": "agent", "name": "Store Ops"}, {"type": "playbook", "name": None}]
    assert "store-ops" not in json.dumps(listed) and "weekly-numbers" not in json.dumps(listed)
    assert listed["tags"] == ["shopify"] and listed["category"] is None and listed["icon"] is None


def test_an_agent_s_or_a_playbook_s_icon_is_answered_as_a_url_only_when_it_is_https(shop):
    agents = {icon: _agent(shop, f"Icon {rank}", icon=icon) for rank, icon in enumerate(STORED_ICONS)}
    playbooks = {icon: _playbook(shop, f"Icon {rank}", icon=icon) for rank, icon in enumerate(STORED_ICONS)}
    listed_agents, listed_playbooks = _by_key(shop, "agents"), _by_key(shop, "playbooks")
    for stored, answered in STORED_ICONS.items():
        assert listed_agents[str(agents[stored].id)]["icon"] == answered, stored
        assert listed_playbooks[str(playbooks[stored].id)]["icon"] == answered, stored


@pytest.mark.parametrize("stored, answered", [
    *STORED_ICONS.items(),
    ("JavaScript:alert(1)", None), ("/\\cdn.example/x.png", None), ("http://[::1/x.png", None),
    ("https://user:pw@cdn.example/x.png", None), ("https://cdn.example/a b.png", None), ("https:/x.png", None),
    ("java\tscript:alert(1)", None), ("\x01javascript:alert(1)", None), ("/\t/cdn.example/x.png", None),
    ("/\n/cdn.example/x.png", None), ("\\\r\\cdn.example/x.png", None), ("\x01//cdn.example/x.png", None),
    ("\x01\\/cdn.example/x.png", None), (f"  {HTTPS_ICON} ", HTTPS_ICON), (" 🧭 ", "🧭"),
    ("", None), (None, None), (7, None),
])
def test_the_icon_mapper_keeps_a_url_only_when_it_is_https_and_any_other_icon_as_its_text(stored, answered):
    assert fields.icon_or_none(stored) == answered
    assert fields.display_item(fields.AGENT_FIELDS, key=1, name="Closer", icon=stored)["icon"] == answered


def test_the_socials_package_is_listed_as_a_brand_new_workspace_is_shown_it(shop, monkeypatch):
    if shop.db.query(MarketplacePackage).filter(MarketplacePackage.slug == SOCIALS_PACKAGE_SLUG).first() is None:
        _package(shop, slug=SOCIALS_PACKAGE_SLUG)
    assert SOCIALS_PACKAGE_SLUG in _by_key(shop, "packages")
    shop.socials["master"] = False
    assert SOCIALS_PACKAGE_SLUG not in _by_key(shop, "packages")
    shop.socials["master"] = True
    monkeypatch.setattr(config, "SOCIALS_ENABLED_DEFAULT", False)
    assert SOCIALS_PACKAGE_SLUG not in _by_key(shop, "packages")


def test_pages_are_stable_more_says_whether_one_follows_and_the_limit_is_capped(shop):
    top = [_agent(shop, f"Top {rank}", installs=TOP + rank) for rank in range(4)]
    first = _page(shop, "agents")
    assert first["type"] == "agents" and first["limit"] == 50 and first["offset"] == 0
    one_by_one = [_page(shop, "agents", limit=1, offset=offset) for offset in range(3)]
    assert all(page["more"] and len(page["items"]) == 1 for page in one_by_one)
    assert [page["items"][0] for page in one_by_one] == _page(shop, "agents", limit=3)["items"]
    assert [item["key"] for item in _page(shop, "agents", limit=3)["items"]] == [str(a.id) for a in top[:0:-1]]
    last = _pages(shop, "agents")[-1]
    assert last["more"] is False and len(last["items"]) <= PAGE_LIMIT
    assert len(_page(shop, "agents", limit=100)["items"]) <= 100
    for params in ({"limit": 101}, {"limit": 0}, {"offset": -1}, {"limit": "many"}):
        assert _get(shop, "agents", **params).status_code == 422, params


@pytest.mark.parametrize("kind", TYPES)
def test_a_page_past_the_end_is_empty_with_nothing_more(shop, kind):
    page = _page(shop, kind, offset=public_api.MAX_OFFSET)
    assert page == {"type": kind, "items": [], "limit": 50, "offset": public_api.MAX_OFFSET, "more": False}
    assert _get(shop, kind, offset=public_api.MAX_OFFSET + 1).status_code == 422


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_cache_header_and_a_weak_etag_give_a_304_until_the_page_changes(shop):
    _agent(shop, "Closer")
    first = _get(shop, "agents")
    assert first.status_code == 200
    assert first.headers["cache-control"] == "public, max-age=300"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert first.headers["content-type"].startswith("application/json")
    etag = first.headers["etag"]
    assert etag.startswith('W/"')
    again = _get(shop, "agents", headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["etag"] == etag and again.headers["cache-control"] == "public, max-age=300"
    _agent(shop, "Newest", installs=TOP + 1)
    changed = _get(shop, "agents", headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_route_s_own_bucket(shop, monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    headers = {"X-Forwarded-For": "198.51.100.7"}
    assert [_get(shop, kind, headers=headers).status_code for kind in ("agents", "packages")] == [200, 200]
    refused = _get(shop, "playbooks", headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    assert len(shop.redis.z) == 1 and next(iter(shop.redis.z)).endswith(":marketplace-public:ip:198.51.100.7")
    assert _get(shop, "agents", headers={"X-Forwarded-For": "198.51.100.8"}).status_code == 200


def test_the_local_edition_answers_404_before_the_limit_counts(shop, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    for kind in (*TYPES, "nope"):
        resp = _get(shop, kind, limit=500)
        assert resp.status_code == 404 and resp.json()["detail"] == public_api.NOT_HOSTED
        assert resp.headers["cache-control"] == "no-store"
    assert shop.redis.z == {}


@pytest.mark.parametrize("kind", ["themes", "nope", "Agents", "agents.json"])
def test_an_unknown_type_is_a_404(shop, kind):
    resp = _get(shop, kind)
    assert resp.status_code == 404 and resp.json()["detail"] == public_api.NO_SUCH_TYPE
    assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers


def test_a_read_the_database_fails_is_a_logged_503(shop, monkeypatch, caplog):
    def broken(db, kind, limit, offset):
        raise OperationalError("SELECT", {}, Exception("connection lost"))

    monkeypatch.setattr(public_api, "public_page", broken)
    with caplog.at_level(logging.ERROR, logger=public_api.__name__):
        resp = _get(shop, "agents")
    assert resp.status_code == 503 and resp.json()["detail"] == public_api.READS_DOWN
    assert resp.headers["cache-control"] == "no-store"
    assert any("agents read failed" in record.getMessage() for record in caplog.records)


def test_a_field_off_the_allowlist_is_refused_before_it_is_answered():
    with pytest.raises(fields.FieldNotAllowed, match="configuration"):
        fields.display_item(fields.AGENT_FIELDS, key=1, name="Closer", configuration={"api_key": "x"})
    with pytest.raises(fields.FieldNotAllowed, match="ref"):
        fields.allowed_item(fields.PACKAGE_MEMBER_FIELDS, type="agent", name="Store Ops", ref="store-ops")


def test_the_route_is_public_a_plain_def_in_the_manifest_and_mounted_by_router_spec():
    routes = [r for r in public_api.router.routes if isinstance(r, APIRoute)]
    assert [(r.path, r.methods) for r in routes] == [("/api/public/marketplace/{type}", {"GET"})]
    route = routes[0]
    assert not inspect.iscoroutinefunction(route.endpoint)
    calls = _calls(route.dependant)
    assert get_request_context_hybrid not in calls
    assert calls[:2] == [public_api.require_hosted_edition, public_api.limit_public_reads] and get_db in calls
    assert not any("permission" in getattr(call, "__qualname__", "") for call in calls)
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": "/api/public/marketplace/{type}"} in manifest["routes"]
    assert RouterSpec("api.public_marketplace") in MANIFEST_ROUTERS
