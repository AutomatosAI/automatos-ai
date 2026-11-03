"""PRD-270 US-270-07b (§5.7, §8-Q9, K10) — the public Marketplace reads for applications, plugins, skills and LLMs.

``GET /api/public/marketplace/{type}`` (``api/public_marketplace.py`` over ``services/marketplace_public/``)
on Postgres, as US-270-07a's suite runs it: the real router on a mini FastAPI app with no request-context
override, so a 200 is an unauthenticated one; rows flushed on one session and rolled back; every walk
reads every page, so rows the suite seeded are held to the same rules; the real ``RateLimitStore`` over
a fake Redis. Proven here:

* each type's catalogue rows only: an app that isn't ``ACTIVE`` or is one of the platform's internal
  tools, a pending, rejected or deactivated plugin, an inactive skill or a workspace's own skill, an
  inactive LLM route never appear; one LLM item per route;
* the allowlist, per type: a walk of every key, nested ones too, fails on any key outside ``fields.py``,
  on a scan or security report, an S3 bucket, path or key, a zip or source URL, and on any workspace,
  credential, connection, installed, enabled or API-key field in any spelling; no value of a column off
  the list (each holds a marker) reaches the body;
* nothing of a workspace changes an answer: a skill enabled and a route installed in a workspace leave
  every item as it was, and the member routes' connection, install and key-presence reads are never
  called;
* each type's plain facts (counts, an ``https``-only icon, the plugin's category and author as a display
  name, the skill's version, the route's provider, vendor, context window, price tier and capabilities);
* the four answer through the route with US-270-07a's rules: stable pages and ``more``, the limit cap,
  an empty page past the end, ``Cache-Control: public, max-age=300`` with a weak ETag and 304, 429 with
  ``Retry-After`` in the ``marketplace-public:ip`` bucket, 404 on a local edition before the limit counts.
"""
from __future__ import annotations

import json
import re
import uuid
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text

import api.llm_marketplace as llm_marketplace
import api.public_marketplace as public_api
import api.public_rate_limit as rate_limit
from api.tools import INTERNAL_APP_NAMES
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.composio.entity_manager import EntityManager
from core.database.database import get_db
from core.models.composio_cache import ComposioAppCache
from core.models.core import LLMModel, Skill, WorkspaceModel
from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory, WorkspaceEnabledSkill
from services.marketplace_public import fields
from services.marketplace_public.reads import READERS
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w1_public_reads import _keys_anywhere, _spelling

ROUTE = "/api/public/marketplace/"
TYPES = ("applications", "plugins", "skills", "llms")
MARK = "PRD270-PRIVATE"
TOP = 10 ** 9  # above any seeded row's count, so the test's rows lead the first page
FORBIDDEN = {
    "security_status", "security_scan_id", "security_scan", "scan", "s3_bucket", "s3_path", "s3_key", "zip_url",
    "source_url", "source_repo", "approval_status", "approved_by", "rejection_reason", "materialized_skill_ids",
    "workspace_id", "workspace", "credential", "credentials", "connection", "connection_id", "is_connected",
    "connected", "is_installed", "installed", "is_enabled", "enabled", "key_available", "api_key", "has_api_key",
    "prompt_template", "implementation", "parameters", "tools_schema", "skill_metadata", "app_metadata",
    "metadata", "auth_schemes", "triggers", "external_id", "requires_plan", "created_by", "id",
}
STEMS = (
    "scan", "security", "s3", "zip", "url", "repo", "path", "bucket", "workspace", "credential", "connect",
    "install", "enabled", "apikey", "token", "secret", "email", "author", "owner", "user",
)
# The tables each reader may name, and the workspace tables none may (its SQL is recorded as it runs).
READS_FROM = {
    "applications": {"composio_apps_cache"}, "plugins": {"marketplace_plugins", "plugin_categories"},
    "skills": {"skills"}, "llms": {"llm_models"},
}
WORKSPACE_TABLES = (
    "workspaces", "workspace_members", "workspace_enabled_skills", "workspace_enabled_plugins", "workspace_models",
    "composio_connections", "composio_entities", "user_api_keys", "credentials",
)
PAGE_LIMIT, MAX_PAGES = 100, 100


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    tables = ("composio_apps_cache", "marketplace_plugins", "plugin_categories", "skills", "llm_models",
              "workspace_enabled_skills", "workspace_models", "workspaces")
    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in tables:
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the public reads tests need Postgres with the marketplace tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def shop(engine, new_session, monkeypatch):
    """The router on a mini app over one uncommitted session; the hosted edition and a fake Redis."""
    db, redis = new_session(), _FakeRedis()
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    app = FastAPI()
    app.include_router(public_api.router)
    app.dependency_overrides[get_db] = lambda: db
    try:
        yield NS(db=db, client=TestClient(app), redis=redis, tag=uuid.uuid4().hex[:10])
    finally:
        new_session.sweep().close()  # rolls the flushed rows back: nothing was committed


def _add(shop, row):
    shop.db.add(row)
    shop.db.flush()
    return row


def _workspace(shop):
    ws = uuid.uuid4()
    shop.db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd270 public reads')"),
                    {"id": str(ws)})
    return ws


def _app(shop, name, status="ACTIVE", actions=TOP, logo="https://logos.prd270.test/app.svg"):
    slug = f"prd270-{name}-{shop.tag}".lower()
    return _add(shop, ComposioAppCache(
        app_name=slug.upper(), app_slug=slug, display_name=f"{name} {shop.tag}", description=f"{name} for teams.",
        logo_url=logo, categories=["crm", "", "sales"], auth_schemes=[f"{MARK}-auth"], action_count=actions,
        trigger_count=3, status=status, app_metadata={"triggers": [{"config": f"{MARK}-trigger"}]},
    ))


def _internal_app(shop):
    """An internal tool's cache row, ACTIVE and at the top (the seeded one when it exists)."""
    name = sorted(INTERNAL_APP_NAMES)[0]
    row = shop.db.query(ComposioAppCache).filter(ComposioAppCache.app_name == name).first()
    if row is None:
        return _app_named(shop, name)
    row.status, row.action_count = "ACTIVE", TOP + 9
    shop.db.flush()
    return row


def _app_named(shop, name):
    return _add(shop, ComposioAppCache(app_name=name, app_slug=name.lower(), display_name=name, categories=[],
                                       auth_schemes=[], action_count=TOP + 9, status="ACTIVE"))


def _category(shop):
    return _add(shop, PluginCategory(slug=f"prd270-{shop.tag}", name="Sales tools", description=f"{MARK}-category",
                                     icon=f"{MARK}-icon"))


def _plugin(shop, name, approval="approved", active=True, enables=TOP, author="Anthropic", category=None):
    return _add(shop, MarketplacePlugin(
        slug=f"{MARK}-{name}-{shop.tag}".lower(), name=f"{name} {shop.tag}", version="1.2.0",
        description=f"{name} tools.", long_description=f"{MARK}-long", tags=["sales", ""],
        category_id=None if category is None else category.id, skills_count=2, commands_count=1, agents_count=1,
        hooks_count=0, source_type=f"{MARK}-type", source_url=f"https://example.test/{MARK}/plugin.zip",
        source_repo=f"{MARK}-repo", original_author=author, license=f"{MARK}-licence", token_estimate=1234,
        recommended_models=[f"{MARK}-model"], s3_bucket=f"{MARK}-bucket", s3_path=f"{MARK}/plugins/{name}.zip",
        security_status=f"{MARK}-scan", is_active=active, approval_status=approval,
        approved_by=f"{MARK}-approver", rejection_reason=f"{MARK}-rejected", enable_count=enables,
        materialized_skill_ids=[f"{MARK}-skill"],
    ))


def _skill(shop, name, active=True, workspace=False):
    return _add(shop, Skill(
        name=f"prd270-{name}-{shop.tag}", description=f"{name}, step by step.", skill_type="cognitive",
        category="writing", implementation=f"{MARK}-implementation", parameters={MARK: 1},
        performance_data={"runs": MARK}, is_active=active, created_by=f"{MARK}-creator@prd270.test",
        workspace_id=_workspace(shop) if workspace else None, package_slug=f"{MARK}-package",
        prompt_template=f"{MARK}-prompt", tools_schema={"tools": [MARK]}, skill_version="1.0.0",
        skill_source=f"{MARK}-source", git_repo_url=f"https://example.test/{MARK}.git", git_commit_sha=f"{MARK}-sha",
        git_branch=f"{MARK}-branch", filesystem_path=f"/{MARK}/skill", content_hash=f"{MARK}-hash",
        tags=["writing", "", 7], skill_metadata={"note": f"{MARK}-metadata"},
    ))


def _route(shop, name, serving="openrouter", status="active", installs=TOP, prices=(0.002, 0.01), **extra):
    return _add(shop, LLMModel(
        provider="moonshotai", serving_provider=serving, model_id=extra.get("model_id", f"prd270-{shop.tag}/{name}"),
        display_name=f"{name} {shop.tag}", model_family=f"{MARK}-family", context_window=128000,
        capabilities=extra.get("capabilities", {"reasoning": "excellent"}), max_output_tokens=4096,
        input_cost_per_1k_tokens=prices[0], output_cost_per_1k_tokens=prices[1], description=f"{name} reads long.",
        status=status, recommended_for=[f"{MARK}-use"], sourcing="aggregator", tags=["long-context"],
        category="general", install_count=installs, requires_plan=f"{MARK}-plan", external_id=f"{MARK}-external",
    ))


def _seed_every_type(shop):
    return NS(app=_app(shop, "Pipeline"), plugin=_plugin(shop, "Closer kit", category=_category(shop)),
              skill=_skill(shop, "brief-writing"), route=_route(shop, "Kimi"))


def _get(shop, kind, headers=None, **params):
    return shop.client.get(ROUTE + kind, params=params, headers=headers or {})


def _page(shop, kind, **params):
    resp = _get(shop, kind, **params)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _pages(shop, kind):
    pages = []
    for number in range(MAX_PAGES):
        pages.append(_page(shop, kind, limit=PAGE_LIMIT, offset=number * PAGE_LIMIT))
        if not pages[-1]["more"]:
            return pages
    raise AssertionError(f"more than {MAX_PAGES} pages of {kind}")


def _by_key(shop, kind):
    return {item["key"]: item for page in _pages(shop, kind) for item in page["items"]}


def _route_key(route):
    return f"{route.serving_provider}:{route.model_id}"


@contextmanager
def _statements(shop):
    """Every SQL statement the session's engine runs inside the block, as a list that fills as they run."""
    ran, bind = [], shop.db.get_bind()

    def record(conn, cursor, statement, parameters, context, executemany):
        ran.append(statement)

    event.listen(bind, "before_cursor_execute", record)
    try:
        yield ran
    finally:
        event.remove(bind, "before_cursor_execute", record)


def _names(statement, table):
    return re.search(rf"\b{table}\b", statement) is not None


# ---------------------------------------------------------------------------
# What is listed
# ---------------------------------------------------------------------------


def test_every_public_type_has_a_reader_and_an_allowlist():
    assert set(READERS) == set(fields.ITEM_FIELDS) == {"agents", "playbooks", "packages", *TYPES}


def test_only_active_applications_appear_and_never_an_internal_tool(shop):
    listed, retired, internal = _app(shop, "Pipeline"), _app(shop, "Retired", status="INACTIVE"), _internal_app(shop)
    apps = _by_key(shop, "applications")
    assert listed.app_slug in apps and retired.app_slug not in apps and internal.app_slug not in apps
    assert not any(key.upper() in INTERNAL_APP_NAMES for key in apps)


def test_only_approved_active_plugins_appear(shop):
    listed = _plugin(shop, "Closer kit")
    hidden = [_plugin(shop, "Pending kit", approval="pending"), _plugin(shop, "Rejected kit", approval="rejected"),
              _plugin(shop, "Retired kit", active=False)]
    plugins = _by_key(shop, "plugins")
    assert str(listed.id) in plugins
    assert not {str(plugin.id) for plugin in hidden} & set(plugins)


def test_only_active_global_skills_appear_never_a_workspace_s_own(shop):
    listed, retired = _skill(shop, "brief"), _skill(shop, "retired", active=False)
    ours = _skill(shop, "ours", workspace=True)
    skills = _by_key(shop, "skills")
    assert listed.name in skills and retired.name not in skills and ours.name not in skills


def test_only_active_llm_routes_appear_one_item_per_route(shop):
    openrouter = _route(shop, "Kimi")
    nvidia = _route(shop, "Kimi NIM", serving="nvidia", model_id=openrouter.model_id)
    retired = _route(shop, "Retired", status="deprecated")
    routes = _by_key(shop, "llms")
    assert _route_key(retired) not in routes
    served = [routes[_route_key(route)] for route in (openrouter, nvidia)]
    assert [(item["provider"], item["publisher"], item["vendor"]) for item in served] == [
        ("openrouter", "OpenRouter", "moonshotai"), ("nvidia", "NVIDIA", "moonshotai"),
    ]


# ---------------------------------------------------------------------------
# The allowlist
# ---------------------------------------------------------------------------


def test_every_key_of_every_answer_is_on_the_allowlist_and_nothing_private_leaves(shop):
    _seed_every_type(shop)
    on_lists = fields.ANSWER_FIELDS.union(*fields.ITEM_FIELDS.values(), *fields.NESTED_FIELDS.values())
    assert not {_spelling(key) for key in on_lists} & {_spelling(key) for key in FORBIDDEN}
    assert not [key for key in on_lists if any(stem in _spelling(key) for stem in STEMS)]
    for kind in TYPES:
        for page in _pages(shop, kind):
            seen = _keys_anywhere(page)
            assert seen <= on_lists, f"{kind}: off the allowlist: {seen - on_lists}"
            assert set(page) == fields.ANSWER_FIELDS
            for item in page["items"]:
                assert set(item) == fields.ITEM_FIELDS[kind], f"{kind}: {set(item) ^ fields.ITEM_FIELDS[kind]}"
                for capability in item.get("capabilities", []):
                    assert set(capability) == fields.LLM_CAPABILITY_FIELDS
            body = json.dumps(page, ensure_ascii=False)
            assert MARK not in body and MARK.lower() not in body, kind


def test_nothing_of_a_workspace_changes_an_answer(shop, monkeypatch):
    seeded = _seed_every_type(shop)
    before = {kind: _by_key(shop, kind) for kind in TYPES}
    ws = _workspace(shop)
    _add(shop, WorkspaceEnabledSkill(workspace_id=ws, skill_id=seeded.skill.id))
    _add(shop, WorkspaceModel(workspace_id=ws, model_id=seeded.route.id, is_active=True))

    def never(*args, **kwargs):
        raise AssertionError("a public read asked for a workspace's state")

    for name in ("_get_installed_ids", "_get_available_providers"):
        monkeypatch.setattr(llm_marketplace, name, never)
    for name in ("get_entity_by_workspace", "get_entity_connections"):
        monkeypatch.setattr(EntityManager, name, never)
    assert {kind: _by_key(shop, kind) for kind in TYPES} == before


@pytest.mark.parametrize("kind", TYPES)
def test_each_reader_s_sql_names_its_catalogue_tables_and_never_a_workspace_s(shop, kind):
    _seed_every_type(shop)
    with _statements(shop) as ran:
        _pages(shop, kind)
    assert ran, kind
    for statement in ran:
        assert any(_names(statement, table) for table in READS_FROM[kind]), statement
        assert not [table for table in WORKSPACE_TABLES if _names(statement, table)], statement


def test_a_field_off_the_allowlist_is_refused_before_it_is_answered():
    with pytest.raises(fields.FieldNotAllowed, match="security_status"):
        fields.display_item(fields.PLUGIN_FIELDS, key="p", name="Closer kit", security_status="safe")
    with pytest.raises(fields.FieldNotAllowed, match="isConnected"):
        fields.display_item(fields.APPLICATION_FIELDS, key="gmail", name="Gmail", isConnected=True)
    with pytest.raises(fields.FieldNotAllowed, match="isInstalled"):
        fields.display_item(fields.LLM_FIELDS, key="openrouter:m", name="M", isInstalled=True)
    with pytest.raises(fields.FieldNotAllowed, match="score"):
        fields.allowed_item(fields.LLM_CAPABILITY_FIELDS, name="reasoning", rating="excellent", score=9)


# ---------------------------------------------------------------------------
# Each type's plain facts
# ---------------------------------------------------------------------------


def test_an_application_answers_its_slug_categories_counts_and_an_https_logo_only(shop):
    listed = _app(shop, "Pipeline")
    unsafe = [_app(shop, f"Logo {n}", logo=logo, actions=TOP - n) for n, logo in enumerate(
        ("http://logos.prd270.test/a.svg", "javascript:alert(1)", "https://user:pw@logos.prd270.test/a.svg",
         "https://logos.prd270.test/a b.svg", "https://[::1"))]
    apps = _by_key(shop, "applications")
    item = apps[listed.app_slug]
    assert item["name"] == listed.display_name and item["icon"] == "https://logos.prd270.test/app.svg"
    assert item["tags"] == ["crm", "sales"] and item["category"] == "crm" and item["requiredApps"] == []
    assert item["actionCount"] == TOP and item["triggerCount"] == 3 and item["publisher"] == "Automatos"
    assert all(apps[app.app_slug]["icon"] is None for app in unsafe)


def test_a_plugin_answers_its_category_author_version_and_counts(shop):
    category = _category(shop)
    named = _plugin(shop, "Closer kit", category=category)
    emailed, nameless = _plugin(shop, "Mailed kit", author="dev@prd270.test"), _plugin(shop, "Plain kit", author=None)
    plugins = _by_key(shop, "plugins")
    item = plugins[str(named.id)]
    assert item["category"] == "Sales tools" and item["icon"] is None and item["publisher"] == "Anthropic"
    assert item["version"] == "1.2.0" and item["tags"] == ["sales"] and item["requiredApps"] == []
    assert (item["skillsCount"], item["commandsCount"], item["agentsCount"], item["hooksCount"]) == (2, 1, 1, 0)
    assert plugins[str(emailed.id)]["publisher"] == plugins[str(nameless.id)]["publisher"] == "Automatos"
    assert plugins[str(emailed.id)]["category"] is None and "dev@prd270.test" not in json.dumps(plugins)


def test_a_skill_answers_its_name_as_key_and_its_version(shop):
    skill = _skill(shop, "brief-writing")
    item = _by_key(shop, "skills")[skill.name]
    assert item == {
        "key": skill.name, "name": skill.name, "description": "brief-writing, step by step.", "category": "writing",
        "icon": None, "tags": ["writing"], "publisher": "Automatos", "requiredApps": [], "version": "1.0.0",
    }


def test_an_llm_route_answers_its_context_window_and_text_capabilities(shop):
    route = _route(shop, "Kimi", capabilities={"reasoning": "excellent", "speed": 3, "": "x", "coding": " high "})
    listed = _route(shop, "Listed", capabilities=["chat"])
    routes = _by_key(shop, "llms")
    item = routes[_route_key(route)]
    assert item["name"] == route.display_name and item["contextWindow"] == 128000 and item["category"] == "general"
    rated = sorted((c["name"], c["rating"]) for c in item["capabilities"])
    assert rated == [("coding", "high"), ("reasoning", "excellent")]
    assert item["tags"] == ["long-context"] and item["requiredApps"] == [] and item["icon"] is None
    assert routes[_route_key(listed)]["capabilities"] == []


@pytest.mark.parametrize("prices, model, tier", [
    ((0.0, 0.0), "plain", "free"),
    ((0.0004, 0.001), "plain", "budget"),
    ((0.002, 0.01), "plain", "mid"),
    ((0.01, 0.03), "plain", "premium"),
    ((None, None), "anthropic/claude-haiku-5-5", "budget"),
    ((None, None), "anthropic/claude-sonnet-5-5", "mid"),
    ((None, None), "mystery", "unknown"),
])
def test_an_llm_route_s_price_tier_follows_the_member_catalogue(shop, prices, model, tier):
    route = _route(shop, "Priced", prices=prices, model_id=f"prd270-{shop.tag}/{model}")
    assert _by_key(shop, "llms")[_route_key(route)]["priceTier"] == tier
    assert llm_marketplace._model_to_out(route).price_tier == tier  # the member card shows the same tier


# ---------------------------------------------------------------------------
# The route's rules, for the four types
# ---------------------------------------------------------------------------


def _top_three(shop, kind):
    """Three rows of ``kind`` that lead its first page, in their answer order (most first)."""
    if kind == "applications":
        rows = [_app(shop, f"Top {rank}", actions=TOP + rank) for rank in range(3)]
        return [row.app_slug for row in rows[::-1]]
    if kind == "plugins":
        rows = [_plugin(shop, f"Top {rank}", enables=TOP + rank) for rank in range(3)]
        return [str(row.id) for row in rows[::-1]]
    if kind == "llms":
        rows = [_route(shop, f"Top {rank}", installs=TOP + rank) for rank in range(3)]
        return [_route_key(row) for row in rows[::-1]]
    for rank in range(3):  # skills run by name: their place among the suite's is the collation's
        _skill(shop, f"stable-{rank}")
    return None


def _change_the_first_page(shop, kind, page):
    """Change what ``kind``'s first page answers: three new leaders, or (skills, by name) the first one retired."""
    if kind != "skills":
        _top_three(shop, kind)
        return
    first = shop.db.query(Skill).filter(Skill.workspace_id.is_(None), Skill.name == page["items"][0]["key"]).one()
    first.is_active = False
    shop.db.flush()


@pytest.mark.parametrize("kind", TYPES)
def test_pages_are_stable_more_says_whether_one_follows_and_the_limit_is_capped(shop, kind):
    expected = _top_three(shop, kind)
    first = _page(shop, kind)
    assert first["type"] == kind and first["limit"] == 50 and first["offset"] == 0
    one_by_one = [_page(shop, kind, limit=1, offset=offset) for offset in range(3)]
    assert all(len(page["items"]) == 1 for page in one_by_one) and one_by_one[1]["more"]
    assert [page["items"][0] for page in one_by_one] == _page(shop, kind, limit=3)["items"]
    if expected is not None:
        assert [item["key"] for item in _page(shop, kind, limit=3)["items"]] == expected
    assert _pages(shop, kind)[-1]["more"] is False
    for params in ({"limit": 101}, {"limit": 0}, {"offset": -1}, {"limit": "many"}):
        assert _get(shop, kind, **params).status_code == 422, params


@pytest.mark.parametrize("kind", TYPES)
def test_a_page_past_the_end_is_empty_with_nothing_more(shop, kind):
    page = _page(shop, kind, offset=public_api.MAX_OFFSET)
    assert page == {"type": kind, "items": [], "limit": 50, "offset": public_api.MAX_OFFSET, "more": False}


@pytest.mark.parametrize("kind", TYPES)
def test_the_cache_header_and_a_weak_etag_give_a_304_until_the_page_changes(shop, kind):
    _seed_every_type(shop)
    first = _get(shop, kind)
    assert first.status_code == 200 and first.headers["cache-control"] == "public, max-age=300"
    assert first.headers["x-content-type-options"] == "nosniff" and first.headers["etag"].startswith('W/"')
    etag = first.headers["etag"]
    again = _get(shop, kind, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag
    assert again.headers["cache-control"] == "public, max-age=300"
    _change_the_first_page(shop, kind, first.json())
    changed = _get(shop, kind, headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_shared_bucket(shop, monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 3)
    headers = {"X-Forwarded-For": "198.51.100.9"}
    assert [_get(shop, kind, headers=headers).status_code for kind in TYPES[:3]] == [200, 200, 200]
    refused = _get(shop, "llms", headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    assert len(shop.redis.z) == 1 and next(iter(shop.redis.z)).endswith(":marketplace-public:ip:198.51.100.9")
    assert _get(shop, "llms", headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 200


def test_the_local_edition_answers_404_before_the_limit_counts(shop, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    for kind in TYPES:
        resp = _get(shop, kind, limit=500)
        assert resp.status_code == 404 and resp.json()["detail"] == public_api.NOT_HOSTED
        assert resp.headers["cache-control"] == "no-store"
    assert shop.redis.z == {}
