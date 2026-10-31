"""The one Marketplace page (PRD-270 §5.7): the signed-out page's Featured and counts reads.

``GET /api/public/marketplace/featured`` (``{featured, popular}``) and ``GET /api/public/marketplace/counts``
(``{items, installs}``), on ``api/public_marketplace.py`` over ``services/marketplace_public/featured.py``
and ``counts.py``. Two halves:

* on Postgres, as US-270-07a's suite runs it (``test_prd270w1_public_reads.py``): the real router on a mini
  app with no request-context override, rows flushed on one session and rolled back. Proven: only approved
  ``owner_type='marketplace'`` rows count or appear (never an unapproved one, a workspace's or another
  owner's); Featured is the featured agents only, most installed first, at most five; Popular mixes agents
  and Playbooks by installs, never a featured one, at most six; the allowlist walk (every key on
  ``FEATURED_FIELDS`` / ``FEATURED_ANSWER_FIELDS`` / ``COUNTS_FIELDS``, nothing private, no column off the
  list, no creator's email); the publisher's display-name rules; required apps; the counts, an install
  count that is NULL read as 0;
* without a database (a stubbed read): ``featured`` and ``counts`` reach their own routes, never the
  ``{type}`` 404; ``Cache-Control: public, max-age=300``, nosniff, a weak ETag and 304; 429 with
  ``Retry-After`` in the public reads' bucket; 404 on a local edition before the limit counts; a failed
  read a logged 503; every refusal ``no-store``; Popular's ranking rules; the allowlist refuses a field off it.

Both routes are plain ``def`` s with no request-context or permission dependency, in the route manifest and
mounted through ``RouterSpec``: ``test_prd270w1_public_reads.py`` and ``test_prd270w1_wire.py`` pin that,
on the router and on the live app.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

import api.public_marketplace as public_api
import api.public_rate_limit as rate_limit
from api.widgets.rate_limit import RateLimitStore
from config import config
from core.database.database import get_db
from services.marketplace_public import featured as featured_reads
from services.marketplace_public import fields
from tests.test_prd264_content_provenance import _FakeRedis
from tests.test_prd270w1_public_reads import MARK, TOP, _agent, _keys_anywhere, _playbook, _spelling, _user

FEATURED, COUNTS = "/api/public/marketplace/featured", "/api/public/marketplace/counts"
FORBIDDEN = {
    "configuration", "model_config", "system_prompt", "custom_persona_prompt", "steps", "inputs", "outputs",
    "execution_config", "schedule_config", "template_definition", "learning_data", "workspace_id", "owner_id",
    "created_by", "created_by_user_id", "original_creator_id", "cloned_from_id", "public_id", "email",
    "is_approved", "is_featured", "created_at",
}
STUB_FEATURED = {"featured": [{"key": "1", "name": "Closer"}], "popular": []}
STUB_COUNTS = {"items": 3, "installs": 9}


# ---------------------------------------------------------------------------
# On Postgres
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            for table in ("users", "agents", "workflow_recipes", "agent_tool_assignments"):
                c.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the public Featured and counts tests need Postgres with the marketplace tables: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def shop(engine, new_session, monkeypatch):
    """The router on a mini app over one uncommitted session; the hosted edition and a fake Redis."""
    db = new_session()
    redis = _FakeRedis()
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


def _featured_agent(shop, name, installs, **kwargs):
    agent = _agent(shop, name, installs=installs, **kwargs)
    agent.is_featured = True
    shop.db.flush()
    return agent


def _answer(shop, path):
    resp = shop.client.get(path)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _keyed(items):
    return [(item["type"], item["key"]) for item in items]


def test_featured_is_the_approved_featured_marketplace_agents_most_installed_first(shop):
    third, first = _featured_agent(shop, "Third", TOP + 1), _featured_agent(shop, "First", TOP + 3)
    second = _featured_agent(shop, "Second", TOP + 2)
    unapproved = _featured_agent(shop, "Draft", TOP + 9, approved=False)
    theirs = _featured_agent(shop, "Ours", TOP + 9, owner="workspace")
    playbook = _playbook(shop, "Featured review")
    playbook.is_featured = True
    shop.db.flush()
    featured = _answer(shop, FEATURED)["featured"]
    assert _keyed(featured)[:3] == [("agents", str(a.id)) for a in (first, second, third)]
    assert len(featured) <= featured_reads.FEATURED_LIMIT == 5
    keys = {item["key"] for item in featured}
    assert str(unapproved.id) not in keys and str(theirs.id) not in keys
    assert all(item["type"] == "agents" for item in featured), "Featured lists agents only, as the member route"
    assert featured[0]["installCount"] == TOP + 3 and featured[0]["version"] == "1.0.0"
    assert featured[0]["name"] == first.name and featured[0]["category"] == "sales"


def test_popular_mixes_agents_and_playbooks_by_installs_never_a_featured_one(shop):
    top_featured = _featured_agent(shop, "Star", TOP + 100)
    best, third = _agent(shop, "Best", installs=TOP + 60), _agent(shop, "Third", installs=TOP + 40)
    second, fourth = _playbook(shop, "Second"), _playbook(shop, "Fourth")
    second.install_count, fourth.install_count = TOP + 50, TOP + 30
    unapproved = _agent(shop, "Draft", approved=False, installs=TOP + 90)
    theirs = _playbook(shop, "Their review", owner="workspace")
    theirs.install_count = TOP + 80
    shop.db.flush()
    popular = _answer(shop, FEATURED)["popular"]
    assert _keyed(popular)[:4] == [
        ("agents", str(best.id)), ("playbooks", str(second.id)), ("agents", str(third.id)),
        ("playbooks", str(fourth.id)),
    ]
    assert len(popular) <= featured_reads.POPULAR_LIMIT == 6
    keyed = set(_keyed(popular))
    assert ("agents", str(top_featured.id)) not in keyed and ("agents", str(unapproved.id)) not in keyed
    assert ("playbooks", str(theirs.id)) not in keyed
    assert popular[1]["installCount"] == TOP + 50 and popular[1]["requiredApps"] == ["gmail"]


def test_every_key_is_on_the_allowlist_and_nothing_private_leaves(shop):
    creator = _user(shop, name="Ada Lovelace")
    agent = _featured_agent(shop, "Closer", TOP + 5, creator=creator, tools=[("GMAIL", True)])
    _agent(shop, "Opener", creator=creator, installs=TOP + 4)
    _playbook(shop, "Pipeline review", creator=creator)
    assert not {_spelling(key) for key in fields.FEATURED_FIELDS} & {_spelling(key) for key in FORBIDDEN}
    answer = _answer(shop, FEATURED)
    assert set(answer) == fields.FEATURED_ANSWER_FIELDS
    seen = _keys_anywhere(answer)
    assert seen <= fields.FEATURED_FIELDS | fields.FEATURED_ANSWER_FIELDS, seen
    for item in answer["featured"] + answer["popular"]:
        assert set(item) == fields.FEATURED_FIELDS, set(item) ^ fields.FEATURED_FIELDS
    counts = _answer(shop, COUNTS)
    assert set(counts) == fields.COUNTS_FIELDS and all(type(value) is int for value in counts.values())
    for body in (json.dumps(answer, ensure_ascii=False), json.dumps(counts)):
        assert MARK not in body and creator.email not in body and str(agent.public_id) not in body


def test_the_publisher_is_a_display_name_and_required_apps_are_lower_case_slugs(shop):
    named, emailed = _user(shop, name=" Ada Lovelace ", label="named"), _user(shop, name="g@x.test", label="mail")
    agent = _featured_agent(shop, "Named", TOP + 7, creator=named,
                            tools=[("GMAIL", True), ("gmail", True), ("SLACK", False)])
    mailed = _featured_agent(shop, "Mailed", TOP + 6, creator=emailed)
    orphan = _featured_agent(shop, "Orphan", TOP + 5)
    by_key = {item["key"]: item for item in _answer(shop, FEATURED)["featured"]}
    assert by_key[str(agent.id)]["publisher"] == "Ada Lovelace"
    assert by_key[str(mailed.id)]["publisher"] == by_key[str(orphan.id)]["publisher"] == fields.DEFAULT_PUBLISHER
    assert by_key[str(agent.id)]["requiredApps"] == ["gmail"]
    body = json.dumps(_answer(shop, FEATURED))
    assert named.email not in body and emailed.email not in body and "g@x.test" not in body


def test_the_counts_are_the_approved_marketplace_agents_and_playbooks_and_their_installs(shop):
    before = _answer(shop, COUNTS)
    _agent(shop, "Counted", installs=7)
    nulled = _playbook(shop, "Counted review")
    shop.db.execute(text("UPDATE workflow_recipes SET install_count = NULL WHERE id = :id"), {"id": nulled.id})
    _agent(shop, "Draft", approved=False, installs=100)
    _agent(shop, "Ours", owner="workspace", installs=100)
    _playbook(shop, "Draft review", approved=False)
    _playbook(shop, "Their review", owner="workspace")
    after = _answer(shop, COUNTS)
    assert after == {"items": before["items"] + 2, "installs": before["installs"] + 7}


# ---------------------------------------------------------------------------
# Without a database: the routes over a stubbed read
# ---------------------------------------------------------------------------


@pytest.fixture
def stub(monkeypatch):
    """The router on a mini app whose reads answer fixed bodies; every tab read is an unknown type (404)."""
    redis = _FakeRedis()
    answers = {"featured": STUB_FEATURED, "counts": STUB_COUNTS}
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 1000)
    monkeypatch.setattr(rate_limit, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setattr(public_api, "read_featured", lambda db: answers["featured"])
    monkeypatch.setattr(public_api, "read_counts", lambda db: answers["counts"])
    monkeypatch.setattr(public_api, "public_page", lambda db, kind, limit, offset: None)
    app = FastAPI()
    app.include_router(public_api.router)
    app.dependency_overrides[get_db] = lambda: None
    return NS(client=TestClient(app), redis=redis, answers=answers)


@pytest.mark.parametrize("path, body", [(FEATURED, STUB_FEATURED), (COUNTS, STUB_COUNTS)])
def test_featured_and_counts_reach_their_own_routes_never_the_type_route(stub, path, body):
    resp = stub.client.get(path)
    assert resp.status_code == 200 and resp.json() == body
    unknown = stub.client.get("/api/public/marketplace/agents")
    assert unknown.status_code == 404 and unknown.json()["detail"] == public_api.NO_SUCH_TYPE


@pytest.mark.parametrize("path, read", [(FEATURED, "featured"), (COUNTS, "counts")])
def test_the_cache_header_and_a_weak_etag_give_a_304_until_the_answer_changes(stub, path, read):
    first = stub.client.get(path)
    assert first.headers["cache-control"] == "public, max-age=300"
    assert first.headers["x-content-type-options"] == "nosniff"
    assert first.headers["content-type"].startswith("application/json")
    etag = first.headers["etag"]
    assert etag.startswith('W/"')
    again = stub.client.get(path, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b"" and again.headers["etag"] == etag
    assert again.headers["cache-control"] == "public, max-age=300"
    stub.answers[read] = {**stub.answers[read], "changed": True}
    changed = stub.client.get(path, headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_over_the_per_ip_limit_is_a_429_with_retry_after_in_the_public_reads_bucket(stub, monkeypatch):
    monkeypatch.setattr(config, "MARKETPLACE_PUBLIC_RATE_PER_MINUTE", 2)
    headers = {"X-Forwarded-For": "198.51.100.9"}
    assert [stub.client.get(path, headers=headers).status_code for path in (FEATURED, COUNTS)] == [200, 200]
    refused = stub.client.get(FEATURED, headers=headers)
    assert refused.status_code == 429
    assert int(refused.headers["retry-after"]) > 0 and refused.headers["cache-control"] == "no-store"
    assert len(stub.redis.z) == 1
    assert next(iter(stub.redis.z)).endswith(f":{public_api.PUBLIC_RATE_BUCKET}:198.51.100.9")
    assert stub.client.get(COUNTS, headers={"X-Forwarded-For": "198.51.100.10"}).status_code == 200


def test_the_local_edition_answers_404_before_the_limit_counts(stub, monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    for path in (FEATURED, COUNTS):
        resp = stub.client.get(path)
        assert resp.status_code == 404 and resp.json()["detail"] == public_api.NOT_HOSTED
        assert resp.headers["cache-control"] == "no-store" and "etag" not in resp.headers
    assert stub.redis.z == {}


@pytest.mark.parametrize("path, read, label", [
    (FEATURED, "read_featured", "featured read failed"), (COUNTS, "read_counts", "counts read failed"),
])
def test_a_read_the_database_fails_is_a_logged_503(stub, monkeypatch, caplog, path, read, label):
    def broken(db):
        raise OperationalError("SELECT", {}, Exception("connection lost"))

    monkeypatch.setattr(public_api, read, broken)
    with caplog.at_level(logging.ERROR, logger=public_api.__name__):
        resp = stub.client.get(path)
    assert resp.status_code == 503 and resp.json()["detail"] == public_api.READS_DOWN
    assert resp.headers["cache-control"] == "no-store"
    assert any(label in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Popular's ranking and the allowlist, as plain functions
# ---------------------------------------------------------------------------


def _row(row_id, installs, created=datetime(2026, 1, 1)):
    return NS(id=row_id, install_count=installs, created_at=created)


def test_popular_ranks_by_installs_then_newest_and_drops_only_the_featured_agents():
    older, newer = datetime(2026, 1, 1), datetime(2026, 6, 1)
    agents = [_row(1, 50), _row(2, 30, older), _row(3, None)]
    playbooks = [_row(1, 40), _row(2, 30, newer), _row(9, 10, None)]
    ranked = featured_reads.popular_from(agents, playbooks, featured_ids=[1])
    assert [(kind, row.id) for kind, row in ranked] == [
        ("playbooks", 1), ("playbooks", 2), ("agents", 2), ("playbooks", 9), ("agents", 3),
    ], "agent 1 is featured; Playbook 1 shares its id and stays; a NULL count ranks as 0"


def test_popular_takes_the_pool_before_dropping_the_featured_and_shows_at_most_six():
    pool = featured_reads.POPULAR_POOL
    agents = [_row(row_id, 1000 - row_id) for row_id in range(1, pool + 5)]
    ranked = featured_reads.popular_from(agents, [], featured_ids=range(1, pool - 2))
    assert [row.id for _kind, row in ranked] == [pool - 2, pool - 1, pool], "only the pool's last three remain"
    assert len(featured_reads.popular_from(agents, [], featured_ids=[])) == featured_reads.POPULAR_LIMIT


def test_a_field_off_the_featured_allowlist_is_refused_before_it_is_answered():
    item = fields.display_item(fields.FEATURED_FIELDS, key=1, name="Closer", type="agents", version="1.0",
                               installCount=3)
    assert set(item) == fields.FEATURED_FIELDS
    with pytest.raises(fields.FieldNotAllowed, match="configuration"):
        fields.display_item(fields.FEATURED_FIELDS, key=1, name="Closer", configuration={"api_key": "x"})
    with pytest.raises(fields.FieldNotAllowed, match="workspace_id"):
        fields.allowed_item(fields.COUNTS_FIELDS, items=1, installs=2, workspace_id="w")


def _listing(row_id, installs, creator=None, **extra):
    return NS(
        id=row_id, name=f"Item {row_id}", description=" Does its job. ", marketplace_category="sales",
        marketplace_icon="javascript:alert(1)", tags=["a", "", 7], original_creator_id=creator, version=" 2.1 ",
        install_count=installs, created_at=datetime(2026, 1, 1), **extra,
    )


def test_read_featured_maps_each_row_through_the_allowlist(monkeypatch):
    star = _listing(1, 90, creator=5)
    agents, playbooks = [star, _listing(2, 70)], [_listing(1, 80, required_tools=["GMAIL", "gmail", 3])]
    asked = {}

    def names(db, ids):
        asked["ids"] = list(ids)
        return {5: "Ada"}

    def apps(db, ids):
        asked["apps"] = list(ids)
        return {1: ["gmail"]}

    monkeypatch.setattr(featured_reads, "_agent_rows", lambda db, limit, featured_only: [star] if featured_only
                        else agents)
    monkeypatch.setattr(featured_reads, "_playbook_rows", lambda db, limit: playbooks)
    monkeypatch.setattr(featured_reads, "publisher_names", names)
    monkeypatch.setattr(featured_reads, "apps_by_agent", apps)
    answer = featured_reads.read_featured(db=None)
    assert [item["key"] for item in answer["featured"]] == ["1"]
    assert _keyed(answer["popular"]) == [("playbooks", "1"), ("agents", "2")]
    assert answer["featured"][0] == {
        "key": "1", "name": "Item 1", "description": "Does its job.", "category": "sales", "icon": None,
        "tags": ["a"], "publisher": "Ada", "requiredApps": ["gmail"], "type": "agents", "version": "2.1",
        "installCount": 90,
    }
    assert answer["popular"][0]["requiredApps"] == ["gmail"] and answer["popular"][0]["publisher"] == "Automatos"
    assert asked == {"ids": [5, None, None], "apps": [1, 2]}
