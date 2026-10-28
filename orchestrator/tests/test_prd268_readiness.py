"""PRD-268 US-C18 (§6.11.3) — readiness: what Auto depends on, and what couldn't be checked.

Proven here:

* **the probes** (``services.readiness.probes``), each in its three states: ``core`` (the
  database answers ``SELECT 1``, on Postgres; a refused connection needs setting up),
  ``model_route`` (Auto's model resolves through the check-in's own ``heartbeat_llm``,
  with a key; unset or keyless needs setting up), ``web_search`` (the
  engine ``resolve_backend`` names for the workspace; none, or switched off, needs setting
  up) and ``push`` (always on SaaS; on a local install once the operator agreed); and for
  each, a probe that does not answer within ``READINESS_PROBE_TIMEOUT_SECONDS`` reads
  ``not_checked``, without the check waiting for it;
* **the runner** (``services.readiness.service``): a probe that raises needs setting up
  (``unreadable``) and its session is closed; a fix is shown only on an item that needs
  setting up, and every fix is an app path (one that isn't cannot be built); SaaS details
  carry no host and no latency, the local edition's do; every sentence is in English and
  Portuguese (Brazil);
* **the routes**: any member reads ``Readiness@1`` (a stranger 403, no workspace 404), in
  both editions and in the language ``Accept-Language`` asks for; ``GET`` answers the held
  check and ``POST /check`` checks again, limited per workspace (429 ``checked_recently``
  with ``Retry-After``, nothing checked); plain ``def``s, mounted through
  ``RouterSpec("api.readiness")``, in the route manifest; the timeout in the config surface.
"""
from __future__ import annotations

import inspect
import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from services.readiness import items as i
from services.readiness import probes, service
from services.readiness.contract import ReadinessFix
from services.readiness.items import FIXES, Finding, Item
from services.readiness.strings import CHECKED_RECENTLY, NO_WORKSPACE, STRINGS

ORCH = Path(__file__).resolve().parents[1]
MANIFEST = ORCH / "reports" / "route-manifest.json"
CONFIG_SURFACE = ORCH / "reports" / "config-surface.json"
READINESS = "/api/readiness"
CHECK = "/api/readiness/check"
ITEM_IDS = ["core", "model_route", "web_search", "push"]
SHORT_TIMEOUT = 0.05
HANG_SECONDS = 5.0
WHERE = "db.internal.example:5432"
LIMIT = 2
DETAILS = [i.CORE_READY, i.CORE_DOWN, i.MODEL_READY, i.MODEL_UNSET, i.MODEL_NO_KEY, i.SEARCH_OPENROUTER,
           i.SEARCH_COMPOSIO, i.SEARCH_SEARXNG, i.SEARCH_NONE, i.SEARCH_OFF, i.PUSH_READY, i.PUSH_UNANSWERED,
           i.PUSH_DECLINED, i.NOT_ANSWERED, i.UNREADABLE]


def _item(item_id: str) -> Item:
    return next(item for item in probes.ITEMS if item.id == item_id)


@pytest.fixture
def saas(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "saas")


@pytest.fixture
def local(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "local")


@pytest.fixture(autouse=True)
def nothing_held(monkeypatch):
    """Every test starts with no check held."""
    monkeypatch.setattr(service, "held", service.HeldChecks(60))


@pytest.fixture
def sessions(monkeypatch):
    """The runner's sessions: each a stand-in recording that it was closed."""
    closed = []
    monkeypatch.setattr(service, "_open_session", lambda: NS(close=lambda: closed.append(True)))
    return closed


# ── the probes, each in its three states ────────────────────────────────────

def test_core_is_ready_when_the_database_answers(db_session, test_engine):
    url = test_engine.url

    finding = probes.core(db_session, uuid.uuid4())

    assert (finding.state, finding.detail, finding.fix) == ("ready", "core_ready", None)
    assert finding.where == (f"{url.host}:{url.port}" if url.host and url.port else url.host)


def test_core_needs_setup_when_the_database_does_not_answer():
    def refuse(_statement):
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    assert probes.core(NS(execute=refuse), uuid.uuid4()) == Finding("needs_setup", "core_down")


def _manager(monkeypatch, api_key=None, refuse=False):
    built = []

    def build(workspace_id, agent_id):
        built.append((workspace_id, agent_id))
        if refuse:
            raise ValueError("LLM provider not configured for service 'heartbeat'")
        return NS(config=NS(api_key=api_key))

    monkeypatch.setattr(probes, "heartbeat_llm", build)
    return built


def test_model_route_is_ready_when_auto_s_model_resolves_with_a_key(monkeypatch):
    ws = uuid.uuid4()
    built = _manager(monkeypatch, api_key="a-test-key")

    assert probes.model_route(None, ws) == Finding("ready", "model_ready")
    assert built == [(ws, None)]  # the check-in's own manager (K9), built, never called


def test_model_route_builds_the_check_in_s_manager():
    from services.proactive import runner

    assert probes.heartbeat_llm is runner.heartbeat_llm and runner.HEARTBEAT_SERVICE == "heartbeat"


def test_model_route_needs_setup_with_no_model_or_no_key(monkeypatch):
    _manager(monkeypatch, refuse=True)
    assert probes.model_route(None, uuid.uuid4()) == Finding("needs_setup", "model_unset", fix="fix_choose_model")

    _manager(monkeypatch, api_key=None)
    assert probes.model_route(None, uuid.uuid4()) == Finding("needs_setup", "model_no_key", fix="fix_add_key")


def _backend(monkeypatch, backend, provider="auto"):
    from config import config

    asked = []
    monkeypatch.setattr(config, "WEB_SEARCH_PROVIDER", provider)
    monkeypatch.setattr(config, "SEARXNG_URL", "http://searxng:8080")
    monkeypatch.setattr(probes.search, "resolve_backend", lambda db, ws: asked.append((db, ws)) or backend)
    return asked


def test_web_search_is_ready_with_the_engine_the_workspace_would_use(monkeypatch):
    ws, db = uuid.uuid4(), object()
    asked = _backend(monkeypatch, "openrouter")
    assert probes.web_search(db, ws) == Finding("ready", "search_openrouter")
    assert asked == [(db, ws)]

    _backend(monkeypatch, "searxng")
    assert probes.web_search(db, ws) == Finding("ready", "search_searxng", where="searxng:8080")


def test_web_search_never_shows_the_searxng_url_s_user_or_password(monkeypatch):
    from config import config

    _backend(monkeypatch, "searxng")
    monkeypatch.setattr(config, "SEARXNG_URL", "http://reader:pw@searxng:8080/")

    assert probes.web_search(None, uuid.uuid4()).where == "searxng:8080"


def test_web_search_needs_setup_with_no_engine_or_switched_off(monkeypatch):
    _backend(monkeypatch, None)
    assert probes.web_search(None, uuid.uuid4()) == Finding("needs_setup", "search_none", fix="fix_add_key")

    asked = _backend(monkeypatch, "openrouter", provider="off")
    assert probes.web_search(None, uuid.uuid4()) == Finding("needs_setup", "search_off")
    assert asked == []


def _consent(monkeypatch, agreed: bool, answered: bool):
    monkeypatch.setattr(probes.consent, "consent_state", lambda: NS(agreed=agreed, answered=answered))


def test_push_is_ready_on_saas_and_on_a_local_install_its_operator_agreed(monkeypatch, saas):
    _consent(monkeypatch, agreed=False, answered=False)
    assert probes.push(None, uuid.uuid4()) == Finding("ready", "push_ready")

    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    _consent(monkeypatch, agreed=True, answered=True)
    assert probes.push(None, uuid.uuid4()) == Finding("ready", "push_ready", where="exp.host")


def test_push_needs_setup_on_a_local_install_not_agreed(monkeypatch, local):
    _consent(monkeypatch, agreed=False, answered=False)
    assert probes.push(None, uuid.uuid4()) == Finding("needs_setup", "push_unanswered", fix="fix_answer_push")

    _consent(monkeypatch, agreed=False, answered=True)
    assert probes.push(None, uuid.uuid4()) == Finding("needs_setup", "push_declined", fix="fix_answer_push")


@pytest.fixture
def hanging(monkeypatch):
    """What every probe reads hangs past a short timeout (the database, Auto's model, the
    search engine, the operator's answer); the hang ends with the test."""
    from config import config

    release = threading.Event()
    monkeypatch.setattr(config, "READINESS_PROBE_TIMEOUT_SECONDS", SHORT_TIMEOUT)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "WEB_SEARCH_PROVIDER", "auto")

    def after_the_hang(answer):
        def hang(*_args, **_kwargs):
            release.wait(HANG_SECONDS)
            return answer
        return hang

    no_host = NS(engine=NS(url=NS(host=None, port=None)))
    session = NS(execute=after_the_hang(None), get_bind=lambda: no_host, close=lambda: None)
    monkeypatch.setattr(service, "_open_session", lambda: session)
    monkeypatch.setattr(probes, "heartbeat_llm", after_the_hang(NS(config=NS(api_key="a-test-key"))))
    monkeypatch.setattr(probes.search, "resolve_backend", after_the_hang("openrouter"))
    monkeypatch.setattr(probes.consent, "consent_state", after_the_hang(NS(agreed=True, answered=True)))
    yield
    release.set()


@pytest.mark.parametrize("item_id", ITEM_IDS)
def test_a_probe_that_does_not_answer_in_time_reads_not_checked(hanging, item_id):
    started = time.monotonic()

    [result] = service.check(uuid.uuid4(), [_item(item_id)]).results

    assert time.monotonic() - started < HANG_SECONDS / 2  # the check never waits for it
    assert (result.finding.state, result.finding.detail, result.ms) == ("not_checked", "not_answered", None)


# ── the runner and Readiness@1 ──────────────────────────────────────────────

def test_a_probe_that_raises_needs_setup_and_its_session_is_closed(sessions):
    def broken(db, workspace_id):
        raise RuntimeError("the settings table is gone")

    [result] = service.check(uuid.uuid4(), [Item("core", ("brief",), broken)]).results

    assert result.finding == Finding("needs_setup", "unreadable") and sessions == [True]


def test_reads_that_find_nothing_held_wait_for_one_check(sessions, monkeypatch, saas):
    asked, answers, go = [], [], threading.Event()

    def ready_when_told(db, workspace_id):
        asked.append(workspace_id)
        go.wait(HANG_SECONDS)
        return Finding("ready", "core_ready")

    monkeypatch.setattr(probes, "ITEMS", (Item("core", ("brief",), ready_when_told),))
    ws = uuid.uuid4()
    readers = [threading.Thread(target=lambda: answers.append(service.current(ws, "en"))) for _ in range(3)]
    for reader in readers:
        reader.start()
    time.sleep(SHORT_TIMEOUT)
    go.set()
    for reader in readers:
        reader.join(HANG_SECONDS)

    assert asked == [ws] and len(answers) == 3 and answers.count(answers[0]) == 3


def _checked(*results) -> service.Checked:
    return service.Checked(datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc), tuple(results))


def test_every_fix_opens_an_app_path():
    assert FIXES and all(fix.href.startswith("/") and not fix.href.startswith("//") for fix in FIXES.values())


@pytest.mark.parametrize("href", ["https://example.com/settings", "//example.com/x", "/\\example.com", "settings"])
def test_a_fix_that_is_not_an_app_path_cannot_be_built(href):
    with pytest.raises(ValidationError):
        ReadinessFix(label="Add a key", href=href)


def test_a_fix_is_shown_only_on_an_item_that_needs_setting_up(saas):
    needs = service.Result(_item("model_route"), Finding("needs_setup", "model_unset", fix="fix_choose_model"))
    ready = service.Result(_item("web_search"), Finding("ready", "search_openrouter", fix="fix_add_key"))

    first, second = service.render(_checked(needs, ready), "en")["items"]

    assert first["fix"] == {"label": "Choose Auto's model", "href": "/settings?tab=orchestrator"}
    assert first["fix"]["href"].startswith("/") and "fix" not in second


@pytest.mark.parametrize("detail", DETAILS)
def test_saas_details_carry_no_host_and_no_latency(saas, detail):
    result = service.Result(_item("core"), Finding("needs_setup", detail, where=WHERE), 12)

    [item] = service.render(_checked(result), "en")["items"]

    assert item["detail"] == STRINGS["en"][detail]
    assert "db.internal" not in json.dumps(item) and not re.search(r"\d+\s*ms\b", item["detail"])


def test_the_local_edition_shows_where_and_how_long(local):
    result = service.Result(_item("core"), Finding("ready", "core_ready", where=WHERE), 12)

    [item] = service.render(_checked(result), "en")["items"]

    assert item["detail"] == f"The server and its database answer. · {WHERE} · 12 ms"


def test_readiness_is_v1_with_every_server_item_and_what_needs_it(saas, sessions, monkeypatch):
    monkeypatch.setattr(probes, "ITEMS", tuple(Item(item.id, item.needed_by, lambda db, ws: Finding("ready", "core_ready"))
                                              for item in probes.ITEMS))

    body = service.check_now(uuid.uuid4(), "en")

    assert body["v"] == 1 and datetime.fromisoformat(body["checked_at"]).tzinfo is not None
    assert [item["id"] for item in body["items"]] == ITEM_IDS
    assert {item["id"]: item["needed_by"] for item in body["items"]} == {
        "core": ["brief", "voice", "push", "check_in"], "model_route": ["brief", "check_in"],
        "web_search": ["voice"], "push": ["push"]}
    assert len(sessions) == len(ITEM_IDS)  # each probe on its own session, each closed


def test_every_sentence_is_in_english_and_portuguese():
    keys = {item.id for item in probes.ITEMS} | set(FIXES) | set(DETAILS) | {CHECKED_RECENTLY, NO_WORKSPACE}

    assert set(STRINGS["en"]) == set(STRINGS["pt-BR"]) == keys
    assert all(STRINGS[locale][key].strip() for locale in STRINGS for key in keys)


# ── the routes ──────────────────────────────────────────────────────────────

@pytest.fixture
def team(db_session) -> NS:
    """A workspace, its owner and a viewer; another workspace and its owner."""
    def person(tag):
        clerk = f"user_prd268rd_{tag}_{uuid.uuid4().hex[:10]}"
        pk = db_session.execute(text(
            "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
            {"u": clerk, "e": f"{clerk}@t.test", "c": clerk}).scalar()
        return NS(pk=int(pk), clerk=clerk)

    def workspace(owner):
        ws = uuid.uuid4()
        db_session.execute(text(
            "INSERT INTO workspaces (id, name, slug, plan, owner_id) VALUES (CAST(:id AS uuid), "
            "'prd268 readiness', :slug, 'pro', :o)"), {"id": str(ws), "slug": f"prd268-ready-{ws.hex[:10]}", "o": owner.pk})
        return ws

    owner, other_owner, viewer = person("owner"), person("other"), person("viewer")
    ws, other_ws = workspace(owner), workspace(other_owner)
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :uid, 'viewer', true)"), {"ws": str(ws), "uid": viewer.pk})
    db_session.flush()
    return NS(db=db_session, ws=ws, owner=owner, viewer=viewer, other_owner=other_owner, other_ws=other_ws)


@pytest.fixture
def probed(monkeypatch, sessions) -> list:
    """Fast stand-in probes: each records the workspace it was asked about."""
    asked = []

    def probe(item_id, finding):
        def run(db, workspace_id):
            asked.append((item_id, workspace_id))
            return finding
        return run

    findings = {"core": Finding("ready", "core_ready", where=WHERE), "model_route": Finding("ready", "model_ready"),
                "web_search": Finding("needs_setup", "search_none", fix="fix_add_key"),
                "push": Finding("ready", "push_ready")}
    monkeypatch.setattr(probes, "ITEMS", tuple(Item(item.id, item.needed_by, probe(item.id, findings[item.id]))
                                              for item in probes.ITEMS))
    return asked


class _FakePipeline:
    def __init__(self, redis):
        self._redis, self._ops = redis, []

    def __getattr__(self, name):
        def queue(*args, **kwargs):
            self._ops.append((name, args, kwargs))
            return self
        return queue

    def execute(self):
        return [getattr(self._redis, name)(*args, **kwargs) for name, args, kwargs in self._ops]


class _FakeRedis:
    """The sorted-set commands ``RateLimitStore`` uses, on a dict."""

    def __init__(self):
        self.z = {}

    def zremrangebyscore(self, key, low, high):
        marks = self.z.get(key, {})
        for member in [m for m, score in marks.items() if float(low) <= score <= float(high)]:
            marks.pop(member)

    def zcard(self, key):
        return len(self.z.get(key, {}))

    def zrange(self, key, start, end, withscores=False):
        marks = sorted(self.z.get(key, {}).items(), key=lambda kv: kv[1])[start:end + 1]
        return marks if withscores else [m for m, _ in marks]

    def zadd(self, key, mapping):
        self.z.setdefault(key, {}).update(mapping)

    def expire(self, key, ttl):
        return True

    def pipeline(self):
        return _FakePipeline(self)


@pytest.fixture
def limited(monkeypatch):
    import api.readiness as readiness_api
    from api.widgets.rate_limit import RateLimitStore
    from core.security import rate_limiter

    redis = _FakeRedis()
    monkeypatch.setattr(readiness_api, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    monkeypatch.setitem(rate_limiter.DEFAULT_LIMITS, service.CHECK_OPERATION, (LIMIT, 60))
    return redis


def _client(db, ctx) -> TestClient:
    import api.readiness as readiness_api
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(readiness_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def _as(team, person, ws=None, system_role: str = "user") -> TestClient:
    from core.auth.dependencies import RequestContext, UserContext

    user = UserContext(id=person.clerk, clerk_user_id=person.clerk, system_role=system_role)
    return _client(team.db, RequestContext(workspace_id=ws or team.ws, user=user, auth_type="clerk"))


def test_any_member_reads_readiness_scoped_to_their_workspace(team, saas, probed):
    reply = _as(team, team.viewer).get(READINESS)

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["v"] == 1 and [item["id"] for item in body["items"]] == ITEM_IDS
    assert {ws for _, ws in probed} == {team.ws}
    [search] = [item for item in body["items"] if item["id"] == "web_search"]
    assert search == {"id": "web_search", "label": "Web search", "detail": "No search engine is set up.",
                      "state": "needs_setup", "needed_by": ["voice"],
                      "fix": {"label": "Add a key", "href": "/settings?tab=api-keys"}}
    assert WHERE not in reply.text and not re.search(r"\d+ ms", reply.text)  # SaaS


def test_a_stranger_is_403_and_no_workspace_is_404(team, saas, probed, limited):
    stranger = _as(team, team.other_owner)
    assert stranger.get(READINESS).status_code == 403
    assert stranger.post(CHECK).status_code == 403

    from core.auth.dependencies import RequestContext, UserContext

    nobody = UserContext(id="user_prd268rd_admin", clerk_user_id="user_prd268rd_admin", system_role="super_admin")
    client = _client(team.db, RequestContext(workspace_id=None, user=nobody, auth_type="clerk"))
    for reply in (client.get(READINESS), client.post(CHECK)):
        assert reply.status_code == 404 and reply.json()["code"] == NO_WORKSPACE
    assert probed == []


def test_get_answers_the_held_check_and_check_runs_again(team, saas, probed, limited):
    client = _as(team, team.owner)

    first, second = client.get(READINESS).json(), client.get(READINESS).json()
    assert first == second and len(probed) == len(ITEM_IDS)  # one check, held

    again = client.post(CHECK)
    assert again.status_code == 200 and len(probed) == 2 * len(ITEM_IDS)
    assert client.get(READINESS).json() == again.json()


def test_check_again_is_rate_limited_per_workspace(team, saas, probed, limited):
    client = _as(team, team.owner)

    assert [client.post(CHECK).status_code for _ in range(LIMIT)] == [200] * LIMIT
    refused = client.post(CHECK, headers={"Accept-Language": "pt-BR"})

    assert refused.status_code == 429
    assert refused.json() == {"code": CHECKED_RECENTLY, "detail": STRINGS["pt-BR"][CHECKED_RECENTLY]}
    assert int(refused.headers["Retry-After"]) >= 1
    assert len(probed) == LIMIT * len(ITEM_IDS)  # nothing checked once refused
    assert _as(team, team.other_owner, ws=team.other_ws).post(CHECK).status_code == 200  # its own window


def test_the_copy_follows_accept_language(team, saas, probed):
    body = _as(team, team.owner).get(READINESS, headers={"Accept-Language": "pt-BR,en;q=0.5"}).json()

    assert [item["label"] for item in body["items"]] == [STRINGS["pt-BR"][item_id] for item_id in ITEM_IDS]


def test_the_local_operator_reads_readiness_with_where_and_how_long(team, local, probed):
    from core.auth.dependencies import RequestContext, UserContext

    operator = UserContext(id="operator@local.test", email="operator@local.test", system_role="super_admin",
                           raw_claims={"source": "local_operator", "user_id": team.owner.pk})
    reply = _client(team.db, RequestContext(workspace_id=team.ws, user=operator, auth_type="anonymous")).get(READINESS)

    assert reply.status_code == 200, reply.text
    [core] = [item for item in reply.json()["items"] if item["id"] == "core"]
    assert core["detail"].startswith(f"The server and its database answer. · {WHERE} · ")
    assert core["detail"].endswith(" ms")


def test_the_routes_are_plain_defs_in_the_route_manifest():
    import api.readiness as readiness_api
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(readiness_api.get_readiness)
    assert not inspect.iscoroutinefunction(readiness_api.check_readiness)
    assert RouterSpec("api.readiness") in MANIFEST_ROUTERS
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": READINESS} in manifest["routes"]
    assert {"method": "POST", "path": CHECK} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_probe_timeout_is_a_setting():
    from config import config

    assert config.READINESS_PROBE_TIMEOUT_SECONDS > 0
    assert "READINESS_PROBE_TIMEOUT_SECONDS" in json.loads(CONFIG_SURFACE.read_text(encoding="utf-8"))["settings"]
