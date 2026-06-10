"""PRD-143 S16 — negative boundary sweep: NO operator path crosses the obs lock.

One cross-cutting net over the whole su perimeter, exhaustive and
source-of-truth-driven so a FUTURE su action or locked router is swept
automatically:

  * the su ACTION set is read from the LIVE registry
    (``register_all_actions()`` → ``super_admin_only``), never hardcoded;
  * the locked ROUTER set is parsed from the signed-off manifest
    (docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md), whose tool table the parity
    suite already pins to the registry;
  * every su action is proven absent from the OpenAI tool surface, the
    dispatcher enum, semantic ranking (even as the STRONGEST match — its own
    description is the query), and graph ranking (even as an edge target);
  * the executor refuses every su action for an operator, under full
    autonomy, and for an API-key admin (hybrid.py system_role='admin');
  * EVERY route on EVERY locked router 403s "Super admin only" for member,
    workspace admin/owner, and API-key principals — full route enumeration,
    not a representative endpoint;
  * the autonomy dial (``platform_set_autonomy_level``) is absent from
    Auto's surface at any elevation while the read stays operator;
  * the Wave 4 audit marker is a queryable filter — mixed telemetry rows
    split exactly on ``router_decision->>'autonomous'``.

Idioms: S2's fake-POSTGRES preamble (closed port 59432), the manifest suite's
real-catalogue registry, S3's surface/graph fakes, S13's deterministic
hash-bag embeddings, the S6/S7 router TestClient shape. No DB, no network.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util as _ilu
import os
import re
import sys
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — the port points at nothing so the
# modules.tools import chain's fail-soft DB connect refuses instantly instead
# of hanging on a wedged local proxy. CI exports real POSTGRES_* so these
# setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    """Stub apscheduler ONLY when truly absent (camelot-shim philosophy).

    This file collects alphabetically FIRST in the prd143 family — an
    unconditional fake would shadow the real package for every later test
    module (the concierge journey imports apscheduler.triggers.cron via
    services.heartbeat_service)."""
    if "apscheduler" in sys.modules:
        return
    try:
        if _ilu.find_spec("apscheduler") is not None:
            return
    except ValueError:  # pragma: no cover - env-dependent
        pass
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


_install_fake_apscheduler()


# Lean-venv shim: modules/tools/__init__ pulls modules.rag's ingestion chain
# (camelot at module top). Stub the missing leaf only when truly absent.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute, APIRouter  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import modules.tools.discovery.platform_executor as pe  # noqa: E402,F401
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.super_admin import require_super_admin  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.tools import tool_router as tr  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402
from modules.tools.execution.telemetry import write_telemetry  # noqa: E402

# ===========================================================================
# Sources of truth (built at collection time — parametrization needs them)
# ===========================================================================


def _build_real_registry() -> ActionRegistry:
    """The REAL catalogue, registered directly (not the singleton) so
    _ensure_initialized cannot trigger a second full init."""
    reg = ActionRegistry()
    register_all_actions(reg)
    reg._initialized = True
    return reg


_REGISTRY = _build_real_registry()

# THE su set — from the registry, never hardcoded (a future su action is
# automatically swept).
_SU_ACTIONS: List[str] = sorted(
    a.name for a in _REGISTRY.get_all() if a.super_admin_only
)

# Operator controls (non-vacuity anchors; tier pinned in the sanity test).
# Promoted operators are first-class schemas and NEVER enum members, so the
# enum control must be a non-promoted operator — derived from the registry
# so future promotions cannot silently break the sweep.
_OPERATOR_CONTROL = "platform_list_agents"
_OPERATOR_EXPANSION = "platform_create_agent"
_DIAL_READ = "platform_get_autonomy_level"
_DIAL_SET = "platform_set_autonomy_level"
_ENUM_CONTROL = sorted(
    a.name
    for a in _REGISTRY.get_all()
    if not a.super_admin_only and not a.admin_only and not a.promoted
)[0]

# The locked-router set — parsed from the manifest (the router half of the
# obs perimeter; the tool half is parity-tested against the registry).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO_ROOT / "docs" / "PRDS" / "PRD-143-OBS-TIER-MANIFEST.md"
_LOCKED_API_MODULES: List[str] = sorted(set(
    re.findall(r"`orchestrator/api/(\w+)\.py`", _MANIFEST_PATH.read_text(encoding="utf-8"))
)) if _MANIFEST_PATH.exists() else []

_WS = uuid.uuid4()

MEMBER = UserContext(id="u-member", role="member", system_role="user")
WS_ADMIN = UserContext(id="u-ws-admin", role="admin", system_role="user")
WS_OWNER = UserContext(id="u-ws-owner", role="owner", system_role="user")
# hybrid.py:783 — API-key principals carry system_role='admin'.
API_KEY_ADMIN = UserContext(id="api_key", email=None, role="admin", system_role="admin")

_MEMBER_CTX = {"workspace_role": "member", "user_id": 7}
_API_KEY_CTX = {"system_role": "admin", "workspace_role": "owner", "user_id": 7}


def test_sweep_sources_are_nonvacuous():
    """The sweep's two sources of truth are live and the controls hold the
    tiers the negatives lean on — an empty parametrization can never pass
    silently."""
    assert len(_SU_ACTIONS) >= 7, f"su tier collapsed: {_SU_ACTIONS}"
    assert _DIAL_SET in _SU_ACTIONS
    for name in (_OPERATOR_CONTROL, _OPERATOR_EXPANSION, _DIAL_READ):
        action = _REGISTRY.get(name)
        assert action is not None, f"operator control missing from catalogue: {name}"
        assert action.super_admin_only is False
    assert len(_LOCKED_API_MODULES) >= 13, (
        f"manifest router table parsed to {_LOCKED_API_MODULES} — manifest moved?"
    )


# ===========================================================================
# Surface: OpenAI tools + dispatcher enum (tool_router)
# ===========================================================================


@contextmanager
def _tool_surface(registry: ActionRegistry):
    """Patch get_tools_for_agent's collaborators: empty ToolRegistry, no-op
    session factory, and the real-catalogue ActionRegistry (S3 idiom)."""
    fake_tool_registry = MagicMock()
    fake_tool_registry.get_all_tools.return_value = []
    with patch.object(tr, "registry_get_tool_registry", return_value=fake_tool_registry), \
            patch.object(tr, "SessionLocal", return_value=MagicMock()), \
            patch("modules.tools.discovery.get_action_registry", return_value=registry):
        yield


def _names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t["function"]["name"] for t in tools]


def _dispatcher_enum(tools: List[Dict[str, Any]]) -> List[str]:
    for t in tools:
        if t["function"]["name"] == "platform_execute":
            return t["function"]["parameters"]["properties"]["action"].get("enum") or []
    raise AssertionError(f"platform_execute not in tools list: {_names(tools)}")


def _surface_variants():
    """Every elevation an operator principal can reach: plain, the
    full-autonomy is_admin elevation (W3), and the PRD-122 workspace-owner
    fallback — none may flip the su surface."""
    owner_session = MagicMock()
    owner_session.query.return_value.filter.return_value.first.return_value = object()
    return [
        ("default", dict(agent_id=None, workspace_id=None)),
        ("admin_elevated", dict(agent_id=None, workspace_id=None, is_admin=True)),
        (
            "owner_fallback",
            dict(agent_id=None, workspace_id=uuid.uuid4(), db_session=owner_session, is_admin=False),
        ),
    ]


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_absent_from_openai_tools_surface(su_action):
    """No su action appears as a first-class schema for any operator
    elevation — even if a future change promotes one."""
    for variant, kwargs in _surface_variants():
        with _tool_surface(_REGISTRY):
            tools = tr.get_tools_for_agent(**kwargs)
        names = _names(tools)
        assert "platform_execute" in names, f"surface empty under {variant} — fixture broken"
        assert su_action not in names, f"su schema leaked on the {variant} surface"


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_absent_from_dispatcher_enum(su_action):
    """The platform_execute action enum — the list the LLM actually picks
    from — never names an su action for any operator elevation."""
    for variant, kwargs in _surface_variants():
        with _tool_surface(_REGISTRY):
            tools = tr.get_tools_for_agent(**kwargs)
        enum = _dispatcher_enum(tools)
        assert _ENUM_CONTROL in enum, f"enum empty under {variant} — fixture broken"
        assert su_action not in enum, f"su action leaked into the {variant} enum"


# ===========================================================================
# Semantic ranking (real ActionSemanticIndex, deterministic embeddings)
# ===========================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _HashBagEmbeddingManager:
    """Deterministic hash-bag-of-words embeddings (S13 idiom): identical text
    → identical vector → cosine 1.0, so an su action queried by its OWN
    description is the strongest possible match."""

    DIM = 1024

    def __init__(self) -> None:
        self.provider = MagicMock()
        self.provider.config = MagicMock()
        self.provider.config.model = "hash-bag"

    def get_provider_info(self) -> dict:
        return {"provider": "fake", "model": "hash-bag", "dimension": self.DIM, "status": "active"}

    def get_dimension(self) -> int:
        return self.DIM

    @classmethod
    def _vec(cls, text: str) -> List[float]:
        vec = [0.0] * cls.DIM
        for tok in _TOKEN_RE.findall(text.lower()):
            bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % cls.DIM
            vec[bucket] += 1.0
        return vec

    async def generate_embedding(self, text: str) -> List[float]:
        return self._vec(text)

    async def generate_embeddings_batch(
        self, texts: List[str], max_concurrent: int = 5
    ) -> List[List[float]]:
        return [self._vec(t) for t in texts]


class _FakeCache:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, List[float]]] = {}

    def get_embeddings_batch(
        self, texts: List[str], model: str = "default"
    ) -> Dict[str, Optional[List[float]]]:
        bucket = self.store.get(model, {})
        return {t: bucket.get(t) for t in texts}

    def set_embeddings_batch(self, embeddings: Dict[str, List[float]], model: str = "default") -> None:
        self.store.setdefault(model, {}).update(embeddings)


@pytest.fixture(scope="module")
def semantic_index():
    """Real ActionSemanticIndex over the real catalogue with injected fakes."""
    from modules.tools.discovery.action_semantic_index import ActionSemanticIndex

    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = _HashBagEmbeddingManager()
    idx._cache = _FakeCache()
    idx._registry = _REGISTRY
    idx._action_embeddings = {}
    idx._indexed = False
    idx._lock = None
    return idx


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_absent_from_semantic_ranking(su_action, semantic_index):
    """Ranked with the su action's OWN description (cosine 1.0 — the
    strongest match in the catalogue) over an UNBOUNDED top_k with the
    promoted filter OFF (two su actions are promoted; the sweep must cover
    them on this path too): still never ranked for an operator.
    Counter-proof: include_super_admin=True ranks it, so the negative is
    not vacuous."""
    query = _REGISTRY.get(su_action).description

    ranked = asyncio.run(
        semantic_index.rank_actions(query, top_k=500, exclude_promoted=False)
    )
    names = [n for n, _ in ranked]
    assert names, "ranking returned nothing — fixture broken"
    assert su_action not in names

    ranked_su = asyncio.run(
        semantic_index.rank_actions(
            query, top_k=500, exclude_promoted=False, include_super_admin=True
        )
    )
    assert su_action in [n for n, _ in ranked_su], (
        "counter-proof failed: the query does not rank its own action even "
        "when su is included — the negative above proves nothing"
    )


# ===========================================================================
# Graph ranking (real GraphRouter; su as edge-expansion target)
# ===========================================================================


class _RecordingIndex:
    """Async fake for ActionSemanticIndex that records rank_actions kwargs."""

    def __init__(self, results: List[Tuple[str, float]], registry: ActionRegistry):
        self._results = list(results)
        self._registry = registry
        self.calls: List[Dict[str, Any]] = []

    async def rank_actions(self, query: str, top_k: int = 15, **kw: Any) -> List[Tuple[str, float]]:
        self.calls.append({"query": query, "top_k": top_k, **kw})
        return list(self._results)[:top_k]


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **kw):
        return self

    def order_by(self, *a, **kw):
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def scalar(self):
        return self._rows

    def all(self):
        return self._rows


class _FakeDBSession:
    def __init__(self, edges):
        self._edges = edges

    def query(self, model, *args):
        if "Edge" in getattr(model, "__name__", ""):
            return _FakeQuery(list(self._edges))
        return _FakeQuery([])


class _EdgeRow:
    def __init__(self, from_action: str, to_action: str):
        self.from_action = from_action
        self.to_action = to_action
        self.confidence = 0.9
        self.weight = 5.0
        self.agent_id = None


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_absent_from_graph_ranking(su_action):
    """An edge learned from super-admin usage points AT the su action — the
    operator chain ranking drops it while keeping the operator expansion, and
    threads include_super_admin=False into entry ranking."""
    from modules.tools.discovery.graph_router import GraphRouter

    fake_index = _RecordingIndex(results=[(_OPERATOR_CONTROL, 0.9)], registry=_REGISTRY)
    router = GraphRouter.__new__(GraphRouter)
    router._semantic_index = fake_index

    edges = [
        _EdgeRow(_OPERATOR_CONTROL, su_action),            # su target → dropped
        _EdgeRow(_OPERATOR_CONTROL, _OPERATOR_EXPANSION),  # operator → kept
    ]

    @contextmanager
    def _fake_db_ctx():
        yield _FakeDBSession(edges)

    with patch.object(GraphRouter, "_get_cache", return_value=None), \
            patch("core.database.database.get_db_session", _fake_db_ctx):
        chains = asyncio.run(router.rank_chains("set things up", agent_id=None, top_k=10))

    assert chains, "rank_chains returned nothing — fixture broken"
    chain_actions = {name for _, _, chain in chains for name in chain}
    assert su_action not in chain_actions
    assert _OPERATOR_EXPANSION in chain_actions  # the drop is surgical
    assert fake_index.calls and fake_index.calls[-1]["include_super_admin"] is False


# ===========================================================================
# Executor: refusal pre-execution on every channel
# ===========================================================================


def _executor() -> PlatformActionExecutor:
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


def _stub_handler(ex: PlatformActionExecutor, action: str) -> AsyncMock:
    leak_detector = AsyncMock(return_value={"success": True, "_sentinel": "handler-ran"})
    ex._handlers[action] = leak_detector
    return leak_detector


def _assert_su_refused(result: dict):
    assert result["success"] is False
    assert result.get("permission_denied") is True
    assert result.get("requires_confirmation") is None
    assert "super admin" in result.get("error", "").lower()


def _execute(su_action: str, caller_context, *, full_autonomy: bool) -> dict:
    ex = _executor()
    handler = _stub_handler(ex, su_action)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=full_autonomy), \
            patch("modules.tools.discovery.get_action_registry", return_value=_REGISTRY):
        result = asyncio.run(ex.execute(su_action, {}, caller_context))
    handler.assert_not_awaited()
    return result


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_executor_refuses_direct_call(su_action):
    """A plain operator caller_context is refused pre-execution."""
    _assert_su_refused(_execute(su_action, dict(_MEMBER_CTX), full_autonomy=False))


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_executor_refuses_under_full_autonomy(su_action):
    """Trap 1: the full_autonomy → is_admin elevation never crosses the gate."""
    _assert_su_refused(_execute(su_action, dict(_MEMBER_CTX), full_autonomy=True))


@pytest.mark.parametrize("su_action", _SU_ACTIONS)
def test_executor_refuses_api_key_admin(su_action):
    """Trap 3: API keys carry system_role='admin' (hybrid.py:783) — refused
    even paired with a workspace-owner role."""
    _assert_su_refused(_execute(su_action, dict(_API_KEY_CTX), full_autonomy=False))


# ===========================================================================
# HTTP perimeter: every route on every locked router 403s
# ===========================================================================


def _fill_path_params(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "1", path)


def _module_routers(module_name: str) -> List[Tuple[str, APIRouter]]:
    module = importlib.import_module(f"api.{module_name}")
    routers = [
        (attr, obj) for attr, obj in vars(module).items() if isinstance(obj, APIRouter)
    ]
    assert routers, f"no APIRouter found in api.{module_name} — manifest stale?"
    return routers


def _client(routers: List[Tuple[str, APIRouter]], user: UserContext) -> TestClient:
    app = FastAPI()
    for _, router_obj in routers:
        app.include_router(router_obj)

    auth_type = "api_key" if user is API_KEY_ADMIN else "clerk"

    def _override_ctx():
        return RequestContext(workspace_id=_WS, user=user, auth_type=auth_type)

    def _override_db():
        yield MagicMock()

    app.dependency_overrides[get_request_context_hybrid] = _override_ctx
    app.dependency_overrides[get_db] = _override_db
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("module_name", _LOCKED_API_MODULES)
def test_router_403_for_operator_and_api_key(module_name):
    """Structural + behavioural lock on the whole module: every APIRouter in
    a locked module carries require_super_admin router-wide, and EVERY route
    (all methods, path params filled) 403s 'Super admin only' for member,
    workspace admin/owner, and API-key principals."""
    routers = _module_routers(module_name)

    # Structural: the router-wide dependency IS the canonical S5 symbol — a
    # future endpoint added to these routers is locked at birth.
    for attr, router_obj in routers:
        dep_fns = [getattr(d, "dependency", None) for d in (router_obj.dependencies or [])]
        assert require_super_admin in dep_fns, (
            f"api.{module_name}.{attr} lost the router-wide require_super_admin"
        )

    # Behavioural: full route enumeration × every non-su principal.
    requests = []
    for _, router_obj in routers:
        for route in router_obj.routes:
            if not isinstance(route, APIRoute):
                continue
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                requests.append((method, _fill_path_params(route.path)))
    assert requests, f"api.{module_name} enumerated zero routes — sweep is vacuous"

    for user in (MEMBER, WS_ADMIN, WS_OWNER, API_KEY_ADMIN):
        client = _client(routers, user)
        for method, path in requests:
            resp = client.request(method, path)
            assert resp.status_code == 403, (
                f"{user.id} reached {method} {path} on api.{module_name}: "
                f"{resp.status_code} {resp.text[:200]}"
            )
            assert resp.json()["detail"] == "Super admin only", (
                f"{method} {path} 403'd outside the su gate: {resp.text[:200]}"
            )


# ===========================================================================
# The dial is human-held
# ===========================================================================


def test_set_autonomy_level_not_in_auto_surface_at_any_autonomy():
    """platform_set_autonomy_level never reaches Auto's surface — plain,
    admin-elevated (the full-autonomy analog) or owner-fallback — while the
    read (platform_get_autonomy_level) stays operator-reachable, proving the
    asymmetry: Auto may read its dial, never set it. Reachable = dispatcher
    enum ∪ first-class schemas (both dial actions are promoted, so the
    first-class list is where each would surface)."""
    for variant, kwargs in _surface_variants():
        with _tool_surface(_REGISTRY):
            tools = tr.get_tools_for_agent(**kwargs)
        reachable = set(_dispatcher_enum(tools)) | set(_names(tools))
        assert _DIAL_SET not in reachable, f"the dial leaked into the {variant} surface"
        assert _DIAL_READ in reachable, f"dial READ missing under {variant} — operator tier broken"


# ===========================================================================
# Audit: the autonomous marker is a queryable filter
# ===========================================================================


def test_audit_distinguishes_autonomous_actions():
    """Mixed telemetry rows split EXACTLY on router_decision->>'autonomous'
    (the Wave 4 / S8 marker) — the filter an audit query uses, proven over
    rows written by the real persistence hook."""
    ws_id = uuid.uuid4()
    rows: List[Any] = []
    db = MagicMock()
    db.add.side_effect = rows.append

    async def _write(tool: str, result: Dict[str, Any]):
        await write_telemetry(
            db,
            tool_name=tool,
            parameters={},
            agent_id=9,
            workspace_id=ws_id,
            result=result,
            execution_time_ms=1,
            caller_context={"user_id": 7},
        )

    # Two confirmation-skipped (autonomous) invocations, two plain ones.
    asyncio.run(_write("platform_delete_agent", {"success": True, "autonomous": True}))
    asyncio.run(_write("platform_set_member_role", {"success": True, "autonomous": True}))
    asyncio.run(_write("platform_list_agents", {"success": True}))
    asyncio.run(_write("platform_get_autonomy_level", {"success": True}))

    assert len(rows) == 4

    autonomous = {r.action_name for r in rows if (r.router_decision or {}).get("autonomous") is True}
    assert autonomous == {"platform_delete_agent", "platform_set_member_role"}

    plain = {r.action_name for r in rows if not (r.router_decision or {}).get("autonomous")}
    assert plain == {"platform_list_agents", "platform_get_autonomy_level"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
