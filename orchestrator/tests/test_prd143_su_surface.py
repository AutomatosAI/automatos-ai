"""PRD-143 S3 — su tools never offered to operators on ANY surface/selection path.

Every surface-building and ranking path must exclude ``super_admin_only``
actions for non-su principals, so Auto never even sees an obs tool schema,
at any autonomy level:

  * ``tool_router.get_tools_for_agent`` — dispatcher enum, first-class
    schemas, and the semantic-narrowing rank call all carry the caller's
    ``is_super_admin`` (default False, fail-closed);
  * the ``full_autonomy → is_admin=True`` elevation and the PRD-122
    workspace-owner fallback may flip ``is_admin``, NEVER the su surface;
  * ``ActionSemanticIndex.rank_actions`` never ranks an su action unless
    ``include_super_admin=True`` is passed explicitly;
  * ``GraphRouter.rank_chains`` excludes su actions from entry nodes AND
    from edge-expansion targets (edges learned from super-admin usage can
    point AT su actions — fail-closed at the source);
  * ``PlatformActionsSection`` (Auto context path) never renders an su
    action, full or filtered.

Synthetic ActionDefinitions are injected via fake registries so these tests
pin the SURFACE LOGIC, independent of the live catalogue (S4 reclassifies
it). Import idiom mirrors tests/test_prd143_su_executor_gate.py.
"""
from __future__ import annotations

import asyncio
import importlib.util as _ilu
import os
import sys as _sys
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

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


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which
# pulls modules.rag's ingestion chain (camelot at module top). Stub the missing
# *leaf* only when truly absent — never the modules.rag package.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

# CI collection-order guard: earlier-collected tests stub modules.*/consumers.*
# in sys.modules (bare ModuleType, no __spec__). On Linux collection order the
# stubs are still live HERE, so the real imports below resolve against them and
# die at collection ("unknown location" ImportError — see PR #434 CI). Purge
# origin-less entries so the real packages import fresh; conftest's autouse
# repair fixture re-binds everything else at test time.
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

from modules.tools import tool_router as tr
from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry

_SU = "platform_su_obs_probe"
_SU_PROMOTED = "platform_su_promoted_probe"
_ADMIN = "platform_admin_probe"
_OPERATOR = "platform_operator_probe"
_OPERATOR_SECOND = "platform_operator_second_probe"
_OPERATOR_PROMOTED = "platform_operator_promoted_probe"


def _action(
    name: str,
    *,
    category: str = "monitoring",
    description: str = "PRD-143 S3 surface probe",
    admin_only: bool = False,
    super_admin_only: bool = False,
    promoted: bool = False,
) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description,
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        admin_only=admin_only,
        super_admin_only=super_admin_only,
        promoted=promoted,
    )


def _surface_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True  # bypass the live platform_actions registrar
    reg.register(_action(_OPERATOR, category="agents", description="List the agents"))
    reg.register(_action(_OPERATOR_SECOND, category="agents", description="Create an agent"))
    reg.register(_action(_ADMIN, category="admin", admin_only=True))
    reg.register(_action(_OPERATOR_PROMOTED, category="agents", promoted=True))
    reg.register(_action(_SU, super_admin_only=True, description="Query the loki logs"))
    reg.register(_action(_SU_PROMOTED, super_admin_only=True, promoted=True))
    return reg


class _RecordingIndex:
    """Async fake for ActionSemanticIndex that records rank_actions kwargs."""

    def __init__(self, results: List[Tuple[str, float]], registry: Optional[ActionRegistry] = None):
        self._results = list(results)
        self.calls: List[Dict[str, Any]] = []
        if registry is not None:
            self._registry = registry

    async def rank_actions(
        self,
        query: str,
        top_k: int = 15,
        exclude_admin: bool = True,
        exclude_promoted: bool = True,
        **kw: Any,
    ) -> List[Tuple[str, float]]:
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "exclude_admin": exclude_admin,
            "exclude_promoted": exclude_promoted,
            **kw,
        })
        return list(self._results)[:top_k]


@contextmanager
def _tool_surface(registry: ActionRegistry):
    """Patch get_tools_for_agent's collaborators: empty ToolRegistry, no-op
    session factory, and the synthetic ActionRegistry."""
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


# ===========================================================================
# get_tools_for_agent — the OpenAI tool surface
# ===========================================================================


def test_su_tool_absent_from_openai_tools_by_default():
    """Default caller (no flags): no su action in the dispatcher enum, no
    su first-class schema."""
    with _tool_surface(_surface_registry()):
        tools = tr.get_tools_for_agent(agent_id=None, workspace_id=None)

    enum = _dispatcher_enum(tools)
    assert _SU not in enum
    assert _SU_PROMOTED not in enum
    assert _OPERATOR in enum

    names = _names(tools)
    assert _SU not in names
    assert _SU_PROMOTED not in names
    assert _OPERATOR_PROMOTED in names


def test_su_tool_absent_under_full_autonomy():
    """Surface paths take no autonomy input — full autonomy manifests as
    is_admin=True (the W3 elevation; S2 trap-1 analog for the surface).
    The admin elevation must work AND must not cross the su boundary."""
    with _tool_surface(_surface_registry()):
        tools = tr.get_tools_for_agent(agent_id=None, workspace_id=None, is_admin=True)

    enum = _dispatcher_enum(tools)
    assert _ADMIN in enum  # the elevation itself works
    assert _SU not in enum
    assert _SU_PROMOTED not in enum
    assert _SU_PROMOTED not in _names(tools)


def test_workspace_owner_fallback_does_not_include_su():
    """The PRD-122 workspace-owner fallback may flip is_admin — proven by
    the admin action appearing — but must NEVER flip the su surface."""
    session = MagicMock()
    session.query.return_value.filter.return_value.first.return_value = object()

    with _tool_surface(_surface_registry()):
        tools = tr.get_tools_for_agent(
            agent_id=None,
            workspace_id=uuid.uuid4(),
            db_session=session,
            is_admin=False,
        )

    enum = _dispatcher_enum(tools)
    assert _ADMIN in enum  # fallback DID flip is_admin (the path ran)
    assert _SU not in enum
    assert _SU_PROMOTED not in enum
    assert _SU_PROMOTED not in _names(tools)


def test_su_tool_present_for_super_admin_principal():
    """The ONLY inclusion path: an explicit is_super_admin=True principal.
    Also proves include_super_admin is threaded into the semantic-narrowing
    rank call (not just the schema builders).

    The routing flags are patched at the helper level — mutating the shared
    ``config.config`` instance here leaves instance-attr residue that
    shadows sibling suites' class-level flag writes."""
    recorder = _RecordingIndex(results=[(_SU, 0.95), (_OPERATOR, 0.6)])
    with _tool_surface(_surface_registry()), \
            patch.object(tr, "_semantic_routing_enabled", return_value=True), \
            patch.object(tr, "_semantic_routing_top_k", return_value=10), \
            patch(
                "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
                return_value=recorder,
            ):
        tools = tr.get_tools_for_agent(
            agent_id=None,
            workspace_id=None,
            is_admin=True,
            is_super_admin=True,
            query="show me the loki logs",
        )

    enum = _dispatcher_enum(tools)
    assert _SU in enum
    assert _SU_PROMOTED in _names(tools)
    assert recorder.calls, "semantic narrowing was not invoked"
    assert recorder.calls[-1]["include_super_admin"] is True


# ===========================================================================
# ActionSemanticIndex — ranking path
# ===========================================================================


class _FakeEmbeddingManager:
    """Deterministic embeddings: loki/logs → su axis, agent → operator axis."""

    DIM = 4

    def __init__(self) -> None:
        self.provider = MagicMock()
        self.provider.config = MagicMock()
        self.provider.config.model = "fake-model"

    def get_provider_info(self) -> dict:
        return {"provider": "fake", "model": "fake-model", "dimension": self.DIM, "status": "active"}

    def get_dimension(self) -> int:
        return self.DIM

    @staticmethod
    def _vec(text: str) -> List[float]:
        t = text.lower()
        if "loki" in t or "logs" in t:
            return [0.0, 0.0, 0.0, 1.0]
        if "agent" in t:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.25, 0.25, 0.25, 0.25]

    async def generate_embedding(self, text: str) -> List[float]:
        return self._vec(text)

    async def generate_embeddings_batch(self, texts: List[str], max_concurrent: int = 5) -> List[List[float]]:
        return [self._vec(t) for t in texts]


class _FakeCache:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, List[float]]] = {}

    def get_embeddings_batch(self, texts: List[str], model: str = "default") -> Dict[str, Optional[List[float]]]:
        bucket = self.store.get(model, {})
        return {t: bucket.get(t) for t in texts}

    def set_embeddings_batch(self, embeddings: Dict[str, List[float]], model: str = "default") -> None:
        self.store.setdefault(model, {}).update(embeddings)


def _make_index(registry: ActionRegistry):
    """Real ActionSemanticIndex with injected fakes (skip heavy __init__)."""
    from modules.tools.discovery.action_semantic_index import ActionSemanticIndex

    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = _FakeEmbeddingManager()
    idx._cache = _FakeCache()
    idx._registry = registry
    idx._action_embeddings = {}
    idx._indexed = False
    idx._lock = None
    return idx


def test_semantic_rank_never_returns_su_for_operator():
    """An su action that is the STRONGEST semantic match still never ranks
    for an operator; include_super_admin=True is the only way in."""
    index = _make_index(_surface_registry())

    ranked = asyncio.run(index.rank_actions("show me the loki logs", top_k=10))
    names = [n for n, _ in ranked]
    assert names, "ranking returned nothing — fixture broken"
    assert _SU not in names
    assert _SU_PROMOTED not in names

    ranked_su = asyncio.run(
        index.rank_actions("show me the loki logs", top_k=10, include_super_admin=True)
    )
    su_names = [n for n, _ in ranked_su]
    assert _SU in su_names
    assert su_names[0] == _SU  # strongest match ranks first once included


# ===========================================================================
# GraphRouter — chain ranking path
# ===========================================================================


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
        name = getattr(model, "__name__", "")
        if "Edge" in name:
            return _FakeQuery(list(self._edges))
        return _FakeQuery([])


class _EdgeRow:
    def __init__(self, from_action: str, to_action: str):
        self.from_action = from_action
        self.to_action = to_action
        self.confidence = 0.9
        self.weight = 5.0
        self.agent_id = None


def test_graph_router_never_returns_su_for_operator():
    """Graph edges learned from super-admin usage can point AT su actions —
    the operator path must drop those chains while keeping operator-only
    expansions, and must thread include_super_admin into entry ranking."""
    from modules.tools.discovery.graph_router import GraphRouter

    registry = _surface_registry()
    fake_index = _RecordingIndex(results=[(_OPERATOR, 0.9)], registry=registry)
    router = GraphRouter.__new__(GraphRouter)
    router._semantic_index = fake_index

    edges = [
        _EdgeRow(_OPERATOR, _SU),               # su expansion target → dropped
        _EdgeRow(_OPERATOR, _OPERATOR_SECOND),  # operator expansion → kept
    ]

    @contextmanager
    def _fake_db_ctx():
        yield _FakeDBSession(edges)

    with patch.object(GraphRouter, "_get_cache", return_value=None), \
            patch("core.database.database.get_db_session", _fake_db_ctx):
        chains = asyncio.run(router.rank_chains("probe ops", agent_id=None, top_k=10))

    assert chains, "rank_chains returned nothing — fixture broken"
    all_chain_actions = {name for _, _, chain in chains for name in chain}
    assert _SU not in all_chain_actions
    assert _SU_PROMOTED not in all_chain_actions
    assert _OPERATOR_SECOND in all_chain_actions  # operator expansion survived

    assert fake_index.calls, "entry ranking was not invoked"
    assert fake_index.calls[-1]["include_super_admin"] is False


# ===========================================================================
# PlatformActionsSection — Auto context path
# ===========================================================================


def test_platform_actions_section_never_includes_su():
    """The Auto context section never renders an su action: not in the full
    catalog, not via the filtered path even when the ranker names one."""
    from modules.context.sections.base import SectionContext
    from modules.context.sections.platform_actions import PlatformActionsSection

    registry = _surface_registry()
    section = PlatformActionsSection()

    # Full-catalog path (no query → _build()).
    with patch(
        "modules.tools.discovery.action_registry.get_action_registry",
        return_value=registry,
    ):
        ctx = SectionContext(agent=None, workspace_id="ws-prd143")
        content = asyncio.run(section.render(ctx))
    assert _OPERATOR in content, "catalog empty — fixture broken"
    assert _SU not in content
    assert _SU_PROMOTED not in content

    # Filtered path: ranker names an su action; the summary still drops it,
    # and the section's rank call pins include_super_admin=False explicitly.
    recorder = _RecordingIndex(results=[(_SU, 0.95), (_OPERATOR, 0.7)])
    with patch(
        "modules.tools.discovery.action_registry.get_action_registry",
        return_value=registry,
    ), patch(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
        return_value=recorder,
    ):
        filtered = asyncio.run(section._build_filtered("show me the loki logs"))

    assert filtered is not None
    assert _OPERATOR in filtered
    assert _SU not in filtered
    assert recorder.calls[-1]["include_super_admin"] is False
