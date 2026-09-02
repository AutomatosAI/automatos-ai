"""PRD-143 S13 — selection robustness at ~200 tools (tests-only).

A synthetic 200-action registry (operator surface across realistic
categories + the 7-action su tier mirroring the real S4 manifest) is ranked
by the REAL ActionSemanticIndex over deterministic hash-bag embeddings —
no network, no model, stable across runs. At the scale the catalogue is
heading for (PRD-143 US-003/US-006, FR-5) this proves:

  * every representative setup/admin intent lands its expected tool inside
    the configured top-K (SEMANTIC_TOOL_ROUTING_TOP_K read from config,
    never hardcoded);
  * su tools never rank for an operator principal — even on obs-flavoured
    intents where they are the strongest semantic match (counter-proven by
    include_super_admin=True ranking them);
  * the selected set is bounded by the configured top-K;
  * the dispatcher enum the LLM actually sees equals the ranked set.

Import header + fake-registry/embedding idioms mirror
tests/test_prd143_su_surface.py (S3).
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util as _ilu
import os
import re
import sys as _sys
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

from config import config as _config
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

TOP_K = int(getattr(_config, "SEMANTIC_TOOL_ROUTING_TOP_K", 15))
FIXTURE_SIZE = 200

# ===========================================================================
# Synthetic 200-action fixture
# ===========================================================================

# Operator anchors mirror REAL catalogue names (S10/S11 + pre-existing) so
# the representative intents exercise the same names the live surface has.
# (name, category, description, tags, examples, permission_level)
_ANCHORS: List[Tuple[str, str, str, List[str], List[str], str]] = [
    (
        "platform_create_agent", "agents",
        "Create a new agent in the workspace with a name, persona and skills",
        ["agents", "create"],
        ["create an agent called Sales Bot", "add a new agent"], "write",
    ),
    (
        "platform_delete_agent", "agents",
        "Delete an agent from the workspace permanently",
        ["agents", "delete"],
        ["delete the sales agent"], "destructive",
    ),
    (
        "platform_connect_channel", "channels",
        "Connect a messaging channel driver such as Slack or email for the workspace",
        ["channels", "connect", "slack"],
        ["connect the slack channel"], "write",
    ),
    (
        "platform_upload_document", "documents",
        "Upload a document into the workspace knowledge base",
        ["documents", "knowledge", "upload"],
        ["upload knowledge", "upload this document to the knowledge base"], "write",
    ),
    (
        "platform_invite_member", "team",
        "Invite a member to the workspace by email",
        ["team", "members", "invite"],
        ["invite a member to the workspace"], "write",
    ),
    (
        "platform_create_mission", "missions",
        "Create and launch a mission for the agent team",
        ["missions", "launch"],
        ["launch a mission"], "write",
    ),
    (
        "platform_execute_playbook", "playbooks",
        "Execute a playbook now and return its execution id",
        ["playbooks", "execute", "run"],
        ["run the onboarding playbook"], "write",
    ),
    (
        "platform_update_widget_config", "widgets",
        "Update the chat widget configuration for the workspace",
        ["widgets", "config"],
        ["update the widget config"], "write",
    ),
    (
        "platform_revoke_api_key", "api_keys",
        "Revoke an SDK api key for the workspace",
        ["api_keys", "revoke"],
        ["revoke an api key"], "destructive",
    ),
    (
        "platform_update_workspace_settings", "settings",
        "Update workspace settings such as defaults and overrides",
        ["settings", "workspace"],
        ["change the workspace settings"], "write",
    ),
    (
        "platform_get_autonomy_level", "governance",
        "Read the current Auto autonomy level",
        ["governance", "autonomy"],
        ["what is the autonomy level"], "read",
    ),
]

# Promoted operator actions (first-class schemas; excluded from the
# dispatcher enum and from dispatcher-path ranking by design).
_PROMOTED: List[Tuple[str, str, str, List[str], List[str], str]] = [
    (
        "platform_board_summary", "tasks",
        "Summarize the kanban board state for the workspace",
        ["tasks", "board"],
        ["summarize the board"], "read",
    ),
    (
        "platform_workspace_stats", "workspace",
        "Get headline workspace statistics",
        ["workspace", "stats"],
        ["workspace stats"], "read",
    ),
]

# The su tier mirrors the REAL 7-action manifest (S4,
# docs/PRDS/PRD-143-OBS-TIER-MANIFEST.md) name-for-name.
_SU_TIER: List[Tuple[str, str, str, List[str], List[str], str]] = [
    (
        "platform_query_loki_logs", "monitoring",
        "Query Loki for recent log lines across services",
        ["monitoring", "logs", "loki"],
        ["query loki for errors", "show me the logs"], "read",
    ),
    (
        "platform_query_prometheus", "monitoring",
        "Run a PromQL query against Prometheus metrics",
        ["monitoring", "metrics", "prometheus"],
        ["query prometheus for error rates"], "read",
    ),
    (
        "platform_get_alerts", "monitoring",
        "Get firing alerts from the monitoring stack",
        ["monitoring", "alerts"],
        ["what alerts are firing"], "read",
    ),
    (
        "platform_get_logs", "monitoring",
        "Fetch recent platform logs for debugging",
        ["monitoring", "logs"],
        ["show me the logs"], "read",
    ),
    (
        "platform_list_services", "monitoring",
        "List running platform services and their status",
        ["monitoring", "services"],
        ["list the running platform services"], "read",
    ),
    (
        "platform_get_system_health", "monitoring",
        "Check overall system health and uptime",
        ["monitoring", "health"],
        ["check the system health"], "read",
    ),
    (
        "platform_set_autonomy_level", "governance",
        "Set the Auto autonomy level dial",
        ["governance", "autonomy"],
        ["set autonomy to full"], "write",
    ),
]

# Filler vocabulary deliberately avoids the anchor/su distinctive tokens so
# the fixture is realistic (lots of plausible neighbours) without being
# rigged: fillers still share verbs and the workspace/platform boilerplate.
_FILLER_VERBS = ["create", "list", "get", "update", "delete"]
_FILLER_NOUNS = [
    "template", "persona", "snippet", "schedule", "webhook", "folder",
    "tag", "deliverable", "board", "note", "draft", "theme",
    "locale", "quota", "backup", "invoice", "ticket", "contact",
    "lead", "campaign", "segment", "audience", "survey", "form",
    "calendar", "event", "reminder", "bookmark", "label", "comment",
    "attachment", "transcript", "summary", "glossary", "faq", "banner",
]
_FILLER_CATEGORIES = [
    "tasks", "blog", "reports", "memory", "graph",
    "scheduling", "marketplace", "analytics", "field",
]
_PERMISSION_BY_VERB = {
    "create": "write", "list": "read", "get": "read",
    "update": "write", "delete": "destructive",
}


def _action(
    name: str,
    category: str,
    description: str,
    tags: List[str],
    examples: List[str],
    permission_level: str,
    *,
    super_admin_only: bool = False,
    promoted: bool = False,
) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description=description,
        category=category,
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level=permission_level,
        super_admin_only=super_admin_only,
        promoted=promoted,
        tags=list(tags),
        examples=list(examples),
    )


def _fixture_actions() -> List[ActionDefinition]:
    actions = [_action(*spec) for spec in _ANCHORS]
    actions += [_action(*spec, promoted=True) for spec in _PROMOTED]
    actions += [_action(*spec, super_admin_only=True) for spec in _SU_TIER]

    filler_target = FIXTURE_SIZE - len(actions)
    fillers: List[ActionDefinition] = []
    for i, noun in enumerate(_FILLER_NOUNS):
        category = _FILLER_CATEGORIES[i % len(_FILLER_CATEGORIES)]
        for verb in _FILLER_VERBS:
            fillers.append(
                _action(
                    f"platform_{verb}_{noun}",
                    category,
                    f"{verb.title()} the {noun} records for this workspace",
                    [category, verb],
                    [f"{verb} the {noun}"],
                    _PERMISSION_BY_VERB[verb],
                )
            )
    if len(fillers) < filler_target:
        raise AssertionError(
            f"fixture vocabulary too small: {len(fillers)} fillers < {filler_target}"
        )
    return actions + fillers[:filler_target]


def _build_registry() -> ActionRegistry:
    reg = ActionRegistry()
    reg._initialized = True  # bypass the live platform_actions registrar
    for adef in _fixture_actions():
        reg.register(adef)
    total = len(reg.get_all())
    if total != FIXTURE_SIZE:
        raise AssertionError(f"fixture must hold exactly {FIXTURE_SIZE} actions, got {total}")
    su = [a.name for a in reg.get_all() if a.super_admin_only]
    if len(su) != len(_SU_TIER):
        raise AssertionError(f"fixture su tier drifted: {su}")
    return reg


# ===========================================================================
# Deterministic embeddings: hash-bag-of-words (md5-bucketed token counts).
# Cosine similarity then reflects token overlap — realistic enough to rank,
# fully deterministic across runs/platforms (never the salted builtin hash).
# ===========================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _HashBagEmbeddingManager:
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


def _make_index(registry: ActionRegistry):
    """Real ActionSemanticIndex with injected fakes (skip heavy __init__)."""
    from modules.tools.discovery.action_semantic_index import ActionSemanticIndex

    idx = ActionSemanticIndex.__new__(ActionSemanticIndex)
    idx._embedding_manager = _HashBagEmbeddingManager()
    idx._cache = _FakeCache()
    idx._registry = registry
    idx._action_embeddings = {}
    idx._indexed = False
    idx._lock = None
    return idx


@pytest.fixture(scope="module")
def scale_registry() -> ActionRegistry:
    return _build_registry()


@pytest.fixture(scope="module")
def scale_index(scale_registry):
    idx = _make_index(scale_registry)
    asyncio.run(idx.ensure_indexed())
    return idx


def _rank(index, query: str, **kw: Any) -> List[Tuple[str, float]]:
    """Rank with the dispatcher path's exact defaults (top_k from config)."""
    return asyncio.run(index.rank_actions(query, top_k=TOP_K, **kw))


def _su_names(registry: ActionRegistry) -> frozenset:
    return frozenset(a.name for a in registry.get_all() if a.super_admin_only)


# ===========================================================================
# Intents
# ===========================================================================

REPRESENTATIVE_INTENTS: List[Tuple[str, str]] = [
    ("create an agent", "platform_create_agent"),
    ("delete the sales agent", "platform_delete_agent"),
    ("connect the slack channel", "platform_connect_channel"),
    ("upload knowledge to the knowledge base", "platform_upload_document"),
    ("invite a member to the workspace", "platform_invite_member"),
    ("launch a mission", "platform_create_mission"),
    ("run the onboarding playbook", "platform_execute_playbook"),
    ("update the widget config", "platform_update_widget_config"),
    ("revoke an api key", "platform_revoke_api_key"),
    ("change the workspace settings", "platform_update_workspace_settings"),
]

# Obs-flavoured intents where an su tool IS the strongest semantic match —
# paired with the su action that must rank once include_super_admin=True.
OBS_INTENTS: List[Tuple[str, str]] = [
    ("show me the logs", "platform_get_logs"),
    ("query prometheus for error rates", "platform_query_prometheus"),
    ("query loki for errors", "platform_query_loki_logs"),
    ("what alerts are firing right now", "platform_get_alerts"),
    ("list the running platform services", "platform_list_services"),
    ("check the system health", "platform_get_system_health"),
    ("set the autonomy level to full", "platform_set_autonomy_level"),
]


# ===========================================================================
# Tool-surface plumbing (mirrors test_prd143_su_surface.py)
# ===========================================================================


@contextmanager
def _tool_surface(registry: ActionRegistry, index):
    """Patch get_tools_for_agent's collaborators: empty ToolRegistry, no-op
    session factory, the synthetic registry, the deterministic index, and
    SEMANTIC_TOOL_ROUTING forced on (env-proof). top_k stays the REAL
    config-read helper — the bound under test is the configured value."""
    fake_tool_registry = MagicMock()
    fake_tool_registry.get_all_tools.return_value = []
    with patch.object(tr, "registry_get_tool_registry", return_value=fake_tool_registry), \
            patch.object(tr, "SessionLocal", return_value=MagicMock()), \
            patch.object(tr, "_semantic_routing_enabled", return_value=True), \
            patch("modules.tools.discovery.get_action_registry", return_value=registry), \
            patch(
                "modules.tools.discovery.action_semantic_index.get_action_semantic_index",
                return_value=index,
            ):
        yield


def _names(tools: List[Dict[str, Any]]) -> List[str]:
    return [t["function"]["name"] for t in tools]


def _dispatcher_enum(tools: List[Dict[str, Any]]) -> List[str]:
    for t in tools:
        if t["function"]["name"] == "platform_execute":
            return t["function"]["parameters"]["properties"]["action"].get("enum") or []
    raise AssertionError(f"platform_execute not in tools list: {_names(tools)}")


# ===========================================================================
# Tests
# ===========================================================================


@pytest.mark.parametrize("intent,expected", REPRESENTATIVE_INTENTS)
def test_relevant_tool_in_topk_for_representative_intents(scale_index, intent, expected):
    """At 200 actions, every representative intent's expected tool must land
    inside the configured top-K — the starvation detector for FR-5."""
    ranked = _rank(scale_index, intent)
    names = [n for n, _ in ranked]
    assert names, f"ranking returned nothing for {intent!r} — fixture broken"
    assert expected in names, (
        f"{expected!r} starved out of top-{TOP_K} at {FIXTURE_SIZE} tools "
        f"for intent {intent!r}; got {names}"
    )


@pytest.mark.parametrize(
    "intent,expected_su",
    OBS_INTENTS + [(intent, None) for intent, _ in REPRESENTATIVE_INTENTS],
)
def test_su_tools_never_in_topk_for_operator(scale_registry, scale_index, intent, expected_su):
    """No intent — setup, admin, or obs-flavoured — ever ranks an su tool for
    an operator principal. For obs intents the paired su action ranking under
    include_super_admin=True proves exclusion (not weak embeddings) keeps it
    out."""
    su_set = _su_names(scale_registry)
    names = [n for n, _ in _rank(scale_index, intent)]
    assert names, f"ranking returned nothing for {intent!r} — fixture broken"
    leaked = set(names) & su_set
    assert not leaked, f"su tools leaked into operator top-K for {intent!r}: {sorted(leaked)}"

    if expected_su:
        su_included = [
            n for n, _ in _rank(scale_index, intent, include_super_admin=True)
        ]
        assert expected_su in su_included, (
            f"{expected_su!r} did not rank even with include_super_admin=True "
            f"for {intent!r} — the operator exclusion above proves nothing"
        )


def test_topk_is_bounded(scale_registry, scale_index):
    """The selected set is capped by SEMANTIC_TOOL_ROUTING_TOP_K (config,
    not hardcoded) — and the router helper reads the same configured value."""
    assert len(scale_registry.get_all()) == FIXTURE_SIZE
    assert len(_su_names(scale_registry)) == len(_SU_TIER)
    assert tr._semantic_routing_top_k() == TOP_K

    for intent, _ in REPRESENTATIVE_INTENTS[:3]:
        ranked = _rank(scale_index, intent)
        assert 0 < len(ranked) <= TOP_K, (
            f"ranked set size {len(ranked)} breaches top-K bound {TOP_K} for {intent!r}"
        )


@pytest.mark.parametrize(
    "intent",
    ["create an agent", "invite a member to the workspace", "show me the logs"],
)
def test_dispatcher_enum_matches_ranked_set(scale_registry, scale_index, intent):
    """The platform_execute enum the LLM sees is exactly the ranked top-K MINUS
    the promoted actions that attach first-class this turn (PRD-232 US-014 §6.2:
    config pins + whatever promoted ranked in) — no fallback to the full
    catalogue, no su members. A promoted action is NEVER a bare enum member: it is
    first-class when pinned/ranked, else reachable via platform_find_tools."""
    with _tool_surface(scale_registry, scale_index):
        tools = tr.get_tools_for_agent(agent_id=None, workspace_id=None, query=intent)

    enum = _dispatcher_enum(tools)

    # §6.2: the surface ranks the FULL set (exclude_promoted=False), attaches the
    # ranked/pinned promoted first-class, and the enum is the ranked remainder.
    # Recompute that expected split exactly the way tool_router's loader does.
    full_ranked = [n for n, _ in _rank(scale_index, intent, exclude_promoted=False)]
    promoted_all = {spec[0] for spec in _PROMOTED}
    pins = tr._promotion_pins()
    expected_first_class = {n for n in full_ranked if n in promoted_all} | (pins & promoted_all)
    expected_enum = [n for n in full_ranked if n not in expected_first_class]

    assert enum, f"dispatcher enum empty for {intent!r}"
    assert set(enum) == set(expected_enum), (
        f"dispatcher enum diverged from the §6.2 ranked-minus-first-class set for "
        f"{intent!r}: enum-only={sorted(set(enum) - set(expected_enum))}, "
        f"expected-only={sorted(set(expected_enum) - set(enum))}"
    )
    assert len(enum) <= TOP_K
    assert not (set(enum) & _su_names(scale_registry))

    # Promotion-as-prior reachability: a promoted action is never a bare enum
    # member; it is first-class iff pinned or ranked in this turn, otherwise
    # reachable only via platform_find_tools (absent from both first-class + enum).
    surface = _names(tools)
    for promoted_name in promoted_all:
        assert promoted_name not in enum
        if promoted_name in expected_first_class:
            assert promoted_name in surface
        else:
            assert promoted_name not in surface
