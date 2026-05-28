"""Shared test fixtures for the Shopify widget plugin tests.

PRD-141 US-011 — the snapshot equivalence tests need three things on top
of the per-test setup already in ``test_widget_proactive.py``:

1. The US-004 fixture files (synthetic INBUILD graph, two page contexts,
   two expected opener strings) loaded into Python objects.
2. A ``GraphifyService`` stub whose ``load_graph`` returns the fixture
   graph regardless of workspace id — the lifted resolvers look it up
   lazily so we have to inject it before the helpers run.
3. (Through Phase 1 only) a fake ``api.widgets.chat`` module exposing
   the four proactive helpers the US-003 shim still imports. We
   AST-extract them from real ``chat.py`` so the test exercises the
   actual production source — no separate fixture copy that could
   silently drift.

As US-006/007/008 move the helpers into ``integrations/shopify/
widget_proactive.py``, the ``api.widgets.chat`` injection in
``real_chat_with_graph`` should be tightened (or eventually removed)
in lockstep. The ``GraphifyService`` stub is needed in every phase.

Synthetic-fixture caveat: the graph is hand-crafted to exercise the
same code branches the production INBUILD data exercises. The
byte-equality contract holds against this fixture; behavioural parity
against real INBUILD data is confirmed during the US-020 canary soak.
See ``fixtures/README.md`` for the full rationale.
"""

from __future__ import annotations

import ast
import json
import logging
import sys
import types
from pathlib import Path
from typing import Optional

import networkx as nx
import pytest
from networkx.readwrite import json_graph


_THIS_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR / "fixtures"
# fixtures/ -> tests/ -> shopify/ -> integrations/ -> orchestrator
_ORCH_ROOT = _THIS_DIR.parents[2]
_CHAT_PY = _ORCH_ROOT / "api" / "widgets" / "chat.py"


# Names AST-extracted from chat.py — the proactive helpers still inline
# there. ``_OPENER_CONTEXT_FIELDS`` and ``_format_opener_context_value``
# were lifted to ``integrations/shopify/context_fields.py`` in
# PRD-141 US-005; they're now imported directly into the exec namespace
# (see ``_extract_chat_helpers``) so the function bodies can close over
# them without NameError. ``_resolve_graph_related_products`` was lifted
# to ``integrations/shopify/widget_proactive.py`` in PRD-141 US-006 and
# ``_resolve_cart_recommendations`` was lifted there in US-007; the
# shim's resolver paths now run locally, so neither needs AST
# extraction from chat.py. As US-008 moves the remaining builders into
# ``widget_proactive.py``, drop them from this set in lockstep.
_WANTED_NAMES = frozenset({
    "_build_proactive_opener_message",
    "_build_cart_idle_opener_message",
})


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return _FIXTURES_DIR


@pytest.fixture(scope="session")
def inbuild_graph() -> nx.Graph:
    """Rehydrate the US-004 synthetic INBUILD-flavoured graph.

    Stored as NetworkX ``node_link_data`` JSON for git-diff readability;
    round-trips identically through ``json_graph.node_link_graph``.
    """
    data = json.loads((_FIXTURES_DIR / "inbuild_graph_snapshot.json").read_text())
    return json_graph.node_link_graph(data, edges="edges")


@pytest.fixture(scope="session")
def product_page_context() -> dict:
    return json.loads(
        (_FIXTURES_DIR / "product_page_context.json").read_text()
    )


@pytest.fixture(scope="session")
def cart_idle_context() -> dict:
    return json.loads(
        (_FIXTURES_DIR / "cart_idle_context.json").read_text()
    )


@pytest.fixture(scope="session")
def expected_product_page_opener() -> str:
    return (_FIXTURES_DIR / "expected_product_page_opener.txt").read_text()


@pytest.fixture(scope="session")
def expected_cart_idle_opener() -> str:
    return (_FIXTURES_DIR / "expected_cart_idle_opener.txt").read_text()


def _extract_chat_helpers() -> dict:
    """Pull the remaining proactive builders out of chat.py via AST.

    Returns a namespace dict containing:

    * ``_OPENER_CONTEXT_FIELDS`` — field-mapping tuple, imported from
      ``integrations.shopify.context_fields`` (US-005 lifted it out of
      chat.py; it's still required in the namespace because the AST-
      extracted ``_build_proactive_opener_message`` closes over it).
    * ``_format_opener_context_value`` — single-value formatter, same
      story as the field mapping.
    * ``_build_proactive_opener_message`` — product-page directive.
    * ``_build_cart_idle_opener_message`` — cart-idle directive.

    ``_resolve_graph_related_products`` (US-006) and
    ``_resolve_cart_recommendations`` (US-007) were lifted to
    ``integrations.shopify.widget_proactive`` and are no longer
    extracted here — the shim calls the local functions and they do
    the same lazy ``GraphifyService`` import the caller
    (``real_chat_with_graph``) arranges.

    Why AST instead of ``import``: chat.py is a FastAPI router that drags
    in SQLAlchemy, Redis, RAG, multimodal, etc. Loading it just to read
    two builders is wasteful and brittle. AST extraction reads the
    source verbatim and execs only the wanted nodes into an isolated
    namespace, so the test exercises identical bytes to the running
    server without paying the import cost. ``context_fields`` has none
    of that baggage, so we import it normally.
    """
    from integrations.shopify.context_fields import (
        _OPENER_CONTEXT_FIELDS,
        _format_opener_context_value,
    )

    src = _CHAT_PY.read_text()
    tree = ast.parse(src)

    ns: dict = {
        "Optional": Optional,
        "logger": logging.getLogger("us011_snapshot"),
        "__name__": "_chat_extracted_for_us011",
        "_OPENER_CONTEXT_FIELDS": _OPENER_CONTEXT_FIELDS,
        "_format_opener_context_value": _format_opener_context_value,
    }

    for node in tree.body:
        name: Optional[str] = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id

        if name in _WANTED_NAMES:
            module = ast.Module(body=[node], type_ignores=[])
            code = compile(module, str(_CHAT_PY), "exec")
            exec(code, ns)

    missing = _WANTED_NAMES - set(ns)
    if missing:
        raise RuntimeError(
            "chat.py is missing expected proactive builders "
            f"{sorted(missing)}. If a US-008 lift moved them, update "
            "conftest.py to point the AST extractor at the new location "
            "(likely integrations/shopify/widget_proactive.py)."
        )
    return ns


@pytest.fixture
def real_chat_with_graph(monkeypatch, inbuild_graph):
    """Wire real chat.py helpers + a fixture-bound GraphifyService into sys.modules.

    The US-003 shim's ``handle_widget_message`` does lazy ``from
    api.widgets.chat import ...`` at the point of the rewrite, and the
    resolvers themselves do ``from modules.knowledge.graph_service
    import GraphifyService``. Both have to resolve to something callable
    before the test invokes ``widget_proactive.handle_widget_message``.

    This fixture sets up both, scoped to the test (monkeypatch cleans up
    automatically), so the snapshot tests exercise the exact code that
    runs in production — only the graph source is swapped for the
    deterministic US-004 fixture.

    Through US-005/006/007/008 the helpers move into
    ``integrations/shopify/widget_proactive.py`` and the shim's
    ``api.widgets.chat`` imports go away. Each lift story should narrow
    the injection here so it stays a faithful test environment, not a
    nostalgic crutch.
    """
    ns = _extract_chat_helpers()

    fake_chat = types.ModuleType("api.widgets.chat")
    fake_chat._build_proactive_opener_message = ns["_build_proactive_opener_message"]
    fake_chat._build_cart_idle_opener_message = ns["_build_cart_idle_opener_message"]

    fake_graph_service_mod = types.ModuleType("modules.knowledge.graph_service")

    class _FixtureGraphifyService:
        async def load_graph(self, workspace_id):  # noqa: D401, ARG002
            return inbuild_graph

    fake_graph_service_mod.GraphifyService = _FixtureGraphifyService

    # Parent packages must exist for `from modules.knowledge.graph_service
    # import GraphifyService` to walk the chain. ``monkeypatch.setitem``
    # adds-or-replaces and restores cleanly after the test, regardless of
    # whether the parent was already in sys.modules.
    for parent in ("modules", "modules.knowledge"):
        monkeypatch.setitem(
            sys.modules, parent, sys.modules.get(parent, types.ModuleType(parent))
        )
    monkeypatch.setitem(
        sys.modules, "modules.knowledge.graph_service", fake_graph_service_mod
    )

    for parent in ("api", "api.widgets"):
        monkeypatch.setitem(
            sys.modules, parent, sys.modules.get(parent, types.ModuleType(parent))
        )
    monkeypatch.setitem(sys.modules, "api.widgets.chat", fake_chat)

    return {
        "chat": fake_chat,
        "graph_service": fake_graph_service_mod,
        "namespace": ns,
    }
