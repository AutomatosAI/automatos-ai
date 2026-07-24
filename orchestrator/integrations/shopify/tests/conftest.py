"""Shared test fixtures for the Shopify widget plugin tests.

PRD-141 US-011 — the snapshot equivalence tests need two things on top
of the per-test setup already in ``test_widget_proactive.py``:

1. The US-004 fixture files (synthetic INBUILD graph, two page contexts,
   two expected opener strings) loaded into Python objects.
2. A ``GraphifyService`` stub whose ``load_graph`` returns the fixture
   graph regardless of workspace id — the lifted resolvers look it up
   lazily so we have to inject it before the helpers run.

After PRD-141 US-008 every proactive helper
(``_resolve_graph_related_products``, ``_resolve_cart_recommendations``,
``_build_proactive_opener_message``, ``_build_cart_idle_opener_message``)
lives in :mod:`integrations.shopify.widget_proactive`. The fixture
imports them directly — no more AST extraction from chat.py, and no
more ``api.widgets.chat`` injection into ``sys.modules``. Only the
``GraphifyService`` stub is still needed.

Synthetic-fixture caveat: the graph is hand-crafted to exercise the
same code branches the production INBUILD data exercises. The
byte-equality contract holds against this fixture; behavioural parity
against real INBUILD data is confirmed during the US-020 canary soak.
See ``fixtures/README.md`` for the full rationale.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import networkx as nx
import pytest
from networkx.readwrite import json_graph


_THIS_DIR = Path(__file__).resolve().parent
_FIXTURES_DIR = _THIS_DIR / "fixtures"


def _node_link_graph_nx31(data: dict) -> nx.Graph:
    """Rehydrate node_link_data under the pinned ``networkx==3.1``.

    The snapshot stores edges under an ``"edges"`` key (the NetworkX >=3.2
    default). On 3.1, ``node_link_graph`` accepts no ``edges=`` kwarg and
    reads the ``"links"`` key, so normalise ``edges`` -> ``links`` first —
    exactly as the production loader does in
    ``modules.knowledge.graph_service._normalize_node_link_data`` — then call
    ``node_link_graph(data)`` with no kwarg.
    """
    if "links" not in data and "edges" in data:
        data["links"] = data.pop("edges")
    return json_graph.node_link_graph(data)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return _FIXTURES_DIR


@pytest.fixture(scope="session")
def inbuild_graph() -> nx.Graph:
    """Rehydrate the US-004 synthetic INBUILD-flavoured graph.

    Stored as NetworkX ``node_link_data`` JSON for git-diff readability;
    round-trips identically through ``json_graph.node_link_graph`` (see
    ``_node_link_graph_nx31`` for the networkx==3.1 edges->links handling).
    """
    data = json.loads((_FIXTURES_DIR / "inbuild_graph_snapshot.json").read_text())
    return _node_link_graph_nx31(data)


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


@pytest.fixture
def real_chat_with_graph(monkeypatch, inbuild_graph):
    """Inject a fixture-bound ``GraphifyService`` for the lifted resolvers.

    The Shopify plugin's resolvers (``_resolve_graph_related_products``,
    ``_resolve_cart_recommendations``) do ``from
    modules.knowledge.graph_service import GraphifyService`` lazily at
    call time. This fixture replaces that import with a deterministic
    stub returning the US-004 fixture graph, so the snapshot tests
    exercise the exact production code path with one — and only one —
    side-channel: the graph source.

    Fixture name kept for git-history continuity through the Phase 1
    lift (US-005/006/007/008/010). After US-008 there is no longer any
    chat.py injection — the plugin calls local builders directly — so
    "real chat" in the name is now historical.
    """
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

    return {
        "graph_service": fake_graph_service_mod,
    }
