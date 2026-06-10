"""PRD-154 S6 — Knowledge Graph export + communities-tool rewire.

Two backend root causes from the deep review §2 / story P154-S6:

1. ``GraphifyService._export_graph`` serialises the graph with
   ``nx.node_link_data`` — which only carries *node attributes*. Community
   membership lives in the separate ``communities`` map (community_id →
   node_ids) and never reaches the node, so the React panel's *default*
   community colouring (``BusinessGraphPanel`` colorMode='community',
   reads ``node.community``) has nothing to read. Confidence is likewise
   absent on any node that arrived without one (merged graph.json targets).
   The fix annotates every exported node with ``community`` + a backfilled
   ``confidence`` before the JSON is written.

2. ``handle_graph_communities`` READS ``graph/communities.json`` via
   ``core.workspace_client.WorkspaceClient`` (the file/S3 backend) while
   ``_export_graph`` WROTE it via ``core.graph_storage.DbWorkspaceClient``
   (the Postgres ``workspace_graphs`` store). For a DB-backed workspace the
   reader looks in the wrong place and returns *nothing*. The fix points the
   reader at ``DbWorkspaceClient`` — the same store the writer used.

Tests are deterministic and DB-free: the annotation is a pure function over
the node_link_data dict, and the communities handler is exercised with a
recording fake for ``DbWorkspaceClient``.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config import chain with no reachable DB
# (blessed pattern). CI exports real POSTGRES_* so these setdefaults no-op.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

# CI collection-order safety net — this module imports the real modules.*
# chain at collection time; restore the real app modules over any sibling
# stub before importing them (see tests/conftest.py).
import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

HANDLERS_GRAPH_PY = ORCH_ROOT / "modules" / "tools" / "discovery" / "handlers_graph.py"
GRAPH_SERVICE_PY = ORCH_ROOT / "modules" / "knowledge" / "graph_service.py"


# ===========================================================================
# 1. EXPORT ANNOTATION — community + confidence on every exported node.
# ===========================================================================


class TestExportNodeAnnotation:
    """``_export_graph`` must attach ``community`` (from the communities map)
    and a backfilled ``confidence`` to each node before writing graph.json.
    Without it the UI's default community colouring reads ``undefined`` and
    every node renders the neutral fallback grey."""

    @pytest.fixture(scope="class")
    def gs_cls(self):
        from modules.knowledge.graph_service import GraphifyService

        return GraphifyService

    def _graph_data(self) -> dict:
        # node_link_data shape: node id under "id"; node "c" carries no
        # confidence (simulates a merged-in edge target).
        return {
            "directed": False,
            "multigraph": False,
            "graph": {},
            "nodes": [
                {"id": "a", "label": "A", "file_type": "concept", "confidence": "EXTRACTED"},
                {"id": "b", "label": "B", "file_type": "entity", "confidence": "INFERRED"},
                {"id": "c", "label": "C", "file_type": "entity"},
            ],
            "links": [],
        }

    def test_community_attached_to_each_node(self, gs_cls):
        communities = {0: ["a", "b"], 1: ["c"]}
        out = gs_cls._annotate_export_nodes(self._graph_data(), communities)
        by_id = {n["id"]: n for n in out["nodes"]}
        assert all("community" in n for n in out["nodes"]), (
            "every exported node must carry a `community` key for the UI's "
            "default community colouring"
        )
        assert by_id["a"]["community"] == 0
        assert by_id["b"]["community"] == 0
        assert by_id["c"]["community"] == 1

    def test_confidence_present_on_every_node_backfilled_where_missing(self, gs_cls):
        communities = {0: ["a", "b", "c"]}
        out = gs_cls._annotate_export_nodes(self._graph_data(), communities)
        by_id = {n["id"]: n for n in out["nodes"]}
        # Pre-existing confidence preserved …
        assert by_id["a"]["confidence"] == "EXTRACTED"
        assert by_id["b"]["confidence"] == "INFERRED"
        # … and backfilled where the node arrived without one.
        assert by_id["c"]["confidence"], "node 'c' confidence must be backfilled, not null"
        assert all(n.get("confidence") for n in out["nodes"])

    def test_node_outside_any_community_gets_null_community_key(self, gs_cls):
        # Key present (UI reads node.community), value null → neutral colour.
        communities = {0: ["a"]}
        out = gs_cls._annotate_export_nodes(self._graph_data(), communities)
        by_id = {n["id"]: n for n in out["nodes"]}
        assert by_id["b"]["community"] is None
        assert "community" in by_id["c"]

    def test_annotation_is_immutable_does_not_mutate_input(self, gs_cls):
        original = self._graph_data()
        snapshot = json.loads(json.dumps(original))
        gs_cls._annotate_export_nodes(original, {0: ["a", "b", "c"]})
        assert original == snapshot, (
            "annotation must return a new dict, never mutate the caller's "
            "graph_data (CLAUDE coding-style: immutability)"
        )

    def test_export_graph_calls_annotation_before_writing_graph_json(self):
        # Wiring pin: a refactor that drops the annotation call would make the
        # UI colouring silently break again. Assert _export_graph annotates.
        src = GRAPH_SERVICE_PY.read_text()
        assert "_annotate_export_nodes" in src, (
            "_export_graph must call _annotate_export_nodes before writing graph.json"
        )
        # The annotation must sit on the graph.json write path, not e.g. only
        # the history snapshot.
        assert re.search(
            r"_annotate_export_nodes\([^)]*communities", src
        ), "annotation must be fed the communities map"


# ===========================================================================
# 2. COMMUNITIES HANDLER — reads via DbWorkspaceClient (the writer's store).
# ===========================================================================


class _FakeDbWsClient:
    """Records the path read and returns canned communities.json content —
    standing in for the Postgres-backed DbWorkspaceClient."""

    last_path: str | None = None

    def __init__(self, workspace_id: str):
        self.workspace_id = workspace_id

    async def read_file(self, path: str):
        type(self).last_path = path
        return {
            "success": True,
            "content": json.dumps(
                [
                    {"community_id": 0, "member_count": 3, "members": ["a", "b", "c"]},
                    {"community_id": 1, "member_count": 1, "members": ["d"]},
                ]
            ),
        }


class TestCommunitiesHandlerUsesDbStore:
    """``handle_graph_communities`` must read from the SAME store
    ``_export_graph`` wrote to — Postgres ``workspace_graphs`` via
    ``DbWorkspaceClient``. The old ``WorkspaceClient`` (file/S3) read returns
    nothing for a DB-backed workspace."""

    def test_returns_communities_for_db_backed_workspace(self):
        from modules.tools.discovery.handlers_graph import handle_graph_communities

        _FakeDbWsClient.last_path = None
        with patch("core.graph_storage.DbWorkspaceClient", _FakeDbWsClient):
            result = asyncio.run(
                handle_graph_communities(MagicMock(), uuid4(), {})
            )
        assert result["success"] is True, result
        assert result["community_count"] == 2
        ids = {c["community_id"] for c in result["communities"]}
        assert ids == {0, 1}
        assert _FakeDbWsClient.last_path == "graph/communities.json"

    def test_specific_community_lookup_round_trips(self):
        from modules.tools.discovery.handlers_graph import handle_graph_communities

        with patch("core.graph_storage.DbWorkspaceClient", _FakeDbWsClient):
            result = asyncio.run(
                handle_graph_communities(MagicMock(), uuid4(), {"community_id": 1})
            )
        assert result["success"] is True, result
        assert result["community"]["community_id"] == 1
        assert result["community"]["members"] == ["d"]

    def test_handler_source_uses_db_client_not_file_client(self):
        # Belt + braces: the file-backed WorkspaceClient read is the bug.
        # Pin the canonical store so a refactor can't reintroduce the split.
        src = HANDLERS_GRAPH_PY.read_text()
        assert "DbWorkspaceClient" in src, (
            "handle_graph_communities must read via DbWorkspaceClient "
            "(the workspace_graphs store _export_graph wrote to)"
        )
        # The communities read must not go through the file/S3 WorkspaceClient.
        assert "from core.workspace_client import WorkspaceClient" not in src, (
            "the file-backed WorkspaceClient read returns nothing for a "
            "DB-backed workspace — must use DbWorkspaceClient"
        )
