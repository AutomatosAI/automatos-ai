"""F070 (containment) — a tool the executor cannot run is never offered.

DatabaseToolIntegration registers `query_/explore_/run_<source>_…` per source
into the process-global ToolRegistry, with no workspace, and the unified
executor has no route to them: every call is "Unknown tool". So a source
connected in one workspace put its tools — named after that tenant's source —
into every workspace's Auto, where they could only fail.

This is containment, not the design decision: whether to delete the per-source
generation in favour of the generic workspace-scoped query_database /
smart_query_database is Gerard's call, because the generic tools have no
successful run on record yet. Hiding tools that can never run loses nothing.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from modules.tools import tool_router
from modules.tools.services.database_tool_integration import DatabaseToolIntegration


class _CapturingRegistry:
    def __init__(self):
        self.specs = []

    def register_tool(self, spec):
        self.specs.append(spec)


def _source(name="Harbourline Shop"):
    return NS(id=34, name=name, max_rows_limit=500, workspace_id="ws-other", dialect="postgresql",
              credential_id=7, query_cache_ttl=300, schema_cache_ttl=3600)


def test_the_generator_marks_its_tools_as_database_tool_integration():
    registry = _CapturingRegistry()
    gen = DatabaseToolIntegration(tool_registry=registry, capability_mapper=NS(
        add_task_tool_mapping=lambda *a, **k: None, update_mapping=lambda *a, **k: None))
    for tool_def in (gen._create_query_tool(_source()), gen._create_schema_tool(_source())):
        gen._register_tool(tool_def, _source())
    assert registry.specs, "the generator registered nothing"
    assert all(s.executor_class == "DatabaseToolIntegration" for s in registry.specs)
    assert any("harbourline" in s.name for s in registry.specs), "names carry the source's name"


def test_only_the_unroutable_tools_are_dropped(monkeypatch):
    monkeypatch.setattr(tool_router, "composio_available", lambda: True)
    generic = NS(name="query_database", executor_class="UnifiedToolExecutor")
    smart = NS(name="smart_query_database", executor_class="UnifiedToolExecutor")
    per_source = NS(name="query_harbourline_shop_database", executor_class="DatabaseToolIntegration")
    schema = NS(name="explore_harbourline_shop_schema", executor_class="DatabaseToolIntegration")
    offered = tool_router._offerable_candidates([generic, per_source, smart, schema], "t-f070")
    assert [t.name for t in offered] == ["query_database", "smart_query_database"]


def test_nothing_is_dropped_when_there_is_nothing_unroutable(monkeypatch):
    monkeypatch.setattr(tool_router, "composio_available", lambda: True)
    candidates = [NS(name="query_database", executor_class="UnifiedToolExecutor")]
    assert tool_router._offerable_candidates(candidates, "t-f070") is candidates
