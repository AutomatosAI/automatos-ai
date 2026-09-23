"""F081, Gerard's option A — search_multimodal searches the EXTRACTED knowledge.

A document's knowledge_items row is a catalog entry nothing embeds, and the
search skips unembedded rows, so "document" in search_multimodal's defaults
never returned a result. Documents are searched through search_knowledge, over
their embedded chunks. These tests pin both halves: the default no longer
includes documents, and a document question still has a RAG tool to reach.
"""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

try:  # the PDF table extractor is a heavy optional import (the tenancy matrix stubs it too)
    import camelot  # noqa: F401
except Exception:  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from modules.rag.services.multimodal_knowledge_tools import DEFAULT_KB_TYPES, MultimodalKnowledgeTools


def _searched_types(**kwargs):
    db = MagicMock()
    db.execute.return_value.fetchall.return_value = []
    tools = MultimodalKnowledgeTools(db)
    tools._embed_query = AsyncMock(return_value="[0.1,0.2,0.3]")
    result = asyncio.run(tools.search_multimodal("quarterly revenue", workspace_id="ws-A", **kwargs))
    assert result["success"] is True
    return db.execute.call_args.args[1]["kb_types"]


def test_the_default_search_skips_documents():
    assert "document" not in DEFAULT_KB_TYPES
    assert _searched_types() == ["table", "image", "formula", "codegraph"]


def test_an_explicit_document_request_is_still_honoured():
    assert _searched_types(kb_types=["docs", "tables"]) == ["document", "table"]


def test_an_empty_type_list_falls_back_to_the_extracted_defaults():
    assert _searched_types(kb_types=["  "]) == list(DEFAULT_KB_TYPES)


def test_the_tool_no_longer_offers_itself_for_documents():
    from modules.tools.registry.tool_registry import get_tool_registry

    spec = get_tool_registry().get_tool("search_multimodal")
    assert "document" not in spec.description.lower()
    [kb_types] = [p for p in spec.parameters if p.name == "kb_types"]
    assert "document" not in kb_types.default


def test_a_document_question_still_has_a_rag_tool_to_reach():
    """search_knowledge covers uploaded documents, the executor routes it, and
    it is what a research task is given."""
    from modules.tools.execution.unified_executor import UnifiedToolExecutor
    from modules.tools.registry.tool_registry import get_tool_registry
    from modules.tools.services.tool_capability_mapper import ToolCapabilityMapper

    # F085 reworded the text ("this workspace's documents — the owner's uploads"); the intent stands.
    description = get_tool_registry().get_tool("search_knowledge").description.lower()
    assert "documents" in description and "uploads" in description
    assert "search_knowledge" in UnifiedToolExecutor(db_session=MagicMock()).tool_routes
    assert "search_knowledge" in ToolCapabilityMapper.TASK_TOOL_MAPPINGS["research"]["specific"]
