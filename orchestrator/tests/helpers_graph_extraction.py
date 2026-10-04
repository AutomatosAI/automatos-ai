"""Shared harness for the F312 graph-extraction tests.

The model is the only fake: a stand-in LLM manager returns the JSONL lines a
test gives it, and the REAL ``extract_from_document`` parses, normalises and
checks them. No DB, no network, no LLM.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

EXTRACTION_TIMEOUT_SECONDS = 5
DEFAULT_OUTPUT_CEILING = 2000


class FakeExtractionLLM:
    """An LLM manager whose every answer is the given JSONL."""

    def __init__(self, items: list[dict[str, Any]]) -> None:
        self.config = SimpleNamespace(max_tokens=DEFAULT_OUTPUT_CEILING)
        self._answer = "\n".join(json.dumps(item) for item in items)

    async def generate_response(self, messages: list[dict[str, str]]) -> SimpleNamespace:
        return SimpleNamespace(content=self._answer)


def node(node_id: str, label: str, file_type: str) -> dict[str, Any]:
    """A node line as the extraction prompt asks for it."""
    return {"kind": "node", "id": node_id, "label": label, "file_type": file_type}


def edge(source: str, target: str, relation: str, label: str) -> dict[str, Any]:
    """An edge line as the extraction prompt asks for it."""
    return {"kind": "edge", "source": source, "target": target, "relation": relation,
            "relation_label": label, "confidence": "EXTRACTED", "confidence_score": 1.0}


async def extract(monkeypatch: Any, items: list[dict[str, Any]], doc_path: str) -> dict[str, list]:
    """Run the real document extraction over a model that answers ``items``."""
    from modules.knowledge import graph_extraction

    monkeypatch.setattr(graph_extraction, "_extraction_timeout", lambda: EXTRACTION_TIMEOUT_SECONDS)
    return await graph_extraction.extract_from_document(
        doc_text="(document text)", doc_path=doc_path, workspace_id="ws-f312",
        llm=FakeExtractionLLM(items),
    )


def edge_between(graph: dict[str, list], a: str, b: str) -> dict[str, Any]:
    """The one extracted edge joining ``a`` and ``b``, whichever way round."""
    found = [e for e in graph["edges"] if {e["source"], e["target"]} == {a, b}]
    assert len(found) == 1, found
    return found[0]
