"""F206 (night 6) — graph extraction's contract fits its output cap.

20 of 25 extraction calls on night 6 ended at the 2,000-token cap ("1 line(s)
lost to truncation, 35 kept"); across nights 3-6, 350 of 412 did. The contract
allowed 25 nodes, 40 edges and 3 hyperedges, about 68 lines, and every node line
repeated the document's path. The cap held about 36 lines, so what was cut was
mostly edges, the relationships.

Now the contract fits the cap with headroom, the limit is declared in the prompt,
and the path is not repeated: the parser files every line under the call's
document.
"""
from __future__ import annotations

import json

HEADROOM = 0.85
# Night 6's longest document name and relation phrases like its own.
DOC = "2026-09-26_051722_3e95bb_task-reply-to-rosie-about-a-double-charge-and-a-grind-change-draft-only.md"


def _full_answer():
    from config import config

    nodes = [json.dumps({"kind": "node", "id": f"harbourline_coffee_roasters_{i}",
                         "label": f"Harbourline Coffee Roasters {i}", "file_type": "entity"})
             for i in range(config.GRAPH_EXTRACTION_MAX_NODES)]
    edges = [json.dumps({"kind": "edge", "source": f"monthly_coffee_club_{i}", "target": f"tasting_card_{i}",
                         "relation": "includes", "relation_label": "every club bag ships with a tasting card",
                         "confidence": "EXTRACTED", "confidence_score": 1.0})
             for i in range(config.GRAPH_EXTRACTION_MAX_EDGES)]
    hyperedges = [json.dumps({"kind": "hyperedge", "id": f"monday_dispatch_flow_{i}", "label": "Monday Dispatch Flow",
                              "nodes": ["roast_day", "packing_station", "royal_mail_labels", "tasting_card"],
                              "relation": "form", "confidence": "EXTRACTED", "confidence_score": 0.9})
                  for i in range(getattr(config, "GRAPH_EXTRACTION_MAX_HYPEREDGES", 3))]
    return "\n".join(nodes + edges + hyperedges)


def test_a_full_contract_answer_fits_the_cap_with_headroom():
    from config import config
    from core.context_guard import count_tokens

    used = count_tokens(_full_answer())
    assert used <= HEADROOM * config.GRAPH_EXTRACTION_MAX_OUTPUT_TOKENS, used


def test_the_prompt_declares_the_limit_and_asks_for_no_paths():
    from modules.knowledge import graph_extraction

    budget = graph_extraction._output_budget()
    assert budget.startswith("BUDGET: return at most 14 nodes, 16 edges and 1 hyperedge(s).")
    for prompt in (graph_extraction._DOCUMENT_EXTRACTION_PROMPT, graph_extraction._REPORT_EXTRACTION_PROMPT):
        assert '"source_file"' not in prompt and "Maximum 3 per" not in prompt


def test_a_line_without_its_path_is_filed_under_the_calls_document():
    from modules.knowledge.graph_extraction import _normalise_extraction

    graph = _normalise_extraction({"nodes": [{"id": "rosie_tanner", "label": "Rosie Tanner"}], "edges": [],
                                   "hyperedges": [{"id": "refund_flow", "nodes": ["a", "b", "c"]}]}, source_file=DOC)
    assert graph["nodes"][0]["source_file"] == DOC and graph["hyperedges"][0]["source_file"] == DOC
