"""What ``ToolResultFormatter.format_for_llm`` writes for a search, a code search, a
database query and a Composio call: the model's part of each result.

Moved out of ``result_formatter.py`` (over the file-size limit) with each branch's
output unchanged, except one line.

F383 (night 11, 7 Oct): the content-bank research Playbook (task 2153, from saving a
plan) blocked twice asking the shop owner for document IDs: ``platform_add_social_topics``
wants each fact's source ``{kind, ref, label}``, with ``ref`` "the document's or
Deliverable's id", and search_knowledge's text named each passage's file but never its
document id (kept only for the frontend's source cards). Each source line now carries
its document id, "[Source 1: wholesale-terms.md] (document 717; 61.0%)", for citing it
as a fact's source, never for the owner.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SEARCH_TOOLS = ("search_knowledge", "search_documents", "semantic_search")
CODE_TOOLS = ("search_codebase", "search_code")
DATABASE_TOOLS = ("query_database", "smart_query_database")

SEARCH_INSTRUCTION = (
    "Full document content for the top results is below. Synthesize an "
    "answer using this material directly — do not just list the documents. "
    "The UI renders source cards separately; name a file only when the "
    "owner asks where something comes from, and then only a file named here. "
    "A source's document id is what a fact's source ref takes (platform_add_social_topics); "
    "never show it to the owner, and never ask the owner for one."
)
CODE_PREVIEW_CHARS = 500
SQL_PREVIEW_CHARS = 300
ROWS_PREVIEW_CHARS = 2000
COMPOSIO_PREVIEW_ITEMS = 3
COMPOSIO_PREVIEW_CHARS = 1800
COMPOSIO_LOG_CHARS = 500


def source_line(index: int, doc: Dict[str, Any]) -> str:
    """One passage's heading: its file's real name (F088), and its document id (F383)."""
    score = float(doc.get('similarity', 0) or 0) * 100.0
    # F088 (night 3): the file's name was withheld here, so asked "which
    # file says that?" Auto invented one — "harbourline-wholesale-sheet.md",
    # "Q2 2026 Metrics.md" — or answered "Source 1". The real name, always.
    name = doc.get('filename') or doc.get('source') or 'an unnamed document'
    document_id = doc.get('document_id')
    if document_id in (None, ""):
        return f"\n[Source {index}: {name}] ({score:.1f}%)"
    return f"\n[Source {index}: {name}] (document {document_id}; {score:.1f}%)"


def search_lines(results: List[Dict[str, Any]]) -> List[str]:
    """A document search's passages, each under its source line."""
    lines = [SEARCH_INSTRUCTION]
    for i, doc in enumerate(results, start=1):
        # PRD-136: read `content` (full body) — `excerpt` is a 500-char UI preview
        # that crippled synthesis. Fall back to excerpt if no content.
        body = doc.get('content') or doc.get('excerpt') or ''
        lines.append(source_line(i, doc))
        lines.append(body)
    return lines


def code_lines(results: List[Dict[str, Any]]) -> List[str]:
    """A code search's hits, each with its first lines of code."""
    lines: List[str] = []
    for code in results:
        lines.append(f"\n💻 {code.get('symbol_name', 'Code')} ({code.get('file_path', 'unknown')})")
        lines.append(f"```{code.get('language', 'python')}\n{code.get('code', '')[:CODE_PREVIEW_CHARS]}\n```")
    return lines


def database_lines(standardized: Dict[str, Any]) -> List[str]:
    """A database query's SQL, its rows and PandasAI's reading of them."""
    lines = [f"\n🗄️ SQL: {standardized.get('sql', '')[:SQL_PREVIEW_CHARS]}",
             f"Total Rows: {standardized.get('row_count', 0)}"]
    # Include ALL data (already limited at query level) - don't truncate here
    all_data = standardized.get('data', [])
    if all_data:
        lines.append(f"Complete data ({len(all_data)} rows):")
        lines.append(json.dumps(all_data, default=str, indent=2)[:ROWS_PREVIEW_CHARS])
    # Include PandasAI insight if available
    pandas_insight = standardized.get('pandas_ai', {})
    if pandas_insight:
        lines.append(f"\n📊 AI Analysis: {pandas_insight.get('summary', '')}")
        if pandas_insight.get('charts'):
            lines.append("(Chart generated - see visualization)")
    return lines


def composio_lines(standardized: Dict[str, Any]) -> List[str]:
    """A compact, structured preview of an external API's results, so the LLM can
    actually answer using the returned data."""
    preview = standardized["results"][:COMPOSIO_PREVIEW_ITEMS]
    logger.info(f"[LLM-Context] Composio results count: {len(standardized['results'])}, preview items: {len(preview)}")
    if not preview:
        logger.warning("[LLM-Context] Composio returned 0 items - LLM will hallucinate!")
        return ["\nAPI returned 0 items for this query."]
    try:
        preview_json = json.dumps(preview, default=str, indent=2)[:COMPOSIO_PREVIEW_CHARS]
        logger.info(f"[LLM-Context] Composio preview (first 500 chars): {preview_json[:COMPOSIO_LOG_CHARS]}")
    except (TypeError, ValueError) as e:
        preview_json = str(preview)[:COMPOSIO_PREVIEW_CHARS]
        logger.warning(f"[LLM-Context] JSON dump failed: {e}")
    return ["\nAPI result preview (use this to answer):", preview_json]


def tool_lines(tool_name: str, standardized: Dict[str, Any], results: List[Dict[str, Any]]) -> List[str]:
    """The lines ``format_for_llm`` writes for ``tool_name``'s results after its header."""
    if tool_name in SEARCH_TOOLS:
        return search_lines(results)
    if tool_name in CODE_TOOLS:
        return code_lines(results)
    if tool_name in DATABASE_TOOLS:
        return database_lines(standardized)
    if tool_name == "composio_execute" or tool_name.startswith("composio_"):
        return composio_lines(standardized)
    return []


__all__ = ["SEARCH_INSTRUCTION", "source_line", "tool_lines"]
