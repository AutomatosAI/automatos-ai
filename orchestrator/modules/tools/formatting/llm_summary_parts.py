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

F460 (night 18): the same question about the database got 14, 30 and 17. A query that lists rows
reached the model as indent=2 JSON cut at ``ROWS_PREVIEW_CHARS`` part-way through a row, and the
model counted the rows it could see; the SQL was cut at 300 characters, so its WHERE was hidden; and
the answer's notes (F301: a count of the top groups only, dates past the data) never reached it.
Now the notes are printed, the SQL's WHERE clause is shown, and rows that would be cut come as one
whole line each, as many as fit, then "Rows shown k of N", with Total Rows as the count.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from services.contested_facts import RANKING_RULE, ranked, standing_note

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
WHERE_PREVIEW_CHARS = 600
# The WHERE clause of a query, up to the clause after it (or the end).
_WHERE = re.compile(r"\bwhere\b.*?(?=\b(?:group\s+by|order\s+by|having|limit|union|window)\b|$)", re.I | re.S)
CELL = " | "
NOTE_LINE = "Note: {note}"
ROWS_SHOWN = "Rows shown {shown} of {total}. The count is Total Rows, not the rows you can see."
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
    # P256-FIX-N13-3 (F410): the document's date and its standing among the passages.
    parts = [] if document_id in (None, "") else [f"document {document_id}"]
    parts += [note for note in (standing_note(doc),) if note]
    return f"\n[Source {index}: {name}] ({'; '.join([*parts, f'{score:.1f}%'])})"


def search_lines(results: List[Dict[str, Any]]) -> List[str]:
    """A document search's passages, each under its source line, newest document first
    (P256-FIX-N13-3: the ranking of contested facts, ``services.contested_facts``)."""
    lines = [SEARCH_INSTRUCTION, RANKING_RULE]
    for i, doc in enumerate(ranked(results), start=1):
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


def _sql_preview(sql: str) -> str:
    """The SQL's first ``SQL_PREVIEW_CHARS``, and its WHERE clause when the cut hides it (F460)."""
    preview = sql[:SQL_PREVIEW_CHARS]
    where = _WHERE.search(sql)
    if len(sql) <= SQL_PREVIEW_CHARS or where is None or where.end() <= SQL_PREVIEW_CHARS:
        return preview
    return f"{preview} … {' '.join(where.group(0).split())[:WHERE_PREVIEW_CHARS]}"


def _columns(rows: List[Any], columns: Any) -> List[str]:
    """The answer's column names, else the first row's keys."""
    if columns:
        return [str(column) for column in columns]
    return [str(key) for key in rows[0]] if isinstance(rows[0], dict) else []


def _row_line(row: Any, columns: List[str]) -> str:
    if isinstance(row, dict):
        return CELL.join(json.dumps(row.get(column), default=str, ensure_ascii=False) for column in columns)
    return json.dumps(row, default=str, ensure_ascii=False)


def _rows_lines(rows: List[Any], columns: Any, total: Any) -> List[str]:
    """All the rows as JSON when they fit ``ROWS_PREVIEW_CHARS``; else one whole line per row while
    they fit, then how many of ``total`` were shown (F460: never a row cut part-way)."""
    whole = json.dumps(rows, default=str, indent=2)
    if len(whole) <= ROWS_PREVIEW_CHARS:
        return [f"Complete data ({len(rows)} rows):", whole]
    names = _columns(rows, columns)
    head = f"Columns: {CELL.join(names)}"
    lines, used = [], len(head)
    for row in rows:
        line = _row_line(row, names)
        if used + len(line) + 1 > ROWS_PREVIEW_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    return [head, *lines, ROWS_SHOWN.format(shown=len(lines), total=total)]


def database_lines(standardized: Dict[str, Any]) -> List[str]:
    """A database query's SQL, its notes, its rows and PandasAI's reading of them."""
    total = standardized.get('row_count', 0)
    lines = [f"\n🗄️ SQL: {_sql_preview(str(standardized.get('sql') or ''))}", f"Total Rows: {total}"]
    lines += [NOTE_LINE.format(note=note) for note in standardized.get('notes') or []]
    # The rows are already limited at query level; ones that would be cut come one per line.
    all_data = standardized.get('data', [])
    if all_data:
        lines += _rows_lines(all_data, standardized.get('columns'), total or len(all_data))
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
