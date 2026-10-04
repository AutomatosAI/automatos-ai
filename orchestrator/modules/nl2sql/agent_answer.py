"""What a database tool hands back to the agent that asked (F299/F300, night 9).

* F299 — the board agents' answer is ``json.dumps(raw)`` (agent_factory's tool
  callback), which raises on the ``Decimal`` and ``date`` values a shop's money, kilos
  and dates come back as: "Object of type Decimal is not JSON serializable" on #1858
  (twice), #1856, #1869, #1879, #1880, #1881 — 72 Decimal and 18 date failures in the
  backend log on 4 Oct. Auto's chat read the same rows because its formatter writes them
  with ``json.dumps(..., default=str)`` (``ToolResultFormatter.format_for_llm``). The
  answer now goes out in that same form, so every lane gets plain JSON.
* F300 — every answer lists the database's tables, columns, types, small value sets and
  date ranges, so the agent never has to ask the owner for a table or column name; a
  failed query's error names the real columns of the tables it used.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

from .schema.grounding import (
    Facts,
    cached_facts,
    fact_key,
    read_facts,
    remember_facts,
)

logger = logging.getLogger(__name__)

SCHEMA_DIGEST_MAX_CHARS = 6000
NAMES_ARE_HERE = "Use these names and ask again; the owner does not know the database's names."


def json_safe(value: Any) -> Any:
    """``value`` as plain JSON: Decimal, date, datetime and UUID become text, written
    exactly as Auto's chat writes them (``json.dumps(..., default=str)``)."""
    return json.loads(json.dumps(value, default=str))


def _column_text(table: str, column: Dict[str, Any], facts: Facts) -> str:
    """``plan_code text (one of CLUB | REGULAR | TASTER)`` for one column."""
    fact = facts.get(fact_key(table, column["name"])) or {}
    text = f"{column['name']} {column.get('type') or ''}".strip()
    if fact.get("values"):
        return f"{text} (one of {' | '.join(fact['values'])})"
    if fact.get("range"):
        return f"{text} ({fact['range'][0]} to {fact['range'][1]})"
    return text


def _tables(schema_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The schema's tables that have a name."""
    return [t for t in schema_metadata.get("tables") or [] if t.get("name")]


def table_line(table: Dict[str, Any], facts: Facts) -> str:
    """One table and its columns on one line."""
    columns = [c for c in table.get("columns") or [] if c.get("name")]
    return f"{table['name']}: " + ", ".join(_column_text(table["name"], c, facts) for c in columns)


def schema_digest(schema_metadata: Dict[str, Any], facts: Facts) -> List[str]:
    """The database as the agent needs it: a line per table, then how tables join.
    Bounded by ``SCHEMA_DIGEST_MAX_CHARS``; tables past the bound are named only."""
    lines: List[str] = []
    used = 0
    for table in _tables(schema_metadata):
        line = table_line(table, facts)
        if used + len(line) > SCHEMA_DIGEST_MAX_CHARS:
            line = f"{table['name']}: (columns not listed here; query it by name)"
        lines.append(line)
        used += len(line)
    joins = [
        f"{r['from_table']}.{r['from_column']} -> {r['to_table']}.{r['to_column']}"
        for r in schema_metadata.get("relationships") or []
        if all(r.get(k) for k in ("from_table", "from_column", "to_table", "to_column"))
    ]
    if joins:
        lines.append("joins: " + ", ".join(joins))
    return lines


def tables_in_sql(sql: str, schema_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The schema's tables a statement names, as whole words."""
    return [
        t for t in _tables(schema_metadata)
        if re.search(rf"\b{re.escape(str(t['name']))}\b", sql or "", re.IGNORECASE)
    ]


def real_columns_note(sql: str, schema_metadata: Dict[str, Any]) -> str:
    """What a failed query should have used: the real columns of the tables it named,
    or every table's name when it named none of them."""
    named = tables_in_sql(sql, schema_metadata)
    if named:
        listed = "; ".join(
            f"{t['name']} has " + ", ".join(str(c.get("name")) for c in t.get("columns") or [])
            for t in named
        )
        return f"The columns that exist: {listed}. {NAMES_ARE_HERE}"
    names = ", ".join(str(t["name"]) for t in _tables(schema_metadata))
    return f"The tables that exist: {names}. {NAMES_ARE_HERE}"


def load_source(source_id: Any, workspace_id: str) -> Any:
    """The source row, only when it belongs to ``workspace_id`` (tenant isolation, as
    ``DatabaseKnowledgeService._get_source`` does), or None. Blocking: run it on a thread."""
    from core.database.database import SessionLocal
    from core.models.database_knowledge import DatabaseKnowledgeSource

    session = SessionLocal()
    try:
        return (
            session.query(DatabaseKnowledgeSource)
            .filter(
                DatabaseKnowledgeSource.id == int(source_id),
                DatabaseKnowledgeSource.workspace_id == str(workspace_id),
            )
            .first()
        )
    finally:
        session.close()


def _ground(service: Any, source_id: Any, workspace_id: str) -> Dict[str, Any]:
    """The source's schema; its facts read first when none are cached. Blocking."""
    source = load_source(source_id, workspace_id)
    if source is None:
        logger.warning("F300: source %s is not in workspace %s; no schema for the agent", source_id, workspace_id)
        return {}
    schema = dict(source.schema_metadata or {})
    if schema.get("tables") and not cached_facts(source_id):
        _read_and_remember(service, source, source_id)
    return schema


def _read_and_remember(service: Any, source: Any, source_id: Any) -> None:
    """Read the source's facts from the owner's database and cache them. A failure is
    logged and the schema goes out without them. Blocking."""
    try:
        credentials = service._decrypt_source_credentials(source)
        remember_facts(source_id, read_facts(service, source, credentials))
    except Exception:  # noqa: BLE001 — logged; names and types still reach the agent
        logger.exception("F300: facts of source %s not read; the schema goes without them", source_id)


async def ground_source(service: Any, source_id: Any, workspace_id: str) -> Dict[str, Any]:
    """The source's schema, with its facts read (or taken from the cache) before the
    query runs. On a thread:
    it reads the platform database and the owner's (F105). An empty dict when either
    cannot be read: the query still runs, without the extra context, and the failure
    is logged."""
    try:
        return await asyncio.to_thread(_ground, service, source_id, workspace_id)
    except Exception:  # noqa: BLE001 — logged; the query itself does not depend on it
        logger.exception("F300: source %s schema not read for the agent", source_id)
        return {}


def shape_answer(result: Any, schema_metadata: Dict[str, Any], source_id: Any) -> Any:
    """The tool's answer for the agent: the schema beside it and the real columns in a
    failure, as plain JSON (F299)."""
    if not isinstance(result, dict):
        return json_safe(result)
    shaped = dict(result)
    if schema_metadata.get("tables"):
        facts = cached_facts(source_id)
        sql = str(result.get("sql") or "")
        shaped["schema"] = schema_digest(schema_metadata, facts)
        if not result.get("success") and result.get("error"):
            shaped["error"] = f"{result['error']} {real_columns_note(sql, schema_metadata)}"
    return json_safe(shaped)
