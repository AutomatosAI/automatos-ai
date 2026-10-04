"""What a connected database actually holds, beside its column names (F300, night 9).

The stored schema has names and types and, for some columns, the first five values
introspection happened to read. That was not enough to write the right query:

* F300 — board agents asked the owner for "the exact plan_code" (#1886) and for "the
  subscription_orders schema" (#1891), and guessed columns that do not exist
  (``wo.order_date`` on #1888; the column is ``ordered_on``).

For every column this records the complete set of values when it is small (``plan_code``:
CLUB, REGULAR, TASTER) and, for a date column, the first and last date recorded. The
facts are read once per source and kept for ``FACTS_TTL_SECONDS``. They reach the agent
through the tool's answer (``modules.nl2sql.agent_answer``).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

VALUE_SET_MAX = 12            # a column with more distinct values is not a list to choose from
FACTS_TTL_SECONDS = 300       # new orders move a date column's last value; re-read after this
PROBE_TIMEOUT_SECONDS = 5     # per statement, against the owner's database
PROBE_MAX_COLUMNS = 120       # bounds the work on a wide database
PROBE_BUDGET_SECONDS = 20     # all probes of one source together; what is read by then is kept
DATE_TYPES = ("date", "timestamp")
CATEGORY_TYPES = ("char", "text", "enum", "bool", "user-defined")

Facts = Dict[str, Dict[str, List[str]]]
Quote = Callable[[str, str], str]

_FACTS: Dict[str, Tuple[float, Facts]] = {}


def fact_key(table: str, column: str) -> str:
    """The key a column's facts are kept under."""
    return f"{table}.{column}"


def column_kind(column_type: Any) -> Optional[str]:
    """``"date"`` for a date or timestamp column, ``"category"`` for text, enum or
    boolean, None for anything else (numbers, ids, blobs)."""
    lowered = str(column_type or "").lower()
    if any(name in lowered for name in DATE_TYPES):
        return "date"
    if any(name in lowered for name in CATEGORY_TYPES):
        return "category"
    return None


def _column_fact(conn: Any, quote: Quote, dialect: str, table: str, column: Dict[str, Any]) -> Optional[Dict[str, List[str]]]:
    """One column's facts: its date range or its complete small value set."""
    kind = column_kind(column.get("type"))
    if kind is None:
        return None
    tq, cq = quote(dialect, table), quote(dialect, column["name"])
    if kind == "date":
        # Identifiers come from the introspected schema, quoted by the service's quoter.
        row = conn.execute(text(f"SELECT MIN({cq}), MAX({cq}) FROM {tq}")).first()  # noqa: S608
        return {"range": [str(row[0]), str(row[1])]} if row and row[0] is not None else None
    rows = conn.execute(
        text(f"SELECT DISTINCT {cq} FROM {tq} WHERE {cq} IS NOT NULL LIMIT :lim"),  # noqa: S608
        {"lim": VALUE_SET_MAX + 1},
    ).fetchall()
    values = sorted(str(r[0]) for r in rows)
    return {"values": values} if 0 < len(values) <= VALUE_SET_MAX else None


def _columns(schema_metadata: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Every (table name, column) pair of the schema, in order, bounded."""
    pairs = [
        (table.get("name"), column)
        for table in schema_metadata.get("tables") or []
        for column in table.get("columns") or []
        if table.get("name") and column.get("name")
    ]
    return pairs[:PROBE_MAX_COLUMNS]


def probe_facts(conn: Any, dialect: str, schema_metadata: Dict[str, Any], quote: Quote) -> Facts:
    """Read the facts of every column from an open connection.

    A column whose probe fails (a timeout, a type the database cannot compare) is left
    out and logged; the transaction is rolled back so the next probe can run. Probing
    stops after ``PROBE_BUDGET_SECONDS``, keeping what it has read."""
    facts: Facts = {}
    deadline = time.monotonic() + PROBE_BUDGET_SECONDS
    for table, column in _columns(schema_metadata):
        if time.monotonic() > deadline:
            logger.info("F300: fact probing stopped at %s.%s after %ss", table, column.get("name"), PROBE_BUDGET_SECONDS)
            break
        try:
            fact = _column_fact(conn, quote, dialect, table, column)
        except SQLAlchemyError as err:
            logger.info("F300: no facts for %s.%s: %s", table, column.get("name"), type(err).__name__)
            conn.rollback()
            continue
        if fact:
            facts[fact_key(table, column["name"])] = fact
    return facts


def cached_facts(source_id: Any) -> Facts:
    """The facts read for a source within the last ``FACTS_TTL_SECONDS``, or none."""
    entry = _FACTS.get(str(source_id or ""))
    if entry is None or time.monotonic() - entry[0] > FACTS_TTL_SECONDS:
        return {}
    return entry[1]


def remember_facts(source_id: Any, facts: Facts) -> None:
    """Keep a source's facts for the next ``FACTS_TTL_SECONDS``."""
    _FACTS[str(source_id)] = (time.monotonic(), dict(facts))


def read_facts(service: Any, source: Any, credentials: Dict[str, Any]) -> Facts:
    """Open the owner's database the way the query does (the service's connection
    string, a per-statement timeout) and read every column's facts. Blocking: run it
    on a thread."""
    engine = create_engine(service._nl2sql_connection_string(credentials, source.dialect), pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            timeout_sql = service._statement_timeout_sql(source.dialect, PROBE_TIMEOUT_SECONDS)
            if timeout_sql:
                conn.execute(text(timeout_sql))
                conn.commit()  # a probe's rollback must not undo the timeout (SET is transactional)
            return probe_facts(conn, source.dialect, source.schema_metadata or {}, service._quote_ident)
    finally:
        engine.dispose()
