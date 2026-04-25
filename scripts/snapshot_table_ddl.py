"""Schema-only snapshot of named tables from a live Postgres.

Usage:
    python3 scripts/snapshot_table_ddl.py table1 table2 ... > snapshot.sql

Reads DDL via pg_catalog (no version-mismatched pg_dump needed). Emits
CREATE TABLE + indexes + foreign keys for each named table. Used by
PRD-135 §12.4 bucket snapshots — schema-only because all dropped tables
have zero rows.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import RealDictCursor


def get_columns(cur, table: str) -> list[dict]:
    cur.execute(
        """
        SELECT column_name, data_type, udt_name, is_nullable, column_default,
               character_maximum_length, numeric_precision, numeric_scale
          FROM information_schema.columns
         WHERE table_schema='public' AND table_name=%s
         ORDER BY ordinal_position;
        """,
        (table,),
    )
    return cur.fetchall()


def get_indexes(cur, table: str) -> list[dict]:
    cur.execute(
        """
        SELECT indexname, indexdef
          FROM pg_indexes
         WHERE schemaname='public' AND tablename=%s
         ORDER BY indexname;
        """,
        (table,),
    )
    return cur.fetchall()


def get_constraints(cur, table: str) -> list[dict]:
    cur.execute(
        """
        SELECT con.conname AS name,
               pg_get_constraintdef(con.oid) AS def,
               con.contype AS type
          FROM pg_constraint con
          JOIN pg_class c ON c.oid = con.conrelid
          JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname='public' AND c.relname=%s
         ORDER BY con.contype, con.conname;
        """,
        (table,),
    )
    return cur.fetchall()


def column_ddl(col: dict) -> str:
    name = col["column_name"]
    udt = col["udt_name"]
    dtype = col["data_type"]

    # Map common type names — be conservative, fall back to udt_name
    if dtype == "ARRAY":
        # postgres reports element type via udt_name (prefixed with _)
        elem = udt.lstrip("_")
        type_str = f"{elem}[]"
    elif dtype == "USER-DEFINED":
        type_str = udt
    elif dtype == "character varying":
        n = col["character_maximum_length"]
        type_str = f"VARCHAR({n})" if n else "VARCHAR"
    elif dtype == "character":
        n = col["character_maximum_length"]
        type_str = f"CHAR({n})" if n else "CHAR"
    elif dtype == "numeric":
        p = col["numeric_precision"]
        s = col["numeric_scale"]
        if p:
            type_str = f"NUMERIC({p},{s or 0})"
        else:
            type_str = "NUMERIC"
    elif dtype == "timestamp with time zone":
        type_str = "TIMESTAMP WITH TIME ZONE"
    elif dtype == "timestamp without time zone":
        type_str = "TIMESTAMP"
    elif dtype == "double precision":
        type_str = "DOUBLE PRECISION"
    else:
        type_str = dtype.upper()

    parts = [f'"{name}"', type_str]
    if col["is_nullable"] == "NO":
        parts.append("NOT NULL")
    if col["column_default"] is not None:
        parts.append(f"DEFAULT {col['column_default']}")
    return " ".join(parts)


def emit_table(cur, table: str) -> str:
    out: list[str] = []
    out.append(f"-- ============================================")
    out.append(f"-- Table: {table}")
    out.append(f"-- ============================================")

    cols = get_columns(cur, table)
    if not cols:
        out.append(f"-- (table {table} not found)")
        return "\n".join(out) + "\n"

    out.append(f'CREATE TABLE IF NOT EXISTS public."{table}" (')
    col_lines = [f"    {column_ddl(c)}" for c in cols]
    out.append(",\n".join(col_lines))
    out.append(");")
    out.append("")

    # Constraints (PK, UNIQUE, CHECK, FK)
    for con in get_constraints(cur, table):
        # Skip not-null pseudo-constraints
        if con["type"] == "n":
            continue
        out.append(
            f'ALTER TABLE public."{table}" '
            f'ADD CONSTRAINT "{con["name"]}" {con["def"]};'
        )

    # Non-constraint indexes
    for idx in get_indexes(cur, table):
        # pg_indexes returns the full CREATE INDEX statement; replays cleanly
        out.append(f"{idx['indexdef']};")

    out.append("")
    return "\n".join(out)


def main() -> None:
    tables = sys.argv[1:]
    if not tables:
        print("usage: snapshot_table_ddl.py <table> [<table> ...]", file=sys.stderr)
        sys.exit(2)

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(2)

    conn = psycopg2.connect(dsn, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    print(f"-- Schema-only snapshot")
    print(f"-- Captured at: {datetime.now(timezone.utc).isoformat()}")
    print(f"-- Tables: {len(tables)}")
    print(f"-- Source: pg_catalog (Railway live)")
    print()
    print("BEGIN;")
    print()

    for t in tables:
        print(emit_table(cur, t))

    print("COMMIT;")


if __name__ == "__main__":
    main()
