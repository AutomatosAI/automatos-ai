"""PRD-273 §5.3 — what this database holds, read from ``information_schema``, and the guard.

``read_schema`` gives every public base table's columns (name, SQL type), the workspace
tables (``workspace_purge._discover_scoped_tables``, the same self-maintaining discovery the
purge and the GDPR export use) and the tables without ``workspace_id`` that reference a
workspace table by foreign key. ``unclassified_tables`` and ``unnamed_secret_columns`` are the
guard: a table no rule classifies, and a secret-looking column no rule names. The export
skips the first and drops the second; the guard test fails on either, so neither lasts.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from services.workspace_move.rules import looks_secret
from services.workspace_move.table_map import TABLE_RULES

_COLUMNS_SQL = text(
    """
    SELECT c.table_name, c.column_name, c.data_type, c.udt_name
      FROM information_schema.columns c
      JOIN information_schema.tables t
        ON t.table_name = c.table_name AND t.table_schema = c.table_schema
     WHERE c.table_schema = 'public' AND t.table_type = 'BASE TABLE'
     ORDER BY c.table_name, c.ordinal_position
    """
)
# Tables with no workspace_id column whose foreign key names a table that has one.
_CHILDREN_SQL = text(
    """
    WITH scoped AS (
      SELECT DISTINCT table_name FROM information_schema.columns
       WHERE column_name = 'workspace_id' AND table_schema = 'public'
    )
    SELECT DISTINCT tc.table_name, kcu.column_name, ccu.table_name AS parent
      FROM information_schema.table_constraints tc
      JOIN information_schema.key_column_usage kcu
        ON kcu.constraint_name = tc.constraint_name AND kcu.table_schema = tc.table_schema
      JOIN information_schema.constraint_column_usage ccu
        ON ccu.constraint_name = tc.constraint_name AND ccu.table_schema = tc.table_schema
     WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
       AND ccu.table_name IN (SELECT table_name FROM scoped)
       AND tc.table_name NOT IN (SELECT table_name FROM scoped)
     ORDER BY 1, 2
    """
)
_ALEMBIC_TABLE_SQL = text("SELECT to_regclass('public.alembic_version') IS NOT NULL")
_ALEMBIC_SQL = text("SELECT version_num FROM alembic_version ORDER BY version_num")


@dataclass(frozen=True)
class ColumnInfo:
    """One column: its name and SQL type (``data_type``, and ``udt_name`` for a user type)."""

    name: str
    data_type: str
    udt_name: str


@dataclass(frozen=True)
class SchemaView:
    """The tables an export can read: columns by table, the workspace tables, and each child
    table's foreign keys to workspace tables as (column, parent) pairs."""

    columns: Mapping[str, Tuple[ColumnInfo, ...]]
    scoped: Tuple[str, ...]
    children: Mapping[str, Tuple[Tuple[str, str], ...]]

    def has(self, table: str) -> bool:
        return table in self.columns

    def column_names(self, table: str) -> Tuple[str, ...]:
        return tuple(column.name for column in self.columns.get(table, ()))


def _columns(db: Session) -> Mapping[str, Tuple[ColumnInfo, ...]]:
    found: Dict[str, List[ColumnInfo]] = {}
    for table, name, data_type, udt_name in db.execute(_COLUMNS_SQL):
        found.setdefault(table, []).append(ColumnInfo(name, data_type or "", udt_name or ""))
    return MappingProxyType({table: tuple(columns) for table, columns in found.items()})


def _children(db: Session) -> Mapping[str, Tuple[Tuple[str, str], ...]]:
    found: Dict[str, List[Tuple[str, str]]] = {}
    for table, column, parent in db.execute(_CHILDREN_SQL):
        found.setdefault(table, []).append((column, parent))
    return MappingProxyType({table: tuple(links) for table, links in found.items()})


def read_schema(db: Session) -> SchemaView:
    """This database's tables, as an export and the guard read them."""
    from services.workspace_purge import _discover_scoped_tables

    return SchemaView(columns=_columns(db), scoped=tuple(_discover_scoped_tables(db)), children=_children(db))


def unclassified_tables(schema: SchemaView) -> Tuple[str, ...]:
    """Workspace tables and their child tables that no rule classifies, sorted."""
    tables = set(schema.scoped) | set(schema.children)
    return tuple(sorted(table for table in tables if table not in TABLE_RULES))


def unnamed_secret_columns(schema: SchemaView) -> Tuple[Tuple[str, str], ...]:
    """(table, column) for every secret-looking column of a classified table that its rule
    does not name as secret. A secret table is never exported, so its columns don't count."""
    found = []
    for table, rule in TABLE_RULES.items():
        if rule.secret_table or not schema.has(table):
            continue
        found.extend((table, name) for name in schema.column_names(table)
                     if looks_secret(name) and name not in rule.secret_columns)
    return tuple(sorted(found))


def alembic_head(db: Session) -> Optional[str]:
    """The database's Alembic revision (several joined by ','); None when it was never stamped."""
    if not db.execute(_ALEMBIC_TABLE_SQL).scalar():
        return None
    versions = [row[0] for row in db.execute(_ALEMBIC_SQL)]
    return ",".join(versions) or None


__all__ = ["ColumnInfo", "SchemaView", "alembic_head", "read_schema", "unclassified_tables", "unnamed_secret_columns"]
