"""PRD-273 §5.1 — one table's rows for ``sql/<table>.jsonl``: which columns, which rows, read in batches.

The columns are the table's own, read from ``information_schema`` at export time, less:

* **secret** ones: those its rule names, and any whose name ``rules.SECRET_COLUMN_NAME``
  flags (a column added later, or one only another schema has, never leaves by default);
* **derived** ones: those its rule names, and every pgvector or tsvector column. The target
  rebuilds them with its own embeddings.

The rows are the workspace's: ``workspace_id`` equal to it, or, for a table without that
column, a ``parent`` row that is the workspace's (``agent_skills`` of the workspace's agents).
The query is SQLAlchemy Core with the workspace bound as a parameter; identifiers come from
``information_schema`` and the rules, never from a request. Rows stream from a server-side
cursor ``batch_rows`` at a time, so a table of any size never sits in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import bindparam, column, select, table
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.types import String

from services.workspace_move.rules import DERIVED_COLUMN_TYPES, TableRule, looks_secret
from services.workspace_move.schema import ColumnInfo, SchemaView

WORKSPACE_COLUMN = "workspace_id"
PARENT_KEY = "id"
SECRET = "secret"
DERIVED = "derived"


@dataclass(frozen=True)
class TablePlan:
    """What one table's part holds: the columns kept, the ones dropped and why, and whether
    its rows are chosen by its own ``workspace_id`` or by its parent's."""

    rule: TableRule
    kept: Tuple[ColumnInfo, ...]
    dropped: Mapping[str, str]
    scoped: bool

    @property
    def table(self) -> str:
        return self.rule.table


def _why_dropped(rule: TableRule, info: ColumnInfo) -> Optional[str]:
    if info.name in rule.secret_columns or looks_secret(info.name):
        return SECRET
    if info.name in rule.derived_columns or info.udt_name in DERIVED_COLUMN_TYPES:
        return DERIVED
    return None


def plan_table(schema: SchemaView, rule: TableRule) -> Tuple[Optional[TablePlan], Optional[str]]:
    """(the plan, None) for a moved table this database can export, else (None, why not)."""
    if not schema.has(rule.table):
        return None, "not in this database"
    scoped = rule.table in schema.scoped
    if not scoped and (rule.parent is None or rule.parent[0] not in schema.scoped):
        return None, "no workspace column and no workspace parent in this database"
    if not scoped and rule.parent[1] not in schema.column_names(rule.table):
        return None, "the parent column is not in this database"
    kept, dropped = [], {}
    for info in schema.columns[rule.table]:
        why = _why_dropped(rule, info)
        if why is None:
            kept.append(info)
        else:
            dropped[info.name] = why
    return TablePlan(rule, tuple(kept), MappingProxyType(dropped), scoped), None


def _workspace_match(schema: SchemaView, name: str, target: Any, workspace_id: UUID | str) -> ColumnElement:
    """``target.workspace_id = :ws``, the parameter typed like the column (uuid or text)."""
    types = {info.name: info.data_type for info in schema.columns[name]}
    if types.get(WORKSPACE_COLUMN) == "uuid":
        value = bindparam("ws", UUID(str(workspace_id)), type_=PGUUID(as_uuid=True), unique=True)
    else:
        value = bindparam("ws", str(workspace_id), type_=String(), unique=True)
    return target.c[WORKSPACE_COLUMN] == value


def workspace_select(schema: SchemaView, plan: TablePlan, workspace_id: UUID | str,
                     names: Optional[Tuple[str, ...]] = None):
    """A SELECT of ``names`` (the kept columns by default) over the workspace's rows of the table."""
    names = list(names) if names is not None else [info.name for info in plan.kept]
    if plan.scoped:
        own = table(plan.table, *(column(name) for name in {*names, WORKSPACE_COLUMN}))
        return select(*(own.c[name] for name in names)).where(_workspace_match(schema, plan.table, own, workspace_id))
    parent_name, link = plan.rule.parent
    own = table(plan.table, *(column(name) for name in {*names, link}))
    parent = table(parent_name, column(PARENT_KEY), column(WORKSPACE_COLUMN))
    parent_ids = select(parent.c[PARENT_KEY]).where(_workspace_match(schema, parent_name, parent, workspace_id))
    return select(*(own.c[name] for name in names)).where(own.c[link].in_(parent_ids))


def redacted(row: Mapping[str, Any], rule: TableRule) -> Dict[str, Any]:
    """The row with the keys its rule removes from JSON columns taken out (a new dict)."""
    out = dict(row)
    for name, keys in rule.redacted_keys.items():
        value = out.get(name)
        if isinstance(value, Mapping):
            out[name] = {key: item for key, item in value.items() if key not in keys}
    return out


def table_rows(db: Session, schema: SchemaView, plan: TablePlan, workspace_id: UUID | str,
               batch_rows: int) -> Iterator[Dict[str, Any]]:
    """The workspace's rows of ``plan``'s table, kept columns only, ``batch_rows`` per fetch."""
    result = db.execute(workspace_select(schema, plan, workspace_id), execution_options={"yield_per": batch_rows})
    for row in result.mappings():
        yield redacted(row, plan.rule)


def columns_manifest(plan: TablePlan) -> Dict[str, str]:
    """The kept columns and their SQL types, for the manifest (a reader decodes by them)."""
    return {info.name: info.udt_name if info.data_type == "USER-DEFINED" else info.data_type for info in plan.kept}


__all__ = [
    "DERIVED", "SECRET", "TablePlan", "columns_manifest", "plan_table", "redacted", "table_rows", "workspace_select",
]
