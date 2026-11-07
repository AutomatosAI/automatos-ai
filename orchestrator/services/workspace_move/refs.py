"""PRD-273 §5.4 step 4 — ``refs/<table>.jsonl``: the install-level rows moved rows point at by id.

A moved row can name a row that belongs to the install, not the workspace: a global skill an
agent uses or the workspace has switched on, a plugin, a shared persona. Its id means nothing
on another install, so the file carries each such row's natural key (a skill's name and
package, a plugin's or a persona's slug) beside the id the moved rows use, and an import
resolves the id to the target's row with the same key, or lists it as unresolved. Only the
rows the workspace's moved rows name are read; nothing else of the install leaves.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Tuple
from uuid import UUID

from sqlalchemy import String, bindparam, cast, column, or_, select, table
from sqlalchemy.orm import Session

from services.workspace_move.rows import WORKSPACE_COLUMN, TablePlan, workspace_select
from services.workspace_move.schema import SchemaView

REFS_PREFIX = "refs/"
TARGET_KEY = "id"


@dataclass(frozen=True)
class Reference:
    """``source.column`` names a row of ``target``, which an import finds again by ``keys``."""

    source: str
    column: str
    target: str
    keys: Tuple[str, ...]


REFERENCES: Tuple[Reference, ...] = (
    Reference("agent_skills", "skill_id", "skills", ("name", "package_slug")),
    Reference("workspace_enabled_skills", "skill_id", "skills", ("name", "package_slug")),
    Reference("agent_assigned_plugins", "plugin_id", "marketplace_plugins", ("slug",)),
    Reference("workspace_enabled_plugins", "plugin_id", "marketplace_plugins", ("slug",)),
    Reference("agents", "persona_id", "personas", ("slug",)),
)


def _targets(plans: Mapping[str, TablePlan], schema: SchemaView) -> Dict[str, Tuple[Reference, ...]]:
    """The references whose source table this export writes, by target table."""
    found: Dict[str, Tuple[Reference, ...]] = {}
    for reference in REFERENCES:
        present = set(schema.column_names(reference.target))
        named = reference.column in schema.column_names(reference.source)
        if reference.source in plans and named and {TARGET_KEY, *reference.keys} <= present:
            found[reference.target] = (*found.get(reference.target, ()), reference)
    return found


def _rows(db: Session, schema: SchemaView, plans: Mapping[str, TablePlan], references: Tuple[Reference, ...],
          workspace_id: UUID | str) -> Iterator[Dict[str, Any]]:
    """The target rows any of ``references`` names, with their id and natural key."""
    first = references[0]
    scoped = WORKSPACE_COLUMN in schema.column_names(first.target)
    extra = (column(WORKSPACE_COLUMN),) if scoped else ()
    target = table(first.target, column(TARGET_KEY), *(column(key) for key in first.keys), *extra)
    named = [workspace_select(schema, plans[ref.source], workspace_id, (ref.column,)) for ref in references]
    wanted = named[0].union(*named[1:]) if len(named) > 1 else named[0]
    statement = select(target.c[TARGET_KEY], *(target.c[key] for key in first.keys)).where(
        target.c[TARGET_KEY].in_(select(wanted.subquery().c[0]))
    )
    if scoped:  # an install-level row, or the workspace's own: never another workspace's
        own = bindparam("ref_ws", str(workspace_id), type_=String(), unique=True)
        statement = statement.where(or_(target.c[WORKSPACE_COLUMN].is_(None),
                                        cast(target.c[WORKSPACE_COLUMN], String()) == own))
    for row in db.execute(statement).mappings():
        yield dict(row)


def write_refs(db: Session, writer: Any, schema: SchemaView, plans: Mapping[str, TablePlan],
               workspace_id: UUID | str) -> Dict[str, int]:
    """Write one refs part per target table the written parts point at; rows per part."""
    counts: Dict[str, int] = {}
    for target, references in sorted(_targets(plans, schema).items()):
        name = f"{REFS_PREFIX}{target}.jsonl"
        counts[name] = writer.write_rows(name, _rows(db, schema, plans, references, workspace_id))["rows"]
    return counts


__all__ = ["REFERENCES", "Reference", "write_refs"]
