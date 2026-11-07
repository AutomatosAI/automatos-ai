"""PRD-273 §5.1 — ``manifest.json``: what the file holds, readable before any passphrase.

The format, where it came from (edition, the database's Alembic head, the workspace), when
and by whom, the categories, the encryption block when sealed, and every part's row count,
plaintext size and SHA-256. Per table: its category, the columns it carries with their SQL
types, and the ones it left behind and why. Then what was not carried and why: workspace
tables no rule classifies (``skipped_tables``), files that could not be read
(``skipped_files``), rows the bundle format refused (``unbundled``).

Core has no release number of its own: its Alembic head is the version an import compares
(§5.4 step 1: the target's head must be the file's or newer).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Tuple

from services.workspace_move.rows import TablePlan, columns_manifest

FORMAT = "automatos.workspace/v1"
FILE_EXTENSION = ".automatos-workspace"
SQL_PREFIX = "sql/"
WORKSPACE_PART = "workspace.json"
AUTO_PART = "auto.json"
MAX_SLUG = 60
DEFAULT_SLUG = "workspace"


def sql_part(table: str) -> str:
    """The part name of a table's rows."""
    return f"{SQL_PREFIX}{table}.jsonl"


@dataclass(frozen=True)
class Source:
    """Where and when the file was made."""

    edition: str
    alembic_head: Optional[str]
    workspace_id: str
    workspace_name: str
    exported_at: datetime
    exported_by: Optional[str]


@dataclass(frozen=True)
class Contents:
    """What an export wrote, and what it left out."""

    categories: Tuple[str, ...]
    plans: Mapping[str, TablePlan]
    parts: Mapping[str, Mapping[str, Any]]
    skipped_tables: Tuple[Mapping[str, str], ...] = ()
    skipped_files: Tuple[Mapping[str, Any], ...] = ()
    unbundled: Tuple[Mapping[str, Any], ...] = ()
    memory: Optional[Mapping[str, bool]] = None
    files: int = 0
    encryption: Optional[Mapping[str, Any]] = field(default=None)


def _tables(plans: Mapping[str, TablePlan]) -> Dict[str, Dict[str, Any]]:
    return {
        table: {"category": plan.rule.category, "columns": columns_manifest(plan), "dropped": dict(plan.dropped)}
        for table, plan in sorted(plans.items())
    }


def build_manifest(source: Source, contents: Contents) -> Dict[str, Any]:
    """The manifest document."""
    return {
        "format": FORMAT,
        "source": {
            "edition": source.edition, "alembic_head": source.alembic_head,
            "workspace_id": source.workspace_id, "workspace_name": source.workspace_name,
        },
        "exported_at": source.exported_at.isoformat(),
        "exported_by": source.exported_by,
        "categories": list(contents.categories),
        "encryption": dict(contents.encryption) if contents.encryption else None,
        "parts": {name: dict(record) for name, record in sorted(contents.parts.items())},
        "tables": _tables(contents.plans),
        "memory": dict(contents.memory) if contents.memory is not None else None,
        "skipped_tables": [dict(entry) for entry in contents.skipped_tables],
        "skipped_files": [dict(entry) for entry in contents.skipped_files],
        "unbundled": [dict(entry) for entry in contents.unbundled],
    }


def summary(contents: Contents) -> Dict[str, Any]:
    """The job's short record: rows per category, files carried, and what was left out."""
    rows: Dict[str, int] = {category: 0 for category in contents.categories}
    for table, plan in contents.plans.items():
        record = contents.parts.get(sql_part(table), {})
        rows[plan.rule.category] = rows.get(plan.rule.category, 0) + int(record.get("rows", 0))
    return {
        "rows": rows,
        "files": contents.files,
        "skipped_tables": len(contents.skipped_tables),
        "skipped_files": len(contents.skipped_files),
        "unbundled": len(contents.unbundled),
    }


def file_name(workspace_name: str, when: datetime) -> str:
    """``<workspace-name>-<date>.automatos-workspace``: the name lower-cased to letters, digits
    and dashes (``workspace`` when nothing is left)."""
    cleaned = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in (workspace_name or "").lower())
    slug = "-".join(part for part in cleaned.split("-") if part)[:MAX_SLUG].rstrip("-") or DEFAULT_SLUG
    return f"{slug}-{when.date().isoformat()}{FILE_EXTENSION}"


__all__ = [
    "AUTO_PART", "Contents", "FILE_EXTENSION", "FORMAT", "Source", "WORKSPACE_PART", "build_manifest",
    "file_name", "sql_part", "summary",
]
