"""PRD-273 §5.2–§5.3 — what a table rule says, the categories a rule belongs to, and the
secret-looking column names every export drops.

A rule classifies one table, by name, as ``MOVED`` (in one category of §5.2) or
``NOT_MOVED``, with a one-line reason. A table without ``workspace_id`` that hangs off a
workspace table by foreign key (``agent_skills``, ``votes``) names that parent: its rows
are the ones whose parent row is the workspace's. A rule also names its table's secret
columns (never in any export, the GDPR export included), its derived columns (vectors and
search indexes the target rebuilds), and keys removed from a JSON column (a Playbook
trigger's webhook id). A ``secret_table`` holds nothing but secrets and connections and
is never exported at all.

The map of rules is ``table_map.py``; this module is the vocabulary only, so the guard
and the GDPR export read the same definitions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

# §5.2's categories, in the order an import writes them (§5.4 step 6).
AUTO = "auto"
AGENTS = "agents"
KNOWLEDGE = "knowledge"
MEMORY = "memory"
CHATS = "chats"
DELIVERABLES = "deliverables"
BOARD = "board"
CATEGORIES: Tuple[str, ...] = (AUTO, AGENTS, KNOWLEDGE, MEMORY, CHATS, DELIVERABLES, BOARD)
# Auto is always exported; Board is off unless ticked (§8-Q3).
ALWAYS_ON: Tuple[str, ...] = (AUTO,)
DEFAULT_CATEGORIES: Tuple[str, ...] = tuple(category for category in CATEGORIES if category != BOARD)

# A column whose name says it holds a secret, a credential link or a webhook key. An export
# drops every such column a rule does not already name, so a column added later (or one
# only Automatos Cloud's schema has) never leaves by default; the guard test fails until
# its rule names it.
SECRET_COLUMN_NAME = re.compile(
    r"(^|_)(encrypted|secret|secrets|password|passphrase|private_key|api_key|key_hash|token_hash|"
    r"code_hash|push_key|credential_id|webhook_key|webhook_id|webhook_url)(_|$)|(^|_)token$"
)
# Column types the target rebuilds from the data (pgvector, full-text search).
DERIVED_COLUMN_TYPES = frozenset({"vector", "tsvector"})

_EMPTY: Mapping[str, Tuple[str, ...]] = MappingProxyType({})


@dataclass(frozen=True)
class TableRule:
    """How one table is exported. ``category`` is set exactly when ``moved``; ``parent`` is
    ``(parent_table, column)`` for a table that has no ``workspace_id`` of its own."""

    table: str
    moved: bool
    reason: str
    category: Optional[str] = None
    parent: Optional[Tuple[str, str]] = None
    secret_columns: Tuple[str, ...] = ()
    derived_columns: Tuple[str, ...] = ()
    redacted_keys: Mapping[str, Tuple[str, ...]] = field(default_factory=lambda: _EMPTY)
    secret_table: bool = False


def moved(table: str, category: str, reason: str, **details) -> TableRule:
    """A table the move file carries, in ``category``."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r} for {table}")
    return TableRule(table=table, moved=True, category=category, reason=reason, **details)


def not_moved(table: str, reason: str, **details) -> TableRule:
    """A table the move file never carries."""
    return TableRule(table=table, moved=False, reason=reason, **details)


def child(table: str, parent: str, column: str, category: Optional[str], reason: str, **details) -> TableRule:
    """A table with no ``workspace_id``, whose ``column`` names a row of ``parent``; moved in
    ``category``, or not moved when ``category`` is None."""
    link = (parent, column)
    if category is None:
        return not_moved(table, reason, parent=link, **details)
    return moved(table, category, reason, parent=link, **details)


def looks_secret(column: str) -> bool:
    """Whether a column's name says it holds a secret (``SECRET_COLUMN_NAME``)."""
    return SECRET_COLUMN_NAME.search(column) is not None


def resolve_categories(requested: Optional[Tuple[str, ...]]) -> Tuple[str, ...]:
    """The categories an export writes: the requested ones (the defaults when None) plus
    the always-on ones, in §5.4's order. Raises ValueError on an unknown category."""
    chosen = set(DEFAULT_CATEGORIES if requested is None else requested)
    unknown = sorted(chosen - set(CATEGORIES))
    if unknown:
        raise ValueError(f"unknown categories: {', '.join(unknown)}")
    chosen |= set(ALWAYS_ON)
    return tuple(category for category in CATEGORIES if category in chosen)


__all__ = [
    "AGENTS", "ALWAYS_ON", "AUTO", "BOARD", "CATEGORIES", "CHATS", "DEFAULT_CATEGORIES", "DELIVERABLES",
    "DERIVED_COLUMN_TYPES", "KNOWLEDGE", "MEMORY", "SECRET_COLUMN_NAME", "TableRule", "child",
    "looks_secret", "moved", "not_moved", "resolve_categories",
]
