"""Saving an agent without renaming it must not trip the name-conflict check.

2026-09-03: two agents named "Bob" existed in one workspace (legacy rows from
2026-08-29). The frontend sends ``name`` on every save, and the update handler
re-validated the unchanged name against the other row, so every
Configure -> Save of either agent answered 400 "Agent with this name already
exists" for the name it had all along. The check now runs only on a real rename.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from api.agents import _renamed_to_taken_name  # noqa: E402


class _Query:
    def __init__(self, hit):
        self._hit = hit

    def filter(self, *_clauses):
        return self

    def first(self):
        return self._hit


class _DB:
    """Answers every conflict query with ``hit`` and counts how often it was asked."""

    def __init__(self, hit=None):
        self.hit = hit
        self.queries = 0

    def query(self, *_args, **_kwargs):
        self.queries += 1
        return _Query(self.hit)


def test_unchanged_name_is_not_revalidated_even_with_a_legacy_duplicate():
    other_bob = SimpleNamespace(id=14, name="Bob")
    db = _DB(hit=other_bob)  # the duplicate WOULD be found if the handler asked
    agent = SimpleNamespace(id=15, name="Bob")
    assert _renamed_to_taken_name(db, "ws", agent, "Bob") is False
    assert db.queries == 0


def test_real_rename_to_a_taken_name_is_refused():
    db = _DB(hit=SimpleNamespace(id=14, name="Alice"))
    agent = SimpleNamespace(id=15, name="Bob")
    assert _renamed_to_taken_name(db, "ws", agent, "Alice") is True
    assert db.queries == 1


def test_real_rename_to_a_free_name_is_allowed():
    db = _DB(hit=None)
    agent = SimpleNamespace(id=15, name="Bob")
    assert _renamed_to_taken_name(db, "ws", agent, "Carol") is False
    assert db.queries == 1
