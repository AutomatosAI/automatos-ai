"""F063 — no paid extraction for a table nothing reads.

Every document upload ran two LLM calls (entities, then relationships) and wrote
the results to `kb_entities` — a table no migration ever created. The INSERT
failed, the except swallowed it, and nothing anywhere reads the table. The
knowledge graph (PRD-165) is where entities live.
"""
from __future__ import annotations

import inspect

import modules.rag.ingestion.manager as manager


class _Cursor:
    def __init__(self, exists=None, boom=False):
        self.exists, self.boom, self.calls = exists, boom, 0

    def execute(self, _sql):
        self.calls += 1
        if self.boom:
            raise RuntimeError("no database")

    def fetchone(self):
        return (self.exists,)


def _fresh():
    manager._KB_ENTITIES_EXISTS = None


def test_a_missing_table_means_no_extraction():
    _fresh()
    assert manager._kb_entities_table_exists(_Cursor(exists=False)) is False


def test_the_answer_is_read_once_per_process():
    _fresh()
    cur = _Cursor(exists=True)
    assert manager._kb_entities_table_exists(cur) is True
    assert manager._kb_entities_table_exists(cur) is True
    assert cur.calls == 1


def test_a_failed_check_fails_safe_to_not_paying():
    _fresh()
    assert manager._kb_entities_table_exists(_Cursor(boom=True)) is False


def test_the_extraction_block_sits_behind_the_guard():
    src = inspect.getsource(manager)
    guard = src.index("if _kb_entities_table_exists(cursor):")
    extract = src.index("entity_extractor.extract_entities(")
    assert guard < extract, "the paid call must be inside the guard"
