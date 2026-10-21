"""PRD-265 — a dev onboarding reset never erases the credit ledger.

``wipe_built`` deletes what onboarding built by discovering every table with a
``workspace_id`` (``services/workspace_purge.purge_built_artifacts``). The credit
ledger and balances carry one, so without a survivor rule a reset would erase the
workspace's billing history and the grant that keeps the trial once per person.
They survive the reset (the trial is topped back up through the ledger instead);
full erasure still deletes them.
"""
from __future__ import annotations

import uuid

from services import workspace_purge


class _Result:
    def __init__(self, rows=None, rowcount=0):
        self._rows = rows or []
        self.rowcount = rowcount

    def fetchall(self):
        return self._rows


class _Savepoint:
    def commit(self):
        pass

    def rollback(self):
        pass


class _DB:
    """Answers table discovery with the given tables and records every DELETE."""

    def __init__(self, tables):
        self._tables = [(t,) for t in tables]
        self.deleted = set()

    def execute(self, statement, params=None):
        sql = str(statement)
        if "referential_constraints" in sql:
            return _Result()
        if "information_schema.columns" in sql:
            return _Result(rows=self._tables)
        if "DELETE FROM" in sql:
            self.deleted.add(sql.split('DELETE FROM "', 1)[1].split('"', 1)[0])
            return _Result(rowcount=1)
        return _Result()

    def begin_nested(self):
        return _Savepoint()


def test_a_reset_keeps_the_ledger_and_balances():
    db = _DB(["documents", "agents", "credit_ledger", "credit_balances"])
    workspace_purge.purge_built_artifacts(db, uuid.uuid4(), wipe_s3=False)
    assert "credit_ledger" not in db.deleted and "credit_balances" not in db.deleted
    assert "documents" in db.deleted  # the wipe itself still ran


def test_full_erasure_still_deletes_them():
    db = _DB(["credit_ledger", "credit_balances", "documents"])
    workspace_purge._delete_rows(db, uuid.uuid4())
    assert {"credit_ledger", "credit_balances"} <= db.deleted
