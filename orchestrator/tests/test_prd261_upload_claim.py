"""PRD-261 C-02a — ``services/themes/upload_claim.py``: one upload of the same bytes at a time.

The route's 409 for an overlapping upload is proven in ``test_prd261_theme_import.py``; these
prove the claim itself: exclusive per (workspace, digest), released when the block exits, and on
Postgres the transaction-scoped advisory lock decides.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes.upload_claim import UPLOAD_LOCK_NAMESPACE, claim_key, upload_claim  # noqa: E402

DIGEST = "sha256:" + "a" * 64


class _Db:
    """A session stand-in: its dialect, and the advisory lock's answer on Postgres."""

    def __init__(self, dialect="sqlite", locked=True):
        self.dialect = dialect
        self.locked = locked
        self.statements = []

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name=self.dialect))

    def execute(self, statement, params):
        self.statements.append((str(statement), params))
        return SimpleNamespace(scalar=lambda: self.locked)


def test_a_second_claim_on_the_same_bytes_is_refused_until_the_first_ends():
    workspace = uuid4()
    with upload_claim(_Db(), workspace, DIGEST) as first:
        with upload_claim(_Db(), workspace, DIGEST) as second:
            assert (first, second) == (True, False)
    with upload_claim(_Db(), workspace, DIGEST) as again:
        assert again


def test_other_bytes_or_another_workspace_are_never_held_up():
    workspace = uuid4()
    with upload_claim(_Db(), workspace, DIGEST) as first:
        with upload_claim(_Db(), workspace, "sha256:" + "b" * 64) as other_bytes:
            with upload_claim(_Db(), uuid4(), DIGEST) as other_workspace:
                assert (first, other_bytes, other_workspace) == (True, True, True)


def test_the_claim_is_released_when_the_upload_raises():
    workspace = uuid4()
    try:
        with upload_claim(_Db(), workspace, DIGEST):
            raise RuntimeError("the upload failed")
    except RuntimeError:
        pass
    with upload_claim(_Db(), workspace, DIGEST) as again:
        assert again


def test_on_postgres_the_advisory_lock_decides():
    workspace = uuid4()
    held = _Db("postgresql", locked=False)
    with upload_claim(held, workspace, DIGEST) as claimed:
        assert not claimed
    ((statement, params),) = held.statements
    assert "pg_try_advisory_xact_lock" in statement
    assert params == {"namespace": UPLOAD_LOCK_NAMESPACE, "key": claim_key(workspace, DIGEST)}
    with upload_claim(_Db("postgresql"), workspace, DIGEST) as claimed:
        assert claimed


def test_without_postgres_no_lock_statement_is_sent():
    db = _Db("sqlite")
    with upload_claim(db, uuid4(), DIGEST) as claimed:
        assert claimed and db.statements == []
