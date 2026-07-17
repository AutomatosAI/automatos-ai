"""PRD-221 S10 — digest_feedback model + endpoint validation (pure).

The migration itself is exercised by the alembic-from-zero CI lane; here we
lock the model shape (rating check constraint, workspace index) and the
endpoint's 422-on-bad-rating contract without a DB.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "core" or n.startswith("core."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import pytest  # noqa: E402


def test_digest_feedback_model_shape():
    from core.models.core import DigestFeedback

    assert DigestFeedback.__tablename__ == "digest_feedback"
    cols = DigestFeedback.__table__.columns
    assert "workspace_id" in cols and not cols["workspace_id"].nullable
    assert "state_hash" in cols and not cols["state_hash"].nullable
    assert "rating" in cols and not cols["rating"].nullable
    # rating check constraint present
    constraints = {c.name for c in DigestFeedback.__table__.constraints if c.name}
    assert "ck_digest_feedback_rating" in constraints
    # workspace index present
    index_names = {ix.name for ix in DigestFeedback.__table__.indexes}
    assert "ix_digest_feedback_workspace" in index_names


def test_migration_chains_on_single_head():
    # alembic/versions is not a package — read the revision file by path.
    import pathlib
    import re

    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "alembic" / "versions" / "prd221_digest_feedback.py"
    )
    src = path.read_text(encoding="utf-8")
    rev = re.search(r"^revision\s*=\s*['\"]([^'\"]+)['\"]", src, re.M)
    down = re.search(r"^down_revision\s*=\s*['\"]([^'\"]+)['\"]", src, re.M)
    assert rev and rev.group(1) == "prd221_digest_feedback"
    # chains directly on the current single head (Auto Speaks), no second join
    assert down and down.group(1) == "prd205_auto_speaks"


def test_feedback_endpoint_rejects_bad_rating():
    """The handler raises 422 for rating outside {-1, 1} BEFORE any DB write,
    and persists a row for a valid rating."""
    import asyncio
    from unittest.mock import MagicMock

    from fastapi import HTTPException
    from api.activity import DigestFeedbackRequest, submit_digest_feedback

    ctx = MagicMock(workspace_id="ws-1", clerk_user_id="user-1")

    # bad rating → 422, no DB write
    db_bad = MagicMock()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(submit_digest_feedback(
            DigestFeedbackRequest(state_hash="h", rating=0), db=db_bad, ctx=ctx))
    assert ei.value.status_code == 422
    db_bad.add.assert_not_called()
    db_bad.commit.assert_not_called()

    # valid rating → persists exactly one row
    db_ok = MagicMock()
    result = asyncio.run(submit_digest_feedback(
        DigestFeedbackRequest(state_hash="h", rating=1), db=db_ok, ctx=ctx))
    assert result["ok"] is True
    db_ok.add.assert_called_once()
    db_ok.commit.assert_called_once()
    persisted = db_ok.add.call_args[0][0]
    assert persisted.state_hash == "h"
    assert persisted.rating == 1
    assert persisted.workspace_id == "ws-1"
