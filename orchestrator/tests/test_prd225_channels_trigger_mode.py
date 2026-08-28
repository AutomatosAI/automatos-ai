"""PRD-225 P225-RVW-7 — a config-only channel update must not drop trigger_mode.

Pure test: ``update_channel`` runs raw SQL (``db.execute(text(...))``), so a
small fake db intercepts the existence SELECT, the ``SELECT config`` read, and
the UPDATE (whose params it captures). No Postgres.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


class _Result:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeChannelDB:
    """Intercepts update_channel's raw SQL: the existence SELECT, the
    ``SELECT config`` read, and the UPDATE (whose params it captures)."""

    def __init__(self, stored_config):
        self.stored_config = stored_config
        self.update_params = None
        self.committed = False

    def execute(self, statement, params=None):
        sql = " ".join(str(statement).split()).upper()
        if sql.startswith("SELECT ID FROM CHANNEL_CONNECTIONS"):
            return _Result(SimpleNamespace(id="chan-1"))
        if sql.startswith("SELECT CONFIG FROM CHANNEL_CONNECTIONS"):
            return _Result(SimpleNamespace(config=self.stored_config))
        if sql.startswith("UPDATE CHANNEL_CONNECTIONS"):
            self.update_params = params
            return _Result(None)
        return _Result(None)

    def commit(self):
        self.committed = True


def _written_config(db):
    assert db.update_params is not None, "expected an UPDATE to run"
    return json.loads(db.update_params["config"])


@pytest.mark.asyncio
async def test_config_only_update_preserves_stored_trigger_mode():
    """PUT {config:{bot_token:'NEW'}} against a channel stored as allow_all keeps
    allow_all — a creds edit must not silently reset the mode to strict."""
    from api.channels import update_channel

    db = _FakeChannelDB({"bot_token": "OLD", "trigger_mode": "allow_all"})
    ctx = SimpleNamespace(workspace_id=uuid.uuid4())
    res = await update_channel("chan-1", {"config": {"bot_token": "NEW"}}, ctx=ctx, db=db)

    assert res == {"status": "updated"}
    written = _written_config(db)
    assert written["trigger_mode"] == "allow_all"  # preserved across the edit
    assert written["bot_token"] == "NEW"           # creds actually updated
    assert db.committed


@pytest.mark.asyncio
async def test_explicit_trigger_mode_overrides_stored():
    """An explicit trigger_mode in the payload still wins over the stored one."""
    from api.channels import update_channel

    db = _FakeChannelDB({"bot_token": "OLD", "trigger_mode": "allow_all"})
    ctx = SimpleNamespace(workspace_id=uuid.uuid4())
    await update_channel(
        "chan-1",
        {"config": {"bot_token": "NEW"}, "trigger_mode": "communication_only"},
        ctx=ctx, db=db,
    )
    written = _written_config(db)
    assert written["trigger_mode"] == "communication_only"
    assert written["bot_token"] == "NEW"


@pytest.mark.asyncio
async def test_modeless_channel_config_edit_materializes_no_mode():
    """A channel that never set a mode isn't handed an explicit 'strict' on a
    config edit — it stays modeless (still strict by default), no clutter."""
    from api.channels import update_channel

    db = _FakeChannelDB({"bot_token": "OLD"})  # no trigger_mode key
    ctx = SimpleNamespace(workspace_id=uuid.uuid4())
    await update_channel("chan-1", {"config": {"bot_token": "NEW"}}, ctx=ctx, db=db)
    written = _written_config(db)
    assert "trigger_mode" not in written  # no materialized default
    assert written["bot_token"] == "NEW"
