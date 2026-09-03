"""PRD-233 — a brand-new local install starts Auto-led onboarding.

Found 2026-09-03 on the isolated lab: the operator workspace was inserted
stage-less, then the container's `alembic upgrade heads` ran PRD-222's veteran
backfill (which stamps every stage-less workspace `skipped`) — so the onboarding
chat never appeared on a fresh install. The seed now inserts the workspace with
an explicit `not_started` document, which the backfill leaves alone.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

from core.seeds.seed_local_first_run import FRESH_INSTALL_ONBOARDING, _ensure_workspace


def _db(rowcount: int):
    db = MagicMock()
    db.execute.return_value.rowcount = rowcount
    return db


def test_fresh_workspace_is_inserted_at_not_started():
    db = _db(1)
    assert _ensure_workspace(db, uuid4()) == "created"
    sql, params = db.execute.call_args.args
    assert "onboarding" in str(sql) and "CAST(:onboarding AS jsonb)" in str(sql)
    doc = json.loads(params["onboarding"])
    assert doc == {"stage": "not_started", "stages": {}, "segment": {}}
    assert json.loads(FRESH_INSTALL_ONBOARDING)["stage"] == "not_started"


def test_existing_workspace_is_left_alone():
    db = _db(0)
    assert _ensure_workspace(db, uuid4()) == "present"
    assert "ON CONFLICT (id) DO NOTHING" in str(db.execute.call_args.args[0])


def test_entrypoint_inserts_the_default_workspace_at_not_started():
    """The entrypoint creates the default workspace BEFORE the migration replay
    (found on the lab 2026-09-03: a fresh boot still read `skipped` after the
    seed fix alone) — its INSERT must carry the stage too."""
    import pathlib

    src = (pathlib.Path(__file__).resolve().parents[2] / "docker-entrypoint.sh").read_text()
    insert = next(line for line in src.splitlines() if "INSERT INTO workspaces" in line)
    assert "onboarding" in insert and "not_started" in insert
