"""F156 — the HARNESS task ledger is in the database.

Which done [HARNESS] board tasks self-management applied, and which wait for
an owner's or admin's /approve, lived in harness/applied_tasks.json on the
workspace volume: a path the generic workspace_write_file tool writes. A
prompted agent could seed task ids there, suppress the "needs /approve" note,
or have a change counted as applied. The ledger is now the harness_task_ledger
table, which only the HARNESS service writes. A workspace's old file is
imported once, the first time its table is empty, so a change applied before
the move is not applied again; a ledger or old file that cannot be read means
nothing is applied (/approve is refused).

PRD-263 US-016 deleted the weekly tick's card-to-Done path, which read the ledger
too; these cases now read it as an owner's /approve does (_read_applied_tasks).
"""
from __future__ import annotations

import json
from uuid import UUID

from sqlalchemy import text

from config import config


def test_the_ledger_round_trips_in_the_database(db_session, seed_workspace):
    from services.harness_service import HarnessService

    ws, other = UUID(seed_workspace()), UUID(seed_workspace())
    entry = {"task_id": "7", "change_type": "heartbeat_tune", "current_value_before": {"interval_minutes": 30}}
    HarnessService._write_applied_tasks(db_session, ws, [entry], ["9"])
    assert HarnessService._read_applied_tasks(db_session, ws) == {
        "applied_task_ids": ["7"], "needs_approve_task_ids": ["9"]}
    stored = db_session.execute(text("SELECT entry FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                                     "AND board_task_id = 7"), {"ws": str(ws)}).scalar()
    assert stored["current_value_before"] == {"interval_minutes": 30}

    # A held task approved later is applied; an applied one never goes back to held.
    HarnessService._write_applied_tasks(db_session, ws, [{"task_id": "9"}], ["7"])
    assert HarnessService._read_applied_tasks(db_session, ws) == {
        "applied_task_ids": ["7", "9"], "needs_approve_task_ids": []}
    assert HarnessService._read_applied_tasks(db_session, other) == {
        "applied_task_ids": [], "needs_approve_task_ids": []}


def _old_file(tmp_path, ws, content):
    folder = tmp_path / str(ws) / "harness"
    folder.mkdir(parents=True)
    (folder / "applied_tasks.json").write_text(content)


def _read(db, ws):
    """The ledger as an owner's /approve reads it: None means nothing may be applied."""
    from services.harness_service import HarnessService

    return HarnessService._read_applied_tasks(db, ws)


def test_the_old_file_is_imported_once_so_nothing_is_applied_twice(db_session, seed_workspace, monkeypatch, tmp_path):
    ws = UUID(seed_workspace())
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    _old_file(tmp_path, ws, json.dumps({"applied_task_ids": [7], "needs_approve_task_ids": ["9"],
                                        "entries": [{"task_id": "7", "current_value_before": {"interval_minutes": 30}}]}))
    ledger = {"applied_task_ids": ["7"], "needs_approve_task_ids": ["9"]}
    assert _read(db_session, ws) == ledger
    entry = db_session.execute(text("SELECT entry FROM harness_task_ledger WHERE workspace_id = CAST(:ws AS uuid) "
                                    "AND board_task_id = 7"), {"ws": str(ws)}).scalar()
    assert entry["current_value_before"] == {"interval_minutes": 30}
    # The table is the ledger from now on: what the file says later changes nothing.
    (tmp_path / str(ws) / "harness" / "applied_tasks.json").write_text(json.dumps({"applied_task_ids": []}))
    assert _read(db_session, ws) == ledger


def test_an_unreadable_ledger_applies_nothing(db_session, seed_workspace, monkeypatch, tmp_path):
    ws = UUID(seed_workspace())
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    db_session.execute(text("ALTER TABLE harness_task_ledger RENAME TO harness_task_ledger_away"))
    assert _read(db_session, ws) is None


def test_an_old_file_that_cannot_be_read_applies_nothing(db_session, seed_workspace, monkeypatch, tmp_path):
    ws = UUID(seed_workspace())
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    _old_file(tmp_path, ws, "{ not json")
    assert _read(db_session, ws) is None


def test_an_old_file_that_cannot_be_read_says_so_on_the_board_once(db_session, seed_workspace, monkeypatch, tmp_path):
    """Nothing is applied until the file is fixed or removed, and one blocked
    [HARNESS] card names the file and the fix; later reads file no second one."""
    from core.models.core import BoardTask
    from services.harness_service import LEGACY_LEDGER_FILE, UNREADABLE_LEDGER_TAG

    ws = UUID(seed_workspace())
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    _old_file(tmp_path, ws, "{ not json")
    assert _read(db_session, ws) is None
    assert _read(db_session, ws) is None
    cards = db_session.query(BoardTask).filter(BoardTask.workspace_id == ws,
                                               BoardTask.tags.contains([UNREADABLE_LEDGER_TAG])).all()
    assert [(card.status, card.title.startswith("[HARNESS]")) for card in cards] == [("blocked", True)]
    assert LEGACY_LEDGER_FILE in cards[0].description and "valid JSON" in cards[0].description
