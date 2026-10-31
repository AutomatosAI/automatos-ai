"""F460 (night 18): Auto could not open a file it had just listed.

``deliverables/sessions/3131/x.md`` was "File not found", ``/Users/<host>/Development/deliverables/
sessions/3103/…`` was "Absolute path not allowed", and ``harbour-lights-launch/1-club-email.md`` (relative
to the session's folder) was not there; ``sessions/3131/x.md`` opened. Now the read-only workspace tools
map a path the way a session's files are mapped (``result_files.worker_read_path``), never reaching past
the root, and platform_get_task lists the card's files as the worker reads them.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from modules.tools.discovery.card_files import FILES_KEY, FILES_NOTE
from modules.tools.execution import exec_workspace

ROOT = "/x/deliverables"
IN_THE_ROOT = "sessions/3131/a.md"
READS = ("workspace_read_file", "workspace_list_dir", "workspace_grep")


@pytest.fixture
def worker(monkeypatch):
    """The workspace worker as the client calls it: each path it is sent, by method."""
    import core.workspace_client as wc
    from config import config

    sent = []

    async def read_file(self, path):
        sent.append(("read", path))
        return {"success": True, "path": path, "name": path.rsplit("/", 1)[-1], "content": "Hello", "size": 5}

    async def list_dir(self, path="."):
        sent.append(("list", path))
        return {"success": True, "path": path, "entries": []}

    async def grep(self, pattern, path=".", include="", max_results=50):
        sent.append(("grep", path))
        return {"success": True, "matches": []}

    async def write_file(self, path, content):
        sent.append(("write", path))
        return {"success": True, "path": path}

    for name, method in (("read_file", read_file), ("list_dir", list_dir), ("grep", grep), ("write_file", write_file)):
        monkeypatch.setattr(wc.WorkspaceClient, name, method)
    monkeypatch.setattr(config, "AUTOMATOS_WORKSPACE_DIR", ROOT, raising=False)
    monkeypatch.setattr(config, "LOCAL_PROJECTS_DIR", "", raising=False)
    return sent


def _call(tool, path, **extra):
    params = {"path": path, "pattern": "club", "content": "x", **extra}
    asyncio.run(exec_workspace.execute_workspace_action(None, tool, params, workspace_id=uuid4()))


@pytest.mark.parametrize("tool", READS)
@pytest.mark.parametrize("path", [f"deliverables/{IN_THE_ROOT}", f"{ROOT}/{IN_THE_ROOT}", IN_THE_ROOT])
def test_a_path_written_from_the_root_or_the_host_reaches_the_worker_in_its_form(worker, tool, path):
    _call(tool, path)

    assert worker[0][1] == IN_THE_ROOT


@pytest.mark.parametrize("path", ["/etc/passwd", "../etc/passwd", f"{ROOT}/../../etc/passwd", "/x/other/a.md"])
def test_a_path_outside_the_root_is_never_mapped_into_it(worker, path):
    _call("workspace_read_file", path)

    assert worker[0][1] == path                                   # as it came: the worker's own check refuses it


def test_a_step_out_after_the_roots_name_still_steps_out(worker):
    _call("workspace_read_file", "deliverables/../../etc/passwd")

    assert worker[0][1] == "deliverables/../../etc/passwd"          # never rewritten: the worker refuses it
    _call("workspace_read_file", "deliverables/./sessions/3131/a.md")
    assert worker[1][1] == "deliverables/./sessions/3131/a.md"      # a dot step is not rewritten either


def test_a_write_goes_as_it_came(worker):
    _call("workspace_write_file", f"deliverables/{IN_THE_ROOT}")

    assert worker == [("write", f"deliverables/{IN_THE_ROOT}")]


# ── platform_get_task names the card's files ───────────────────────────────────

def _at(day):
    return datetime(2026, 10, day, 20, 0, tzinfo=timezone.utc)


@pytest.fixture
def board(db_session, seed_workspace):
    from tests import test_f161_a_later_step_reads_an_earlier_steps_file as f161

    db_session.execute(text(f161.OUTPUTS_VIEW))
    return db_session, UUID(seed_workspace())


def _card(db, ws, seq):
    from core.models.core import BoardTask

    card = BoardTask(workspace_id=ws, title="Harbour Lights Box launch wording", status="review", priority="medium",
                     workspace_seq=seq, result="Wrote harbour-lights-launch/1-club-email.md and 2-site-banner.md.")
    db.add(card)
    db.flush()
    return card


def _file(db, ws, card, path, day):
    db.execute(text(
        "INSERT INTO deliverables (workspace_id, source_type, source_id, artifact_type, title, file_path, created_at) "
        "VALUES (CAST(:ws AS uuid), 'task', :card, 'document', :title, :path, :made)"),
        {"ws": str(ws), "card": str(card.id), "title": path.rsplit("/", 1)[-1], "path": path, "made": _at(day)})
    db.flush()


def test_get_task_lists_the_files_the_card_records(board):
    from modules.tools.discovery.handlers_board_tasks import get_board_task

    db, ws = board
    card = _card(db, ws, 3125)
    _file(db, ws, card, "sessions/3125/harbour-lights-launch/1-club-email.md", 18)
    _file(db, ws, card, "sessions/3125/harbour-lights-launch/2-site-banner.md", 19)
    bare = _card(db, ws, 3126)

    answer = asyncio.run(get_board_task(db, ws, {"task_id": "#3125"}))

    assert answer["success"] is True and answer["task"][FILES_KEY] == [
        "sessions/3125/harbour-lights-launch/2-site-banner.md", "sessions/3125/harbour-lights-launch/1-club-email.md"]
    assert answer["task"]["files_note"] == FILES_NOTE
    none = asyncio.run(get_board_task(db, ws, {"task_id": f"#{bare.workspace_seq}"}))
    assert none["success"] is True and FILES_KEY not in none["task"]
