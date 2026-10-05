"""F333 (night 10) — a second workspace's session files open, pass the card check and reach the next step.

A CLI host paired to a second workspace (root ``~/Development/deliverables-sim``) wrote
``~/Development/deliverables-sim/sessions/2050/…html``. The backend maps that to
``projects/deliverables-sim/sessions/2050/…html`` (the owner's projects folder) but asked
the worker for it under the SECOND workspace, while compose mounts the projects folder
only under the default one. With every file on disk: the Deliverables link answered 404
"File not found", the card check sent #2051, #2053, #2061… to review ("the workspace has no
such file"), and ``read_step_file`` said "File not found" (#2058, #2059, #2090–#2092).

The real workspace worker runs here over a temporary volume laid out as the local stack
mounts it: ``<volume>/<default>/projects/`` is the projects folder. No database.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from types import SimpleNamespace as NS

import pytest

from api import workspace_files
from core import local_projects_mount as mount
from core import workspace_client
from services import cli_host_service as svc
from services import result_files
from tests.helpers_workspace_worker import worker_server

DEFAULT_WS = "00000000-0000-0000-0000-0000000000c1"
SECOND_WS = str(uuid.UUID("febae41b-0000-4000-8000-000000000333"))
PROJECTS_DIR = "/Users/me/Development"
DELIVERABLES_DIR = "/Users/me/Development/deliverables"
SIM_SESSION = f"{PROJECTS_DIR}/deliverables-sim/sessions/2050"
SAVED = "projects/deliverables-sim/sessions/2050/weekly-report.html"
PAGE = "<h1>Week 40</h1>"


def _local_stack(monkeypatch, edition="local", projects_dir=PROJECTS_DIR):
    monkeypatch.setattr(mount.config, "AUTH_EDITION", edition)
    monkeypatch.setattr(mount.config, "LOCAL_PROJECTS_DIR", projects_dir)
    monkeypatch.setattr(mount.config, "DEFAULT_WORKSPACE_ID", DEFAULT_WS)
    monkeypatch.setattr(mount.config, "AUTOMATOS_WORKSPACE_DIR", DELIVERABLES_DIR)
    monkeypatch.setattr(mount.config, "WORKER_INTERNAL_TOKEN", "")


def _saved_on_disk(volume):
    """The session's file where the local stack's projects mount puts it."""
    target = volume / DEFAULT_WS / SAVED
    target.parent.mkdir(parents=True)
    target.write_text(PAGE)
    (volume / DEFAULT_WS / "reports").mkdir()
    (volume / DEFAULT_WS / "reports" / "private.md").write_text("the default workspace's own report")


def _against_the_worker(monkeypatch, volume, work):
    """Run ``work()`` with the backend's worker client pointed at a live worker over ``volume``."""
    async def run():
        async with worker_server(monkeypatch, volume) as (base, _http):
            monkeypatch.setattr(workspace_client.config, "WORKER_INTERNAL_URL", base)
            monkeypatch.setattr(workspace_client, "_client", None)
            try:
                return await work()
            finally:
                await workspace_client._get_client().aclose()
    return asyncio.run(run())


# ── what the owner saw: the Deliverables link, the card check, the next step ──

def test_the_deliverables_link_of_a_second_workspace_opens_the_file(monkeypatch, tmp_path):
    _local_stack(monkeypatch)
    _saved_on_disk(tmp_path)
    ctx = NS(workspace_id=SECOND_WS)

    body = _against_the_worker(monkeypatch, tmp_path, lambda: workspace_files.get_file_content(
        workspace_id=SECOND_WS, path=SAVED, ctx=ctx))

    assert body["success"] is True and body["content"] == PAGE


def test_a_later_mission_step_reads_the_file_an_earlier_step_saved(monkeypatch, tmp_path):
    """``read_step_file`` reads a registered file through ``WorkspaceClient.read_file``."""
    _local_stack(monkeypatch)
    _saved_on_disk(tmp_path)

    read = _against_the_worker(monkeypatch, tmp_path,
                               lambda: workspace_client.WorkspaceClient(SECOND_WS).read_file(SAVED))

    assert read["success"] is True and read["content"] == PAGE


def test_the_card_check_finds_the_file_and_still_catches_one_that_is_not_there(monkeypatch, tmp_path):
    _local_stack(monkeypatch)
    _saved_on_disk(tmp_path)
    task = NS(id=2050, runtime_ref={"host_id": "sim-host", "cwd": SIM_SESSION})
    found = "Saved the report: `" + SIM_SESSION + "/weekly-report.html`."
    invented = "Saved the report: `" + SIM_SESSION + "/never-written.html`."

    async def both():
        return (await result_files.check_named_files(task, found, SECOND_WS, projects_dir=PROJECTS_DIR),
                await result_files.check_named_files(task, invented, SECOND_WS, projects_dir=PROJECTS_DIR))

    on_disk, missing = _against_the_worker(monkeypatch, tmp_path, both)

    assert on_disk is None                       # the card closes done, no "no such file"
    assert missing is not None and missing.review and "never-written.html" in missing.note


# ── what stays as it was ─────────────────────────────────────────────────────

@pytest.mark.parametrize("escaping", [
    "projects/../reports/private.md", "projects/./../reports/private.md", "projects//../reports/private.md",
    "/projects/deliverables-sim/x.html", "./projects/deliverables-sim/x.html", "projectsX/a.md", "reports/a.md",
])
def test_a_path_that_is_not_plainly_under_projects_stays_in_its_own_workspace(monkeypatch, escaping):
    _local_stack(monkeypatch)
    assert mount.worker_workspace_for(SECOND_WS, escaping) == SECOND_WS


def test_a_path_stepping_out_of_projects_never_reads_the_default_workspace(monkeypatch, tmp_path):
    _local_stack(monkeypatch)
    _saved_on_disk(tmp_path)

    read = _against_the_worker(monkeypatch, tmp_path, lambda: workspace_client.WorkspaceClient(SECOND_WS)
                               .read_file("projects/../reports/private.md"))

    assert read["success"] is False and "own report" not in str(read)


@pytest.mark.parametrize("edition, projects_dir", [("saas", PROJECTS_DIR), ("local", "")])
def test_the_hosted_edition_and_a_stack_without_a_projects_folder_are_unchanged(
        monkeypatch, tmp_path, edition, projects_dir):
    _local_stack(monkeypatch, edition=edition, projects_dir=projects_dir)
    _saved_on_disk(tmp_path)

    read = _against_the_worker(monkeypatch, tmp_path,
                               lambda: workspace_client.WorkspaceClient(SECOND_WS).read_file(SAVED))

    assert mount.worker_workspace_for(SECOND_WS, SAVED) == SECOND_WS
    assert read["success"] is False and read.get("status_code") == 404


def test_the_default_workspace_reads_its_projects_folder_as_before(monkeypatch):
    _local_stack(monkeypatch)
    assert mount.worker_workspace_for(DEFAULT_WS, SAVED) == DEFAULT_WS
    assert mount.worker_workspace_for(SECOND_WS, SAVED) == DEFAULT_WS


# ── a session in a folder the platform cannot read says so when it starts ─────

def test_a_session_outside_every_readable_folder_says_so_once(monkeypatch, caplog):
    _local_stack(monkeypatch)
    caplog.set_level(logging.WARNING)
    ref = {}
    task = NS(id=2099, workspace_id=SECOND_WS)

    svc._record_session_cwd(ref, task, "/Volumes/elsewhere/sim/sessions/2099")
    svc._record_session_cwd(ref, task, "/Volumes/elsewhere/sim/sessions/2099")   # the result reports it again

    notes = ref[svc.SESSION_NOTES_KEY]
    assert ref["explorer_root"] is None and len(notes) == 1
    assert "/Volumes/elsewhere/sim/sessions/2099" in notes[0]["note"] and "cannot read" in notes[0]["note"]
    assert DELIVERABLES_DIR in notes[0]["note"] and PROJECTS_DIR in notes[0]["note"]
    assert sum("outside every folder the platform reads" in r.getMessage() for r in caplog.records) == 1


def test_a_session_under_the_projects_folder_carries_no_such_note(monkeypatch):
    _local_stack(monkeypatch)
    ref = {}

    svc._record_session_cwd(ref, NS(id=2050, workspace_id=SECOND_WS), SIM_SESSION)

    assert ref["explorer_root"] == "projects/deliverables-sim/sessions/2050"
    assert svc.SESSION_NOTES_KEY not in ref
