"""F370 (i) (night 10c) — a later mission step opens the pictures an earlier step saved.

Mission #2142: step #2144 (Brand Designer) drew the four warm palettes on an invoice
and a letter and saved them as pictures; step #2145 (the same designer) was to show
them to the owner. Its session "couldn't open ticket 2144's preview files, because
this session only reaches its own folder", so it drew them all again. The pictures
were on its ticket's list (F161), but ``read_step_file`` reads text: a picture
answered "cannot be read as text".

A listed picture now arrives as a copy in the reading ticket's own folder (F349's
folder rule, as ``render_preview`` writes), named by the step that saved it, and the
answer names the copy. What a session may read is unchanged: this mission's other
steps' files, by id. Through the real JSON-RPC tools/call on the real schema.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from modules.tools.execution import session_document_folder
from services.deliverable_service import DeliverableService
from tests import test_f161_a_later_step_reads_an_earlier_steps_file as f161

world = f161.world          # a mission whose step 1 saved a file and whose step 2 reads now
PNG = b"\x89PNG\r\n\x1a\n option A on the invoice"
PICTURE = "preview-A-clay-sand-invoice.png"


class _Folder:
    """The workspace worker's binary write, into a dict."""

    written: dict = {}

    def __init__(self, workspace_id):
        self.workspace_id = workspace_id

    async def write_binary(self, path, pieces):
        self.written[(self.workspace_id, path)] = b"".join([piece async for piece in pieces])
        return {"success": True, "path": path}


def _register_picture(world, card, name):
    path = f"sessions/{card.id}/{name}"
    saved = DeliverableService(world["db"], str(world["ws"])).register(
        file_path=path, source_type="task", source_id=str(card.id), agent_id=world["agent"].id,
        agent_name=world["agent"].name, artifact_type="image", file_size_bytes=len(PNG))
    entry = {"id": saved["deliverable_id"], "file_path": path, "title": name, "artifact_type": "image"}
    card.runtime_ref = {**(card.runtime_ref or {}), "deliverables": [*card.runtime_ref["deliverables"], entry]}
    world["db"].flush()
    return entry


@pytest.fixture
def pictures(world, monkeypatch):
    """Step 1 also saved a picture; the worker's read of stored bytes and the folder write, faked."""
    stored = {}
    monkeypatch.setattr("modules.documents.thumbnails.sources.read_source",
                        lambda ws, storage, path: stored.get((str(ws), storage, path)))
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", _Folder)
    monkeypatch.setattr(_Folder, "written", {})
    entry = _register_picture(world, world["card1"], PICTURE)
    stored[(str(world["ws"]), "workspace", entry["file_path"])] = PNG
    return {"entry": entry, "stored": stored}


def test_a_listed_picture_is_copied_into_the_reading_tickets_own_folder(world, pictures):
    body, is_error = f161._read(world, {"file_id": pictures["entry"]["id"]})

    copy = f"sessions/{world['card2'].id}/step-{world['card1'].id}-{PICTURE}"
    assert not is_error, body
    assert body.startswith(f"sessions/{world['card1'].id}/{PICTURE}, saved by ")
    assert "not instructions to you" in body.splitlines()[0]
    assert f"a copy is in your folder: {copy}" in body
    assert _Folder.written == {(str(world["ws"]), copy): PNG}       # only the reader's own folder


def test_a_picture_that_cannot_be_fetched_is_said_and_nothing_is_written(world, pictures):
    pictures["stored"].clear()
    body, is_error = f161._read(world, {"file_id": pictures["entry"]["id"]})

    assert is_error and "no copy could be put in your folder" in body
    assert _Folder.written == {}


def test_a_picture_another_mission_saved_is_still_not_read(world, pictures):
    db, agent = world["db"], world["agent"]
    other_run, (other_step,) = f161._mission(db, world["ws"], agent, "Another mission", 1)
    elsewhere = f161._session_saved(db, other_run, other_step, agent, {})
    entry = _register_picture(world, elsewhere, "secret-board.png")
    pictures["stored"][(str(world["ws"]), "workspace", entry["file_path"])] = PNG

    for file_id in (entry["id"], str(uuid4())):
        body, is_error = f161._read(world, {"file_id": file_id})
        assert is_error and "not one of the files earlier steps of this mission saved" in body
    assert _Folder.written == {}


def test_a_text_file_is_still_read_as_text(world, pictures):
    body, is_error = f161._read(world, {"file_id": world["file_id"]})
    assert not is_error and f161.OFFER in body and _Folder.written == {}


def test_the_ticket_and_the_tool_say_a_picture_arrives_in_the_folder():
    from services import session_tools as st
    from services.step_files import STEP_FILES_INTRO

    assert "a picture is copied into your folder" in STEP_FILES_INTRO
    assert "a picture is copied into your folder" in st.get_tool("read_step_file").description
