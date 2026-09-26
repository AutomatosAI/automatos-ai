"""F179 (B) — publishing an image asks first, unless an owner or admin asked for it.

workspace_get_public_url puts a workspace image at a link anyone can open. It was
declared a read, and the workspace tools never met the confirmation gate at all,
so a ticket, mission or playbook run could publish on its own. It is a write with
requires_confirmation now, and a workspace tool clears the gates its definition
declares, the platform actions' own: an owner or admin who asks for it in chat is
the approval; every other lane gets a card that names the file and the link. The
hierarchy audit (F151) reads every file that registers actions, so it counts the
workspace tools too.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PATH = "content/social/instagram/launch.png"
IMAGE_ID = "11111111-2222-4333-8444-555555555555"
LINK = "/api/generated-images/<new id>"


def test_publishing_is_a_confirmed_write():
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get("workspace_get_public_url")
    assert (action.permission_level, action.requires_confirmation) == ("write", True)


# ── the audit ───────────────────────────────────────────────────────────────

def _audit():
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_audit_counts_the_workspace_tools():
    registrations = _audit().collect_registrations()
    assert registrations["workspace_get_public_url"].permission_level == "write"
    assert registrations["workspace_write_file"].permission_level == "write"


def test_an_ungated_write_in_any_file_fails_the_audit(tmp_path, capsys):
    audit = _audit()
    (tmp_path / "workspace_actions.py").write_text(
        "from .action_registry import ActionDefinition\n\n\ndef register(registry):\n"
        '    registry.register(ActionDefinition(name="workspace_publish_folder", permission_level="write"))\n')
    (tmp_path / "platform_executor.py").write_text("_HIERARCHY_TARGETS = {\n}\n")
    audit.ACTIONS_DIR, audit.EXECUTOR = tmp_path, tmp_path / "platform_executor.py"
    audit.ALLOW_LIST = set()
    assert audit.main() == 1
    assert "workspace_publish_folder [write]" in capsys.readouterr().err


# ── through the executor ────────────────────────────────────────────────────

def _person(db, ws, name, role):
    clerk = f"user_{name}_{uuid4().hex[:8]}"
    user = db.execute(text("INSERT INTO users (email, username, clerk_user_id) VALUES (:e, :u, :c) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}",
                       "c": clerk}).scalar()
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})
    return NS(id=user, clerk=clerk)


@pytest.fixture
def studio(db_session, seed_workspace, monkeypatch):
    db = db_session
    ws = UUID(seed_workspace())
    downloads, stored, announced, reads = [], [], [], []
    broken = {"on": False}

    class Worker:  # the workspace worker, holding one image
        def __init__(self, workspace_id):
            pass

        async def download_file(self, path):
            downloads.append(path)
            if broken["on"]:
                return {"success": False, "error": "File not found"}
            return {"success": True, "content": PNG}

        async def read_file(self, path):
            reads.append(path)
            return {"success": True, "content": "notes"}

    class Store:
        async def save_image(self, b64, mime_type="image/png", workspace_id=None):
            stored.append(mime_type)
            return IMAGE_ID

    monkeypatch.setattr("core.workspace_client.WorkspaceClient", Worker)
    monkeypatch.setattr("core.services.image_store.get_image_store", lambda: Store())
    monkeypatch.setattr("modules.tools.execution.tool_grants._notify_approval_pending",
                        lambda grant, workspace_id: announced.append(grant.reason))
    return NS(db=db, ws=ws, owner=_person(db, ws, "priya", "owner"), editor=_person(db, ws, "sam", "editor"),
              downloads=downloads, stored=stored, announced=announced, reads=reads, broken=broken)


def _publish(studio, caller_context):
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    return asyncio.run(UnifiedToolExecutor(db_session=studio.db).execute_tool(
        tool_name="workspace_get_public_url", parameters={"path": PATH},
        workspace_id=studio.ws, caller_context=caller_context))


def _for(person, **lane):
    return {"user_id": person.clerk, "driving_user_id": str(person.id), **lane}


@pytest.mark.parametrize("lane", [{"board_task_id": 41}, {"mission_id": "m-7"}, {"playbook_id": 12}],
                         ids=["ticket", "mission", "playbook"])
def test_a_run_gets_a_card_naming_the_image_and_its_link(studio, lane):
    reply = _publish(studio, _for(studio.owner, **lane))
    assert reply["requires_confirmation"] is True and isinstance(reply.get("grant_id"), int)
    for said in (reply["message"], *studio.announced):
        assert f"'{PATH}'" in said and LINK in said and "anyone with the link" in said
    assert len(studio.announced) == 1
    assert studio.downloads == [] and studio.stored == []


def test_the_owner_asking_in_chat_publishes_without_a_card(studio):
    reply = _publish(studio, _for(studio.owner, conversation_id="c-1"))
    assert reply["success"] is True and reply["human_directed"] is True
    assert reply["public_url"].endswith(f"/api/generated-images/{IMAGE_ID}")
    assert studio.stored == ["image/png"] and studio.announced == []


def test_an_editor_asking_in_chat_gets_the_card(studio):
    reply = _publish(studio, _for(studio.editor, conversation_id="c-1"))
    assert reply["requires_confirmation"] is True and f"'{PATH}'" in reply["message"]
    assert studio.stored == []


def _approved(studio):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    ask = _publish(studio, {"mission_id": "m-7"})
    grant = studio.db.get(ApprovalGrant, ask["grant_id"])
    grant_grant(grant, granted_by=f"user:{studio.owner.id}")
    studio.db.flush()
    return grant


def test_an_approved_card_publishes_the_image_once(studio):
    """One yes, one public link (the TESTER's call): a repeat asks again."""
    from api.approval_grants import _resume_tool_call

    grant = _approved(studio)
    asyncio.run(_resume_tool_call(studio.db, grant))
    assert grant.details["executed_result"]["success"] is True
    assert studio.downloads == [PATH] and studio.stored == ["image/png"]

    again = _publish(studio, {"mission_id": "m-7"})
    assert again["requires_confirmation"] is True and again["grant_id"] != grant.id
    assert studio.stored == ["image/png"]


def test_an_approval_whose_publish_did_nothing_is_given_back(studio):
    """Review MEDIUM: the file was gone, so nothing was published; the yes stays."""
    from api.approval_grants import _resume_tool_call
    from core.models.approval_grants import GrantStatus

    grant = _approved(studio)
    studio.broken["on"] = True
    asyncio.run(_resume_tool_call(studio.db, grant))
    assert grant.details["executed_result"]["success"] is False and studio.stored == []
    assert grant.status == GrantStatus.GRANTED.value

    studio.broken["on"] = False
    again = _publish(studio, {"mission_id": "m-7"})
    assert again["success"] is True and again["approved_via_grant_id"] == grant.id
    assert studio.stored == ["image/png"]


def test_a_publish_on_an_approved_card_is_marked_with_its_grant(studio):
    grant = _approved(studio)
    reply = _publish(studio, {"mission_id": "m-7"})
    assert reply["success"] is True and reply["approved_via_grant_id"] == grant.id


def test_a_file_tool_meets_the_gates_its_workspace_tool_declares(studio, monkeypatch):
    """Review MEDIUM: read_file, write_file and the other file tools dispatch workspace
    tools by those tools' names, so a gate one of them declares holds there too."""
    import dataclasses

    from modules.tools.discovery import get_action_registry
    from modules.tools.execution import exec_file_ops
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    registry = get_action_registry()
    read = registry.get("workspace_read_file")
    monkeypatch.setitem(registry._actions, "workspace_read_file", dataclasses.replace(read, requires_confirmation=True))
    reply = asyncio.run(exec_file_ops.execute_file_op(
        UnifiedToolExecutor(db_session=studio.db), "read_file", {"path": "notes/plan.md"}, 0,
        workspace_id=studio.ws, caller_context={"mission_id": "m-7"}))
    assert reply["requires_confirmation"] is True and studio.reads == []


# ── a floor below the full-autonomy dial ────────────────────────────────────

@pytest.fixture
def full_autonomy(monkeypatch):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    monkeypatch.setattr(PlatformActionExecutor, "_full_autonomy", lambda self: True)


def test_under_full_autonomy_a_run_still_gets_the_card(studio, full_autonomy):
    """The TESTER's call: a public link cannot be taken back, so the dial never skips this card."""
    reply = _publish(studio, {"mission_id": "m-7"})
    assert reply["requires_confirmation"] is True and studio.stored == []


def test_under_full_autonomy_the_owner_asking_in_chat_still_publishes(studio, full_autonomy):
    reply = _publish(studio, _for(studio.owner, conversation_id="c-1"))
    assert reply["success"] is True and reply["human_directed"] is True and "autonomous" not in reply


def test_the_floor_names_only_publishing():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import _dial_skips_the_card

    registry = get_action_registry()
    assert _dial_skips_the_card(registry.get("workspace_get_public_url"), True) is False
    assert _dial_skips_the_card(registry.get("platform_delete_memory"), True) is True
    assert _dial_skips_the_card(registry.get("platform_delete_memory"), False) is False
