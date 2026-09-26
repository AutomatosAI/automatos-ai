"""F151 — the hierarchy audit keys on each action's declared type.

scripts/check_hierarchy_gate.py checked only writes whose name carried a
per-target verb. It now checks every write and destructive action: each is
hierarchy-gated, admin_only / super_admin_only, or on ALLOW_LIST with its
reason. Nine writes it had not seen are an owner's or admin's now, as their
REST equivalents are (workspace:manage); two of them are also confirmed. The
chat has no API key create tool: a key's full value exists only in its create
response, so keys are created in Settings.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

ORCH = Path(__file__).resolve().parents[1]

# action: whether it is also confirmed
GATED = {
    "platform_revoke_api_key": True,
    "platform_connect_channel": False,  # F212: no card; its grant would store the credentials
    "platform_start_channel": False,
    "platform_stop_channel": False,
    "platform_uninstall_plugin": True,
    "platform_set_skill_script_execution": True,
    "platform_set_power_mode": False,
    "platform_create_routing_rule": False,
    "platform_install_package": False,
}


@pytest.mark.parametrize("name,confirmed", sorted(GATED.items()))
def test_the_write_is_an_owners_or_admins(name, confirmed):
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get(name)
    assert (action.admin_only, action.requires_confirmation) == (True, confirmed)


# ── through the executor ────────────────────────────────────────────────────

def _user(db, name):
    return db.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                      {"e": f"{name}-{uuid4().hex[:8]}@harbourline.test", "u": f"{name}-{uuid4().hex[:8]}"}).scalar()


@pytest.fixture
def cafe(db_session, seed_workspace):
    db = db_session
    ws = UUID(seed_workspace())
    admin, editor = _user(db, "priya"), _user(db, "sam")
    for user, role in ((admin, "admin"), (editor, "editor")):
        db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                        "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"), {"ws": str(ws), "user": user, "role": role})
    return NS(db=db, ws=ws, admin=admin, editor=editor)


def _run(cafe, action, params, caller_context):
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(cafe.db, cafe.ws)
    executor._full_autonomy = lambda: False
    handler = AsyncMock(return_value={"success": True, "handler": "ran"})
    executor._handlers[action] = handler
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(executor.execute(action, params, caller_context)), handler


def _for(user):
    return {"driving_user_id": str(user), "conversation_id": "c1"}


@pytest.mark.parametrize("action,params", [
    ("platform_revoke_api_key", {"key_id": str(uuid4())}),
    ("platform_connect_channel", {"platform": "telegram", "config": {}}),
    ("platform_set_skill_script_execution", {"skill_id": 1, "enabled": True}),
    ("platform_set_power_mode", {"power_mode": "max"}),
])
@pytest.mark.parametrize("caller", ["editor", "nobody"])
def test_an_editor_or_a_lane_for_nobody_is_refused(cafe, action, params, caller):
    context = _for(cafe.editor) if caller == "editor" else None
    reply, handler = _run(cafe, action, params, context)
    assert reply["success"] is False and reply.get("permission_denied") is True
    handler.assert_not_called()


def test_an_admins_new_channel_runs_with_no_card(cafe):
    """F212: connecting a channel asks for no card (its grant would store the
    credentials); admin_only is the gate, so an admin's call runs."""
    reply, handler = _run(cafe, "platform_connect_channel", {"platform": "telegram", "config": {}},
                          {"driving_user_id": str(cafe.admin)})
    assert reply == {"success": True, "handler": "ran"}
    handler.assert_called_once()


def test_the_chat_cannot_create_an_api_key():
    from modules.tools.discovery import get_action_registry
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    registry = get_action_registry()
    assert registry.get("platform_create_api_key") is None
    assert "platform_create_api_key" not in PlatformActionExecutor(None, None)._handlers
    assert "Settings → Widget SDK → API keys" in registry.get("platform_list_api_keys").description


def test_an_admin_sets_the_power_mode(cafe):
    reply, handler = _run(cafe, "platform_set_power_mode", {"power_mode": "max"}, _for(cafe.admin))
    assert reply == {"success": True, "handler": "ran"}


# ── the audit ───────────────────────────────────────────────────────────────

def _audit(tmp_path, registrations, *, allow=(), targets=""):
    spec = importlib.util.spec_from_file_location("check_hierarchy_gate", ORCH / "scripts" / "check_hierarchy_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "actions_cafe.py").write_text(
        "from .action_registry import ActionDefinition\n\n\ndef register(registry):\n" + registrations)
    (tmp_path / "platform_executor.py").write_text("_HIERARCHY_TARGETS = {\n" + targets + "\n}\n")
    module.ACTIONS_DIR, module.EXECUTOR = tmp_path, tmp_path / "platform_executor.py"
    module.ALLOW_LIST = set(allow)
    return module


def _registration(name, *lines):
    body = "".join(f"        {line}\n" for line in (f'name="{name}",', *lines))
    return f"    registry.register(ActionDefinition(\n{body}    ))\n"


def test_an_ungated_write_fails_whatever_its_name(tmp_path, capsys):
    audit = _audit(tmp_path, _registration("platform_mint_key", 'permission_level="write",'))
    assert audit.main() == 1
    assert "platform_mint_key [write]" in capsys.readouterr().err


def test_a_flag_named_in_a_comment_does_not_gate(tmp_path):
    audit = _audit(tmp_path, _registration("platform_mint_key", 'permission_level="write",',
                                           "# admin_only=True once the owner decides"))
    assert audit.main() == 1


def test_every_kind_of_gate_accounts_for_a_write(tmp_path):
    registrations = "".join([
        _registration("platform_mint_key", 'permission_level="write",', "admin_only=True,"),
        _registration("platform_set_rates", 'permission_level="write",', "super_admin_only=True,"),
        _registration("platform_update_menu", 'permission_level="destructive",'),
        _registration("platform_post_notice", 'permission_level="write",'),
        _registration("platform_read_menu"),  # read is ActionDefinition's default
    ])
    audit = _audit(tmp_path, registrations, allow={"platform_post_notice"},
                   targets='    "platform_update_menu": ("agent", "agent_id"),')
    assert audit.main() == 0


@pytest.mark.parametrize("stale", ["platform_mint_key", "platform_retired_tool"])
def test_a_gated_or_retired_allow_list_entry_fails(tmp_path, capsys, stale):
    audit = _audit(tmp_path, _registration("platform_mint_key", 'permission_level="write",', "admin_only=True,"),
                   allow={stale})
    assert audit.main() == 1
    assert f"  - {stale}\n" in capsys.readouterr().err


@pytest.mark.parametrize("registrations", [
    '    NAME = "platform_mint_key"\n'
    '    registry.register(ActionDefinition(name=NAME, permission_level="write"))\n',
    '    action = ActionDefinition(name="platform_mint_key", permission_level="write")\n'
    '    registry.register(action)\n',
    '    registry.register(make_action("platform_mint_key", permission_level="write"))\n',
], ids=["a name that is not a literal", "a definition built outside register", "a register fed by a factory"])
def test_a_registration_it_cannot_read_stops_the_audit(tmp_path, registrations):
    audit = _audit(tmp_path, registrations)
    with pytest.raises(SystemExit) as stopped:
        audit.main()
    assert stopped.value.code == 2


def test_a_registration_in_a_subfolder_is_audited(tmp_path):
    audit = _audit(tmp_path, "    pass\n")
    (tmp_path / "cafe").mkdir()
    (tmp_path / "cafe" / "actions_menu.py").write_text(
        "def register(registry):\n" + _registration("platform_mint_key", 'permission_level="write",'))
    assert audit.main() == 1
