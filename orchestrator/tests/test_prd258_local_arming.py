"""PRD-258 US-003 — Auto Live runs on the local stack through a tunnel.

* ``retell_api.build_agent_endpoints`` builds the agent's socket and webhook
  URLs from one host; creation and the re-arm both use it.
* A re-arm with an existing agent id sends ONE PATCH carrying the tuning AND
  both URLs for the current ``PUBLIC_API_HOST``, so an agent is moved to it.
* A local-edition first arm names the agent 'Auto Live (local)'; saas keeps
  'Auto Live'.
* ``PUBLIC_API_HOST`` is assigned once in config.py, the Railway-aware one.
* The tunnel script parses as bash.

No network: Retell's create/update calls are replaced by recorders.
"""
from __future__ import annotations

import ast
import asyncio
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_REPO_ROOT = _orchestrator_root.parent
_TUNNEL_HOST = "auto-live-test.example.dev"
_OLD_HOST = "api.example.dev"


def _admin_ctx():
    return SimpleNamespace(
        workspace_id=uuid.uuid4(),
        user=SimpleNamespace(id=7, system_role="admin"),
        auth_type="clerk",
    )


def _arm_env(monkeypatch, *, agent_id, host=_TUNNEL_HOST, edition="saas"):
    """Recorders for the arm route's collaborators; no DB, no Retell."""
    import api.voice_retell as vr
    from config import config
    from modules.voice.live_settings import RetellCredentials

    monkeypatch.setattr(config, "PUBLIC_API_HOST", host)
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(vr.live_settings, "set_voice_setting", lambda db, k, v: None)
    monkeypatch.setattr(
        vr.live_settings, "retell_credentials", lambda: RetellCredentials("key_k", "sec_k", agent_id)
    )
    calls = {"create": [], "patch": []}

    async def fake_create(api_key, **kwargs):
        calls["create"].append(kwargs)
        return "agent_new_1"

    async def fake_update(api_key, agent_id_arg, settings):
        calls["patch"].append((agent_id_arg, settings))

    monkeypatch.setattr(vr.retell_api, "create_custom_llm_agent", fake_create)
    monkeypatch.setattr(vr.retell_api, "update_agent", fake_update)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    return vr, calls, db


def _arm(vr, db):
    from api.voice_retell import ArmVoiceRequest

    return asyncio.run(vr.arm_voice_live(ArmVoiceRequest(), ctx=_admin_ctx(), db=db))


# ---------------------------------------------------------------------------
# The endpoints helper (pure)
# ---------------------------------------------------------------------------

def test_endpoints_are_built_from_the_host():
    from modules.voice.retell_api import build_agent_endpoints

    assert build_agent_endpoints(_TUNNEL_HOST) == {
        "response_engine": {
            "type": "custom-llm",
            "llm_websocket_url": f"wss://{_TUNNEL_HOST}/api/voice/retell/llm-websocket",
        },
        "webhook_url": f"https://{_TUNNEL_HOST}/api/voice/retell/events",
    }


def test_endpoints_drop_a_trailing_slash():
    from modules.voice.retell_api import build_agent_endpoints

    out = build_agent_endpoints(f" {_TUNNEL_HOST}/ ")
    assert out["webhook_url"] == f"https://{_TUNNEL_HOST}/api/voice/retell/events"


@pytest.mark.parametrize("bad", ["", "   ", f"https://{_TUNNEL_HOST}", f"{_TUNNEL_HOST}/api"])
def test_endpoints_refuse_an_empty_host_a_scheme_or_a_path(bad):
    from modules.voice.retell_api import build_agent_endpoints

    with pytest.raises(ValueError):
        build_agent_endpoints(bad)


# ---------------------------------------------------------------------------
# Re-arm: one PATCH with the tuning AND both URLs for the current host
# ---------------------------------------------------------------------------

def test_rearm_sends_one_patch_with_tuning_and_both_urls(monkeypatch):
    from modules.voice.retell_api import build_agent_tuning

    vr, calls, db = _arm_env(monkeypatch, agent_id="agent_existing")
    out = _arm(vr, db)

    assert out["agent_id"] == "agent_existing"
    assert calls["create"] == []
    assert len(calls["patch"]) == 1
    agent_id, body = calls["patch"][0]
    assert agent_id == "agent_existing"
    for key in build_agent_tuning():
        assert key in body, f"tuning key {key} missing from the re-arm PATCH"
    assert body["response_engine"] == {
        "type": "custom-llm",
        "llm_websocket_url": f"wss://{_TUNNEL_HOST}/api/voice/retell/llm-websocket",
    }
    assert body["webhook_url"] == f"https://{_TUNNEL_HOST}/api/voice/retell/events"


def test_rearm_follows_the_host_it_is_given(monkeypatch):
    """An agent armed against one host moves to whatever PUBLIC_API_HOST is now."""
    vr, calls, db = _arm_env(monkeypatch, agent_id="agent_existing", host=_OLD_HOST)
    _arm(vr, db)
    vr, calls_after, db = _arm_env(monkeypatch, agent_id="agent_existing", host=_TUNNEL_HOST)
    _arm(vr, db)

    assert calls["patch"][0][1]["webhook_url"].startswith(f"https://{_OLD_HOST}/")
    assert calls_after["patch"][0][1]["webhook_url"].startswith(f"https://{_TUNNEL_HOST}/")


def test_a_failing_rearm_patch_still_arms(monkeypatch):
    from modules.voice.retell_api import RetellApiError

    vr, calls, db = _arm_env(monkeypatch, agent_id="agent_existing")

    async def refuse(api_key, agent_id, settings):
        raise RetellApiError("Retell refused the agent update (HTTP 500)")

    monkeypatch.setattr(vr.retell_api, "update_agent", refuse)
    out = _arm(vr, db)
    assert out == {"armed": True, "platform_enabled": True, "agent_id": "agent_existing"}
    assert db.commit.called


def test_a_bad_public_host_rearms_with_the_tuning_alone(monkeypatch):
    """A re-arm never failed on the host before US-003; a bad host still arms,
    re-tunes, and leaves the agent's endpoints where they were."""
    from modules.voice.retell_api import build_agent_tuning

    vr, calls, db = _arm_env(monkeypatch, agent_id="agent_existing", host=f"https://{_TUNNEL_HOST}")
    out = _arm(vr, db)

    assert out["agent_id"] == "agent_existing"
    (_, body), = calls["patch"]
    assert body == build_agent_tuning()
    assert "response_engine" not in body and "webhook_url" not in body


def test_a_bad_public_host_refuses_a_first_arm_with_an_honest_500(monkeypatch):
    from fastapi import HTTPException

    vr, calls, db = _arm_env(monkeypatch, agent_id="", host=f"{_TUNNEL_HOST}/api")
    with pytest.raises(HTTPException) as exc:
        _arm(vr, db)
    assert exc.value.status_code == 500
    assert "PUBLIC_API_HOST" in exc.value.detail
    assert calls["create"] == [] and not db.commit.called


# ---------------------------------------------------------------------------
# First arm: the agent's name follows the edition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "edition, name", [("local", "Auto Live (local)"), ("saas", "Auto Live")]
)
def test_a_new_agent_is_named_for_its_edition(monkeypatch, edition, name):
    vr, calls, db = _arm_env(monkeypatch, agent_id="", edition=edition)
    out = _arm(vr, db)

    assert out["agent_id"] == "agent_new_1"
    assert calls["patch"] == []
    (kwargs,) = calls["create"]
    assert kwargs["agent_name"] == name
    assert kwargs["endpoints"]["webhook_url"] == f"https://{_TUNNEL_HOST}/api/voice/retell/events"


def test_agent_names_are_named_constants():
    from modules.voice import arming

    assert arming.AGENT_NAME_LOCAL == "Auto Live (local)"
    assert arming.AGENT_NAME_SAAS == "Auto Live"


# ---------------------------------------------------------------------------
# config.py: PUBLIC_API_HOST assigned once, the Railway-aware definition
# ---------------------------------------------------------------------------

def _public_api_host_assignments() -> list[ast.AST]:
    tree = ast.parse((_orchestrator_root / "config.py").read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "PUBLIC_API_HOST":
            found.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "PUBLIC_API_HOST" for t in node.targets
        ):
            found.append(node)
    return found


def test_public_api_host_is_assigned_once_and_reads_railway():
    assignments = _public_api_host_assignments()
    assert len(assignments) == 1, f"PUBLIC_API_HOST assigned {len(assignments)} times in config.py"
    assert "RAILWAY_PUBLIC_DOMAIN" in ast.unparse(assignments[0])


# ---------------------------------------------------------------------------
# The tunnel script (D3): parses, and commits no domain
# ---------------------------------------------------------------------------

def test_tunnel_script_parses_and_names_no_domain():
    script = _REPO_ROOT / "scripts" / "voice" / "tunnel.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    source = script.read_text()
    assert "NGROK_DOMAIN" in source
    assert ".ngrok-free.app" not in source and ".ngrok.app" not in source


def test_the_tunnel_lets_through_only_retells_routes():
    """On the local edition an anonymous request is the instance's super
    admin, so the tunnel must refuse everything but /api/voice/retell/."""
    from modules.voice.retell_api import RETELL_EVENTS_PATH, RETELL_LLM_WEBSOCKET_PATH

    voice_dir = _REPO_ROOT / "scripts" / "voice"
    assert "--traffic-policy-file" in (voice_dir / "tunnel.sh").read_text()
    policy = (voice_dir / "ngrok-policy.yml").read_text()
    assert "!req.url.path.startsWith('/api/voice/retell/')" in policy
    assert "type: deny" in policy
    for path in (RETELL_EVENTS_PATH, RETELL_LLM_WEBSOCKET_PATH):
        assert path.startswith("/api/voice/retell/")
