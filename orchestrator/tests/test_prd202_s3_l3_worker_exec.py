"""PRD-202 S3 (P2-21) — L3 script execution via the workspace worker.

Pins:
1. run_skill_script materializes the skill bundle into the worker and invokes
   WorkspaceClient.exec_command (NOT the in-process ActionExecutor).
2. Only the script OUTPUT (stdout/stderr) returns to context — never the source.
3. A non-enabled skill's script is refused (S4 gate: import != executable).

Pure/mocked — the workspace worker is mocked at the WorkspaceClient HTTP
boundary; the bundle + enablement are stubbed; no live worker, no network.
"""
import asyncio
import os
import pathlib
import sys
from unittest.mock import AsyncMock, MagicMock

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _skill_session(skill):
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.return_value = skill
    db = MagicMock()
    db.query.return_value = q
    return db


def _fake_skill(name="docx", fs="/tmp/skills/docx"):
    s = MagicMock()
    s.id = 55
    s.name = name
    s.filesystem_path = fs
    return s


def _install_worker_mock(monkeypatch, exec_result):
    """Patch WorkspaceClient at its source; return the mock client for asserts."""
    import core.workspace_client as wc

    client = MagicMock()
    client.write_file = AsyncMock(return_value={"success": True})
    client.exec_command = AsyncMock(return_value=exec_result)
    monkeypatch.setattr(wc, "WorkspaceClient", MagicMock(return_value=client))
    return client


def _patch_bundle(monkeypatch, bundle):
    import modules.agents.services.skill_portability as sp
    monkeypatch.setattr(sp, "collect_skill_bundle", lambda _p: bundle)


def _patch_enabled(monkeypatch, enabled):
    import core.services.skill_l3_execution as l3
    monkeypatch.setattr(l3, "is_l3_execution_enabled", lambda *a, **k: enabled)


# ---------------------------------------------------------------------------
# 1 + 2. worker exec + output-only
# ---------------------------------------------------------------------------

def test_run_skill_script_invokes_worker_exec(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import run_skill_script

    _patch_enabled(monkeypatch, True)
    _patch_bundle(monkeypatch, {"scripts/convert.py": "print('SECRET SOURCE do not leak')"})
    client = _install_worker_mock(monkeypatch, {
        "success": True, "stdout": "converted 3 files", "stderr": "", "exit_code": 0,
    })
    db = _skill_session(_fake_skill())

    result = asyncio.run(run_skill_script(db, "ws-1", {"skill": "docx", "script": "convert.py"}))

    assert result["success"] is True
    # bundle materialized into the worker filesystem
    assert client.write_file.await_count >= 1
    # the sandboxed worker exec was invoked (not the in-process executor)
    client.exec_command.assert_awaited_once()
    call = client.exec_command.await_args
    assert "scripts/convert.py" in call.kwargs["command"]
    assert call.kwargs["command"].startswith("python ")
    assert call.kwargs["cwd"] == ".skills/docx"


def test_run_skill_script_returns_output_only(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import run_skill_script

    _patch_enabled(monkeypatch, True)
    _patch_bundle(monkeypatch, {"scripts/convert.py": "print('SECRET SOURCE do not leak')"})
    _install_worker_mock(monkeypatch, {
        "success": True, "stdout": "the output", "stderr": "a warning", "exit_code": 0,
    })
    db = _skill_session(_fake_skill())

    result = asyncio.run(run_skill_script(db, "ws-1", {"skill": "docx", "script": "convert.py"}))

    assert result["stdout"] == "the output"
    assert result["stderr"] == "a warning"
    # the script SOURCE never enters the result
    blob = str(result)
    assert "SECRET SOURCE" not in blob


def test_run_skill_script_caps_output(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import run_skill_script

    _patch_enabled(monkeypatch, True)
    _patch_bundle(monkeypatch, {"scripts/x.py": "print('x')"})
    _install_worker_mock(monkeypatch, {
        "success": True, "stdout": "A" * 100000, "stderr": "", "exit_code": 0,
    })
    db = _skill_session(_fake_skill())

    result = asyncio.run(run_skill_script(db, "ws-1", {"skill": "docx", "script": "x.py"}))
    assert len(result["stdout"]) < 100000  # capped
    assert "truncated" in result["stdout"]


# ---------------------------------------------------------------------------
# 3. enablement gate (S4)
# ---------------------------------------------------------------------------

def test_l3_execution_requires_enablement(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import run_skill_script

    _patch_enabled(monkeypatch, False)  # NOT enabled for this workspace
    _patch_bundle(monkeypatch, {"scripts/convert.py": "print('x')"})
    client = _install_worker_mock(monkeypatch, {"success": True, "stdout": "x"})
    db = _skill_session(_fake_skill())

    result = asyncio.run(run_skill_script(db, "ws-1", {"skill": "docx", "script": "convert.py"}))

    assert result["success"] is False
    assert result.get("enablement_required") is True
    # the worker was never touched — a disabled skill's script never runs
    client.exec_command.assert_not_awaited()


def test_run_skill_script_unknown_script_refused(monkeypatch):
    from modules.tools.discovery.handlers_skill_runtime import run_skill_script

    _patch_enabled(monkeypatch, True)
    _patch_bundle(monkeypatch, {"scripts/convert.py": "print('x')"})
    _install_worker_mock(monkeypatch, {"success": True, "stdout": "x"})
    db = _skill_session(_fake_skill())

    result = asyncio.run(run_skill_script(db, "ws-1", {"skill": "docx", "script": "missing.py"}))
    assert result["success"] is False
    assert "not found" in result["error"].lower()
