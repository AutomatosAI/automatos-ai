"""F191 (night 6) — a command that failed is not reported as a success.

At 03:30:07 agent 325 ran `python3 scripts/profile.py harbourline-club-members-
export-2026-09-26.csv --out sqa_out`. The script was not there, python3 exited 2
("can't open file"), and the call was logged as a success. The worker answers
/exec with HTTP 200 and the exit code in its body, and the executor marked every
workspace action without an ``error`` successful. workspace_exec now fails on
any non-zero exit except 1 with nothing on stderr, and keeps the command's
output. Exit 1 with nothing on stderr (grep found nothing) stays a success, and a
step that ends on one does not trip F131.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from modules.tools.execution import exec_workspace

PROFILE = {"exit_code": 2, "stdout": "", "duration_ms": 41,
           "stderr": "python3: can't open file '/workspace/scripts/profile.py': [Errno 2] No such file or directory"}


@pytest.fixture
def run(monkeypatch):
    import core.workspace_client as wc

    def _run(answer, command):
        async def exec_command(self, command, cwd=None, timeout=120):
            return dict(answer)

        async def list_dir(self, path="."):
            return {"path": path, "entries": []}

        monkeypatch.setattr(wc.WorkspaceClient, "exec_command", exec_command)
        monkeypatch.setattr(wc.WorkspaceClient, "list_dir", list_dir)
        return asyncio.run(exec_workspace.execute_workspace_action(
            None, "workspace_exec", {"command": command}, workspace_id=uuid4(), trace_id="t-f191"))
    return _run


def test_night_6s_missing_script_is_a_failure(run):
    result = run(PROFILE, "python3 scripts/profile.py harbourline-club-members-export-2026-09-26.csv --out sqa_out")

    assert result["success"] is False
    assert result["error"] == ("the command exited 2: python3: can't open file '/workspace/scripts/profile.py': "
                               "[Errno 2] No such file or directory")
    assert (result["exit_code"], result["stderr"]) == (2, PROFILE["stderr"])      # its output stays


def test_a_script_that_raised_is_a_failure(run):
    traceback = "Traceback (most recent call last):\n  File \"<string>\", line 1, in <module>\nZeroDivisionError: division by zero"
    result = run({"exit_code": 1, "stdout": "", "stderr": traceback}, "python3 -c '1/0'")
    assert result["success"] is False and result["error"] == "the command exited 1: ZeroDivisionError: division by zero"


def test_a_grep_that_found_nothing_is_a_no_and_its_step_has_not_failed(run):
    from api.recipe_executor import step_failure

    result = run({"exit_code": 1, "stdout": "0\n", "stderr": ""}, "grep -c 'Gull & Anchor' documents/orders.csv")

    assert result["success"] is True and result["exit_code"] == 1
    last_call = {"action": "workspace_exec", "success": result["success"], "result": result["stdout"]}
    assert step_failure([last_call], {"status": "success"}) is None                # F131 does not fire


def test_a_command_that_worked_is_a_success(run):
    result = run({"exit_code": 0, "stdout": "440\n", "stderr": ""}, "wc -l < documents/orders.csv")
    assert result["success"] is True and result["stdout"] == "440\n"


def test_a_command_killed_by_a_signal_is_a_failure():
    assert exec_workspace.exec_failure({"exit_code": -9, "stdout": "", "stderr": ""}) == (
        "the command exited -9: nothing on stderr")


# ── a skill's bundled script: the same exec, run by platform_run_skill_script ──

@pytest.mark.parametrize("answer, ok", [(PROFILE, False), ({"exit_code": 0, "stdout": "profiled", "stderr": ""}, True)],
                         ids=["exit-2", "exit-0"])
def test_a_skill_script_that_failed_is_not_a_success(monkeypatch, answer, ok):
    import core.services.skill_l3_execution as l3
    import core.workspace_client as wc
    import modules.agents.services.skill_portability as portability
    from modules.tools.discovery import handlers_skill_runtime as runtime
    from types import SimpleNamespace

    async def write_file(self, path, content):
        return {"success": True}

    async def exec_command(self, command, cwd=None, timeout=120):
        return dict(answer)

    monkeypatch.setattr(runtime, "_resolve_visible_skill",
                        lambda db, ws, name="", skill_id=None: SimpleNamespace(id=164, name="spreadsheet-qa",
                                                                               filesystem_path="/skills/sqa"))
    monkeypatch.setattr(l3, "is_l3_execution_enabled", lambda db, ws, skill_id: True)
    monkeypatch.setattr(portability, "collect_skill_bundle", lambda path: {"scripts/profile.py": "print('x')"})
    monkeypatch.setattr(wc.WorkspaceClient, "write_file", write_file)
    monkeypatch.setattr(wc.WorkspaceClient, "exec_command", exec_command)

    result = asyncio.run(runtime.run_skill_script(None, uuid4(), {"skill": "spreadsheet-qa", "script": "profile.py"}))

    assert result["success"] is ok and result["exit_code"] == answer["exit_code"]
    assert ("error" in result) is (not ok)
