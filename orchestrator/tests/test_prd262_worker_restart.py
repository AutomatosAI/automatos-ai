"""PRD-262 US-008: the smoke step that kills a uvicorn worker and needs a replacement.

scripts/ci/smoke_worker_restart.py runs against the production-stage backend that
smoke-images' pull-request variant boots through compose. The step itself needs Docker
and runs only in CI; these tests hold its logic:

- it finds the uvicorn process and its ``--workers`` children (multiprocessing's
  ``spawn_main``), never the ``sh -c "alembic … && uvicorn …"`` around them;
- it refuses a single uvicorn process (no manager to restart it: WEB_CONCURRENCY=2);
- a replacement counts only when it is new, logged as ``Started server process``, and
  the worker count is back; the killed worker must be gone;
- the run fails when nothing replaces the worker, when /health stays down, or when the
  container itself restarted;
- the process listing it runs inside the container reads /proc correctly;
- smoke-fresh-clone.sh runs the step in smoke-images' pull-request variant (prebuilt),
  right after the worker-count check, and that variant watches the step's file.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "ci" / "smoke_worker_restart.py"
_SMOKE = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_SMOKE_IMAGES = _REPO / ".github" / "workflows" / "smoke-images.yml"
_spec = importlib.util.spec_from_file_location("smoke_worker_restart", _SCRIPT)
restart = importlib.util.module_from_spec(_spec)
sys.modules["smoke_worker_restart"] = restart
_spec.loader.exec_module(restart)

SPAWN = "/usr/local/bin/python3.11 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=5, pipe_handle=7) --multiprocessing-fork"
UVICORN = "/usr/local/bin/python3.11 /usr/local/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4"
TRACKER = "/usr/local/bin/python3.11 -c from multiprocessing.resource_tracker import main;main(4)"
STATE = {"restart_count": "0", "started_at": "2026-10-18T10:00:00Z"}


def _rows(workers=(21, 22, 23, 24), manager=12):
    return [
        {"pid": 1, "ppid": 0, "args": "/bin/sh /usr/local/bin/docker-entrypoint.sh sh -c alembic upgrade heads && uvicorn main:app --workers 4"},
        {"pid": 11, "ppid": 1, "args": "sh -c alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4"},
        {"pid": manager, "ppid": 11, "args": UVICORN},
        {"pid": 13, "ppid": manager, "args": TRACKER},
    ] + [{"pid": pid, "ppid": manager, "args": SPAWN} for pid in workers]


def _log(*pids):
    return "".join(f"INFO:     Started server process [{pid}]\nINFO:     Waiting for application startup.\n" for pid in pids)


def test_the_workers_are_the_uvicorn_processs_spawned_children():
    assert restart.uvicorn_workers(_rows()) == {"manager": 12, "workers": [21, 22, 23, 24], "servers": 1}


def test_two_uvicorn_processes_or_none_name_no_manager():
    rows = _rows() + [{"pid": 40, "ppid": 1, "args": UVICORN}]
    assert restart.uvicorn_workers(rows)["manager"] is None
    assert restart.uvicorn_workers([r for r in _rows() if r["pid"] != 12])["manager"] is None


def test_the_started_lines_give_the_pids():
    assert restart.started_pids(_log(21, 22) + "Child process [22] died\n" + _log(57)) == [21, 22, 57]


def test_a_single_process_has_no_worker_to_kill():
    assert restart.pick_victim([12]) is None
    assert restart.pick_victim([21, 22]) == 22


@pytest.mark.parametrize(
    "now, logged, expected",
    [
        ([21, 22, 23], [21, 22, 23, 24, 57], None),  # not replaced yet
        ([21, 22, 23, 57], [21, 22, 23, 24], None),  # spawned, not started yet
        ([21, 22, 23, 57], [21, 22, 23, 24, 57], 57),
        ([21, 22, 23, 24], [21, 22, 23, 24], None),  # the kill did not take
        ([21, 22, 23, 57, 58], [57, 58], None),  # the count is not the same
    ],
)
def test_a_replacement_is_new_logged_and_restores_the_count(now, logged, expected):
    assert restart.replacement([21, 22, 23, 24], now, 24, logged) == expected


def test_a_moved_restart_count_or_start_time_is_a_restart():
    assert restart.restart_problem(STATE, dict(STATE)) is None
    assert "restarted" in restart.restart_problem(STATE, {**STATE, "restart_count": "1"})
    assert "restarted" in restart.restart_problem(STATE, {**STATE, "started_at": "2026-10-18T10:01:00Z"})


class _Container:
    """The docker calls the step makes, against a scripted container."""

    def __init__(self, monkeypatch, *, replaced=True, healthy=True, restarts=False, workers=(21, 22, 23, 24)):
        self.killed = []
        self.states = [STATE, {**STATE, "restart_count": "1"} if restarts else STATE]
        self.workers, self.replaced, self.healthy = list(workers), replaced, healthy
        for name, fake in (("container_state", self.state), ("container_processes", self.processes),
                           ("container_log", self.log), ("kill_worker", self.kill), ("health_ok", self.health)):
            monkeypatch.setattr(restart, name, fake)
        monkeypatch.setattr(restart, "POLL_SECONDS", 0)

    def state(self, _container):
        return self.states.pop(0)

    def processes(self, _container):
        alive = [pid for pid in self.workers if pid not in self.killed]
        return _rows(alive + ([57] if self.killed and self.replaced else []))

    def log(self, _container):
        return _log(*self.workers, *([57] if self.killed and self.replaced else []))

    def kill(self, _container, pid):
        self.killed.append(pid)

    def health(self, _url):
        return self.healthy


def test_a_killed_worker_that_comes_back_passes(monkeypatch):
    container = _Container(monkeypatch)
    report = restart.run("automatos_backend", "http://localhost:8000/health", 5)
    assert container.killed == [24]
    assert report["failures"] == [] and report["replacement"] == 57
    assert report["replaced_seconds"] is not None and report["healthy_seconds"] is not None


@pytest.mark.parametrize(
    "scripted, failure",
    [
        ({"replaced": False}, "no new worker logged 'Started server process'"),
        ({"healthy": False}, "did not answer 200"),
        ({"restarts": True}, "the container restarted"),
    ],
)
def test_no_replacement_no_health_or_a_restart_fails(monkeypatch, scripted, failure):
    _Container(monkeypatch, **scripted)
    report = restart.run("automatos_backend", "http://localhost:8000/health", 0.2)
    assert any(failure in text for text in report["failures"]), report["failures"]


def test_one_worker_is_refused_before_anything_is_killed(monkeypatch):
    container = _Container(monkeypatch, workers=(21,))
    report = restart.run("automatos_backend", "http://localhost:8000/health", 5)
    assert container.killed == []
    assert "WEB_CONCURRENCY=2" in report["failures"][0]


def test_the_report_and_the_exit_code(monkeypatch, tmp_path):
    _Container(monkeypatch, replaced=False)
    out, summary = tmp_path / "restart.json", tmp_path / "summary.md"
    code = restart.main(["--container", "c", "--health-url", "http://h/health", "--timeout", "0.2",
                         "--out", str(out), "--summary", str(summary)])
    assert code == 1
    assert json.loads(out.read_text())["victim"] == 24
    assert "FAILED" in summary.read_text()


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="reads Linux /proc, as the container does (CI is Linux)")
def test_the_process_listing_reads_proc():
    """The snippet the step runs inside the container, run here: it lists itself under this process."""
    proc = subprocess.run([sys.executable, "-c", restart.LIST_PROCESSES], capture_output=True, text=True, check=True)
    children = [row for row in json.loads(proc.stdout) if row["ppid"] == os.getpid()]
    assert any("import json, os" in row["args"] for row in children), children


# --------------------------------------------------------------------------- the wiring


def _shell_function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_kills_a_worker_in_the_pull_request_variant_after_the_worker_count():
    text = _SMOKE.read_text(encoding="utf-8")
    assert 'RESTART_TOOL="scripts/ci/smoke_worker_restart.py"' in text
    step = _shell_function(text, "check_worker_restart")
    assert step.lstrip().startswith('[ "$SMOKE_IMAGES" = "prebuilt" ] || return 0'), "the main runs kill nothing"
    assert '"$RESTART_TOOL" --container "$BACKEND_CONTAINER" --health-url "$HEALTH_URL"' in step
    assert "|| fail " in step
    main = _shell_function(text, "main")
    order = [main.index(call) for call in ("check_backend_processes", "check_worker_restart", "boot_frontend")]
    assert order == sorted(order)


def test_the_pull_request_variant_boots_prebuilt_images_and_watches_the_step():
    workflow = yaml.safe_load(_SMOKE_IMAGES.read_text(encoding="utf-8"))
    triggers = workflow.get("on", workflow.get(True))
    assert "scripts/ci/smoke_worker_restart.py" in triggers["pull_request"]["paths"]
    boot = next(step for step in workflow["jobs"]["smoke-images-pr"]["steps"]
                if step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")
    assert boot["env"]["SMOKE_IMAGES"] == "prebuilt"
