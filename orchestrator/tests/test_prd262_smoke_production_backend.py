"""PRD-262 US-035: the smoke checks that compose runs the production backend.

``scripts/ci/smoke_production_backend.py`` runs inside the smoke lanes against a real
stack; these tests feed it recorded shapes and a local HTTP server instead of Docker:

* ``process`` passes the production stage (no bind mount over /app, no ``--reload``,
  uvicorn the entrypoint's own child) and fails the source mount, the development
  stage's ``--reload``, a ``sh -c`` between the entrypoint and uvicorn (today's
  ``alembic … && uvicorn`` CMD), and an entrypoint that exec'd uvicorn with its
  watchdog off;
* ``env`` passes when the value the smoke wrote to envs/api.local is in the backend;
* ``stop`` holds an SSE connection open while the stop command runs, and fails a stop
  at or over the bar, a failed stop, and a stream that never connected;
* ``scripts/ci/smoke-fresh-clone.sh`` runs the three in order, keeps a checkout's own
  envs/api.local and puts it back, and the two smoke workflows watch the tool.
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
ENTRYPOINT_PID = 4101


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


tool = _load("smoke_production_backend")
report = _load("smoke_report")

ENTRYPOINT_ARGS = "/bin/bash /usr/local/bin/docker-entrypoint.sh /bin/sh -c exec uvicorn main:app --host 0.0.0.0"
UVICORN_ARGS = "/usr/local/bin/python3.11 /usr/local/bin/uvicorn main:app --host 0.0.0.0 --port 8000"
SPAWN = "/usr/local/bin/python3.11 -c from multiprocessing.spawn import spawn_main; spawn_main(pipe_handle={})"


def _rows(*rows: tuple) -> list:
    return [{"pid": pid, "ppid": ppid, "args": args} for pid, ppid, args in rows]


# What `docker top` shows once US-029's CMD execs uvicorn under the watchdog: the
# entrypoint, uvicorn as its child (SERVER_PID), and the watchdog's subshell and sleep.
PRODUCTION = _rows(
    (ENTRYPOINT_PID, 4080, ENTRYPOINT_ARGS),
    (4250, ENTRYPOINT_PID, UVICORN_ARGS),
    (4251, ENTRYPOINT_PID, ENTRYPOINT_ARGS),
    (4252, 4251, "sleep 30"),
)
# The production CMD before US-029: `sh -c "alembic upgrade heads && uvicorn …"` sat in between.
SHELL_BETWEEN = _rows(
    (ENTRYPOINT_PID, 4080, ENTRYPOINT_ARGS),
    (4190, ENTRYPOINT_PID, "sh -c alembic upgrade heads && uvicorn main:app --host 0.0.0.0 --workers 4"),
    (4250, 4190, UVICORN_ARGS + " --workers 4"),
    *[(pid, 4250, SPAWN.format(pid)) for pid in (4253, 4254)],
)
VOLUMES = [
    {"Type": "volume", "Name": "automatos_backend_logs", "Source": "/var/lib/docker/volumes/x/_data",
     "Destination": "/app/logs"},
    {"Type": "volume", "Name": "automatos_backend_data", "Source": "/var/lib/docker/volumes/y/_data",
     "Destination": "/app/data"},
    {"Type": "bind", "Source": "/home/runner/work/automatos-ai/workspaces",
     "Destination": "/workspaces/00000000-0000-0000-0000-0000000000c1"},
]


# --------------------------------------------------------------------------- process


def test_volumes_under_app_and_the_workspace_folder_are_not_source_mounts():
    assert tool.source_mounts(VOLUMES) == []


@pytest.mark.parametrize("destination", ["/app", "/app/core", "/usr/local/bin/docker-entrypoint.sh",
                                         "/usr/local/bin/docker-healthcheck.sh",
                                         "/usr/local/bin/docker-secrets-guard.sh"])
def test_a_bind_mount_over_the_code_or_a_script_is_found(destination):
    mount = {"Type": "bind", "Source": "/home/runner/work/automatos-ai/orchestrator", "Destination": destination}
    assert tool.source_mounts(VOLUMES + [mount]) == [f"{mount['Source']} -> {destination}"]


def test_the_production_stage_under_the_watchdog_passes():
    found = tool.server_chain(PRODUCTION, ENTRYPOINT_PID)
    assert found["failures"] == []
    assert found["server_parent_is_entrypoint"] is True and found["reload"] is False
    assert found["server"] == UVICORN_ARGS


def test_several_workers_still_have_uvicorn_as_the_entrypoints_child():
    """WEB_CONCURRENCY=2 (the worker-restart step): uvicorn supervises spawned workers."""
    rows = PRODUCTION + _rows(*[(pid, 4250, SPAWN.format(pid)) for pid in (4260, 4261)])
    assert tool.server_chain(rows, ENTRYPOINT_PID)["failures"] == []


def test_a_shell_between_the_entrypoint_and_uvicorn_fails():
    found = tool.server_chain(SHELL_BETWEEN, ENTRYPOINT_PID)
    assert found["server_parent_is_entrypoint"] is False
    assert len(found["failures"]) == 1
    assert found["failures"][0].startswith("uvicorn's parent is `sh -c alembic upgrade heads && uvicorn")


def test_the_development_stages_reload_fails():
    rows = _rows((ENTRYPOINT_PID, 4080, ENTRYPOINT_ARGS),
                 (4250, ENTRYPOINT_PID, UVICORN_ARGS + " --reload --timeout-graceful-shutdown 5"),
                 (4251, 4250, SPAWN.format(9)))
    failures = tool.server_chain(rows, ENTRYPOINT_PID)["failures"]
    assert len(failures) == 1 and "--reload" in failures[0]


def test_uvicorn_exec_d_in_place_of_the_entrypoint_means_no_watchdog():
    rows = _rows((ENTRYPOINT_PID, 4080, UVICORN_ARGS))
    failures = tool.server_chain(rows, ENTRYPOINT_PID)["failures"]
    assert any("is not docker-entrypoint.sh" in failure for failure in failures)
    assert any("liveness watchdog is off" in failure for failure in failures)


def test_no_uvicorn_fails():
    failures = tool.server_chain(_rows((ENTRYPOINT_PID, 4080, ENTRYPOINT_ARGS)), ENTRYPOINT_PID)["failures"]
    assert failures == ["expected one uvicorn server process, found 0"]


def test_check_process_reads_the_entrypoint_pid_and_the_mounts():
    source = {"Type": "bind", "Source": "/src/orchestrator", "Destination": "/app"}
    clean = tool.check_process({"State": {"Pid": ENTRYPOINT_PID}, "Mounts": VOLUMES}, PRODUCTION)
    assert clean["failures"] == [] and clean["source_mounts"] == []
    mounted = tool.check_process({"State": {"Pid": ENTRYPOINT_PID}, "Mounts": VOLUMES + [source]}, PRODUCTION)
    assert mounted["failures"] == ["a bind mount lays the checkout over the image: /src/orchestrator -> /app"]


def test_the_process_command_reads_docker_and_writes_its_record(tmp_path, monkeypatch):
    top = "PID PPID COMMAND\n" + "\n".join(f"{row['pid']} {row['ppid']} {row['args']}" for row in PRODUCTION)
    calls = []

    def fake_text(*args):
        calls.append(args)
        return top

    monkeypatch.setattr(tool.images, "docker_json", lambda *args: [{"State": {"Pid": ENTRYPOINT_PID}, "Mounts": VOLUMES}])
    monkeypatch.setattr(tool.images, "docker_text", fake_text)
    out, summary = tmp_path / "process.json", tmp_path / "summary.md"
    assert tool.main(["process", "--container", "automatos_backend", "--out", str(out), "--summary", str(summary)]) == 0
    assert calls == [("top", "automatos_backend", "-eo", "pid,ppid,args", "-ww")]
    assert json.loads(out.read_text())["server_parent_is_entrypoint"] is True
    assert "the entrypoint's watchdog signals uvicorn itself" in summary.read_text()


# --------------------------------------------------------------------------- env


def test_env_passes_only_on_the_value_the_smoke_wrote():
    assert tool.check_env("K", "us035-1", "us035-1")["failures"] == []
    assert "is not set in the backend" in tool.check_env("K", "us035-1", None)["failures"][0]
    assert "'stale'" in tool.check_env("K", "us035-1", "stale")["failures"][0]


def test_the_env_command_reads_the_container(tmp_path, monkeypatch):
    seen = {}

    def fake_printenv(container, key):
        seen.update(container=container, key=key)
        return "us035-7"

    monkeypatch.setattr(tool, "printenv", fake_printenv)
    out = tmp_path / "env.json"
    args = ["env", "--container", "automatos_backend", "--key", "K", "--value", "us035-7",
            "--out", str(out), "--summary", str(tmp_path / "summary.md")]
    assert tool.main(args) == 0
    assert seen == {"container": "automatos_backend", "key": "K"}
    assert json.loads(out.read_text())["found"] == "us035-7"


# --------------------------------------------------------------------------- stop

GOOD_STREAM = {"url": "http://localhost:8000/api/v1/tasks/stream", "status": 200,
               "content_type": "text/event-stream; charset=utf-8", "first_bytes": 60,
               "first_line": "event: board_changed"}


class _Clock:
    def __init__(self, *ticks: float):
        self.ticks = list(ticks)

    def __call__(self) -> float:
        return self.ticks.pop(0)


def _stop(stream=GOOD_STREAM, code=0, seconds=5.2, raises=None):
    """stop_with_stream with a fake stream, stop command and clock: (record, events)."""
    events = []

    def opener(url):
        if raises:
            raise raises
        events.append("open")
        return dict(stream), lambda: events.append("close")

    def runner(command):
        events.append(("stop", tuple(command)))
        return code

    record = tool.stop_with_stream(GOOD_STREAM["url"], ["docker", "compose", "stop", "backend"], 8,
                                   opener=opener, runner=runner, clock=_Clock(100.0, 100.0 + seconds))
    return record, events


def test_a_stop_under_the_bar_with_the_stream_open_passes():
    record, events = _stop(seconds=5.2)
    assert record["failures"] == [] and record["seconds"] == 5.2 and record["exit_code"] == 0
    assert events == ["open", ("stop", ("docker", "compose", "stop", "backend")), "close"], (
        "the stream stays open while the stop runs")


@pytest.mark.parametrize("seconds", [8.0, 10.4])
def test_a_stop_at_or_over_the_bar_fails(seconds):
    record, _ = _stop(seconds=seconds)
    assert len(record["failures"]) == 1
    assert record["failures"][0].startswith(f"stopping the backend took {seconds:.1f}s")
    assert "UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN" in record["failures"][0]


def test_a_failed_stop_fails():
    record, _ = _stop(code=1, seconds=1.0)
    assert record["failures"] == ["the stop command exited 1"]


@pytest.mark.parametrize("stream,expected", [
    ({**GOOD_STREAM, "status": 401}, "answered 401"),
    ({**GOOD_STREAM, "content_type": "application/json"}, "not an event stream"),
    ({**GOOD_STREAM, "first_bytes": 0}, "sent no event"),
])
def test_no_stop_runs_without_a_connected_client(stream, expected):
    record, events = _stop(stream=stream)
    assert len(record["failures"]) == 1 and expected in record["failures"][0]
    assert events == ["open", "close"], "the stop never ran"


def test_a_stream_that_cannot_connect_fails_without_stopping():
    record, events = _stop(raises=ConnectionRefusedError(111, "Connection refused"))
    assert record["failures"][0].startswith("no SSE client could connect to")
    assert events == []


@pytest.fixture()
def sse_server():
    """A server that sends one SSE event, then holds the connection until released."""
    release, held = threading.Event(), threading.Event()

    class Stream(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — BaseHTTPRequestHandler's name
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'event: board_changed\ndata: {"workspace_id": "c1"}\n\n')
            self.wfile.flush()
            held.set()
            release.wait(10)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stream)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/api/v1/tasks/stream", release, held
    release.set()
    server.shutdown()
    server.server_close()


def test_the_real_client_holds_the_stream_open_while_the_stop_runs(sse_server):
    url, release, held = sse_server
    during = {}

    def runner(command):
        during["held"] = held.wait(5) and not release.is_set()
        release.set()  # the stop ends the stream, as uvicorn's shutdown does
        return 0

    record = tool.stop_with_stream(url, ["stop"], 8, runner=runner)
    assert record["failures"] == [], record
    assert record["stream"]["first_line"] == "event: board_changed"
    assert during == {"held": True}


def test_the_stop_command_takes_the_words_after_the_double_dash(tmp_path, monkeypatch):
    seen = {}

    def fake_stop(url, command, max_seconds):
        seen.update(url=url, command=command, max_seconds=max_seconds)
        return {"failures": [], "seconds": 4.9}

    monkeypatch.setattr(tool, "stop_with_stream", fake_stop)
    args = ["stop", "--url", "http://localhost:8000/api/v1/tasks/stream", "--max-seconds", "8",
            "--out", str(tmp_path / "stop.json"), "--summary", str(tmp_path / "summary.md"),
            "--", "docker", "compose", "-f", "docker-compose.yml", "stop", "backend"]
    assert tool.main(args) == 0
    assert seen["command"] == ["docker", "compose", "-f", "docker-compose.yml", "stop", "backend"]
    assert seen["max_seconds"] == 8.0
    assert "took 4.9s with an SSE client connected (bar: under 8s)" in (tmp_path / "summary.md").read_text()


# --------------------------------------------------------------------------- the smoke script


def _script() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_runs_the_checks_in_order():
    main = _function(_script(), "main")
    order = ["boot_fresh", "check_backend_processes", "check_production_backend", "boot_frontend",
             "warm_restarts", "check_api_local_reaches_backend", "check_stop_with_sse_client", 'outcome="pass"']
    assert [main.index(step) for step in order] == sorted(main.index(step) for step in order)


def test_the_smoke_checks_the_process_and_stops_with_an_sse_client():
    text = _script()
    assert 'PRODUCTION_TOOL="scripts/ci/smoke_production_backend.py"' in text
    assert '"$PRODUCTION_TOOL" process --container "$BACKEND_CONTAINER"' in _function(text, "check_production_backend")
    assert re.search(r"^STOP_MAX_SECONDS=8$", text, re.M)
    assert 'SSE_URL="${API_BASE}/api/v1/tasks/stream"' in text
    stop = _function(text, "check_stop_with_sse_client")
    assert '--url "$SSE_URL" --max-seconds "$STOP_MAX_SECONDS"' in stop
    assert '-- docker compose "${COMPOSE_ARGS[@]}" stop backend' in stop


def test_the_sse_url_is_the_boards_event_stream():
    source = (_REPO / "orchestrator" / "api" / "board_tasks.py").read_text(encoding="utf-8")
    assert 'APIRouter(prefix="/api/v1/tasks"' in source and '@router.get("/stream")' in source
    hook = (_REPO / "frontend" / "hooks" / "use-board-event-stream.ts").read_text(encoding="utf-8")
    assert "const STREAM_ENDPOINT = '/api/v1/tasks/stream'" in hook


def test_the_smoke_recreates_the_backend_the_way_the_night_tooling_does():
    text = _script()
    compose = yaml.safe_load((_REPO / "docker-compose.yml").read_text(encoding="utf-8"))
    env_file = compose["services"]["backend"]["env_file"]
    assert re.search(rf'^API_LOCAL="{re.escape(env_file[-1]["path"])}"$', text, re.M)
    recreate = _function(text, "check_api_local_reaches_backend")
    steps = ["write_api_local_probe", "docker logs -t", "compose up -d --no-build --no-deps backend",
             "wait_backend recreate", '"$PRODUCTION_TOOL" env']
    assert [recreate.index(step) for step in steps] == sorted(recreate.index(step) for step in steps)
    assert "replaced_backend_log;" in _function(text, "write_report")
    finish = _function(text, "finish")
    assert finish.index("compose down -v") < finish.index("restore_api_local")


def test_the_recreated_boot_is_not_a_warm_restart():
    boots = [{"label": label, "health_s": seconds, "ready_s": seconds}
             for label, seconds in (("fresh", 30.0), ("warm-1", 10.0), ("warm-2", 12.0), ("recreate", 40.0))]
    assert report.warm_median(boots) == {"restarts": 2, "health_s": 11.0, "ready_s": 11.0}


def _probe_harness(tmp_path: Path) -> Path:
    """A bash script with the smoke's two envs/api.local functions, run from tmp_path."""
    text = _script()
    functions = "".join(f"{name}() {{\n{_function(text, name)}}}\n"
                        for name in ("write_api_local_probe", "restore_api_local"))
    harness = tmp_path / "harness.sh"
    harness.write_text(
        'set -uo pipefail\nAPI_LOCAL="envs/api.local"\nAPI_LOCAL_KEY="PROBE"\n'
        f'API_LOCAL_SAVED="{tmp_path}/saved"\napi_local_written=""\n'
        'fail() { echo "FAIL: $1"; exit 1; }\n' + functions
        + 'write_api_local_probe "v1"\ncp envs/api.local during.txt\nrestore_api_local\n',
        encoding="utf-8")
    (tmp_path / "envs").mkdir()
    return harness


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_a_checkouts_own_api_local_is_kept_and_put_back(tmp_path):
    harness = _probe_harness(tmp_path)
    own = "LOG_LEVEL=DEBUG"  # no trailing newline: the probe still lands on a line of its own
    (tmp_path / "envs" / "api.local").write_text(own, encoding="utf-8")
    proc = subprocess.run(["bash", str(harness)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "during.txt").read_text().splitlines() == [own, "PROBE=v1"]
    assert (tmp_path / "envs" / "api.local").read_text() == own


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_api_local_the_smoke_wrote_is_removed(tmp_path):
    harness = _probe_harness(tmp_path)
    proc = subprocess.run(["bash", str(harness)], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "during.txt").read_text().splitlines() == ["", "PROBE=v1"]
    assert not (tmp_path / "envs" / "api.local").exists()


# --------------------------------------------------------------------------- the workflows


def _triggers(name: str) -> dict:
    data = yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))
    return data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True


def test_both_smoke_workflows_watch_the_tool():
    tool_path = "scripts/ci/smoke_production_backend.py"
    fresh = _triggers("smoke-fresh-clone.yml")
    assert tool_path in fresh["push"]["paths"] and tool_path in fresh["pull_request"]["paths"]
    assert tool_path in _triggers("smoke-images.yml")["pull_request"]["paths"]
