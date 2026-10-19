"""F416 (night 14): a backend that stops serving ends its container, so it restarts.

The nightly EdgeBuilder ran the backend out of memory; the kernel killed the
uvicorn worker, and uvicorn's supervisor (``--reload``, as compose ran it then)
stayed up serving nothing. The container stayed "running" and unhealthy for
~5 h: Docker's ``restart: unless-stopped`` never applies to a container that is
only unhealthy.

Now ``orchestrator/docker-healthcheck.sh`` is the one probe (Docker's
healthcheck and the entrypoint's watchdog both run it), and with
``AUTOMATOS_LIVENESS_WATCHDOG=true`` (``envs/api.defaults``) the entrypoint runs
the command as its child: once the server has served, failed probes in a row
stop it and the entrypoint exits non-zero, so the container exits and restarts.

The entrypoint runs for real under bash with a stub server and a stub probe, at
one-second intervals; the healthcheck runs for real against a local HTTP server.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import yaml

_ORCH = Path(__file__).resolve().parents[1]
_REPO = _ORCH.parent
_ENTRYPOINT = _ORCH / "docker-entrypoint.sh"
_HEALTHCHECK = _ORCH / "docker-healthcheck.sh"
_DOCKERFILE = _ORCH / "Dockerfile"
_COMPOSE = _REPO / "docker-compose.yml"
_DEV_COMPOSE = _REPO / "docker-compose.dev.yml"
_API_DEFAULTS = _REPO / "envs" / "api.defaults"

_RUN_TIMEOUT_SECONDS = 30
_RESTART_BUDGET_SECONDS = 150  # "restarted within ~2 minutes"

# A server that records its pid and exits 0 on TERM (as uvicorn does).
_SERVER = """
echo $$ > "$SERVER_PID_FILE"
trap 'echo TERM >> "$SERVER_LOG"; exit 0' TERM
while :; do sleep 0.2; done
"""
# A hung server: it ignores TERM, so only KILL ends it.
_HUNG_SERVER = """
echo $$ > "$SERVER_PID_FILE"
trap '' TERM
while :; do sleep 0.2; done
"""
# The probe passes PROBE_PASSES times, then fails.
_PROBE = """
n=$(cat "$PROBE_COUNT" 2>/dev/null || echo 0)
n=$((n + 1))
echo "$n" > "$PROBE_COUNT"
[ "$n" -le "${PROBE_PASSES:-0}" ]
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!/bin/bash\n{body}\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        "PATH": f"/usr/bin{os.pathsep}/bin",
        "TMPDIR": str(tmp_path),
        "AUTOMATOS_LIVENESS_WATCHDOG": "true",
        "AUTOMATOS_LIVENESS_INTERVAL": "1",
        "AUTOMATOS_LIVENESS_RETRIES": "2",
        "AUTOMATOS_LIVENESS_GRACE": "1",
        "AUTOMATOS_HEALTHCHECK_CMD": str(_script(tmp_path / "probe", _PROBE)),
        "PROBE_COUNT": str(tmp_path / "probe.count"),
        "SERVER_PID_FILE": str(tmp_path / "server.pid"),
        "SERVER_LOG": str(tmp_path / "server.log"),
    }
    return {**env, **overrides}


def _run(tmp_path: Path, server_body: str, **env: str) -> tuple[int, str, float]:
    server = _script(tmp_path / "server", server_body)
    started = time.monotonic()
    proc = subprocess.run(
        ["bash", str(_ENTRYPOINT), str(server)],
        env=_env(tmp_path, **env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=_RUN_TIMEOUT_SECONDS,
    )
    return proc.returncode, proc.stdout + proc.stderr, time.monotonic() - started


def _server_pid(tmp_path: Path) -> int:
    return int((tmp_path / "server.pid").read_text(encoding="utf-8").strip())


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


# --- the entrypoint propagates a dead server as the container's exit ---------


def test_a_server_that_stops_serving_is_stopped_and_the_entrypoint_exits_non_zero(tmp_path):
    """The night-14 shape: the supervisor is alive, nothing answers /health."""
    code, out, elapsed = _run(tmp_path, _SERVER, PROBE_PASSES="1")
    assert code != 0, out
    assert "stopped serving" in out
    assert "(2/2)" in out, "both failed probes are counted before the stop"
    assert "TERM" in (tmp_path / "server.log").read_text(encoding="utf-8"), "TERM first, for a clean shutdown"
    assert not _is_running(_server_pid(tmp_path))
    assert elapsed < _RUN_TIMEOUT_SECONDS


def test_a_hung_server_that_ignores_term_is_killed_after_the_grace(tmp_path):
    code, out, _ = _run(tmp_path, _HUNG_SERVER, PROBE_PASSES="1")
    assert code != 0, out
    assert "stopped serving" in out
    assert not _is_running(_server_pid(tmp_path))


def test_a_server_that_exits_passes_its_exit_status_on(tmp_path):
    """A supervisor that dies with its worker ends the container with its status."""
    body = 'echo $$ > "$SERVER_PID_FILE"\nsleep 1\nexit 3'
    code, out, _ = _run(tmp_path, body, PROBE_PASSES="100")
    assert code == 3, out
    assert "stopped serving" not in out


def test_a_slow_boot_is_never_stopped_before_it_first_serves(tmp_path):
    """Every probe fails, but the server never served: Docker's start_period judges a boot."""
    body = 'echo $$ > "$SERVER_PID_FILE"\nsleep 4\nexit 0'
    code, out, _ = _run(tmp_path, body, PROBE_PASSES="0", AUTOMATOS_LIVENESS_RETRIES="1")
    assert code == 0, out
    assert "stopped serving" not in out
    assert "armed" not in out


def test_a_healthy_server_is_left_alone(tmp_path):
    body = 'echo $$ > "$SERVER_PID_FILE"\nsleep 4\nexit 0'
    code, out, _ = _run(tmp_path, body, PROBE_PASSES="100", AUTOMATOS_LIVENESS_RETRIES="1")
    assert code == 0, out
    assert "armed" in out
    assert "stopped serving" not in out


def test_docker_stop_reaches_the_server_and_the_entrypoint_exits_with_it(tmp_path):
    """TERM to the entrypoint (PID 1 in the container) is forwarded to the server."""
    server = _script(tmp_path / "server", _SERVER)
    proc = subprocess.Popen(
        ["bash", str(_ENTRYPOINT), str(server)],
        env=_env(tmp_path, PROBE_PASSES="100"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + _RUN_TIMEOUT_SECONDS
    while not (tmp_path / "server.pid").exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=_RUN_TIMEOUT_SECONDS)
    # The server's own 0, or 143 (128 + TERM) when it exits before the entrypoint
    # looks again: either way a stop, never the watchdog's failure.
    assert proc.returncode in (0, 128 + signal.SIGTERM), out
    assert "stopped serving" not in out
    assert "TERM" in (tmp_path / "server.log").read_text(encoding="utf-8")
    assert not _is_running(_server_pid(tmp_path))


def test_without_the_flag_the_command_is_exec_d_as_before(tmp_path):
    """The image default (Railway): no watchdog, the command replaces the entrypoint."""
    server = _script(tmp_path / "server", "exit 0")
    env = _env(tmp_path)
    env.pop("AUTOMATOS_LIVENESS_WATCHDOG")
    proc = subprocess.run(["bash", str(_ENTRYPOINT), str(server)], env=env, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (tmp_path / "probe.count").exists(), "no probe ran"


# --- the healthcheck: a failed check exits non-zero ---------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _serve(status: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — the http.server API
            self.send_response(status if self.path == "/health" else 404)
            self.end_headers()

        def log_message(self, *_args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _check(port: int) -> int:
    env = {"PATH": f"/usr/bin{os.pathsep}/bin", "PORT": str(port), "AUTOMATOS_HEALTHCHECK_TIMEOUT": "5"}
    return subprocess.run(["sh", str(_HEALTHCHECK)], env=env, capture_output=True, timeout=15).returncode


@pytest.mark.skipif(shutil.which("curl") is None, reason="the healthcheck runs curl, as the image does")
@pytest.mark.parametrize("status,healthy", [(200, True), (500, False), (503, False)])
def test_the_healthcheck_passes_only_on_a_2xx_health_answer(status, healthy):
    server = _serve(status)
    try:
        code = _check(server.server_address[1])
    finally:
        server.shutdown()
    assert (code == 0) is healthy


@pytest.mark.skipif(shutil.which("curl") is None, reason="the healthcheck runs curl, as the image does")
def test_the_healthcheck_fails_when_nothing_serves():
    assert _check(_free_port()) != 0


# --- the wiring: compose and the image run the same probe, and restart --------


def _compose_backend(path: Path = _COMPOSE) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["services"]["backend"]


def _seconds(value: str) -> int:
    match = re.fullmatch(r"(\d+)s", str(value))
    assert match, f"healthcheck durations are written in seconds: {value!r}"
    return int(match.group(1))


def _entrypoint_default(name: str) -> int:
    match = re.search(rf"^{name}=(\d+)$", _ENTRYPOINT.read_text(encoding="utf-8"), re.M)
    assert match, f"{name} is missing from docker-entrypoint.sh"
    return int(match.group(1))


def test_compose_runs_the_probe_restarts_and_turns_the_watchdog_on():
    backend = _compose_backend()
    assert backend["restart"] == "unless-stopped"
    assert backend["healthcheck"]["test"] == ["CMD", "docker-healthcheck.sh"]
    # PRD-262 US-035: the default stack runs the image's own probe; `make dev` mounts the
    # checkout's copy over it, so an edit to it applies without a rebuild.
    mount = "./orchestrator/docker-healthcheck.sh:/usr/local/bin/docker-healthcheck.sh:ro"
    assert not any("docker-healthcheck.sh" in str(volume) for volume in backend["volumes"])
    assert mount in _compose_backend(_DEV_COMPOSE)["volumes"]
    lines = _API_DEFAULTS.read_text(encoding="utf-8").splitlines()
    assert "AUTOMATOS_LIVENESS_WATCHDOG=true" in lines


def test_the_watchdog_matches_compose_and_restarts_within_about_two_minutes():
    check = _compose_backend()["healthcheck"]
    interval = _entrypoint_default("LIVENESS_DEFAULT_INTERVAL_SECONDS")
    retries = _entrypoint_default("LIVENESS_DEFAULT_RETRIES")
    grace = _entrypoint_default("LIVENESS_DEFAULT_GRACE_SECONDS")
    assert interval == _seconds(check["interval"]) and retries == int(check["retries"])
    worst_case = retries * (interval + _seconds(check["timeout"])) + grace
    assert worst_case <= _RESTART_BUDGET_SECONDS


def test_the_image_carries_the_probe_and_the_dev_server_replaces_its_shell():
    text = _DOCKERFILE.read_text(encoding="utf-8")
    assert text.count("COPY --chmod=0755 docker-healthcheck.sh /usr/local/bin/docker-healthcheck.sh") == 2
    assert text.count("CMD docker-healthcheck.sh") == 2
    assert "curl -f http://localhost" not in text, "one probe, not a second copy"
    development = text.split("FROM base as development", 1)[1].split("FROM base as production", 1)[0]
    cmd = next(line for line in development.splitlines() if line.startswith("CMD "))
    assert "&& exec uvicorn" in cmd, "the watchdog and docker stop must signal uvicorn, not a shell"
