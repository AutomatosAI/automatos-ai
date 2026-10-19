"""PRD-262 US-038: health gates in seconds, on a stated Docker floor.

* ``docker-compose.yml`` probes postgres (over TCP: ``pg_isready -h 127.0.0.1``, so it
  never passes on initdb's socket-only temporary server), redis, minio, the backend, the
  workspace worker and media-render every second while each starts
  (``start_interval: 1s``), and each start period is longer than that interval.
* The backend's and the frontend's Dockerfile HEALTHCHECKs gain ``--start-interval``,
  with ``# syntax=docker/dockerfile:1`` as line 1. (The worker's and media-render's
  come with US-045 and US-051.)
* ``make env``, which ``make up``, ``make build`` and ``make dev`` run first, runs
  ``scripts/stack_docker_floor.sh`` first: it stops, naming Docker Engine 25 and
  Compose 2.24.4, on an older Docker. It runs here for real under sh, with ``docker``
  replaced by a stub (``FAKE_ENGINE`` is what ``docker version`` prints,
  ``FAKE_COMPOSE`` what ``docker compose version --short`` prints, ``FAKE_DOCKER_RC``
  and ``FAKE_COMPOSE_RC`` make either fail).
* CI reads the gates off Docker's start and health_status events
  (``scripts/ci/smoke_health_gates.py``): redis and minio healthy within 2 s of their
  start, postgres reported, and, where compose starts the frontend (make up), the
  frontend started within 3 s of the backend turning healthy. Both smoke lanes run it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_COMPOSE = _REPO / "docker-compose.yml"
_DEV_COMPOSE = _REPO / "docker-compose.dev.yml"
_MAKEFILE = _REPO / "Makefile"
_FLOOR = _REPO / "scripts" / "stack_docker_floor.sh"
_FRESH_SMOKE = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_MAKE_UP_SMOKE = _REPO / "scripts" / "ci" / "smoke-make-up.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
_GATES_PATH = "scripts/ci/smoke_health_gates.py"
_FLOOR_PATH = "scripts/stack_docker_floor.sh"
_SYNTAX = "# syntax=docker/dockerfile:1"
# Every compose service with a healthcheck, and so a start_interval (US-038's list,
# with media-render's probe as US-051 asks).
_PROBED = {"postgres", "redis", "minio", "backend", "workspace-worker", "media-render"}
_FLOOR_MESSAGE = "Automatos needs Docker Engine 25 or later and Docker Compose 2.24.4 or later"
NS = 1_000_000_000
T0 = 1_792_100_000  # a whole epoch second


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


gates = _load("smoke_health_gates")

SERVICES = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]


def _seconds(value) -> int:
    match = re.fullmatch(r"(\d+)s", str(value).strip())
    assert match, f"healthcheck durations are written in seconds: {value!r}"
    return int(match.group(1))


# --------------------------------------------------------------------------- compose


def test_every_probed_service_is_probed_every_second_while_it_starts():
    probed = {name for name, spec in SERVICES.items() if "healthcheck" in spec}
    assert probed == _PROBED, "a new healthcheck needs start_interval too (PRD-262 US-038)"
    for name in sorted(_PROBED):
        check = SERVICES[name]["healthcheck"]
        assert check.get("start_interval") == "1s", name
        # start_interval applies only inside start_period, which Compose also requires.
        assert _seconds(check["start_period"]) > _seconds(check["start_interval"]), name


def test_postgres_is_probed_over_tcp_so_initdbs_temporary_server_never_passes():
    test = SERVICES["postgres"]["healthcheck"]["test"]
    assert test[0] == "CMD-SHELL"
    assert re.search(r"\bpg_isready -h 127\.0\.0\.1\b", test[1]), test
    assert "-U ${POSTGRES_USER:-postgres}" in test[1]


def test_the_backend_keeps_its_slow_boot_budget_and_the_frontend_its_gate():
    """F228 still holds: the 1 s probes run inside the 300 s start period."""
    assert _seconds(SERVICES["backend"]["healthcheck"]["start_period"]) >= 300
    assert SERVICES["frontend"]["depends_on"]["backend"]["condition"] == "service_healthy"


def test_make_dev_keeps_the_probes():
    dev = yaml.safe_load(_DEV_COMPOSE.read_text(encoding="utf-8"))["services"]
    assert not any("healthcheck" in spec for spec in dev.values())


# --------------------------------------------------------------------------- Dockerfiles


@pytest.mark.parametrize("path", ["orchestrator/Dockerfile", "frontend/Dockerfile"])
def test_the_dockerfile_healthchecks_probe_every_second_while_starting(path):
    text = (_REPO / path).read_text(encoding="utf-8")
    assert text.splitlines()[0] == _SYNTAX, "a builder older than --start-interval reads it through the directive"
    checks = [line for line in text.splitlines() if line.startswith("HEALTHCHECK ")]
    assert len(checks) == 2, "the development and production stages"
    for line in checks:
        assert "--start-interval=1s" in line, line
        assert re.search(r"--start-period=\d+s", line), line


# --------------------------------------------------------------------------- the floor


def test_make_env_checks_the_docker_floor_before_anything_else():
    makefile = _MAKEFILE.read_text(encoding="utf-8")
    assert re.search(r"^STACK_FLOOR\s*\?=\s*sh scripts/stack_docker_floor\.sh$", makefile, re.M)
    env = makefile.split("\nenv:\n", 1)[1].split("\n\n", 1)[0].splitlines()
    assert [line.strip() for line in env if line.startswith("\t")][0] == "@$(STACK_FLOOR)"
    for target in ("up", "build", "dev"):
        recipe = makefile.split(f"\n{target}:\n", 1)[1].split("\n\n", 1)[0].splitlines()
        assert recipe[0].strip() == "@$(MAKE) --no-print-directory env", f"make {target} runs make env first"


_FAKE_DOCKER = """#!/bin/sh
echo "docker $*" >> "$CALL_LOG"
case "$1" in
  version)
    if [ "${FAKE_DOCKER_RC:-0}" != 0 ]; then
      echo "Cannot connect to the Docker daemon at unix:///var/run/docker.sock." >&2
      exit "$FAKE_DOCKER_RC"
    fi
    echo "$FAKE_ENGINE" ;;
  compose)
    if [ "${FAKE_COMPOSE_RC:-0}" != 0 ]; then
      echo "docker: 'compose' is not a docker command." >&2
      exit "$FAKE_COMPOSE_RC"
    fi
    echo "$FAKE_COMPOSE" ;;
esac
"""
_VERSION_CALL = "docker version --format {{.Server.Version}} {{.Server.APIVersion}}"
_COMPOSE_CALL = "docker compose version --short"


def _floor(tmp_path: Path, **env: str) -> tuple:
    """``sh scripts/stack_docker_floor.sh`` with the stub docker: (exit code, calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(_FAKE_DOCKER, encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    defaults = {"FAKE_ENGINE": "28.4.0 1.51", "FAKE_COMPOSE": "2.39.1"}
    proc = subprocess.run(["sh", str(_FLOOR)],
                          env={"PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin", "CALL_LOG": str(log),
                               **defaults, **env},
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=30)
    return proc.returncode, log.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr


needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="needs sh")


@needs_sh
@pytest.mark.parametrize("engine,compose", [
    ("28.4.0 1.51", "2.39.1"),
    ("25.0.0 1.44", "2.24.4"),               # exactly the floor
    ("27.3.1 1.47", "v2.29.7-desktop.1"),    # Docker Desktop's Compose spelling
    ("26.1.3+dfsg1 1.45", "2.24.6+ds1-0ubuntu1"),  # distribution package suffixes
])
def test_a_docker_at_or_above_the_floor_passes_silently(tmp_path, engine, compose):
    code, calls, out = _floor(tmp_path, FAKE_ENGINE=engine, FAKE_COMPOSE=compose)
    assert code == 0, out
    assert out == ""
    assert calls == [_VERSION_CALL, _COMPOSE_CALL]


@needs_sh
@pytest.mark.parametrize("engine", ["24.0.7 1.43", "20.10.24+dfsg1 1.41", "5.2.0 1.41"])
def test_an_engine_older_than_25_stops_with_the_floor_named(tmp_path, engine):
    """Engine 24 (API 1.43), Debian's docker.io 20.10, and a Podman answering API 1.41."""
    code, calls, out = _floor(tmp_path, FAKE_ENGINE=engine)
    assert code == 1
    assert _FLOOR_MESSAGE in out
    assert f"Docker Engine {engine.split()[0]} (API {engine.split()[1]})" in out
    assert "start_interval" in out and "docs.docker.com/engine/install" in out
    assert calls == [_VERSION_CALL], "the engine is checked first"


@needs_sh
def test_without_an_api_version_the_engine_release_decides(tmp_path):
    code, _, out = _floor(tmp_path, FAKE_ENGINE="24.0.7 ")
    assert code == 1 and _FLOOR_MESSAGE in out
    assert _floor(tmp_path, FAKE_ENGINE="25.0.3 ")[0] == 0


@needs_sh
@pytest.mark.parametrize("compose", ["2.24.3", "2.21.0", "v2.20.2"])
def test_compose_older_than_2_24_4_stops_with_the_floor_named(tmp_path, compose):
    code, calls, out = _floor(tmp_path, FAKE_COMPOSE=compose)
    assert code == 1
    assert _FLOOR_MESSAGE in out and f"Docker Compose {compose}." in out
    assert calls == [_VERSION_CALL, _COMPOSE_CALL]


@needs_sh
def test_docker_not_answering_stops_with_dockers_own_words(tmp_path):
    code, calls, out = _floor(tmp_path, FAKE_DOCKER_RC="1")
    assert code == 1
    assert "Docker does not answer" in out and "Cannot connect to the Docker daemon" in out
    assert calls == [_VERSION_CALL]


@needs_sh
def test_a_missing_compose_plugin_stops_with_the_floor_named(tmp_path):
    code, _, out = _floor(tmp_path, FAKE_COMPOSE_RC="1")
    assert code == 1
    assert _FLOOR_MESSAGE in out and "Compose v2 plugin is missing" in out


@needs_sh
def test_a_version_it_cannot_read_passes_with_a_warning(tmp_path):
    """A development build: Compose itself still refuses an engine that is too old."""
    code, _, out = _floor(tmp_path, FAKE_COMPOSE="dev")
    assert code == 0
    assert "could not read the Docker Compose version from 'dev'" in out


# --------------------------------------------------------------------------- the CI gates


def _at(offset_s: float) -> int:
    return T0 * NS + round(offset_s * 1000) * 1_000_000


def _event(service: str, action: str, at_s: float) -> dict:
    attributes = {"name": f"automatos_{service}", "com.docker.compose.service": service}
    return {"Type": "container", "Action": action, "Actor": {"ID": "c0ffee", "Attributes": attributes},
            "time": T0 + int(at_s), "timeNano": _at(at_s)}


def _healthy(service: str, at_s: float) -> dict:
    return _event(service, "health_status: healthy", at_s)


def _stack(redis_s=1.2, minio_s=1.4, postgres_s=4.6, backend_healthy=40.0, frontend_start=40.6) -> list:
    """A make up run: datastores start at 0, the backend at 5, the frontend once it is healthy."""
    events = [_event("postgres", "start", 0.0), _healthy("postgres", postgres_s),
              _event("redis", "start", 0.0), _event("minio", "start", 0.1), _event("backend", "start", 5.0)]
    if redis_s is not None:
        events.append(_healthy("redis", redis_s))
    if minio_s is not None:
        events.append(_healthy("minio", 0.1 + minio_s))
    if backend_healthy is not None:
        events.append(_healthy("backend", backend_healthy))
    if frontend_start is not None:
        events.append(_event("frontend", "start", frontend_start))
    return events


def test_a_fast_stack_passes_every_gate():
    record = gates.check(_stack(), gates.GATE)
    assert record["failures"] == []
    rows = {row["service"]: row for row in record["datastores"]}
    assert rows["redis"]["healthy_s"] == 1.2 and rows["redis"]["bar_s"] == 2.0
    assert rows["minio"]["healthy_s"] == 1.4
    assert rows["postgres"]["healthy_s"] == 4.6 and rows["postgres"]["bar_s"] is None
    assert record["frontend"] == {"seconds": 0.6, "problem": None, "gated": True, "bar_s": 3.0, "failure": None}


@pytest.mark.parametrize("field,value,expected", [
    ("redis_s", 2.3, "redis (start 1) turned healthy 2.3 s after it started (bar: 2 s)"),
    ("minio_s", 6.0, "minio (start 1) turned healthy 6.0 s after it started (bar: 2 s)"),
    ("minio_s", None, "minio (start 1) never reported healthy"),
])
def test_a_slow_or_never_healthy_redis_or_minio_fails(field, value, expected):
    assert gates.check(_stack(**{field: value}), gates.REPORT)["failures"] == [expected]


def test_a_datastore_that_never_started_fails():
    events = [event for event in _stack() if event["Actor"]["Attributes"]["com.docker.compose.service"] != "redis"]
    assert gates.check(events, gates.GATE)["failures"] == ["redis never started"]


def test_postgres_is_reported_however_long_initdb_takes():
    record = gates.check(_stack(postgres_s=19.0), gates.GATE)
    assert record["failures"] == []
    assert next(row for row in record["datastores"] if row["service"] == "postgres")["healthy_s"] == 19.0


def test_the_frontend_gap_is_gated_only_where_compose_starts_it():
    slow = _stack(frontend_start=43.5)
    assert gates.check(slow, gates.GATE)["failures"] == [
        "the frontend started 3.5 s after the backend turned healthy (bar: 3 s)"]
    reported = gates.check(slow, gates.REPORT)
    assert reported["failures"] == []
    assert reported["frontend"]["seconds"] == 3.5 and reported["frontend"]["gated"] is False


def test_the_gap_runs_from_the_backends_last_healthy_before_the_frontend_started():
    """A backend that restarted while booting: the healthy that let the frontend start counts."""
    events = _stack(backend_healthy=20.0, frontend_start=41.0)
    events += [_event("backend", "start", 25.0), _healthy("backend", 40.0), _event("backend", "start", 90.0),
               _healthy("backend", 99.0)]
    assert gates.check(events, gates.GATE)["frontend"]["seconds"] == 1.0


@pytest.mark.parametrize("field,problem", [
    ("frontend_start", "the frontend never started"),
    ("backend_healthy", "the backend never reported healthy before the frontend started"),
])
def test_a_gated_frontend_needs_both_events(field, problem):
    record = gates.check(_stack(**{field: None}), gates.GATE)
    assert record["failures"] == [problem]
    assert gates.check(_stack(**{field: None}), gates.REPORT)["failures"] == []


def _write_events(path: Path, events: list) -> None:
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + '{"truncated', encoding="utf-8")


def test_the_command_writes_the_record_and_the_summary_and_exits_0_on_a_pass(tmp_path):
    events, out, summary = tmp_path / "events.jsonl", tmp_path / "gates.json", tmp_path / "summary.md"
    _write_events(events, _stack())
    code = gates.main(["--events", str(events), "--frontend", "gate", "--out", str(out), "--summary", str(summary)])
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["failures"] == []
    text = summary.read_text(encoding="utf-8")
    assert "### Health gates (PRD-262 US-038" in text
    assert "| redis | 1 | 1.2 s | ≤ 2 s | pass |" in text
    assert "| postgres | 1 | 4.6 s | reported; its bar comes from this number (§8) | — |" in text
    assert "Frontend: the frontend started 0.6 s after the backend turned healthy (bar: ≤ 3 s): pass." in text


def test_the_command_exits_1_and_marks_the_miss(tmp_path, capsys):
    events, out, summary = tmp_path / "events.jsonl", tmp_path / "gates.json", tmp_path / "summary.md"
    _write_events(events, _stack(minio_s=2.4, frontend_start=50.0))
    code = gates.main(["--events", str(events), "--out", str(out), "--summary", str(summary)])
    assert code == 1
    text = summary.read_text(encoding="utf-8")
    assert "| minio | 1 | 2.4 s | ≤ 2 s | **FAIL** |" in text
    assert "Reported, not gated" in text, "--frontend defaults to report"
    assert "::error title=health gates::minio (start 1) turned healthy 2.4 s" in capsys.readouterr().out


# --------------------------------------------------------------------------- the lanes


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"no {name}() function"
    return match.group(1)


def _steps(text: str) -> list:
    return [line.strip() for line in _function(text, "main").splitlines()]


def test_the_fresh_clone_smoke_gates_once_the_frontend_is_up_and_gates_it_only_after_make_up():
    text = _FRESH_SMOKE.read_text(encoding="utf-8")
    steps = _steps(text)
    order = [steps.index(step) for step in ("start_event_log", "boot_frontend", "finish_make_up",
                                             "check_health_gates", "warm_restarts")]
    assert order == sorted(order)
    check = _function(text, "check_health_gates")
    assert 'if [ "$SMOKE_IMAGES" = "pull" ]; then\n    frontend="gate"' in check
    assert '"$GATES_TOOL" --events "$EVENTS_FILE" --frontend "$frontend"' in check
    assert "|| fail " in check
    assert f'GATES_TOOL="{_GATES_PATH}"' in text


def test_the_make_up_smoke_streams_the_events_and_gates_the_frontend():
    text = _MAKE_UP_SMOKE.read_text(encoding="utf-8")
    steps = _steps(text)
    order = [steps.index(step) for step in ("trap finish EXIT", "start_event_log", "check_nothing_was_built",
                                             "check_health_gates", 'outcome="pass"')]
    assert order == sorted(order)
    assert next(i for i, step in enumerate(steps) if step.startswith("make up 2>&1")) > steps.index("start_event_log")
    assert "--filter event=start --filter event=health_status" in _function(text, "start_event_log")
    check = _function(text, "check_health_gates")
    assert check.lstrip().startswith("stop_event_log")
    assert '"$GATES_TOOL" --events "$EVENTS_FILE" --frontend gate' in check and "|| fail " in check
    assert "stop_event_log" in _function(text, "finish")


def _paths(workflow: str, event: str) -> list:
    data = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    return data.get("on", data.get(True))[event]["paths"]


def test_the_lanes_run_when_the_gates_or_the_floor_change():
    assert {_GATES_PATH, _FLOOR_PATH} <= set(_paths("smoke-images.yml", "pull_request"))
    for event in ("push", "pull_request"):
        assert _GATES_PATH in _paths("smoke-fresh-clone.yml", event), event
