"""PRD-262 US-045: the smoke sees the workspace worker healthy within 20 s and /metrics 200.

scripts/ci/smoke_worker_health.py reads the smoke's Docker events (the start and
health_status events smoke_report.py tables), measures the worker's first start to its
first healthy event, and asks /metrics from inside the container. The smoke
(smoke-fresh-clone.sh, which smoke-images also runs) calls it after the boot, in every
variant. These tests hold the tool's verdicts and the wiring; the real boot runs in CI.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "scripts" / "ci" / "smoke-fresh-clone.sh"
WORKFLOWS = ROOT / ".github" / "workflows"
TOOL_PATH = "scripts/ci/smoke_worker_health.py"
NS = 1_000_000_000
T0 = 1_792_100_000 * NS
SERVICE, CONTAINER = "workspace-worker", "automatos_workspace_worker"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


health = _load("smoke_worker_health")


def _event(action: str, service: str, at_s: float) -> dict:
    return {"Action": action, "timeNano": T0 + round(at_s * NS),
            "Actor": {"Attributes": {"com.docker.compose.service": service}}}


def _write(path: Path, events: list[dict]) -> Path:
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    return path


def _span(events: list[dict]):
    return health.first_start(health.smoke_report.service_spans(events), SERVICE)


# --------------------------------------------------------------------------- the verdict


def test_a_worker_healthy_in_6_s_passes():
    span = _span([_event("start", "redis", 0), _event("start", SERVICE, 2),
                  _event("health_status: healthy", "redis", 3), _event("health_status: healthy", SERVICE, 8)])
    assert health.healthy_seconds(span) == 6.0
    assert health.health_problem(span, SERVICE, 20) is None


def test_a_worker_healthy_after_the_bar_fails():
    # A 30 s interval with no start interval: the first probe comes one interval in.
    span = _span([_event("start", SERVICE, 0), _event("health_status: healthy", SERVICE, 31.5)])
    assert health.health_problem(span, SERVICE, 20) == f"{SERVICE} reported healthy 31.5s after its start; the bar is 20s"


def test_only_the_first_start_counts():
    span = _span([_event("start", SERVICE, 0), _event("start", SERVICE, 40),
                  _event("health_status: healthy", SERVICE, 42)])
    assert span["start_ns"] == T0
    assert health.health_problem(span, SERVICE, 20) == f"{SERVICE} never reported healthy after its first start"


def test_a_worker_that_never_started_fails():
    span = _span([_event("start", "backend", 0), _event("health_status: healthy", "backend", 5)])
    assert span is None
    assert health.health_problem(span, SERVICE, 20) == f"no start of {SERVICE} is in the smoke's Docker events"


def test_the_verdict_stays_open_until_healthy_or_past_the_bar():
    started = _span([_event("start", SERVICE, 0)])
    limit = T0 + round((20 + health.GRACE_SECONDS) * NS)
    assert health.verdict_open(started, T0, limit - 1, 20)
    assert not health.verdict_open(started, T0, limit, 20)
    healthy = _span([_event("start", SERVICE, 0), _event("health_status: healthy", SERVICE, 4)])
    assert not health.verdict_open(healthy, T0, T0, 20)
    # No start yet: the wait runs from when it began.
    assert health.verdict_open(None, T0, T0 + 5 * NS, 20)
    assert not health.verdict_open(None, T0, limit, 20)


def test_the_wait_reads_the_events_again_until_the_healthy_event_lands(tmp_path):
    events = _write(tmp_path / "events.jsonl", [_event("start", SERVICE, 0)])
    clock = {"now": T0 + 3 * NS}

    def sleep(seconds: float) -> None:
        clock["now"] += round(seconds * NS)
        if clock["now"] >= T0 + 5 * NS:
            _write(events, [_event("start", SERVICE, 0), _event("health_status: healthy", SERVICE, 5)])

    span = health.wait_for_first_start(events, SERVICE, 20, clock=lambda: clock["now"], sleep=sleep)
    assert health.healthy_seconds(span) == 5.0
    assert clock["now"] == T0 + 5 * NS


def test_the_wait_gives_up_once_the_bar_and_grace_have_passed(tmp_path):
    events = _write(tmp_path / "events.jsonl", [_event("start", SERVICE, 0)])
    clock = {"now": T0}
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock["now"] += round(seconds * NS)

    span = health.wait_for_first_start(events, SERVICE, 20, clock=lambda: clock["now"], sleep=sleep)
    assert health.healthy_seconds(span) is None
    assert len(sleeps) == 20 + health.GRACE_SECONDS


# --------------------------------------------------------------------------- /metrics


@pytest.mark.parametrize(("stdout", "expected"), [("200", 200), ("500", 500), ("000", None), ("", None)])
def test_metrics_is_asked_inside_the_container_with_its_own_token(monkeypatch, stdout, expected):
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(health.subprocess, "run", run)
    assert health.metrics_status(CONTAINER) == expected
    assert calls == [["docker", "exec", CONTAINER, "sh", "-c", health.METRICS_PROBE]]
    assert '"X-Internal-Token: ${WORKER_INTERNAL_TOKEN:-}"' in health.METRICS_PROBE
    assert "http://localhost:${WORKER_HEALTH_PORT:-8081}/metrics" in health.METRICS_PROBE


def test_metrics_other_than_200_is_a_failure():
    assert health.metrics_problem(200) is None
    assert health.metrics_problem(500) == "the worker's /metrics answered 500, not 200"
    assert health.metrics_problem(None) == "the worker's /metrics answered nothing, not 200"


@pytest.mark.parametrize(("healthy_at", "status", "code"), [(4, 200, 0), (4, 500, 1), (25, 200, 1)])
def test_main_writes_the_record_and_the_summary(tmp_path, monkeypatch, healthy_at, status, code):
    events = _write(tmp_path / "events.jsonl",
                    [_event("start", SERVICE, 0), _event("health_status: healthy", SERVICE, healthy_at)])
    monkeypatch.setattr(health, "metrics_status", lambda container: status)
    out, summary = tmp_path / "out" / "worker-health.json", tmp_path / "summary.md"
    assert health.main(["--events", str(events), "--service", SERVICE, "--container", CONTAINER,
                        "--out", str(out), "--summary", str(summary)]) == code
    record = json.loads(out.read_text())
    assert (record["healthy_seconds"], record["metrics_status"], record["max_seconds"]) == (healthy_at, status, 20)
    assert bool(record["failures"]) == bool(code)
    assert ("FAILED" in summary.read_text()) == bool(code)


# --------------------------------------------------------------------------- the wiring


def _shell_function(text: str, name: str) -> str:
    """The body of shell function `name`: the lines between `name() {` and its closing `}`."""
    header = f"\n{name}() {{\n"
    start = text.index(header) + len(header)
    return text[start:text.index("\n}\n", start)]


def test_the_smoke_checks_the_worker_after_the_boot_in_every_variant():
    text = SMOKE.read_text(encoding="utf-8")
    for line in (f'WORKER_CONTAINER="{CONTAINER}"', f'WORKER_HEALTH_TOOL="{TOOL_PATH}"',
                 "WORKER_HEALTHY_MAX_SECONDS=20", 'WORKER_HEALTH_FILE="${OUT_DIR}/worker-health.json"'):
        assert f"\n{line}\n" in text, line
    step = _shell_function(text, "check_workspace_worker")
    assert step.lstrip().startswith(f'python3 "$WORKER_HEALTH_TOOL" --events "$EVENTS_FILE" --service {SERVICE}')
    assert '--max-seconds "$WORKER_HEALTHY_MAX_SECONDS"' in step
    assert "|| fail " in step
    assert "return 0" not in step, "every variant checks the worker"
    main = _shell_function(text, "main")
    order = [main.index(call) for call in ("start_event_log", "boot_fresh", "check_workspace_worker", "warm_restarts")]
    assert order == sorted(order)
    assert "compose logs --tail 80 workspace-worker" in _shell_function(text, "dump_failure_logs")


def test_the_service_and_container_are_composes_worker():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    worker = compose["services"][SERVICE]
    assert worker["container_name"] == CONTAINER
    assert worker["healthcheck"]["test"], "the verdict reads Docker's health_status events"


def _triggers(name: str) -> dict:
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    return data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True


def test_both_smoke_workflows_watch_the_tool():
    fresh = _triggers("smoke-fresh-clone.yml")
    assert TOOL_PATH in fresh["push"]["paths"] and TOOL_PATH in fresh["pull_request"]["paths"]
    assert TOOL_PATH in _triggers("smoke-images.yml")["pull_request"]["paths"]
