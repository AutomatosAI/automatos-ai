"""PRD-262 US-002: the fresh-clone smoke measures the first run properly.

The lane itself runs only in CI, against a real stack. These tests hold its parts:

* ``scripts/ci/smoke_boot.py`` times each URL from the container's ``StartedAt``, not
  from its own sleeps, polls once a second, and times liveness and readiness apart. For
  a container another process starts (``make up``, PRD-262 US-036) it first waits for
  the start, and stops waiting once that process has ended without starting it;
* ``scripts/ci/smoke_report.py`` turns the boot records, Docker's events and the
  backend log into the step summary: Started → Healthy per service, the warm median,
  the alembic runs counted the §8 way, the /health/bootstrap stages, the frontend's
  address, and a summary on a failed run too;
* ``scripts/ci/smoke-fresh-clone.sh`` starts the frontend, restarts the backend twice
  with the database kept, and writes the report on every exit. Since PRD-262 US-039 the
  source build is ``make build`` itself, whose failure is the run's failure;
* ``smoke-fresh-clone.yml`` watches the boot inputs US-002 names, runs nightly, pins
  its Actions by SHA and uploads the records.
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
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "ci" / "smoke-fresh-clone.sh"
_WORKFLOW = _REPO / ".github" / "workflows" / "smoke-fresh-clone.yml"
NS = 1_000_000_000
T0 = 1_792_100_000  # 2026-10-15 21:33:20 UTC, a whole epoch second
BACKEND, FRONTEND = "automatos_backend", "automatos_frontend"
CONTEXT_IMPL = "INFO  [alembic.runtime.migration] Context impl PostgresqlImpl."


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


boot = _load("smoke_boot")
report = _load("smoke_report")


def _at(offset_s: float) -> int:
    """T0 plus an offset, in nanoseconds, without float rounding."""
    return T0 * NS + round(offset_s * 1000) * 1_000_000


def _started_at(ns: int) -> str:
    """Like ``State.StartedAt``: RFC 3339 with the fraction's trailing zeros trimmed."""
    seconds, fraction = divmod(ns, NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    digits = f"{fraction:09d}".rstrip("0")
    return f"{stamp}.{digits}Z" if digits else f"{stamp}Z"


def _log_stamp(ns: int) -> str:
    """Like ``docker logs -t``: always nine fractional digits."""
    seconds, fraction = divmod(ns, NS)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{fraction:09d}Z"


class FakeClock:
    """A wall clock that the fake probes and sleeps move forward."""

    def __init__(self, now_ns: int):
        self.now = now_ns
        self.sleeps: list[float] = []

    def __call__(self) -> int:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += round(seconds * NS)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(round(seconds, 3))
        self.advance(seconds)


# --------------------------------------------------------------------------- smoke_boot


def test_parse_timestamp_reads_both_docker_formats():
    ns = _at(21.98)
    assert _started_at(ns).endswith(".98Z")
    assert boot.parse_timestamp(_started_at(ns)) == ns
    assert boot.parse_timestamp(_log_stamp(ns)) == ns
    assert boot.parse_timestamp(_started_at(_at(0))) == T0 * NS
    assert boot.parse_timestamp("2026-10-16T13:44:21.5+01:00") == boot.parse_timestamp("2026-10-16T12:44:21.5Z")
    with pytest.raises(ValueError):
        boot.parse_timestamp("16 Oct 2026 12:44")


def test_poll_times_each_url_from_the_container_start_once_a_second():
    started = _at(0)
    clock = FakeClock(_at(20))  # polling begins 20 s after the container started
    calls = {"h": 0, "r": 0}

    def fake_probe(url):
        calls[url] += 1
        clock.advance(0.1)  # each request takes 100 ms
        return (200 if calls[url] >= {"h": 3, "r": 5}[url] else 503), url

    results = boot.poll({"health": "h", "ready": "r"}, started, 60,
                        probe_fn=fake_probe, clock=clock, sleep=clock.sleep)

    # From the container's start to the response, not from the first poll.
    assert results["health"]["seconds"] == pytest.approx(22.1)
    assert results["ready"]["seconds"] == pytest.approx(24.1)
    assert results["ready"]["status"] == 200
    # An answered URL is not probed again; ticks stay one second apart.
    assert calls == {"h": 3, "r": 5}
    assert clock.sleeps == [0.8, 0.8, 0.8, 0.9]
    assert boot.POLL_INTERVAL_SECONDS == 1


def test_poll_gives_up_at_the_timeout():
    started = _at(0)
    clock = FakeClock(started)

    def refused(url):
        clock.advance(0.1)
        return 0, url

    results = boot.poll({"health": "h"}, started, 3, probe_fn=refused, clock=clock, sleep=clock.sleep)
    assert results["health"]["seconds"] is None
    assert results["health"]["status"] == 0
    assert clock.now - started < 4 * NS


def test_wait_appends_a_record_timed_from_the_container_start(tmp_path, monkeypatch):
    started_at = "2026-10-16T12:44:21.98Z"
    monkeypatch.setattr(boot, "container_started_at", lambda container: started_at)
    monkeypatch.setattr(boot, "probe", lambda url: (200, url))
    monkeypatch.setattr(boot, "fetch_json", lambda url: {"stages": [], "total_duration_ms": 5})
    out = tmp_path / "boots.jsonl"

    code = boot.main([
        "--label", "fresh", "--container", BACKEND,
        "--url", "health=http://x/health", "--url", "ready=http://x/health/ready",
        "--fetch", "bootstrap=http://x/health/bootstrap", "--meta", "note=a=b",
        "--timeout", "5", "--out", str(out),
    ])

    assert code == 0
    (record,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert record["started_at"] == record["started_at_after"] == started_at
    assert set(record["probes"]) == {"health", "ready"}
    assert record["probes"]["ready"]["seconds"] > 0
    assert record["fetched"] == {"bootstrap": {"stages": [], "total_duration_ms": 5}}
    assert record["meta"] == {"note": "a=b"}
    assert record["ok"] is True


def test_wait_fails_when_a_url_never_answers_200(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "container_started_at", lambda container: "2026-10-16T12:44:21Z")
    monkeypatch.setattr(boot, "probe", lambda url: (503, url))
    out = tmp_path / "boots.jsonl"

    code = boot.main(["--label", "warm-1", "--container", BACKEND, "--url", "ready=http://x/health/ready",
                      "--fetch", "bootstrap=http://x/health/bootstrap", "--timeout", "0", "--out", str(out)])

    assert code == 1
    record = json.loads(out.read_text())
    assert record["ok"] is False
    assert record["probes"]["ready"] == {"url": "http://x/health/ready", "status": 503,
                                         "final_url": "http://x/health/ready", "seconds": None}
    assert record["fetched"] == {}


def test_wait_refuses_a_container_that_never_started(tmp_path, monkeypatch):
    monkeypatch.setattr(boot, "container_started_at", lambda container: "0001-01-01T00:00:00Z")
    out = tmp_path / "boots.jsonl"
    assert boot.main(["--label", "fresh", "--container", BACKEND, "--url", "health=http://x", "--out", str(out)]) == 1
    assert not out.exists()


def test_wait_for_start_waits_for_a_container_another_process_starts():
    clock = FakeClock(_at(0))
    answers = iter([None, "0001-01-01T00:00:00Z", _started_at(_at(3))])
    assert boot.wait_for_start(BACKEND, 60, read=lambda name: next(answers), clock=clock, sleep=clock.sleep)
    assert clock.sleeps == [1, 1], "polled once a second until it started"


def test_wait_for_start_stops_when_the_starter_ended_or_time_ran_out(tmp_path):
    clock = FakeClock(_at(0))
    ended = tmp_path / "make-up.exit"
    ended.write_text("2\n")
    assert not boot.wait_for_start(BACKEND, 60, ended, read=lambda name: None, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == [], "a start that will never come is not waited for"
    assert not boot.wait_for_start(BACKEND, 3, read=lambda name: None, clock=clock, sleep=clock.sleep)
    assert boot.wait_for_start(BACKEND, 60, ended, read=lambda name: _started_at(_at(1)), clock=clock,
                               sleep=clock.sleep), "started before its starter ended"


def test_wait_with_wait_start_times_from_the_start_and_records_the_published_address(tmp_path, monkeypatch):
    started_at = "2026-10-16T12:44:21.98Z"
    monkeypatch.setattr(boot, "wait_for_start", lambda container, timeout, stop_file: True)
    monkeypatch.setattr(boot, "container_started_at", lambda container: started_at)
    monkeypatch.setattr(boot, "probe", lambda url: (200, url))
    monkeypatch.setattr(boot, "published_address", lambda container, port: f"127.0.0.1:{port}")
    out = tmp_path / "boots.jsonl"
    code = boot.main(["--label", "frontend", "--container", "automatos_frontend", "--url", "ui=http://x/",
                      "--wait-start", "--stop-file", str(tmp_path / "exit"), "--published-port", "3000",
                      "--timeout", "5", "--out", str(out)])
    assert code == 0
    assert json.loads(out.read_text())["meta"] == {"address": "127.0.0.1:3000"}


def test_wait_with_wait_start_fails_when_the_container_never_starts(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(boot, "wait_for_start", lambda container, timeout, stop_file: False)
    out = tmp_path / "boots.jsonl"
    assert boot.main(["--label", "fresh", "--container", BACKEND, "--url", "health=http://x", "--wait-start",
                      "--timeout", "5", "--out", str(out)]) == 1
    assert "did not start within 5s, or the process that starts it ended first" in capsys.readouterr().out
    assert not out.exists()


def test_wait_rejects_a_url_without_a_name(tmp_path):
    with pytest.raises(SystemExit):
        boot.main(["--label", "fresh", "--container", BACKEND, "--url", "http://x", "--out", str(tmp_path / "o")])


# --------------------------------------------------------------------------- smoke_report


def _bootstrap(database_ms: int) -> dict:
    return {
        "stages": [
            {"stage": "database_init", "status": "success", "duration_ms": database_ms, "error": None},
            {"stage": "scheduler_init", "status": "failed", "duration_ms": 12, "error": "boom"},
        ],
        "total_duration_ms": database_ms + 12,
    }


def _record(label: str, container: str, start_s: float, probes: dict, fetched=None, meta=None) -> dict:
    started_at = _started_at(_at(start_s))
    return {
        "label": label,
        "container": container,
        "started_at": started_at,
        "started_at_after": started_at,
        "probes": {name: {"url": f"http://x/{name}", "status": 200, "final_url": final, "seconds": seconds}
                   for name, (seconds, final) in probes.items()},
        "fetched": fetched or {},
        "meta": meta or {},
        "ok": True,
    }


RECORDS = [
    _record("fresh", BACKEND, 6.0, {"health": (35.0, "h"), "ready": (35.0, "r")},
            fetched={"bootstrap": _bootstrap(3545)}),
    _record("frontend", FRONTEND, 45.0, {"ui": (3.2, "http://localhost:3000/chat")},
            meta={"address": "0.0.0.0:3000"}),
    _record("warm-1", BACKEND, 100.0, {"health": (18.0, "h"), "ready": (18.5, "r")},
            fetched={"bootstrap": _bootstrap(1200)}),
    _record("warm-2", BACKEND, 200.0, {"health": (22.0, "h"), "ready": (23.5, "r")},
            fetched={"bootstrap": {"error": "HTTP 404"}}),
]


def _event(service, action: str, at_s: float) -> dict:
    attributes = {"name": f"automatos_{service}"}
    if service:
        attributes["com.docker.compose.service"] = service
    return {"Type": "container", "Action": action, "Actor": {"ID": "c0ffee", "Attributes": attributes},
            "time": T0 + int(at_s), "timeNano": _at(at_s)}


EVENTS = [  # out of order on purpose: the report sorts by time
    _event("backend", "start", 100.0),
    _event("postgres", "start", 0.0),
    _event("postgres", "health_status: healthy", 5.5),
    _event("redis", "start", 0.0),
    _event("redis", "health_status: healthy", 5.6),
    _event("minio", "start", 0.2),
    _event("minio", "health_status: healthy", 6.1),
    _event("backend", "start", 6.0),
    _event("backend", "exec_start: docker-healthcheck.sh", 30.0),
    _event("backend", "health_status: healthy", 41.0),
    _event("frontend", "start", 45.0),
    _event("frontend", "health_status: healthy", 48.0),
    _event("backend", "health_status: healthy", 118.0),
    _event("backend", "start", 200.0),
    _event(None, "start", 1.0),  # no compose label: not ours
]


def _log(at_s: float, text: str) -> str:
    return f"{_log_stamp(_at(at_s))} {text}"


LOG_LINES = [
    _log(6.5, "Automatos AI Backend Starting..."),
    _log(7.0, CONTEXT_IMPL),  # init_fresh_db's replay: one per revision
    _log(7.1, CONTEXT_IMPL),
    _log(7.2, CONTEXT_IMPL),
    _log(8.0, "init_fresh_db: done — 167 tables, alembic at heads."),
    _log(9.0, "🗄️  Running database migrations (alembic upgrade heads)..."),
    _log(9.5, CONTEXT_IMPL),  # the entrypoint's alembic
    _log(30.0, CONTEXT_IMPL),  # the CMD's alembic
    "a line with no timestamp",
    _log(101.0, CONTEXT_IMPL),
    _log(105.0, CONTEXT_IMPL),
    _log(201.0, CONTEXT_IMPL),
]


def _write_run(tmp_path: Path) -> list[str]:
    results, events, log = tmp_path / "boots.jsonl", tmp_path / "events.jsonl", tmp_path / "backend.log"
    results.write_text("".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8")
    events.write_text("".join(json.dumps(event) + "\n" for event in EVENTS) + '{"cut off', encoding="utf-8")
    log.write_text("\n".join(LOG_LINES) + "\n", encoding="utf-8")
    return ["--results", str(results), "--events", str(events), "--backend-log", str(log)]


def test_alembic_runs_count_only_after_the_fresh_init_replay():
    replay = [CONTEXT_IMPL] * 200 + ["init_fresh_db: done — 167 tables, alembic at heads."]
    assert report.alembic_runs(replay) == {"runs": 0, "context_impl_lines": 200, "init_fresh_db": True}
    assert report.alembic_runs(replay + [CONTEXT_IMPL]) == {"runs": 1, "context_impl_lines": 201, "init_fresh_db": True}
    assert report.alembic_runs([CONTEXT_IMPL, "x", CONTEXT_IMPL])["runs"] == 2


def test_report_writes_the_summary_and_the_json(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef")
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    summary, out = tmp_path / "summary.md", tmp_path / "smoke-report.json"

    code = report.main(_write_run(tmp_path) + [
        "--build-seconds", "312", "--outcome", "pass", "--summary", str(summary), "--json", str(out),
    ])

    assert code == 0
    data = json.loads(out.read_text())
    assert [boot_row["label"] for boot_row in data["backend"]] == ["fresh", "warm-1", "warm-2"]
    assert data["backend"][0]["alembic"] == {"runs": 2, "context_impl_lines": 5, "init_fresh_db": True}
    assert data["backend"][1]["alembic"] == {"runs": 2, "context_impl_lines": 2, "init_fresh_db": False}
    assert data["backend"][2]["alembic"]["runs"] == 1
    assert data["warm_median"] == {"restarts": 2, "health_s": 20.0, "ready_s": 21.0}
    assert data["frontend"]["address"] == "0.0.0.0:3000"
    assert data["frontend"]["ui_s"] == 3.2
    healthy = {(row["service"], row["start"]): row["healthy_s"] for row in data["services"]}
    assert healthy == {("postgres", 1): 5.5, ("redis", 1): 5.6, ("minio", 1): 5.9, ("backend", 1): 35.0,
                       ("frontend", 1): 3.0, ("backend", 2): 18.0, ("backend", 3): None}
    # First datastore start (0.0) to the last one healthy (minio, 6.1).
    assert data["datastores"] == {"services": ["postgres", "redis", "minio"], "healthy_s": 6.1, "missing": []}
    assert (data["commit"], data["run_id"], data["build_seconds"], data["outcome"]) == (
        "0123456789abcdef", "42", 312, "pass")

    text = summary.read_text()
    for expected in (
        "Result: pass", "`0123456`", "Image build: 312 s",
        "`/health` 200 | `/health/ready` 200",
        "| fresh |", "| 35.0 s | 35.0 s | 2 | 5 | ran |",
        "| **warm median** (2 restarts) | | 20.0 s | 21.0 s |",
        "`:3000` answered 200 3.2 s after `automatos_frontend` started (final URL http://localhost:3000/chat)",
        "`docker compose port frontend 3000`: `0.0.0.0:3000`",
        "| postgres | 1 |", "| 5.5 s |", "never healthy",
        "Datastores (postgres, redis, minio): all healthy 6.1 s after the first of them started",
        "| Stage | fresh | warm-1 |", "| database_init | 3545 | 1200 |", "| scheduler_init | failed 12 | failed 12 |",
        "| **total** | 3557 | 1212 |", "Not available: warm-2: HTTP 404",
    ):
        assert expected in text, expected


def test_report_still_writes_a_summary_when_the_run_failed_early(tmp_path):
    results = tmp_path / "boots.jsonl"
    results.write_text("", encoding="utf-8")
    summary, out = tmp_path / "summary.md", tmp_path / "smoke-report.json"

    code = report.main([
        "--results", str(results), "--events", str(tmp_path / "missing-events.jsonl"),
        "--backend-log", str(tmp_path / "missing.log"), "--build-seconds", "",
        "--outcome", "fail", "--failure", "docker compose build failed",
        "--summary", str(summary), "--json", str(out),
    ])

    assert code == 0
    text = summary.read_text()
    for expected in ("**FAIL**: docker compose build failed", "Image build: — s", "No backend boot was timed.",
                     "The frontend was not started.", "No container events were recorded."):
        assert expected in text, expected
    data = json.loads(out.read_text())
    assert data["build_seconds"] is None
    assert data["backend"] == [] and data["frontend"] is None
    assert data["datastores"]["healthy_s"] is None
    assert "Datastores (postgres, redis, minio): not all healthy; missing postgres, redis, minio." in text


def test_the_datastore_gate_uses_first_starts_and_waits_for_every_datastore():
    spans = report.service_spans([
        _event("postgres", "start", 0.0), _event("postgres", "health_status: healthy", 5.5),
        _event("redis", "start", 0.1), _event("redis", "health_status: healthy", 4.0),
        _event("minio", "start", 0.3), _event("minio", "health_status: healthy", 6.3),
        _event("postgres", "start", 50.0), _event("postgres", "health_status: healthy", 70.0),  # a later restart
        _event("backend", "start", 6.4),
    ])
    assert report.datastore_gate(spans) == {"services": ["postgres", "redis", "minio"], "healthy_s": 6.3, "missing": []}

    never = report.service_spans([_event("postgres", "start", 0.0), _event("postgres", "health_status: healthy", 5.5),
                                  _event("redis", "start", 0.0), _event("minio", "start", 0.0),
                                  _event("minio", "health_status: healthy", 2.0)])
    assert report.datastore_gate(never) == {"services": ["postgres", "redis", "minio"], "healthy_s": None,
                                            "missing": ["redis"]}


def test_a_backend_that_restarted_mid_boot_is_flagged(tmp_path):
    record = _record("fresh", BACKEND, 6.0, {"health": (35.0, "h"), "ready": (35.0, "r")})
    record = {**record, "started_at_after": _started_at(_at(50.0))}
    (row,) = report.backend_boots([record], [])
    assert row["restarted_while_booting"] is True
    assert "(restarted while booting)" in "\n".join(report.render_backend([row], report.warm_median([row])))


# --------------------------------------------------------------------------- the script


def _script() -> str:
    return _SCRIPT.read_text(encoding="utf-8")


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_script_times_boots_from_the_container_start_not_its_own_sleeps():
    text = _script()
    assert boot.STARTED_AT_FORMAT == "{{.State.StartedAt}}"
    assert "POLL_INTERVAL_SECONDS=5" not in text
    assert not re.search(r"^\s*sleep\b", text, re.M), "boots are timed by smoke_boot.py, never by the script's sleeps"
    assert 'BOOT_TOOL="scripts/ci/smoke_boot.py"' in text
    assert f'--url "{boot.HEALTH}=${{HEALTH_URL}}" --url "{boot.READY}=${{READY_URL}}"' in text
    assert f'--fetch "{boot.BOOTSTRAP}=${{BOOTSTRAP_URL}}"' in text


def test_script_restarts_the_backend_twice_with_the_database_kept():
    text = _script()
    assert re.search(r"^WARM_RESTARTS=2$", text, re.M)
    warm = _function(text, "warm_restarts")
    assert "compose restart backend" in warm
    assert 'wait_backend "warm-${i}"' in warm
    # The only `down -v` is the teardown after every boot, so the warm restarts keep the database.
    assert text.count("down -v") == 1
    assert "compose down -v" in _function(text, "finish")


def test_script_starts_the_frontend_and_records_its_address():
    text = _script()
    frontend = _function(text, "boot_frontend")
    # make build and make up start it once the backend is healthy (PRD-262 US-039, US-036);
    # the prebuilt images' run starts it here.
    assert 'if [ -n "$MAKE_TARGET" ]; then\n    boot_frontend_from_make' in frontend
    assert "compose up -d --no-build frontend" in frontend
    assert 'address="$(compose port frontend 3000)"' in frontend
    assert f'--url "{boot.UI}=${{FRONTEND_URL}}" --meta "address=${{address}}"' in frontend
    assert "--published-port 3000" in _function(text, "boot_frontend_from_make")
    assert re.search(r'^FRONTEND_URL="http://localhost:\$\{FRONTEND_PORT:-3000\}/"$', text, re.M)


def test_script_writes_the_report_on_every_exit():
    text = _script()
    assert "trap finish EXIT" in _function(text, "main")
    assert "write_report" in _function(text, "finish")
    writer = _function(text, "write_report")
    assert '"$REPORT_TOOL"' in writer
    assert '--summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}"' in writer
    assert 'docker logs -t "$BACKEND_CONTAINER"' in writer
    assert 'REPORT_TOOL="scripts/ci/smoke_report.py"' in text


def _call_at(body: str, call: str) -> int:
    match = re.search(rf"^[ \t]+{re.escape(call)}$", body, re.M)
    assert match, f"main() does not run `{call}` on a line of its own"
    return match.start()


def test_script_checks_the_host_before_it_sets_the_teardown_trap():
    main = _function(_script(), "main")
    order = [_call_at(main, call) for call in ("refuse_existing_stack", "trap finish EXIT", "prepare_images")]
    assert order == sorted(order)
    assert re.search(r"^STACK_NAME_PATTERN='\^automatos_'$", _script(), re.M)


# A docker that records its calls and lists FAKE_CONTAINERS and FAKE_VOLUMES, and a make
# that records its call and fails, as a make build whose build failed would; python3 is
# the interpreter running these tests.
_FAKE_DOCKER = """#!/bin/bash
echo "docker $*" >> "$CALL_LOG"
case "$*" in
  "ps -a --format"*) [ -z "${FAKE_CONTAINERS:-}" ] || printf '%s\\n' $FAKE_CONTAINERS ;;
  "volume ls --format"*) [ -z "${FAKE_VOLUMES:-}" ] || printf '%s\\n' $FAKE_VOLUMES ;;
esac
exit 0
"""
_FAKE_MAKE = """#!/bin/bash
echo "make $*" >> "$CALL_LOG"
echo "make-tag ${AUTOMATOS_IMAGE_TAG-unset} make-revision ${AUTOMATOS_REVISION-unset}" >> "$CALL_LOG"
exit 1
"""


def _run_script(tmp_path: Path, **fake: str) -> tuple[int, list[str], str]:
    """Run smoke-fresh-clone.sh against the fake docker; (exit code, docker calls, output)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stubs = {"docker": _FAKE_DOCKER, "make": _FAKE_MAKE, "python3": f'#!/bin/bash\nexec "{sys.executable}" "$@"\n'}
    for name, body in stubs.items():
        (bin_dir / name).write_text(body, encoding="utf-8")
        (bin_dir / name).chmod((bin_dir / name).stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    log.touch()
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "CALL_LOG": str(log),
        "SMOKE_OUT_DIR": str(tmp_path / "out"),
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        **fake,
    }
    proc = subprocess.run(["bash", str(_SCRIPT)], env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=60)
    return proc.returncode, log.read_text(encoding="utf-8").splitlines(), proc.stdout + proc.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("fake", [
    {"FAKE_VOLUMES": "other_volume automatos_postgres_data"},
    {"FAKE_CONTAINERS": "automatos_backend"},
])
def test_script_refuses_a_host_with_its_own_automatos_stack(tmp_path, fake):
    code, calls, out = _run_script(tmp_path, **fake)
    assert code == 1, out
    found = [name for name in " ".join(fake.values()).split() if name.startswith("automatos_")]
    assert f"already has Automatos containers or volumes ({found[0]})" in out
    # It stops before building, pulling or tearing anything down, and writes no report.
    assert calls == ["docker ps -a --format {{.Names}}", "docker volume ls --format {{.Name}}"]
    assert not (tmp_path / "summary.md").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_script_reports_and_tears_down_when_make_build_fails(tmp_path):
    """The source build is make build itself (PRD-262 US-039): its failure fails the run."""
    code, calls, out = _run_script(tmp_path)
    assert code == 1, out
    assert "make build" in calls, "the smoke runs make build, not its own compose build"
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO, capture_output=True, text=True).stdout.strip()
    if head:  # the build is tagged local-<sha7> and labelled with HEAD, exported for make build
        assert f"make-tag local-{head[:7]} make-revision {head}" in calls
    docker = [call for call in calls if call.startswith("docker ")]
    assert not any(" build" in call or " up " in call for call in docker), "the smoke itself builds and starts nothing"
    assert calls[-1] == "docker compose -f docker-compose.yml down -v"
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "**FAIL**: make build failed" in summary
    assert "No backend boot was timed." in summary
    report_json = json.loads((tmp_path / "out" / "smoke-report.json").read_text(encoding="utf-8"))
    assert (report_json["outcome"], report_json["failure"]) == ("fail", "make build failed")


# --------------------------------------------------------------------------- the workflow


def _workflow() -> dict:
    return yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict:
    data = _workflow()
    return data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True


US002_PATHS = {
    "services/workspace-worker/**", "docker-compose.dev.yml", "Makefile", "frontend/Dockerfile",
    "orchestrator/requirements*.txt", "orchestrator/main.py", "orchestrator/core/seeds/**",
    "orchestrator/core/database/load_seed_data.py", "orchestrator/alembic/**", "envs/*.defaults",
    "frontend/package-lock.json",
    # US-022 renames requirements.txt to requirements.in plus lock files (review addendum),
    # and pybuild splits the lock with orchestrator/scripts/split_lock.py.
    "orchestrator/requirements.in", "orchestrator/requirements*.lock", "orchestrator/scripts/split_lock.py",
}
EARLIER_PATHS = {
    "docker-compose.yml", "orchestrator/docker-entrypoint.sh", "orchestrator/Dockerfile",
    "orchestrator/scripts/init_fresh_db.py", "orchestrator/scripts/init_test_db.py",
    "scripts/ci/smoke-fresh-clone.sh", ".github/workflows/smoke-fresh-clone.yml",
    "scripts/ci/smoke_boot.py", "scripts/ci/smoke_report.py",
}


def test_workflow_watches_the_boot_inputs():
    triggers = _triggers()
    push, pull = triggers["push"]["paths"], triggers["pull_request"]["paths"]
    assert set(push) == set(pull), "push and pull_request must watch the same paths"
    assert len(pull) == len(set(pull)), "a path is listed twice"
    missing = (US002_PATHS | EARLIER_PATHS) - set(pull)
    assert not missing, f"smoke-fresh-clone.yml does not watch {sorted(missing)}"


def test_workflow_runs_every_night():
    schedule = _triggers()["schedule"]
    assert schedule and all(len(entry["cron"].split()) == 5 for entry in schedule)
    assert "workflow_dispatch" in _triggers()


def test_workflow_never_cancels_a_measurement_run():
    """Only a pull request's runs share a group; every other run gets its own.

    GitHub keeps one pending run per concurrency group and cancels the older one, even
    with cancel-in-progress off, so a shared main group would drop the boot records of
    a queued nightly or main run (US-004 needs five).
    """
    concurrency = _workflow()["concurrency"]
    group = concurrency["group"]
    assert "github.event_name == 'pull_request' && github.ref || github.run_id" in group, group
    assert concurrency["cancel-in-progress"] == "${{ github.event_name == 'pull_request' }}"


def test_workflow_pins_every_action_by_sha():
    pinned = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
    steps = [step for job in _workflow()["jobs"].values() for step in job["steps"]]
    uses = [step["uses"] for step in steps if "uses" in step]
    assert uses, "the workflow has no action steps"
    assert all(pinned.match(ref) for ref in uses), uses


def test_workflow_uploads_the_records_on_every_run():
    steps = _workflow()["jobs"]["fresh-clone-boot"]["steps"]
    run = next(step for step in steps if step.get("run") == "bash scripts/ci/smoke-fresh-clone.sh")
    upload = next(step for step in steps if "upload-artifact" in step.get("uses", ""))
    assert upload["if"] == "always()"
    assert upload["with"]["path"].rstrip("/") == run["env"]["SMOKE_OUT_DIR"].rstrip("/")
