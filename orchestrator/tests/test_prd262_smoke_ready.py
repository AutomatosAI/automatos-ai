"""PRD-262 US-034: the smokes hold that the API is ready after its core and boots the rest
off the loop.

The lifespan yields after phase-1 core and the trust gate, so the server listens; the
rest boots in the background, its synchronous warm-ups in worker threads. The smokes
(smoke-fresh-clone and smoke-images) prove it on a real stack:

* ``scripts/ci/smoke_ready.py`` reads every backend boot's record and log: the stages of
  its ``/health/bootstrap`` report up to and including ``trust_gate`` sum to 3 s or
  less, and APScheduler logs no ``Run time of job … was missed`` warning in the boot's
  first 60 s. Its table shows ``/health`` 200 and ``/health/ready`` 200 separately;
* ``scripts/ci/smoke-fresh-clone.sh`` waits until the last boot's first 60 s are in the
  log, then runs the check.

These tests hold the parts; the lanes run them only in CI.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from core.models.bootstrap import BootstrapStage

_REPO = Path(__file__).resolve().parents[2]
_CI = _REPO / "scripts" / "ci"
_SMOKE = _CI / "smoke-fresh-clone.sh"
_WORKFLOWS = _REPO / ".github" / "workflows"
NS = 1_000_000_000
T0 = 1_792_100_000  # 2026-10-15 21:33:20 UTC, a whole epoch second
BACKEND, FRONTEND = "automatos_backend", "automatos_frontend"
MISSED = ('Run time of job "telemetry_canary_tick (trigger: interval[0:00:30], next run at: '
          '2026-10-15 21:34:20 UTC)" was missed by 0:00:05.712000')


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _CI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # before exec, as the scripts register their siblings
    spec.loader.exec_module(module)
    return module


ready = _load("smoke_ready")


def _at(offset_s: float) -> int:
    return T0 * NS + round(offset_s * 1000) * 1_000_000


def _started_at(ns: int) -> str:
    seconds, fraction = divmod(ns, NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    digits = f"{fraction:09d}".rstrip("0")
    return f"{stamp}.{digits}Z" if digits else f"{stamp}Z"


def _log(at_s: float, text: str) -> str:
    seconds, fraction = divmod(_at(at_s), NS)
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{fraction:09d}Z {text}"


def _bootstrap(core_ms=(900, 40, 2)) -> dict:
    """A /health/bootstrap report: the core stages, then the background ones."""
    names = ["database_init", "semantic_embeddings", "trust_gate"]
    stages = [{"stage": name, "status": "success", "duration_ms": ms, "error": None}
              for name, ms in zip(names, core_ms)]
    stages += [{"stage": name, "status": "success", "duration_ms": 5000, "error": None}
               for name in ("seed_system_data", "tool_sync", "scheduler_init", "ready")]
    return {"stages": stages, "failed_stages": []}


def _record(label: str, container: str, start_s: float, bootstrap=None) -> dict:
    record = {"label": label, "container": container, "started_at": _started_at(_at(start_s)),
              "probes": {"health": {"seconds": 11.2}, "ready": {"seconds": 19.8}}}
    if container == BACKEND:
        record["fetched"] = {"bootstrap": _bootstrap() if bootstrap is None else bootstrap}
    return record


RECORDS = [
    _record("fresh", BACKEND, 0.0),
    _record("frontend", FRONTEND, 30.0),
    _record("warm-1", BACKEND, 100.0),
    _record("warm-2", BACKEND, 130.0),
]


def _check(log=(), records=RECORDS) -> dict:
    return ready.check_boots(records, list(log), BACKEND)


# --------------------------------------------------------------------------- the core stages


def test_the_core_is_the_stages_up_to_and_including_the_trust_gate():
    assert ready.CORE_LAST_STAGE == BootstrapStage.TRUST_GATE.value
    assert ready.core_stages_ms(_bootstrap((900, 40, 2))) == (942, None)  # the background stages left out
    assert ready.core_stages_ms({"error": "HTTP 500"}) == (None, "no /health/bootstrap stages (HTTP 500)")
    assert ready.core_stages_ms(None) == (None, "no /health/bootstrap stages (not fetched)")
    no_gate = {"stages": [{"stage": "database_init", "duration_ms": 10}]}
    assert ready.core_stages_ms(no_gate) == (None, "/health/bootstrap has no trust_gate stage")


def test_boots_whose_core_fits_the_budget_and_miss_no_job_pass():
    result = _check()
    assert result["failures"] == []
    assert [(row["label"], row["core_ms"], row["health_s"], row["ready_s"]) for row in result["boots"]] == [
        ("fresh", 942, 11.2, 19.8), ("warm-1", 942, 11.2, 19.8), ("warm-2", 942, 11.2, 19.8)]


@pytest.mark.parametrize("core_ms,fails", [((2000, 998, 2), False), ((2000, 999, 2), True)])
def test_a_core_over_three_seconds_fails(core_ms, fails):
    records = [RECORDS[0], _record("warm-1", BACKEND, 100.0, _bootstrap(core_ms))]
    result = _check(records=records)
    problem = f"warm-1: the stages up to trust_gate took {sum(core_ms)} ms, over the 3000 ms budget"
    assert (problem in result["failures"]) is fails, result["failures"]


def test_a_boot_without_its_bootstrap_report_fails():
    records = [_record("fresh", BACKEND, 0.0, {"error": "HTTP 503"})]
    assert _check(records=records)["failures"] == ["fresh: no /health/bootstrap stages (HTTP 503)"]


# --------------------------------------------------------------------------- the missed jobs


def test_the_pattern_matches_the_warning_apscheduler_logs():
    """The pattern is held to the installed APScheduler, whose executors log the warning."""
    import apscheduler.executors.base as executors_base

    assert ready.MISSED_JOB.search(inspect.getsource(executors_base)), "APScheduler no longer logs it"
    assert ready.MISSED_JOB.search(MISSED)
    assert not ready.MISSED_JOB.search('Job "telemetry_canary_tick" executed successfully')


def test_a_missed_job_in_a_boots_first_minute_fails_and_names_the_job():
    result = _check([_log(42.0, f"WARNING:apscheduler.executors.default:{MISSED}")])
    assert len(result["failures"]) == 1
    assert result["failures"][0].startswith("fresh: 1 missed APScheduler job(s) in its first 60 s, the first: ")
    assert "telemetry_canary_tick" in result["failures"][0]


def test_a_missed_job_after_the_first_minute_or_before_the_boot_is_not_its_own():
    log = [_log(61.0, MISSED),   # fresh's process, after its first 60 s
           _log(99.0, MISSED),   # still fresh's process: warm-1 starts at 100 s
           _log(131.0, MISSED)]  # warm-2's first second, so not warm-1's
    result = _check(log)
    assert [len(row["missed_jobs"]) for row in result["boots"]] == [0, 0, 1]
    assert [failure.split(":")[0] for failure in result["failures"]] == ["warm-2"]


def test_no_backend_boot_fails():
    assert _check(records=[RECORDS[1]])["failures"] == [f"no {BACKEND} boot was timed"]


# --------------------------------------------------------------------------- the wait


def test_the_window_left_runs_from_the_last_backend_boot():
    assert ready.window_left_s(RECORDS, BACKEND, _at(130.0)) == 60
    assert ready.window_left_s(RECORDS, BACKEND, _at(150.5)) == 40  # rounded up
    assert ready.window_left_s(RECORDS, BACKEND, _at(500.0)) == 0
    assert ready.window_left_s([RECORDS[1]], BACKEND, _at(0.0)) == 0


def test_the_command_prints_the_window_left(tmp_path, capsys):
    results = tmp_path / "boots.jsonl"
    results.write_text("".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8")
    assert ready.main(["--results", str(results), "--backend-container", BACKEND, "--window-left"]) == 0
    assert capsys.readouterr().out.strip() == "0"  # the records are from 2026-10-15


@pytest.mark.parametrize("log,code,verdict", [([], 0, "passed"), ([_log(5.0, MISSED)], 1, "FAILED")])
def test_the_command_writes_its_record_and_summary(tmp_path, log, code, verdict):
    results, backend_log = tmp_path / "boots.jsonl", tmp_path / "backend.log"
    results.write_text("".join(json.dumps(record) + "\n" for record in RECORDS), encoding="utf-8")
    backend_log.write_text("\n".join(log) + "\n", encoding="utf-8")
    args = ["--results", str(results), "--backend-log", str(backend_log),
            "--out", str(tmp_path / "ready.json"), "--summary", str(tmp_path / "summary.md")]

    assert ready.main(args) == code

    record = json.loads((tmp_path / "ready.json").read_text(encoding="utf-8"))
    assert [boot["label"] for boot in record["boots"]] == ["fresh", "warm-1", "warm-2"]
    assert (record["budget_ms"], record["window_s"]) == (3000, 60)
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert f"### {ready.TITLE}: {verdict}" in summary
    assert "| `/health` 200 (s) | `/health/ready` 200 (s) |" in summary  # timed separately
    assert "| warm-1 | 11.2 | 19.8 | 942 | 0 |" in summary


# --------------------------------------------------------------------------- the wiring


def _function(text: str, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n(.*?)^\}}", text, re.M | re.S)
    assert match, f"smoke-fresh-clone.sh has no {name}() function"
    return match.group(1)


def test_the_smoke_checks_every_boot_after_the_last_one_ran_its_first_minute():
    text = _SMOKE.read_text(encoding="utf-8")
    assert 'READY_TOOL="scripts/ci/smoke_ready.py"' in text
    main = _function(text, "main")
    order = [main.index(call) for call in ("boot_behind_head", "check_seed_stages", "check_ready_after_core",
                                           'outcome="pass"')]
    assert order == sorted(order)
    check = _function(text, "check_ready_after_core")
    steps = [check.index(step) for step in ("--window-left", 'sleep "$left"', 'docker logs -t "$BACKEND_CONTAINER"',
                                            '"$READY_TOOL" --results "$BOOTS_FILE" --backend-log')]
    assert steps == sorted(steps)
    assert '--summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}"' in check
    assert check.count("|| fail ") == 3


@pytest.mark.parametrize("workflow,events", [("smoke-fresh-clone.yml", ("push", "pull_request")),
                                             ("smoke-images.yml", ("pull_request",))])
def test_both_smokes_watch_the_ready_check(workflow, events):
    data = yaml.safe_load((_WORKFLOWS / workflow).read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True))  # PyYAML reads the bare `on:` key as True
    for event in events:
        assert "scripts/ci/smoke_ready.py" in triggers[event]["paths"]


def test_the_fresh_clone_smoke_watches_the_boot_stages():
    data = yaml.safe_load((_WORKFLOWS / "smoke-fresh-clone.yml").read_text(encoding="utf-8"))
    triggers = data.get("on", data.get(True))
    for event in ("push", "pull_request"):
        assert {"orchestrator/core/boot/**", "orchestrator/core/models/bootstrap.py"} <= set(triggers[event]["paths"])
