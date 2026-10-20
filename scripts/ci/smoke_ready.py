#!/usr/bin/env python3
"""The API is ready after its core, and boots the rest off the loop: checked from the smoke.

PRD-262 US-034: the lifespan yields after phase-1 core and the trust gate, so uvicorn
listens and ``/health`` answers; the starters, the Composio catalogue, Phase 2 and
board dispatch run in the background, and ``/health/ready`` answers 200 once they
finished. Their synchronous warm-ups run in worker threads, so the event loop keeps
serving. ``smoke-fresh-clone.sh`` calls this after its last backend boot, with the
records ``smoke_boot.py`` appended (each boot's seconds to ``/health`` 200 and to
``/health/ready`` 200, timed separately, and its ``/health/bootstrap`` report) and the
backend's ``docker logs -t``. Every backend boot must show:

* the stages of its ``/health/bootstrap`` report up to and including ``trust_gate``
  (the core the server waits for) summing to ``CORE_BUDGET_MS`` or less;
* no APScheduler ``Run time of job … was missed`` warning in its log during its first
  ``WINDOW_S`` seconds (or until the next boot started, when that came sooner): a
  missed job means something held the event loop.

``--window-left`` prints the whole seconds until the last backend boot's window has
passed and exits, so the smoke can wait for it before it reads the log.

The standard library only, like the other smoke tools. Exit 0 = every boot passed.

    python3 scripts/ci/smoke_ready.py --results "$OUT_DIR/boots.jsonl" \\
        --backend-log "$OUT_DIR/backend.log"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_report = _sibling("smoke_report")

CORE_BUDGET_MS = 3000
CORE_LAST_STAGE = "trust_gate"
WINDOW_S = 60
MISSED_JOB = re.compile(r"Run time of job .* was missed")
NS_PER_SECOND = smoke_report.NS_PER_SECOND
TITLE = "Ready after the core; the rest boots off the loop (PRD-262 US-034)"


def core_stages_ms(report: Optional[dict]) -> Tuple[Optional[int], Optional[str]]:
    """The milliseconds of the report's stages up to and including ``trust_gate``.

    Returns (None, why) when the report is missing, holds no stages, or has no
    ``trust_gate`` stage.
    """
    stages = (report or {}).get("stages")
    if not isinstance(stages, list):
        reason = (report or {}).get("error") or (report or {}).get("status") or "not fetched"
        return None, f"no /health/bootstrap stages ({reason})"
    total = 0
    for stage in stages:
        total += int(stage.get("duration_ms") or 0)
        if stage.get("stage") == CORE_LAST_STAGE:
            return total, None
    return None, f"/health/bootstrap has no {CORE_LAST_STAGE} stage"


def timed_lines(lines: Sequence[str]) -> List[Tuple[int, str]]:
    """``docker logs -t`` lines as (nanoseconds, text); a line without a timestamp is dropped."""
    timed = []
    for line in lines:
        stamp, _, text = line.partition(" ")
        try:
            timed.append((smoke_report.parse_timestamp(stamp), text))
        except ValueError:
            continue
    return timed


def missed_jobs(timed: Sequence[Tuple[int, str]], start_ns: int, end_ns: int) -> List[str]:
    """The missed-job warnings logged from ``start_ns`` up to, not including, ``end_ns``."""
    return [text for at, text in timed if start_ns <= at < end_ns and MISSED_JOB.search(text)]


def _seconds(record: dict, name: str) -> Optional[float]:
    return ((record.get("probes") or {}).get(name) or {}).get("seconds")


def _backend_boots(records: Sequence[dict], container: str) -> List[dict]:
    boots = [record for record in records if record.get("container") == container]
    return sorted(boots, key=lambda record: smoke_report.parse_timestamp(record["started_at"]))


def boot_row(record: dict, timed: Sequence[Tuple[int, str]], end_ns: int) -> dict:
    """One boot's facts: its probe times, its core stages' sum and its missed jobs."""
    start_ns = smoke_report.parse_timestamp(record["started_at"])
    core_ms, core_problem = core_stages_ms((record.get("fetched") or {}).get(smoke_report.BOOTSTRAP))
    missed = missed_jobs(timed, start_ns, min(end_ns, start_ns + WINDOW_S * NS_PER_SECOND))
    return {
        "label": record["label"],
        "health_s": _seconds(record, smoke_report.HEALTH),
        "ready_s": _seconds(record, smoke_report.READY),
        "core_ms": core_ms,
        "core_problem": core_problem,
        "missed_jobs": missed,
    }


def boot_problems(row: dict) -> List[str]:
    """What one boot's facts break; empty when it passes."""
    label = row["label"]
    problems = []
    if row["core_problem"]:
        problems.append(f"{label}: {row['core_problem']}")
    elif row["core_ms"] > CORE_BUDGET_MS:
        problems.append(f"{label}: the stages up to {CORE_LAST_STAGE} took {row['core_ms']} ms, "
                        f"over the {CORE_BUDGET_MS} ms budget")
    if row["missed_jobs"]:
        problems.append(f"{label}: {len(row['missed_jobs'])} missed APScheduler job(s) in its first "
                        f"{WINDOW_S} s, the first: {row['missed_jobs'][0].strip()}")
    return problems


def check_boots(records: Sequence[dict], log_lines: Sequence[str], container: str) -> dict:
    """Every backend boot's row and the problems the rows show."""
    boots = _backend_boots(records, container)
    timed = timed_lines(log_lines)
    ends = [smoke_report.parse_timestamp(record["started_at"]) for record in boots[1:]] + [math.inf]
    rows = [boot_row(record, timed, end) for record, end in zip(boots, ends)]
    failures = [problem for row in rows for problem in boot_problems(row)]
    if not rows:
        failures.append(f"no {container} boot was timed")
    return {"budget_ms": CORE_BUDGET_MS, "window_s": WINDOW_S, "boots": rows, "failures": failures}


def window_left_s(records: Sequence[dict], container: str, now_ns: int) -> int:
    """Whole seconds until the last backend boot's first ``WINDOW_S`` seconds have passed."""
    boots = _backend_boots(records, container)
    if not boots:
        return 0
    ends_ns = smoke_report.parse_timestamp(boots[-1]["started_at"]) + WINDOW_S * NS_PER_SECOND
    return max(0, math.ceil((ends_ns - now_ns) / NS_PER_SECOND))


def _cell(value) -> str:
    return "—" if value is None else f"{value}"


def summary_markdown(result: dict) -> str:
    verdict = "passed" if not result["failures"] else "FAILED"
    lines = [
        f"### {TITLE}: {verdict}",
        "",
        f"| Boot | `/health` 200 (s) | `/health/ready` 200 (s) | stages to {CORE_LAST_STAGE} (ms, "
        f"≤ {result['budget_ms']}) | missed jobs, first {result['window_s']} s |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {row['label']} | {_cell(row['health_s'])} | {_cell(row['ready_s'])} | "
              f"{_cell(row['core_ms'])} | {len(row['missed_jobs'])} |" for row in result["boots"]]
    lines += [""] + [f"- **FAIL**: {failure}" for failure in result["failures"]]
    return "\n".join(lines) + "\n\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--results", type=Path, required=True, help="the records smoke_boot.py appended")
    parser.add_argument("--backend-log", type=Path, help="docker logs -t output of the backend")
    parser.add_argument("--backend-container", default=smoke_report.BACKEND_CONTAINER)
    parser.add_argument("--window-left", action="store_true",
                        help="print the seconds until the last boot's window has passed, and exit")
    parser.add_argument("--out", type=Path, help="where the JSON record goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    records = smoke_report.read_json_lines(args.results)
    if args.window_left:
        print(window_left_s(records, args.backend_container, time.time_ns()))
        return 0
    if args.backend_log is None:
        print("FAIL: --backend-log is required to check the boots", file=sys.stderr)
        return 1
    log_lines = args.backend_log.read_text(encoding="utf-8", errors="replace").splitlines()
    result = check_boots(records, log_lines, args.backend_container)
    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as handle:
        handle.write(summary_markdown(result))
    for failure in result["failures"]:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
