#!/usr/bin/env python3
"""The workspace worker reports healthy quickly and serves /metrics (PRD-262 US-045).

The smoke (smoke-fresh-clone.sh, which smoke-images runs too) starts the worker as a
dependency of the backend. Its HEALTHCHECK probes every 2 s while it starts
(``--start-interval``), so it turns healthy within seconds of its health server
listening, not one 30 s interval later. This step needs, from the smoke's own records:

1. the worker's first start to reach ``health_status: healthy`` within
   ``--max-seconds`` (20) of that start. Both moments come from the Docker events the
   smoke streams to ``--events`` (the ``start`` and ``health_status`` events
   smoke_report.py tables), so the number never depends on when this step runs. While
   a verdict is still open (no healthy event yet, and the limit not yet past), it reads
   the file again once a second;
2. ``/metrics`` on the worker's health port to answer 200. The port is not published,
   so the request runs inside the container, with the worker's own
   ``WORKER_INTERNAL_TOKEN`` when one is set (curl sends no header for an empty one).

It writes a JSON record to ``--out`` and a short table to stdout and ``--summary``.
Exit 0 = both hold, 1 = not. The standard library only, like the other smoke tools.

    python3 scripts/ci/smoke_worker_health.py --events "$OUT_DIR/events.jsonl" \\
        --service workspace-worker --container automatos_workspace_worker \\
        --out "$OUT_DIR/worker-health.json"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path, as smoke_report.py loads smoke_boot."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_report = _sibling("smoke_report")

NS_PER_SECOND = smoke_report.NS_PER_SECOND
DEFAULT_MAX_SECONDS = 20.0
# Docker writes the healthy event a moment after the probe that passed; the verdict
# waits this long past the limit before it calls a start that never turned healthy.
GRACE_SECONDS = 2.0
POLL_SECONDS = 1.0
EXEC_TIMEOUT_SECONDS = 30
# Run by the container's own sh: its token and health port, never the runner's.
METRICS_PROBE = (
    'curl -sS -o /dev/null -w "%{http_code}" -H "X-Internal-Token: ${WORKER_INTERNAL_TOKEN:-}" '
    '"http://localhost:${WORKER_HEALTH_PORT:-8081}/metrics"'
)

Clock = Callable[[], int]


def first_start(spans: Sequence[dict], service: str) -> Optional[dict]:
    """The service's first start, from smoke_report.service_spans; None if it never started."""
    return next((span for span in spans if span["service"] == service), None)


def healthy_seconds(span: Optional[dict]) -> Optional[float]:
    """Seconds from the start to its first healthy event, or None when there was none."""
    if span is None or span["healthy_ns"] is None:
        return None
    return round((span["healthy_ns"] - span["start_ns"]) / NS_PER_SECOND, 3)


def verdict_open(span: Optional[dict], waited_from_ns: int, now_ns: int, max_seconds: float) -> bool:
    """True while the start may still turn healthy within the limit."""
    if span is not None and span["healthy_ns"] is not None:
        return False
    since = span["start_ns"] if span is not None else waited_from_ns
    return now_ns < since + int((max_seconds + GRACE_SECONDS) * NS_PER_SECOND)


def health_problem(span: Optional[dict], service: str, max_seconds: float) -> Optional[str]:
    """Why the first start does not meet the bar, or None when it does."""
    if span is None:
        return f"no start of {service} is in the smoke's Docker events"
    seconds = healthy_seconds(span)
    if seconds is None:
        return f"{service} never reported healthy after its first start"
    if seconds > max_seconds:
        return f"{service} reported healthy {seconds:.1f}s after its start; the bar is {max_seconds:.0f}s"
    return None


def metrics_problem(status: Optional[int]) -> Optional[str]:
    if status == 200:
        return None
    return f"the worker's /metrics answered {status if status is not None else 'nothing'}, not 200"


def wait_for_first_start(events: Path, service: str, max_seconds: float,
                         clock: Clock = time.time_ns, sleep: Callable[[float], None] = time.sleep) -> Optional[dict]:
    """The service's first start once its verdict is settled: healthy, or past the limit."""
    waited_from = clock()
    while True:
        span = first_start(smoke_report.service_spans(smoke_report.read_json_lines(events)), service)
        if not verdict_open(span, waited_from, clock(), max_seconds):
            return span
        sleep(POLL_SECONDS)


def metrics_status(container: str) -> Optional[int]:
    """The HTTP status of the worker's /metrics, asked from inside the container."""
    try:
        done = subprocess.run(
            ["docker", "exec", container, "sh", "-c", METRICS_PROBE],
            capture_output=True, text=True, errors="replace", timeout=EXEC_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        print(f"::warning title=Worker /metrics::docker exec could not run: {error}")
        return None
    if done.returncode != 0:
        print(f"::warning title=Worker /metrics::docker exec exited {done.returncode}: {done.stderr.strip()}")
    text = done.stdout.strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


def run(events: Path, service: str, container: str, max_seconds: float) -> dict:
    """Both checks; the report lists any failures."""
    span = wait_for_first_start(events, service, max_seconds)
    status = metrics_status(container)
    failures = [problem for problem in (health_problem(span, service, max_seconds), metrics_problem(status))
                if problem]
    return {
        "service": service,
        "container": container,
        "started": smoke_report.clock_time(span["start_ns"]) if span else None,
        "healthy_seconds": healthy_seconds(span),
        "max_seconds": max_seconds,
        "metrics_status": status,
        "failures": failures,
    }


def summary_markdown(report: dict) -> str:
    verdict = "passed" if not report["failures"] else "FAILED"
    rows = [
        ("started (UTC)", report.get("started")),
        ("Started → Healthy (s)", report.get("healthy_seconds")),
        ("bar (s)", report.get("max_seconds")),
        ("/metrics status", report.get("metrics_status")),
    ]
    lines = [f"### The workspace worker is healthy within {report['max_seconds']:.0f} s and serves /metrics "
             f"(PRD-262 US-045): {verdict}", "", "| | |", "|---|---|"]
    lines += [f"| {name} | {value if value is not None else '—'} |" for name, value in rows]
    lines += [f"- {failure}" for failure in report["failures"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--events", type=Path, required=True, help="the smoke's docker events (JSON lines)")
    parser.add_argument("--service", required=True, help="the worker's compose service name")
    parser.add_argument("--container", required=True, help="the worker's container name")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    report = run(args.events, args.service, args.container, args.max_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = summary_markdown(report)
    print(text)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(text)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
