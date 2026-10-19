#!/usr/bin/env python3
"""Kill one uvicorn worker of the running backend and see uvicorn replace it (PRD-262 US-008).

uvicorn 0.24 never restarted a ``--workers`` process that died: the container kept
running on the survivors (or on none). From 0.30 its multiprocess manager starts a
replacement. This step proves it on the production stage booted through compose, in
smoke-images' pull-request variant (PRD-262 US-003), after the boot and the worker
count check:

1. it reads the backend's processes from ``/proc`` inside the container, finds the
   uvicorn process and the ``--workers`` processes it spawned (children running
   multiprocessing's ``spawn_main``), and notes the container's restart count, start
   time and the PIDs its log already reports as ``Started server process [<pid>]``;
2. it kills one worker with SIGKILL, as the kernel's OOM killer would;
3. within ``--timeout`` seconds (30) it needs: a new worker under the same uvicorn
   process, the worker count back where it was, ``Started server process [<new pid>]``
   in the container's log, and ``--health-url`` answering 200 after that; and the same
   restart count and start time, so the container itself never restarted.

A single uvicorn process (one worker) has no manager to restart it, so the step
refuses to run on one: once PRD-262 US-029 takes the worker count from
WEB_CONCURRENCY (envs/api.defaults sets 1), the job sets WEB_CONCURRENCY=2 for it.

The standard library only, like the other smoke tools. Exit 0 = replaced, 1 = not.

    python3 scripts/ci/smoke_worker_restart.py --container automatos_backend \\
        --health-url http://localhost:8000/health --out "$OUT_DIR/worker-restart.json"
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.5
HEALTH_REQUEST_SECONDS = 5
MIN_WORKERS = 2
WORKER_MARK = "spawn_main"  # uvicorn starts each --workers process with multiprocessing's spawn
STARTED = re.compile(r"Started server process \[(\d+)\]")
SIGKILL = 9
# Run inside the container with its own python: every process, its parent and its
# command line, read from /proc (the slim image has no ps).
LIST_PROCESSES = """
import json, os
rows = []
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    try:
        with open("/proc/%s/stat" % name) as stat, open("/proc/%s/cmdline" % name, "rb") as cmd:
            ppid = int(stat.read().rsplit(")", 1)[1].split()[1])
            args = cmd.read().replace(b"\\0", b" ").decode(errors="replace").strip()
    except (OSError, IndexError, ValueError):
        continue
    rows.append({"pid": int(name), "ppid": ppid, "args": args})
print(json.dumps(rows))
"""


def _runs_uvicorn(args: str) -> bool:
    """``uvicorn …`` or ``python …/uvicorn …``; never the ``sh -c "… && uvicorn …"`` around it."""
    return any(os.path.basename(token) == "uvicorn" for token in args.split()[:2])


def uvicorn_workers(rows: Sequence[dict]) -> Dict[str, object]:
    """The uvicorn process (``manager``) and the PIDs of the workers it spawned.

    ``manager`` is None unless exactly one uvicorn process runs.
    """
    servers = [row for row in rows if _runs_uvicorn(row["args"])]
    if len(servers) != 1:
        return {"manager": None, "workers": [], "servers": len(servers)}
    manager = servers[0]["pid"]
    workers = sorted(row["pid"] for row in rows if row["ppid"] == manager and WORKER_MARK in row["args"])
    return {"manager": manager, "workers": workers, "servers": 1}


def started_pids(log_text: str) -> List[int]:
    """Every PID the log reports as ``Started server process [<pid>]``, in order."""
    return [int(pid) for pid in STARTED.findall(log_text)]


def pick_victim(workers: Sequence[int]) -> Optional[int]:
    """The newest worker, or None when there is no manager to replace it."""
    return max(workers) if len(workers) >= MIN_WORKERS else None


def replacement(before: Sequence[int], now: Sequence[int], victim: int, logged: Sequence[int]) -> Optional[int]:
    """The worker that replaced ``victim``: new since ``before``, logged as started, and
    the count back where it was. None until all three hold."""
    fresh = [pid for pid in now if pid not in before]
    if victim in now or len(now) != len(before) or len(fresh) != 1:
        return None
    return fresh[0] if fresh[0] in logged else None


def restart_problem(before: Dict[str, str], after: Dict[str, str]) -> Optional[str]:
    """Why the container counts as restarted (restart count or start time moved), or None."""
    if before != after:
        return f"the container restarted: {before} -> {after}"
    return None


# --------------------------------------------------------------------------- docker


def _docker(*args: str) -> str:
    return subprocess.run(
        ["docker", *args], check=True, capture_output=True, text=True, errors="replace",
    ).stdout


def container_processes(container: str) -> List[dict]:
    return json.loads(_docker("exec", container, "python3", "-c", LIST_PROCESSES))


def container_state(container: str) -> Dict[str, str]:
    fields = _docker("inspect", "--format", "{{.RestartCount}}|{{.State.StartedAt}}", container).strip()
    restarts, started_at = fields.split("|", 1)
    return {"restart_count": restarts, "started_at": started_at}


def container_log(container: str) -> str:
    """The container's log, stdout and stderr together: uvicorn logs to stderr."""
    return subprocess.run(
        ["docker", "logs", container], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace",
    ).stdout


def kill_worker(container: str, pid: int) -> None:
    _docker("exec", container, "python3", "-c", f"import os; os.kill({int(pid)}, {SIGKILL})")


def health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=HEALTH_REQUEST_SECONDS) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


# --------------------------------------------------------------------------- the step


def _replacement_now(container: str, manager: int, before: Sequence[int], victim: int) -> Optional[int]:
    now = uvicorn_workers(container_processes(container))
    if now["manager"] != manager:  # uvicorn itself went: nothing will replace the worker
        return None
    return replacement(before, now["workers"], victim, started_pids(container_log(container)))


def wait_for_replacement(container: str, manager: int, before: Sequence[int], victim: int,
                         deadline: float) -> Optional[int]:
    while time.monotonic() < deadline:
        found = _replacement_now(container, manager, before, victim)
        if found is not None:
            return found
        time.sleep(POLL_SECONDS)
    return None


def wait_for_health(url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if health_ok(url):
            return True
        time.sleep(POLL_SECONDS)
    return False


def _seconds_since(start: float) -> float:
    return round(time.monotonic() - start, 1)


def kill_and_watch(container: str, health_url: str, timeout: float, found: dict, victim: int) -> dict:
    """SIGKILL ``victim``, then wait for its logged replacement and a /health 200."""
    killed = time.monotonic()
    deadline = killed + timeout
    kill_worker(container, victim)
    new = wait_for_replacement(container, found["manager"], found["workers"], victim, deadline)
    if new is None:
        return {"replacement": None,
                "failures": [f"no new worker logged 'Started server process' within {timeout:.0f}s of killing {victim}"]}
    replaced = _seconds_since(killed)
    if not wait_for_health(health_url, deadline):
        return {"replacement": new, "replaced_seconds": replaced,
                "failures": [f"{health_url} did not answer 200 within {timeout:.0f}s of the kill"]}
    return {"replacement": new, "replaced_seconds": replaced, "healthy_seconds": _seconds_since(killed), "failures": []}


def run(container: str, health_url: str, timeout: float) -> dict:
    """Kill one worker and watch for its replacement; the report lists any failures."""
    state_before = container_state(container)
    found = uvicorn_workers(container_processes(container))
    victim = pick_victim(found["workers"])
    report = {"container": container, "manager": found["manager"], "workers_before": found["workers"],
              "victim": victim, "timeout_seconds": timeout, "state_before": state_before}
    if found["manager"] is None or victim is None:
        return {**report, "failures": [
            f"no uvicorn manager with {MIN_WORKERS}+ workers to test (uvicorn processes: {found['servers']}, "
            f"workers: {found['workers']}); a single process has no manager, so run this with WEB_CONCURRENCY=2"
        ]}
    watched = kill_and_watch(container, health_url, timeout, found, victim)
    state_after = container_state(container)
    problem = restart_problem(state_before, state_after)
    failures = watched["failures"] + ([problem] if problem else [])
    return {**report, **watched, "state_after": state_after, "failures": failures}


def summary_markdown(report: dict) -> str:
    verdict = "passed" if not report["failures"] else "FAILED"
    rows = [
        ("uvicorn manager PID", report.get("manager")),
        ("workers before", ", ".join(str(pid) for pid in report.get("workers_before", []))),
        ("killed (SIGKILL)", report.get("victim")),
        ("replacement", report.get("replacement")),
        ("replacement logged after (s)", report.get("replaced_seconds")),
        ("/health 200 after (s)", report.get("healthy_seconds")),
        ("container restarts", f"{report.get('state_before')} -> {report.get('state_after')}"),
    ]
    lines = [f"### A killed uvicorn worker is replaced (PRD-262 US-008): {verdict}", "", "| | |", "|---|---|"]
    lines += [f"| {name} | {value if value is not None else '—'} |" for name, value in rows]
    lines += [f"- {failure}" for failure in report["failures"]]
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--container", required=True)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.container, args.health_url, args.timeout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, ValueError) as error:
        detail = getattr(error, "stderr", "") or ""
        report = {"container": args.container, "failures": [f"docker could not be read: {error} {detail}".strip()]}
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
