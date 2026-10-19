#!/usr/bin/env python3
"""Compose runs the production backend: what the smoke checks about it (PRD-262 US-035).

Since US-035 ``docker-compose.yml`` builds the backend's production stage and mounts no
source, entrypoint or healthcheck over it; hot reload moved to ``make dev``
(``docker-compose.dev.yml``). ``scripts/ci/smoke-fresh-clone.sh`` runs these checks on
the running stack in every mode (the fresh-clone source build, and smoke-images'
published and pull-request images), one command per step:

* ``process``: no bind mount at /app (or under it, or over the entrypoint or the
  healthcheck), no process runs ``--reload``, and the process the entrypoint's liveness
  watchdog signals (``SERVER_PID``, the entrypoint's child) is uvicorn itself, with no
  ``sh`` in between. Read from ``docker inspect`` (the mounts and ``State.Pid``, the
  entrypoint's PID on the host) and ``docker top``, whose PIDs are the host's too;
* ``env``: a key only ``envs/api.local`` sets reached the backend, which the smoke
  recreated with ``docker compose up -d --no-deps backend``, as the night tooling does
  (TESTER, PRD-262 §9-Q1);
* ``stop``: with an SSE client connected (the board's event stream, which the server
  never ends by itself), the command after ``--`` (``docker compose … stop backend``)
  finishes in under ``--max-seconds``. uvicorn gives up on open connections after
  UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN (envs/api.defaults, PRD-262 US-029) instead of
  holding the stop until Docker's 10 s kill.

Each command writes its JSON record to ``--out`` and one line to ``--summary``. The
standard library only, like the other smoke tools. Exit 0 = the check holds, 1 = not.

    python3 scripts/ci/smoke_production_backend.py process --container automatos_backend \\
        --out "$OUT_DIR/production-backend.json"
"""
from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


images = _sibling("smoke_images")  # docker top's parser and the uvicorn test, shared with US-003

APP_DIR = "/app"
SCRIPT_MOUNTS = ("/usr/local/bin/docker-entrypoint.sh", "/usr/local/bin/docker-healthcheck.sh")
ENTRYPOINT = "docker-entrypoint.sh"
BIND = "bind"
EVENT_STREAM = "text/event-stream"
STREAM_TIMEOUT_SECONDS = 30
FIRST_READ_BYTES = 4096
STREAM_ERRORS = (http.client.HTTPException, OSError, ValueError)
TITLE = "Production backend in compose (PRD-262 US-035)"

Closer = Callable[[], None]
Opener = Callable[[str], Tuple[dict, Closer]]


# --------------------------------------------------------------------------- process


def source_mounts(mounts: Sequence[dict]) -> List[str]:
    """The bind mounts that lay the checkout over the image: /app, under it, or a script."""
    found = []
    for mount in mounts:
        target = str(mount.get("Destination", ""))
        over_code = target == APP_DIR or target.startswith(APP_DIR + "/") or target in SCRIPT_MOUNTS
        if mount.get("Type") == BIND and over_code:
            found.append(f"{mount.get('Source', '?')} -> {target}")
    return found


def uvicorn_servers(rows: Sequence[dict]) -> List[dict]:
    """The uvicorn processes whose parent is not uvicorn: one per server, whatever it spawned."""
    pids = {row["pid"] for row in rows if images._runs_uvicorn(row["args"])}
    return [row for row in rows if row["pid"] in pids and row["ppid"] not in pids]


def _parent_problem(server: dict, entrypoint_pid: int, by_pid: dict) -> Optional[str]:
    """Why uvicorn is not the entrypoint's own child, or None."""
    if server["pid"] == entrypoint_pid:
        return ("uvicorn is the container's first process: the entrypoint exec'd it, so its liveness "
                "watchdog is off (AUTOMATOS_LIVENESS_WATCHDOG=true in envs/api.defaults)")
    if server["ppid"] != entrypoint_pid:
        parent = (by_pid.get(server["ppid"]) or {}).get("args", f"process {server['ppid']}")
        return (f"uvicorn's parent is `{parent}`, not the entrypoint: SERVER_PID is that process, so the "
                "watchdog and `docker stop` signal it instead of uvicorn")
    return None


def server_chain(rows: Sequence[dict], entrypoint_pid: int) -> dict:
    """What the entrypoint's watchdog signals, and whether anything reloads.

    ``entrypoint_pid`` is the container's first process (``State.Pid``). With the watchdog
    on, the entrypoint runs the command as its child (``SERVER_PID``), which must be uvicorn.
    """
    by_pid = {row["pid"]: row for row in rows}
    entrypoint = by_pid.get(entrypoint_pid)
    servers = uvicorn_servers(rows)
    reloading = [row["args"] for row in rows if images.RELOAD_FLAG in row["args"].split()]
    failures = []
    if entrypoint is None or ENTRYPOINT not in entrypoint["args"]:
        failures.append(f"the container's first process ({entrypoint_pid}) is not {ENTRYPOINT}")
    if len(servers) == 1:
        problem = _parent_problem(servers[0], entrypoint_pid, by_pid)
        failures.extend([problem] if problem else [])
    else:
        failures.append(f"expected one uvicorn server process, found {len(servers)}")
    if reloading:
        failures.append(f"the backend runs {images.RELOAD_FLAG}, as the development stage does: {reloading[0]}")
    return {
        "entrypoint": entrypoint["args"] if entrypoint else None,
        "server": servers[0]["args"] if len(servers) == 1 else None,
        "server_parent_is_entrypoint": len(servers) == 1 and servers[0]["ppid"] == entrypoint_pid,
        "reload": bool(reloading),
        "processes": list(rows),
        "failures": failures,
    }


def check_process(inspected: dict, rows: Sequence[dict]) -> dict:
    """The backend container's mounts and processes against what US-035 requires."""
    entrypoint_pid = int((inspected.get("State") or {}).get("Pid") or 0)
    chain = server_chain(rows, entrypoint_pid)
    mounts = source_mounts(inspected.get("Mounts") or [])
    failures = [f"a bind mount lays the checkout over the image: {mount}" for mount in mounts]
    return {**chain, "source_mounts": mounts, "failures": failures + chain["failures"]}


# --------------------------------------------------------------------------- env


def printenv(container: str, key: str) -> Optional[str]:
    """``key`` in the container's environment, or None when it is not set."""
    proc = subprocess.run(["docker", "exec", container, "printenv", key], capture_output=True, text=True)
    return proc.stdout.rstrip("\n") if proc.returncode == 0 else None


def check_env(key: str, expected: str, found: Optional[str]) -> dict:
    """Whether the value the smoke wrote to envs/api.local is the one the backend has."""
    failures = []
    if found is None:
        failures.append(f"{key} is not set in the backend: envs/api.local did not reach the container that "
                        "`docker compose up -d --no-deps backend` recreated")
    elif found != expected:
        failures.append(f"{key} is {found!r} in the backend, not the value the smoke wrote to envs/api.local")
    return {"key": key, "expected": expected, "found": found, "failures": failures}


# --------------------------------------------------------------------------- stop


def open_stream(url: str) -> Tuple[dict, Closer]:
    """Connect to an SSE endpoint as the browser does and read its first event.

    The connection stays open: the second value closes it.
    """
    parts = urllib.parse.urlsplit(url)
    connection = http.client.HTTPConnection(parts.hostname or "localhost", parts.port or 80,
                                            timeout=STREAM_TIMEOUT_SECONDS)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    try:
        connection.request("GET", path, headers={"Accept": EVENT_STREAM, "Cache-Control": "no-cache"})
        response = connection.getresponse()
        first = response.read1(FIRST_READ_BYTES) if response.status == 200 else b""
    except STREAM_ERRORS:
        connection.close()
        raise

    def close() -> None:
        response.close()
        connection.close()

    lines = first.decode("utf-8", "replace").splitlines()
    return {
        "url": url,
        "status": response.status,
        "content_type": response.getheader("Content-Type", ""),
        "first_bytes": len(first),
        "first_line": lines[0] if lines else "",
    }, close


def stream_problems(stream: dict) -> List[str]:
    """Why no SSE client is connected, or nothing."""
    if stream.get("error"):
        return [f"no SSE client could connect to {stream['url']}: {stream['error']}"]
    if stream["status"] != 200 or not stream["content_type"].startswith(EVENT_STREAM):
        return [f"{stream['url']} answered {stream['status']} {stream['content_type']!r}, not an event stream"]
    if not stream["first_bytes"]:
        return [f"{stream['url']} sent no event, so no client is known to be connected"]
    return []


def stop_problems(code: int, seconds: float, max_seconds: float) -> List[str]:
    """Why the stop failed its bar, or nothing."""
    failures = [f"the stop command exited {code}"] if code else []
    if seconds >= max_seconds:
        failures.append(f"stopping the backend took {seconds:.1f}s with an SSE client connected; the bar is "
                        f"under {max_seconds:g}s (UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN in envs/api.defaults)")
    return failures


def stop_with_stream(
    url: str,
    command: Sequence[str],
    max_seconds: float,
    opener: Opener = open_stream,
    runner: Callable[[List[str]], int] = subprocess.call,
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Hold an SSE connection open, run ``command`` (the stop) and time it."""
    try:
        stream, close = opener(url)
    except STREAM_ERRORS as error:
        stream, close = {"url": url, "error": str(error)}, (lambda: None)
    record = {"stream": stream, "command": list(command), "max_seconds": max_seconds,
              "exit_code": None, "seconds": None}
    try:
        failures = stream_problems(stream)
        if failures:
            return {**record, "failures": failures}
        started = clock()
        code = runner(list(command))
        seconds = round(clock() - started, 3)
    finally:
        close()
    return {**record, "exit_code": code, "seconds": seconds,
            "failures": stop_problems(code, seconds, max_seconds)}


# --------------------------------------------------------------------------- commands


def _finish(args: argparse.Namespace, record: dict, line: str) -> int:
    """Write the record and the summary line; print the failures as annotations."""
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    failures = record["failures"]
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(f"**{TITLE}**: {line if not failures else '; '.join(failures)}\n\n")
    for failure in failures:
        print(f"::error title=smoke production backend::{failure}")
    if not failures:
        print(f"OK: {line}")
    return 1 if failures else 0


def cmd_process(args: argparse.Namespace) -> int:
    """The backend's mounts and processes."""
    inspected = images.docker_json("inspect", args.container)[0]
    rows = images.parse_top(images.docker_text("top", args.container, *images.TOP_ARGS))
    record = check_process(inspected, rows)
    return _finish(args, record, f"no bind mount over the code, no {images.RELOAD_FLAG}, and the entrypoint's "
                                 f"watchdog signals uvicorn itself (`{record['server']}`)")


def cmd_env(args: argparse.Namespace) -> int:
    """A key only envs/api.local sets, read inside the recreated backend."""
    record = check_env(args.key, args.value, printenv(args.container, args.key))
    return _finish(args, record, f"`{args.key}`, set only in envs/api.local, reached the backend that "
                                 "`docker compose up -d --no-deps backend` recreated")


def cmd_stop(args: argparse.Namespace) -> int:
    """Time the stop command after ``--`` with an SSE client connected."""
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("smoke_production_backend.py stop: give the stop command after --")
    record = stop_with_stream(args.url, command, args.max_seconds)
    seconds = record["seconds"]
    timing = f"{seconds:.1f}s" if seconds is not None else "not run"
    return _finish(args, record, f"`{' '.join(command)}` took {timing} with an SSE client connected "
                                 f"(bar: under {args.max_seconds:g}s)")


def build_parser() -> argparse.ArgumentParser:
    """One subcommand per smoke step."""
    root = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = root.add_subparsers(dest="name", required=True)
    process = commands.add_parser("process", help="no bind mount over the code, no --reload, uvicorn is SERVER_PID")
    process.add_argument("--container", required=True)
    env = commands.add_parser("env", help="a key only envs/api.local sets reached the backend")
    env.add_argument("--container", required=True)
    env.add_argument("--key", required=True)
    env.add_argument("--value", required=True)
    stop = commands.add_parser("stop", help="time the stop command with an SSE client connected")
    stop.add_argument("--url", required=True, help="the SSE endpoint the client holds open")
    stop.add_argument("--max-seconds", type=float, required=True, help="the stop must finish in under this")
    stop.add_argument("command", nargs=argparse.REMAINDER, help="-- docker compose … stop backend")
    for sub in (process, env, stop):
        sub.add_argument("--out", type=Path, required=True)
        sub.add_argument("--summary", default="/dev/stdout")
    return root


COMMANDS = {"process": cmd_process, "env": cmd_env, "stop": cmd_stop}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.name](args)


if __name__ == "__main__":
    sys.exit(main())
