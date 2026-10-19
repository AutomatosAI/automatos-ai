#!/usr/bin/env python3
"""Time one boot of a container for the smoke lanes (PRD-262 US-002).

``smoke_boot.py`` polls URLs every ``POLL_INTERVAL_SECONDS`` until each answers 200.
Each URL is timed from the container's own start (``docker inspect`` ``State.StartedAt``)
to the response that first carried 200, so the number does not depend on the
poller's sleeps. It appends one JSON record per boot to ``--out``: the start time
(read again at the end, so a container that restarted mid-boot shows), each URL's
seconds to its first 200 and last status, any ``--fetch``ed JSON (the backend's
``/health/bootstrap``) and the ``--meta`` values the caller recorded. Exit 1 when a
URL never answered 200 within ``--timeout``.

    smoke_boot.py --label fresh --container automatos_backend \\
        --url health=http://localhost:8000/health --url ready=http://localhost:8000/health/ready \\
        --fetch bootstrap=http://localhost:8000/health/bootstrap --out boots.jsonl

When another process starts the container (``make up`` in the published-image smoke,
PRD-262 US-036, ``make build`` in the fresh-clone smoke, US-039), ``--wait-start`` first
waits for it to start, so the boot is timed while it runs rather than after it returns;
``--stop-file`` names the file that process writes when it ends, so a start that will
never come stops the wait at once. ``--start-timeout`` bounds that wait on its own
(``make build`` builds three images first); it defaults to ``--timeout``.
``--published-port`` records where Docker published one of the container's ports, as
``address``, once it has started.

``scripts/ci/smoke_report.py`` turns the records into the step summary. The standard
library only: the runner's own ``python3`` runs both, with no install step.
"""
from __future__ import annotations

import argparse
import http.client
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

POLL_INTERVAL_SECONDS = 1
REQUEST_TIMEOUT_SECONDS = 5
NS_PER_SECOND = 1_000_000_000
STARTED_AT_FORMAT = "{{.State.StartedAt}}"
NEVER_STARTED_PREFIX = "0001-01-01"  # Docker's zero time: the container never started
# The names smoke-fresh-clone.sh gives its ``--url`` and ``--fetch`` probes.
HEALTH, READY, UI, BOOTSTRAP = "health", "ready", "ui", "bootstrap"
_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})$")
_HTTP_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError)

Probe = Callable[[str], Tuple[int, str]]


def parse_timestamp(text: str) -> int:
    """Nanoseconds since the epoch for one of Docker's RFC 3339 times.

    ``State.StartedAt`` trims trailing zeros from its fraction and ``docker logs -t``
    pads it to nine digits, so the two are compared as integers, never as strings.
    """
    match = _TIMESTAMP.match(text.strip())
    if match is None:
        raise ValueError(f"not an RFC 3339 time: {text!r}")
    whole, fraction, zone = match.groups()
    offset = "+00:00" if zone == "Z" else zone
    seconds = int(datetime.fromisoformat(whole + offset).timestamp())
    return seconds * NS_PER_SECOND + int((fraction or "").ljust(9, "0"))


def container_started_at(container: str) -> str:
    """The container's ``State.StartedAt``, exactly as Docker reports it."""
    result = subprocess.run(
        ["docker", "inspect", "-f", STARTED_AT_FORMAT, container],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def has_started(started_at: Optional[str]) -> bool:
    """Whether ``State.StartedAt`` says the container has started (not Docker's zero time)."""
    return bool(started_at) and not started_at.startswith(NEVER_STARTED_PREFIX)


def wait_for_start(
    container: str,
    timeout_s: float,
    stop_file: Optional[Path] = None,
    read: Callable[[str], Optional[str]] = lambda name: _started_at_or_none(name, quiet=True),
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Poll once a second until ``container`` has started; False on timeout, or once
    ``stop_file`` exists (the process that starts it has ended) and it still has not."""
    deadline = clock() + int(timeout_s * NS_PER_SECOND)
    while True:
        if has_started(read(container)):
            return True
        if stop_file is not None and stop_file.exists():
            return has_started(read(container))
        if clock() >= deadline:
            return False
        sleep(POLL_INTERVAL_SECONDS)


def published_address(container: str, port: str) -> Optional[str]:
    """``docker port <container> <port>``'s first address, or None when Docker has none."""
    result = subprocess.run(["docker", "port", container, port], capture_output=True, text=True)
    lines = result.stdout.split() if result.returncode == 0 else []
    return lines[0] if lines else None


def probe(url: str) -> Tuple[int, str]:
    """One GET: the HTTP status (0 when nothing answered) and the URL that answered.

    Redirects are followed, as a browser opening the address follows them (the
    frontend's ``/`` redirects to ``/chat``).
    """
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return response.status, response.geturl()
    except urllib.error.HTTPError as exc:
        return exc.code, url
    except _HTTP_ERRORS:
        return 0, url


def fetch_json(url: str) -> dict:
    """GET ``url`` and decode its JSON object; on failure, a dict that says what failed."""
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except _HTTP_ERRORS + (ValueError,) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    return body if isinstance(body, dict) else {"error": "the answer is not a JSON object"}


def _probe_once(result: dict, started_ns: int, probe_fn: Probe, clock: Callable[[], int]) -> dict:
    """``result`` after one more probe of its URL, unless it already answered 200."""
    if result["seconds"] is not None:
        return result
    status, final_url = probe_fn(result["url"])
    seconds = round((clock() - started_ns) / NS_PER_SECOND, 3) if status == 200 else None
    return {**result, "status": status, "final_url": final_url, "seconds": seconds}


def poll(
    urls: Dict[str, str],
    started_ns: int,
    timeout_s: float,
    probe_fn: Probe = probe,
    clock: Callable[[], int] = time.time_ns,
    sleep: Callable[[float], None] = time.sleep,
) -> Dict[str, dict]:
    """Probe every URL once a second until each has answered 200 or ``timeout_s`` passes.

    A URL's ``seconds`` runs from ``started_ns`` (the container's start) to the response
    that first carried 200; it stays None while the URL has not answered 200. Ticks are
    one second apart from the first probe, whatever a slow request took.
    """
    results = {name: {"url": url, "status": 0, "final_url": url, "seconds": None} for name, url in urls.items()}
    deadline = clock() + int(timeout_s * NS_PER_SECOND)
    next_tick = clock()
    while True:
        results = {name: _probe_once(result, started_ns, probe_fn, clock) for name, result in results.items()}
        if all(result["seconds"] is not None for result in results.values()) or clock() >= deadline:
            return results
        next_tick += POLL_INTERVAL_SECONDS * NS_PER_SECOND
        sleep(max(0, next_tick - clock()) / NS_PER_SECOND)


def _pair(text: str) -> Tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep or not name:
        raise argparse.ArgumentTypeError(f"expected name=value, got {text!r}")
    return name, value


def _started_at_or_none(container: str, quiet: bool = False) -> Optional[str]:
    """``State.StartedAt``, or None when it cannot be read (``quiet``: the container may not exist yet)."""
    try:
        return container_started_at(container)
    except (subprocess.CalledProcessError, OSError) as exc:
        if not quiet:
            print(f"::warning title=Smoke boot::could not re-read {container}'s start time: {exc}")
        return None


def _say(record: dict) -> None:
    for name, result in record["probes"].items():
        if result["seconds"] is None:
            print(f"FAIL: {name} ({result['url']}) never answered 200; last status {result['status']}")
        else:
            print(f"OK: {name} ({result['url']}) answered 200 {result['seconds']:.1f}s "
                  f"after {record['container']} started [{record['label']}]")


def wait(args: argparse.Namespace) -> int:
    """Time one boot of ``--container`` and append its record to ``--out``."""
    start_timeout = args.timeout if args.start_timeout is None else args.start_timeout
    if args.wait_start and not wait_for_start(args.container, start_timeout, args.stop_file):
        print(f"::error title=Smoke boot::{args.container} did not start within {start_timeout:.0f}s, "
              "or the process that starts it ended first")
        return 1
    try:
        started_at = container_started_at(args.container)
        started_ns = parse_timestamp(started_at)
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"::error title=Smoke boot::cannot read {args.container}'s start time: {exc}")
        return 1
    if started_at.startswith(NEVER_STARTED_PREFIX):
        print(f"::error title=Smoke boot::{args.container} has never started")
        return 1
    probes = poll(dict(args.url), started_ns, args.timeout, probe_fn=probe)
    answered = all(result["seconds"] is not None for result in probes.values())
    record = {
        "label": args.label,
        "container": args.container,
        "started_at": started_at,
        "started_at_after": _started_at_or_none(args.container),
        "probes": probes,
        "fetched": {name: fetch_json(url) for name, url in args.fetch or []} if answered else {},
        "meta": {**_published_meta(args), **dict(args.meta or [])},
        "ok": answered,
    }
    with open(args.out, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    _say(record)
    return 0 if answered else 1


def _published_meta(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    if not args.published_port:
        return {}
    return {"address": published_address(args.container, args.published_port)}


def build_parser() -> argparse.ArgumentParser:
    """``--label``, ``--container``, the URLs, and where the record goes."""
    parser = argparse.ArgumentParser(description="Time one boot of a container (PRD-262 US-002).")
    parser.add_argument("--label", required=True, help="fresh, warm-1, frontend, ...")
    parser.add_argument("--container", required=True)
    parser.add_argument("--url", type=_pair, action="append", required=True, help="name=url, polled until 200")
    parser.add_argument("--fetch", type=_pair, action="append", help="name=url, JSON fetched once all are 200")
    parser.add_argument("--meta", type=_pair, action="append", help="name=value, recorded as given")
    parser.add_argument("--timeout", type=float, default=300, help="seconds before giving up")
    parser.add_argument("--out", type=Path, required=True, help="JSON-lines file the record is appended to")
    parser.add_argument("--wait-start", action="store_true",
                        help="another process starts the container: wait for it (within --start-timeout) first")
    parser.add_argument("--stop-file", type=Path,
                        help="with --wait-start: the file that process writes when it ends")
    parser.add_argument("--start-timeout", type=float, default=None,
                        help="with --wait-start: seconds to wait for the start (default: --timeout)")
    parser.add_argument("--published-port", help="record `docker port <container> <port>` as meta address")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return wait(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
