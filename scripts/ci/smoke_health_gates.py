#!/usr/bin/env python3
"""The health gates in seconds, from Docker's own timestamps (PRD-262 US-038).

docker-compose.yml probes postgres, redis, minio, the backend, the workspace worker and
media-render every second while each one starts (``start_interval: 1s``, Docker Engine
25 or later), so a service that waits on another starts within seconds of it being
ready. The smoke lanes stream Docker's ``start`` and ``health_status`` events to a
JSON-lines file (``docker events --format '{{json .}}'``) and run this on it once the
frontend answers:

* redis and minio each turn healthy within ``DATASTORE_BAR_SECONDS`` of every start
  in the file;
* postgres is reported, not gated: on the smoke's fresh volume its TCP probe passes
  only after initdb's restart, and Gerard sets its bar in PRD-262 §8 from this number;
* with ``--frontend gate``, the frontend's first start comes within
  ``FRONTEND_BAR_SECONDS`` of the backend's last ``healthy`` before it. Only compose
  starts the frontend that way, the moment the backend turns healthy (``make up``). A
  lane that starts the frontend itself, after checks of its own, passes
  ``--frontend report``: the gap is shown, not gated.

Every time is the difference between two of the daemon's event timestamps
(``timeNano``), never this script's clock. It writes one JSON record to ``--out`` and a
table to ``--summary``. The standard library only, like the other smoke tools.
Exit 0 = every gate holds; 1 = a gate missed its bar, or the events it needs are missing.

    python3 scripts/ci/smoke_health_gates.py --events "$OUT_DIR/events.jsonl" \\
        --frontend gate --out "$OUT_DIR/health-gates.json"
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


report = _sibling("smoke_report")  # the event reader and Started → Healthy spans, shared with US-002

NS_PER_SECOND = report.NS_PER_SECOND
# PRD-262 §8 and US-038: redis and minio healthy at most 2 s after they start.
DATASTORE_BAR_SECONDS = 2.0
GATED_DATASTORES = ("redis", "minio")
# Reported only: its bar is set in §8 from the W5 number.
REPORTED_DATASTORES = ("postgres",)
# PRD-262 US-038: the frontend starts at most 3 s after the backend turns healthy.
FRONTEND_BAR_SECONDS = 3.0
BACKEND, FRONTEND = "backend", "frontend"
GATE, REPORT = "gate", "report"
TITLE = "Health gates (PRD-262 US-038, Docker's start and health_status events)"


def datastore_rows(spans: Sequence[dict]) -> List[dict]:
    """Started → Healthy for every start of each datastore, with its bar (None: reported).

    A datastore that never started gets one row with ``start`` None.
    """
    rows = report.service_rows(spans)
    found = []
    for service in GATED_DATASTORES + REPORTED_DATASTORES:
        bar = DATASTORE_BAR_SECONDS if service in GATED_DATASTORES else None
        starts = [{**row, "bar_s": bar} for row in rows if row["service"] == service]
        found += starts or [{"service": service, "start": None, "started_ns": None, "healthy_s": None, "bar_s": bar}]
    return found


def datastore_failures(rows: Sequence[dict]) -> List[str]:
    """Each gated start that never turned healthy, or did so after its bar."""
    failures = []
    for row in rows:
        if row["bar_s"] is None:
            continue
        name = f"{row['service']} (start {row['start']})"
        if row["start"] is None:
            failures.append(f"{row['service']} never started")
        elif row["healthy_s"] is None:
            failures.append(f"{name} never reported healthy")
        elif row["healthy_s"] > row["bar_s"]:
            failures.append(f"{name} turned healthy {row['healthy_s']:.1f} s after it started "
                            f"(bar: {row['bar_s']:g} s)")
    return failures


def frontend_gap(spans: Sequence[dict]) -> dict:
    """Seconds from the backend's last ``healthy`` to the frontend's first start after it."""
    frontend = next((span for span in spans if span["service"] == FRONTEND), None)
    if frontend is None:
        return {"seconds": None, "problem": "the frontend never started"}
    healthy = [span["healthy_ns"] for span in spans
               if span["service"] == BACKEND and span["healthy_ns"] is not None
               and span["healthy_ns"] <= frontend["start_ns"]]
    if not healthy:
        return {"seconds": None, "problem": "the backend never reported healthy before the frontend started"}
    seconds = round((frontend["start_ns"] - max(healthy)) / NS_PER_SECOND, 3)
    return {"seconds": seconds, "problem": None}


def frontend_check(spans: Sequence[dict], mode: str) -> dict:
    """The frontend gap, gated or reported, and its failure when gated."""
    gap = frontend_gap(spans)
    gated = mode == GATE
    failure = None
    if gated and gap["problem"]:
        failure = gap["problem"]
    elif gated and gap["seconds"] > FRONTEND_BAR_SECONDS:
        failure = (f"the frontend started {gap['seconds']:.1f} s after the backend turned healthy "
                   f"(bar: {FRONTEND_BAR_SECONDS:g} s)")
    return {**gap, "gated": gated, "bar_s": FRONTEND_BAR_SECONDS if gated else None, "failure": failure}


def check(events: Sequence[dict], frontend_mode: str) -> dict:
    """The whole record: the datastore rows, the frontend gap and every failure."""
    spans = report.service_spans(events)
    datastores = datastore_rows(spans)
    frontend = frontend_check(spans, frontend_mode)
    failures = datastore_failures(datastores) + ([frontend["failure"]] if frontend["failure"] else [])
    return {"datastores": datastores, "frontend": frontend, "failures": failures}


# --------------------------------------------------------------------------- Markdown


def _seconds(value: Optional[float]) -> str:
    return "never" if value is None else f"{value:.1f} s"


def _datastore_line(row: dict) -> str:
    if row["bar_s"] is None:
        return (f"| {row['service']} | {row['start'] or '—'} | {_seconds(row['healthy_s'])} "
                "| reported; its bar comes from this number (§8) | — |")
    passed = row["healthy_s"] is not None and row["healthy_s"] <= row["bar_s"]
    return (f"| {row['service']} | {row['start'] or '—'} | {_seconds(row['healthy_s'])} "
            f"| ≤ {row['bar_s']:g} s | {'pass' if passed else '**FAIL**'} |")


def _frontend_line(frontend: dict) -> str:
    if frontend["problem"]:
        measured = frontend["problem"]
    else:
        measured = f"the frontend started {frontend['seconds']:.1f} s after the backend turned healthy"
    if not frontend["gated"]:
        return (f"Frontend: {measured}. Reported, not gated: this lane starts the frontend itself, "
                "after its own checks; the make up lanes gate it.")
    verdict = "**FAIL**" if frontend["failure"] else "pass"
    return f"Frontend: {measured} (bar: ≤ {frontend['bar_s']:g} s): {verdict}."


def render_markdown(record: dict) -> str:
    """The step summary's section: one row per datastore start, then the frontend line."""
    lines = [f"### {TITLE}", "", "| Service | Start | Started → Healthy | Bar | |", "|---|---|---|---|---|"]
    lines += [_datastore_line(row) for row in record["datastores"]]
    lines += ["", _frontend_line(record["frontend"]), ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- command


def build_parser() -> argparse.ArgumentParser:
    """Where the events are, whether the frontend is gated, and where the record goes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=Path, required=True, help="docker events --format '{{json .}}' output")
    parser.add_argument("--frontend", choices=(GATE, REPORT), default=REPORT,
                        help="gate: compose started the frontend (make up); report: this lane did")
    parser.add_argument("--out", type=Path, required=True, help="where the JSON record goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    record = check(report.read_json_lines(args.events), args.frontend)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as summary:
        summary.write(render_markdown(record))
    for failure in record["failures"]:
        print(f"::error title=health gates::{failure}")
    if not record["failures"]:
        print("OK: redis and minio turned healthy within their bar"
              + (" and the frontend started within its bar of a healthy backend" if args.frontend == GATE else ""))
    return 1 if record["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
