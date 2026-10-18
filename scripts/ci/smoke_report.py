#!/usr/bin/env python3
"""The fresh-clone smoke's step summary and JSON report (PRD-262 US-002).

``smoke_report.py`` reads what one smoke run left behind:

* the boot records ``scripts/ci/smoke_boot.py`` appended (one per boot: fresh, each
  warm restart, the frontend);
* the run's Docker events (``docker events --format '{{json .}}'``, ``start`` and
  ``health_status``);
* the backend's ``docker logs -t`` output.

It appends the Markdown step summary and writes one JSON file, the artifact PRD-262
US-004 takes its five-run medians from. The summary shows, pass or fail:

* per backend boot, the seconds from the container's start to ``/health`` 200 and to
  ``/health/ready`` 200, separately, and the median of the warm restarts;
* the frontend's seconds to its first 200 and the address compose published it on;
* Started → Healthy per service and per start, from the daemon's events;
* the ``/health/bootstrap`` stage table of each boot;
* the alembic runs of each boot, counted as PRD-262 §8 says: the
  ``Context impl PostgresqlImpl`` lines after ``init_fresh_db: done``, or all of them
  on a boot where init_fresh_db did not run (its replay of the migration forest logs
  one per revision).

The standard library only, like smoke_boot.py.
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _sibling(name: str):
    """``scripts/ci/<name>.py``, loaded by path: the tests load these scripts by path too."""
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # registered before exec, as the tests load them
    spec.loader.exec_module(module)
    return module


smoke_boot = _sibling("smoke_boot")

NS_PER_SECOND = smoke_boot.NS_PER_SECOND
HEALTH, READY, UI, BOOTSTRAP = smoke_boot.HEALTH, smoke_boot.READY, smoke_boot.UI, smoke_boot.BOOTSTRAP
SERVICE_LABEL = "com.docker.compose.service"
START_ACTION = "start"
HEALTHY_ACTION = "health_status: healthy"
ALEMBIC_RUN_LINE = "Context impl PostgresqlImpl"
FRESH_INIT_DONE_LINE = "init_fresh_db: done"
WARM_PREFIX = "warm"
BACKEND_CONTAINER = "automatos_backend"
FRONTEND_CONTAINER = "automatos_frontend"
DEFAULT_TITLE = "Fresh-clone smoke: boot timings (PRD-262 US-002)"
parse_timestamp = smoke_boot.parse_timestamp


def clock_time(ns: int) -> str:
    """``HH:MM:SS.s`` UTC, for the summary tables."""
    moment = datetime.fromtimestamp(ns / NS_PER_SECOND, tz=timezone.utc)
    return moment.strftime("%H:%M:%S.") + str(moment.microsecond // 100_000)


def read_json_lines(path: Optional[Path]) -> List[dict]:
    """The JSON objects of a JSON-lines file; a missing file is empty, a bad line is skipped."""
    if path is None or not path.exists():
        return []
    rows, skipped = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            skipped += 1
    if skipped:
        print(f"::warning title=Smoke report::{path.name}: skipped {skipped} line(s) that are not JSON")
    return rows


def _event_ns(event: dict) -> int:
    return int(event.get("timeNano") or int(event.get("time") or 0) * NS_PER_SECOND)


def service_spans(events: Iterable[dict]) -> List[dict]:
    """Each start of each compose service, with the time it first reported healthy.

    ``events`` are ``docker events --format '{{json .}}'`` objects. A start that never
    reached healthy keeps ``healthy_ns`` None. Events without the compose service label
    (a build container, another project) are ignored.
    """
    spans: List[dict] = []
    open_span: Dict[str, int] = {}
    for event in sorted(events, key=_event_ns):
        service = ((event.get("Actor") or {}).get("Attributes") or {}).get(SERVICE_LABEL)
        action = event.get("Action") or event.get("status") or ""
        if not service:
            continue
        if action == START_ACTION:
            open_span[service] = len(spans)
            spans.append({"service": service, "start_ns": _event_ns(event), "healthy_ns": None})
        elif action == HEALTHY_ACTION and service in open_span:
            index = open_span.pop(service)
            spans[index] = {**spans[index], "healthy_ns": _event_ns(event)}
    return spans


def split_log(lines: Iterable[str], starts_ns: Sequence[int]) -> List[List[str]]:
    """``docker logs -t`` lines cut into one slice per boot, at each boot's start.

    ``starts_ns`` is ascending. A line before the first start goes to the first boot;
    a line without a leading timestamp is dropped.
    """
    slices: List[List[str]] = [[] for _ in starts_ns]
    if not slices:
        return slices
    for line in lines:
        stamp, _, text = line.partition(" ")
        try:
            at = parse_timestamp(stamp)
        except ValueError:
            continue
        slices[max(bisect.bisect_right(starts_ns, at) - 1, 0)].append(text)
    return slices


def alembic_runs(lines: Sequence[str]) -> dict:
    """The alembic runs of one boot, counted as PRD-262 §8 says.

    ``init_fresh_db`` replays the migration forest on an empty database and logs one
    ``Context impl PostgresqlImpl`` line per revision, so only the lines after its
    ``init_fresh_db: done`` count; on a boot where it did not run, all of them count.
    """
    context = [index for index, line in enumerate(lines) if ALEMBIC_RUN_LINE in line]
    done = [index for index, line in enumerate(lines) if FRESH_INIT_DONE_LINE in line]
    counted = [index for index in context if not done or index > done[-1]]
    return {"runs": len(counted), "context_impl_lines": len(context), "init_fresh_db": bool(done)}


def _seconds(record: dict, probe_name: str) -> Optional[float]:
    return ((record.get("probes") or {}).get(probe_name) or {}).get("seconds")


def backend_boots(records: Sequence[dict], log_lines: Sequence[str]) -> List[dict]:
    """One row per backend boot, in start order, with its timings and alembic count."""
    boots = sorted(records, key=lambda record: parse_timestamp(record["started_at"]))
    slices = split_log(log_lines, [parse_timestamp(record["started_at"]) for record in boots])
    return [
        {
            "label": record["label"],
            "started_at": record["started_at"],
            "health_s": _seconds(record, HEALTH),
            "ready_s": _seconds(record, READY),
            "restarted_while_booting": record.get("started_at_after") not in (None, record["started_at"]),
            "alembic": alembic_runs(lines),
            "bootstrap": (record.get("fetched") or {}).get(BOOTSTRAP),
        }
        for record, lines in zip(boots, slices)
    ]


def _median(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return round(statistics.median(present), 3) if present else None


def warm_median(boots: Sequence[dict]) -> dict:
    """The median of the warm restarts' timings (the median of two is their mean)."""
    warm = [boot for boot in boots if boot["label"].startswith(WARM_PREFIX)]
    return {
        "restarts": len(warm),
        "health_s": _median(boot["health_s"] for boot in warm),
        "ready_s": _median(boot["ready_s"] for boot in warm),
    }


def service_rows(spans: Sequence[dict]) -> List[dict]:
    """Started → Healthy per service and per start, numbered from 1 for each service."""
    counts: Dict[str, int] = {}
    rows = []
    for span in spans:
        counts[span["service"]] = counts.get(span["service"], 0) + 1
        healthy = span["healthy_ns"]
        rows.append({
            "service": span["service"],
            "start": counts[span["service"]],
            "started_ns": span["start_ns"],
            "healthy_s": round((healthy - span["start_ns"]) / NS_PER_SECOND, 3) if healthy else None,
        })
    return rows


def frontend_row(records: Sequence[dict]) -> Optional[dict]:
    """The frontend's boot: seconds to its first 200, the final URL and the published address."""
    if not records:
        return None
    record = records[-1]
    ui = (record.get("probes") or {}).get(UI) or {}
    return {
        "label": record["label"],
        "container": record["container"],
        "ui_s": ui.get("seconds"),
        "status": ui.get("status"),
        "final_url": ui.get("final_url"),
        "address": (record.get("meta") or {}).get("address"),
    }


def build_summary(
    records: Sequence[dict],
    events: Sequence[dict],
    log_lines: Sequence[str],
    containers: Tuple[str, str],
    run: dict,
) -> dict:
    """Everything the step summary shows, as one JSON-ready dict.

    ``containers`` is (backend, frontend); ``run`` carries the title, outcome,
    failure, build_seconds, commit and run_id.
    """
    backend, frontend = containers
    boots = backend_boots([record for record in records if record.get("container") == backend], log_lines)
    return {
        **run,
        "backend": boots,
        "warm_median": warm_median(boots),
        "frontend": frontend_row([record for record in records if record.get("container") == frontend]),
        "services": service_rows(service_spans(events)),
    }


# --------------------------------------------------------------------------- Markdown


def _fmt(seconds: Optional[float]) -> str:
    return "—" if seconds is None else f"{seconds:.1f} s"


def _boot_line(boot: dict) -> str:
    alembic = boot["alembic"]
    label = boot["label"] + (" (restarted while booting)" if boot["restarted_while_booting"] else "")
    started = clock_time(parse_timestamp(boot["started_at"]))
    fresh_init = "ran" if alembic["init_fresh_db"] else "no"
    return (f"| {label} | {started} | {_fmt(boot['health_s'])} | {_fmt(boot['ready_s'])} "
            f"| {alembic['runs']} | {alembic['context_impl_lines']} | {fresh_init} |")


def render_backend(boots: Sequence[dict], median: dict) -> List[str]:
    """The backend table: each boot's seconds to /health 200 and /health/ready 200."""
    if not boots:
        return ["### Backend", "", "No backend boot was timed.", ""]
    lines = [
        "### Backend: container start to the first 200",
        "",
        "| Boot | Started (UTC) | `/health` 200 | `/health/ready` 200 | alembic runs "
        "| `Context impl` lines | init_fresh_db |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_boot_line(boot) for boot in boots]
    if median["restarts"]:
        lines.append(f"| **warm median** ({median['restarts']} restarts) | | {_fmt(median['health_s'])} "
                     f"| {_fmt(median['ready_s'])} | | | |")
    return lines + [
        "",
        "Polled every second; each time runs from the container's `State.StartedAt`. alembic runs: "
        "the `Context impl PostgresqlImpl` lines after `init_fresh_db: done`, or all of them on a "
        "boot where init_fresh_db did not run (PRD-262 §8).",
        "",
    ]


def render_frontend(row: Optional[dict]) -> List[str]:
    """The frontend's time to its first 200 and the address compose published."""
    if row is None:
        return ["### Frontend", "", "The frontend was not started.", ""]
    answered = (f"answered 200 {_fmt(row['ui_s'])} after `{row['container']}` started "
                f"(final URL {row['final_url']})" if row["ui_s"] is not None
                else f"never answered 200 (last status {row['status']})")
    return [
        "### Frontend",
        "",
        f"- `:3000` {answered}.",
        f"- `docker compose port frontend 3000`: `{row['address'] or 'not recorded'}`",
        "",
    ]


def render_services(rows: Sequence[dict]) -> List[str]:
    """Started → Healthy per service and start, in the order the containers started."""
    lines = ["### Started → Healthy (Docker's start and health_status events)", ""]
    if not rows:
        return lines + ["No container events were recorded.", ""]
    lines += ["| Service | Start | Started (UTC) | Started → Healthy |", "|---|---|---|---|"]
    lines += [
        f"| {row['service']} | {row['start']} | {clock_time(row['started_ns'])} "
        f"| {_fmt(row['healthy_s']) if row['healthy_s'] is not None else 'never healthy'} |"
        for row in rows
    ]
    return lines + [""]


def _stage_cell(stage: dict) -> str:
    duration = stage.get("duration_ms")
    return f"{duration}" if stage.get("status") == "success" else f"{stage.get('status')} {duration}"


def render_bootstrap(boots: Sequence[dict]) -> List[str]:
    """The /health/bootstrap stage table, one column per boot (milliseconds)."""
    lines = ["### `/health/bootstrap` stages (ms)", ""]
    reports = [(boot["label"], boot["bootstrap"] or {}) for boot in boots]
    usable = [(label, report) for label, report in reports if isinstance(report.get("stages"), list)]
    missing = [f"{label}: {report.get('error') or report.get('status') or 'not fetched'}"
               for label, report in reports if not isinstance(report.get("stages"), list)]
    if usable:
        cells = [{stage.get("stage"): _stage_cell(stage) for stage in report["stages"]} for _, report in usable]
        names = list(dict.fromkeys(name for column in cells for name in column))
        lines += ["| Stage | " + " | ".join(label for label, _ in usable) + " |",
                  "|---" * (len(usable) + 1) + "|"]
        lines += [f"| {name} | " + " | ".join(column.get(name, "—") for column in cells) + " |" for name in names]
        lines.append("| **total** | " + " | ".join(str(report.get("total_duration_ms")) for _, report in usable) + " |")
    if missing:
        lines += ["", "Not available: " + "; ".join(missing)]
    return lines + [""]


def render_markdown(summary: dict) -> str:
    """The whole step summary."""
    result = "pass" if summary["outcome"] == "pass" else f"**FAIL**: {summary.get('failure') or 'see the log'}"
    head = [
        f"## {summary.get('title') or DEFAULT_TITLE}",
        "",
        f"Result: {result}. Commit: `{(summary.get('commit') or 'local')[:7]}`. "
        f"Image build: {summary['build_seconds'] if summary.get('build_seconds') is not None else '—'} s "
        "(report only, no bar).",
        "",
    ]
    body = (render_backend(summary["backend"], summary["warm_median"]) + render_frontend(summary["frontend"])
            + render_services(summary["services"]) + render_bootstrap(summary["backend"]))
    return "\n".join(head + body) + "\n"


# --------------------------------------------------------------------------- commands


def _read_lines(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def report(args: argparse.Namespace) -> int:
    """Write the step summary and the JSON report from a run's records, events and log."""
    run = {
        "title": args.title,
        "outcome": args.outcome,
        "failure": args.failure or None,
        "build_seconds": args.build_seconds,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
    }
    summary = build_summary(
        read_json_lines(args.results),
        read_json_lines(args.events),
        _read_lines(args.backend_log),
        (args.backend_container, args.frontend_container),
        run,
    )
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with open(args.summary, "a", encoding="utf-8") as handle:
        handle.write(render_markdown(summary))
    return 0


def _optional_int(text: str) -> Optional[int]:
    return int(text) if text.strip() else None


def build_parser() -> argparse.ArgumentParser:
    """Where the run's records, events and log are, and where the report goes."""
    parser = argparse.ArgumentParser(description="The fresh-clone smoke's report (PRD-262 US-002).")
    parser.add_argument("--results", type=Path, required=True, help="the records smoke_boot.py appended")
    parser.add_argument("--events", type=Path, help="docker events --format '{{json .}}' output")
    parser.add_argument("--backend-log", type=Path, help="docker logs -t output of the backend")
    parser.add_argument("--backend-container", default=BACKEND_CONTAINER)
    parser.add_argument("--frontend-container", default=FRONTEND_CONTAINER)
    parser.add_argument("--build-seconds", type=_optional_int, default=None, help="empty when the build did not finish")
    parser.add_argument("--outcome", choices=("pass", "fail"), required=True)
    parser.add_argument("--failure", default="", help="why the smoke failed, empty on a pass")
    parser.add_argument("--title", default=DEFAULT_TITLE, help="the summary's heading (smoke-images sets its own)")
    parser.add_argument("--json", type=Path, help="where the JSON report goes")
    parser.add_argument("--summary", default="/dev/stdout", help="the Markdown is appended here")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    return report(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
